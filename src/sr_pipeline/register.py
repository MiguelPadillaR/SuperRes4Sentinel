from pathlib import Path
import argparse
import sys
import math
import cv2
import numpy as np

# If running as module inside your project, prefer imports below.
try:
    from utils import imread, imwrite
    from constants import PROC_DIR, RES_DIR, SCALE as DEFAULT_SCALE
except Exception:
    # Fallback if run as a standalone script
    def imread(p):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        return img
    def imwrite(p, img):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
    PROC_DIR = Path('data/processed')
    RES_DIR = Path('out/res')
    DEFAULT_SCALE = 4

# -------------------------
# Feature detection & matching helpers
# -------------------------

def _sift(nfeatures: int = 5000):
    if hasattr(cv2, 'SIFT_create'):
        return cv2.SIFT_create(nfeatures=nfeatures)
    # Fallback to ORB if SIFT unavailable
    return cv2.ORB_create(nfeatures)


def detect_and_describe(img_bgr, nfeatures=5000):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Improve contrast a bit
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    feat = _sift(nfeatures)
    kps, des = feat.detectAndCompute(gray, None)
    return kps, des, isinstance(feat, cv2.ORB)


def match_descriptors(des1, des2, used_orb=False, ratio=0.75):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []
    if used_orb:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        # FLANN for SIFT (KD-Tree)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def estimate_homography(kp1, kp2, matches, ransac_thresh=3.0):
    if len(matches) < 8:
        return None, None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    return H, mask

# -------------------------
# Warping strategies
# -------------------------

def scale_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def warp_hr_to_lr_canvas(hr_bgr, H_d, lr_shape):
    """
    hr_bgr: original HR image (Hh, Wh)
    H_d: homography mapping hr_down -> lr (hr_down is HR resized to LR size)
    lr_shape: (h, w)
    Returns HR warped onto LR canvas (size == LR). For *visual QA* only.
    """
    h_lr, w_lr = lr_shape[:2]
    # Compute downscale factors from HR to LR
    sy = h_lr / hr_bgr.shape[0]
    sx = w_lr / hr_bgr.shape[1]
    Sdown = scale_matrix(sx, sy)  # maps HR_full -> HR_down (LR size)
    H_full = H_d @ Sdown
    warped = cv2.warpPerspective(hr_bgr, H_full, (w_lr, h_lr))
    return warped


def warp_hr_to_sr_canvas(hr_bgr, H_d, lr_shape, scale: int):
    """
    Produce an SR-ready registered HR: canvas size = (LR_w*scale, LR_h*scale).
    If you downscale this by `scale`, it should align with LR.
    """
    h_lr, w_lr = lr_shape[:2]
    sy = h_lr / hr_bgr.shape[0]
    sx = w_lr / hr_bgr.shape[1]
    Sdown = scale_matrix(sx, sy)           # HR_full -> HR_down (LR size)
    Sup = scale_matrix(scale, scale)       # LR -> SR canvas
    H_out = Sup @ H_d @ Sdown              # HR_full -> SR canvas
    out_size = (w_lr * scale, h_lr * scale)
    warped = cv2.warpPerspective(hr_bgr, H_out, out_size)
    return warped

# -------------------------
# Visualization helpers
# -------------------------

def draw_matches(img1, kp1, img2, kp2, matches, max_draw=80):
    matches = sorted(matches, key=lambda m: m.distance)[:max_draw]
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def overlay(a_bgr, b_bgr, alpha=0.5):
    return cv2.addWeighted(a_bgr, alpha, b_bgr, 1-alpha, 0)


def checkerboard(a, b, tile=32):
    h, w = a.shape[:2]
    out = a.copy()
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x//tile)+(y//tile)) % 2 == 0:
                out[y:y+tile, x:x+tile] = b[y:y+tile, x:x+tile]
    return out

# -------------------------
# Core alignment routine
# -------------------------

def align_pair(lr_path: Path, hr_path: Path, mode: str = 'lrspace', scale: int = DEFAULT_SCALE, out_dir: Path = None, viz: bool = True):
    """
    mode: 'lrspace' for quick visual QA (warps HR onto LR size), 'srspace' for SR-ready (warps HR onto (LR*scale)).
    Returns dict with metrics and output paths.
    """
    lr = imread(lr_path)
    hr = imread(hr_path)

    # Resize HR to LR size for feature matching
    lr_h, lr_w = lr.shape[:2]
    hr_resized = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

    # Detect & match
    kp1, des1, orb1 = detect_and_describe(hr_resized)
    kp2, des2, orb2 = detect_and_describe(lr)
    used_orb = orb1 or orb2
    matches = match_descriptors(des1, des2, used_orb=used_orb, ratio=0.75)

    if len(matches) < 10:
        raise RuntimeError(f"Too few matches ({len(matches)}) between {hr_path.name} and {lr_path.name}. Try different image, increase features, or ensure content overlap.")

    H_d, mask = estimate_homography(kp1, kp2, matches, ransac_thresh=3.0)
    if H_d is None:
        raise RuntimeError('Homography estimation failed.')

    inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = inliers / max(1, len(matches))

    # Warp according to mode
    if mode == 'lrspace':
        registered = warp_hr_to_lr_canvas(hr, H_d, lr.shape)
        # For visual diff, compare LR vs downscaled(registered) == already LR-sized
        reg_for_diff = registered
    elif mode == 'srspace':
        registered = warp_hr_to_sr_canvas(hr, H_d, lr.shape, scale=scale)
        # For visual diff, compare LR vs downscaled(registered)
        reg_for_diff = cv2.resize(registered, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError("mode must be 'lrspace' or 'srspace'")

    # Prepare output paths
    if out_dir is None:
        out_dir = PROC_DIR / 'registered'
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = lr_path.stem
    if mode == 'lrspace':
        out_img = out_dir / f'{stem}_HR_registered_LRspace.png'
    else:
        out_img = out_dir / f'{stem}_HR_registered_x{scale}.png'

    # Save main output
    imwrite(out_img, registered)

    # Visualizations
    out = {
        'out_img': out_img,
        'inliers': inliers,
        'matches': len(matches),
        'inlier_ratio': float(inlier_ratio),
        'mode': mode,
    }

    if viz:
        vis_dir = out_dir / 'viz'
        vis_dir.mkdir(parents=True, exist_ok=True)
        # Matches image (on resized HR)
        match_img = draw_matches(hr_resized, kp1, lr, kp2, matches)
        imwrite(vis_dir / f'{stem}_matches.png', match_img)

        # Overlays
        if mode == 'lrspace':
            overlay_img = overlay(lr, reg_for_diff, 0.5)
            checker_img = checkerboard(lr, reg_for_diff, 64)
        else:
            # Compare LR with downscaled registered HR
            overlay_img = overlay(lr, reg_for_diff, 0.5)
            checker_img = checkerboard(lr, reg_for_diff, 64)
        imwrite(vis_dir / f'{stem}_overlay.png', overlay_img)
        imwrite(vis_dir / f'{stem}_checker.png', checker_img)
        out['viz_matches'] = vis_dir / f'{stem}_matches.png'
        out['viz_overlay'] = vis_dir / f'{stem}_overlay.png'
        out['viz_checker'] = vis_dir / f'{stem}_checker.png'

    return out

# -------------------------
# Batch mode
# -------------------------

def batch_align(lr_dir: Path, hr_dir: Path, out_dir: Path, mode: str, scale: int, pattern: str = '*'):
    lr_dir = Path(lr_dir)
    hr_dir = Path(hr_dir)
    files = sorted([p for p in lr_dir.glob(pattern) if p.is_file()])
    results = []
    for lr_p in files:
        hr_p = (hr_dir / lr_p.name).with_suffix(lr_p.suffix)
        if not hr_p.exists():
            print(f'[WARN] Missing HR for {lr_p.name}')
            continue
        try:
            res = align_pair(lr_p, hr_p, mode=mode, scale=scale, out_dir=out_dir, viz=False)
            print(f"Aligned {lr_p.name}: inliers {res['inliers']}/{res['matches']} (ratio {res['inlier_ratio']:.2f}) → {res['out_img'].name}")
            results.append(res)
        except Exception as e:
            print(f'[FAIL] {lr_p.name}: {e}')
    return results

# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description='Register HR (Google) to LR (Sentinel) tiles.')
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--pair', nargs=2, metavar=('LR', 'HR'), help='Single pair: LR_path HR_path')
    grp.add_argument('--dirs', nargs=2, metavar=('LR_DIR', 'HR_DIR'), help='Batch over directories with matching filenames')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory (default: data/processed/registered)')
    parser.add_argument('--mode', choices=['lrspace', 'srspace'], default='lrspace', help='lrspace = visual QA; srspace = SR-ready')
    parser.add_argument('--scale', type=int, default=DEFAULT_SCALE, help='SR scale when mode=srspace')
    parser.add_argument('--viz', type=int, default=1, help='Save visualization assets (1) or not (0)')
    parser.add_argument('--pattern', type=str, default='*', help='Glob pattern for batch (e.g., "*.png")')
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (PROC_DIR / 'registered')

    if args.pair:
        lr_p, hr_p = map(Path, args.pair)
        res = align_pair(lr_p, hr_p, mode=args.mode, scale=args.scale, out_dir=out_dir, viz=bool(args.viz))
        print(f"Saved: {res['out_img']}")
        print(f"Inliers: {res['inliers']}/{res['matches']} (ratio {res['inlier_ratio']:.2f})")
    else:
        batch_align(Path(args.dirs[0]), Path(args.dirs[1]), out_dir, mode=args.mode, scale=args.scale, pattern=args.pattern)

if __name__ == '__main__':
    main()
