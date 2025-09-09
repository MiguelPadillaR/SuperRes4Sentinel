import io
from PIL import Image, ImageEnhance
import requests
import numpy as np
import argparse

import time
from datetime import datetime, timedelta

from sentinelhub import SHConfig, DataCollection, MimeType, SentinelHubRequest
from .sh_config import CONFIG_NAME, GOOGLE_MAPS_STATIC_API_KEY
from .utils import *

from src.utils.constants import *

config = SHConfig(CONFIG_NAME)

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

# ----------------------------------
# GOOGLE MAPS REQUEST (DEPRECATED)
# ----------------------------------
def download_google_image(lat, lon, size, zoom, filename):
    """
    Fetches a Google Maps satellite image for the given lat, lon, size, and zoom level,
    and saves it to the specified filename.

    Arguments:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        size (tuple): Size of the image in pixels (width, height).
        zoom (int): Zoom level for the image.
        filename (str): Filename to save the image.

    Returns:
        is_image_downloaded (bool): True if the image was downloaded and saved, False otherwise.
    """
    is_valid_image = True

    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size[0]}x{size[1]}&maptype=satellite&style=feature:all|element:labels|visibility:off&key={GOOGLE_MAPS_STATIC_API_KEY}"
    resp = requests.get(url)
    print("Google response status:", resp.status_code, resp.headers.get("Content-Type"))
    if resp.status_code != 200:
        print(f"\nERROR {resp.status_code}:\nResponse text:", resp.text[:500] + "...\n")  # print first 500 chars
    google = Image.open(io.BytesIO(resp.content))

    is_valid_image = perform_image_sanity_check(lat, lon, resp)
    if is_valid_image:
        filepath = HR_DIR / filename
        google.save(filepath)
        print(f"Google HR image saved to {filepath}")

    return is_valid_image

# ----------------------------------
# SENTINEL REQUEST
# ----------------------------------
def adjust_temperature(img: Image.Image, factor: float = -0.1) -> Image.Image:
    """
    Adjust image color temperature.
    Negative factor = colder (more blue), positive factor = warmer (more red/yellow).
    """
    arr = np.array(img).astype(np.float32)

    # factor in range ~ [-1, 1]
    if factor < 0:  # colder
        arr[..., 0] *= 1 + factor  # reduce red
        arr[..., 2] *= 1 - factor  # boost blue
    else:  # warmer
        arr[..., 0] *= 1 + factor  # boost red
        arr[..., 2] *= 1 - factor  # reduce blue

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def download_sentinel_image(lat, lon, size, zoom, filename, evalscript):
    """
    Fetches a Sentinel image for the given lat, lon, size, and zoom level,
    and saves it to the specified filename.
    
    Arguments:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        size (tuple): Size of the image in pixels (width, height).
        zoom (int): Zoom level for the image.
        filename (str): Filename to save the image.
        evalscript (str): Javascript code that defines how the satellite data shall be retrieved and processed.
    """
    bbox = get_bbox_from_zoom(lat, lon, size, zoom)

    # Get a range of dates to ensure cloud-free scenes
    now = datetime.now()
    delta = DELTA_DAYS
    look_from = now - timedelta(days=delta)
    initial_date = f"{look_from.year}-{look_from.month:02d}-{look_from.day:02d}"
    final_date = f"{now.year}-{now.month:02d}-{now.day:02d}"

    request_true_color = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url),
                time_interval=(initial_date, final_date),
                maxcc=0.2  # maximum cloud coverage (20%)
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    # Retrieve image data
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    if not true_color_imgs or true_color_imgs[0] is None:
        print("No Sentinel-2 imagery available for given time and location")
        return False

    # Scale to 0–255 if reflectance (float values ~0–1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        # optional: scale 16-bit down to 8-bit for PNG/JPEG
        image = (image / 256).astype(np.uint8)

    # Convert to PIL image
    img = Image.fromarray(image)

    # Save into bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    is_valid_image = perform_image_sanity_check(lat, lon, img_bytes)
    if is_valid_image:
        sentinel = Image.fromarray(image)
        
        sentinel = adjust_temperature(sentinel, factor=-0.2)  # colder

        # Enhance brightness (>1.0 = brighter)
        enhancer = ImageEnhance.Brightness(sentinel)
        sentinel = enhancer.enhance(4.0)
        filepath = LR_DIR / filename
        sentinel.save(filepath)

        print(f"Sentinel LR image saved to {filepath}")
    return is_valid_image

# ----------------------------------
# ESRI REQUEST
# ----------------------------------

def deg_to_num(lat, lon, zoom):
    """Convert lat/lon to XYZ tile numbers."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def fetch_tile(z, x, y):
    """Fetch a single ESRI World Imagery tile."""
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

def latlon_to_pixel(lat, lon, zoom, tile_size=256):
    """Convert lat/lon to pixel coordinates in the global Web Mercator map."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n * tile_size
    y = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n * tile_size
    return int(x), int(y)

def download_esri_image(lat, lon, filename, size=(512, 512), zoom=17):
    """
    Fetch an ESRI World Imagery image centered at lat/lon.
    Works like Google Static Maps (satellite).
    
    Args:
        lat, lon (float): Center coordinates
        zoom (int): Zoom level
        size (tuple): Output size (width, height) in pixels
    """
    # Determine center tile
    x, y = deg_to_num(lat, lon, zoom)

    # Calculate required number of tiles to cover requested size
    tiles_x = math.ceil(size[0] / TILE_SIZE_ESRI) + 2
    tiles_y = math.ceil(size[1] / TILE_SIZE_ESRI) + 2

    # Fetch tiles around the center
    mosaic = Image.new("RGB", (tiles_x * TILE_SIZE_ESRI, tiles_y * TILE_SIZE_ESRI))
    for dx in range(-tiles_x // 2, tiles_x // 2 + 1):
        for dy in range(-tiles_y // 2, tiles_y // 2 + 1):
            try:
                tile = fetch_tile(zoom, x + dx, y + dy)
                px = (dx + tiles_x // 2) * TILE_SIZE_ESRI
                py = (dy + tiles_y // 2) * TILE_SIZE_ESRI
                mosaic.paste(tile, (px, py))
            except Exception as e:
                print(f"Tile fetch failed: z={zoom}, x={x+dx}, y={y+dy}, err={e}")

    # Approximate resolution
    mpp = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)  # meters per pixel
    print(f"Approximate ESRI resolution: {mpp:.2f} m/px at zoom {zoom}")

    # Find exact pixel of lat/lon
    global_px, global_py = latlon_to_pixel(lat, lon, zoom, TILE_SIZE_ESRI)

    # Compute mosaic origin in global pixels
    origin_tile_x = x - tiles_x // 2
    origin_tile_y = y - tiles_y // 2
    origin_px = origin_tile_x * TILE_SIZE_ESRI
    origin_py = origin_tile_y * TILE_SIZE_ESRI

    # Pixel position of lat/lon in the mosaic
    cx = global_px - origin_px
    cy = global_py - origin_py

    # Crop mosaic to requested size centered at lat/lon
    left = cx - size[0] // 2
    top = cy - size[1] // 2
    right = left + size[0]
    bottom = top + size[1]

    # Save HR image
    filepath = HR_DIR / filename
    mosaic.crop((left, top, right, bottom)).save(filepath)
    print(f"ESRI HR image saved to {filepath}")


def download_image_pairs(lat, lon, size, zoom, bands=None):
    """
    Downloads a pair of images (Google Maps and Sentinel) for the given latitude and longitude.
    
    Arguments:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        size (tuple): Size of the image in pixels (width, height).
        zoom (int): Zoom level for the image.
        filename (str): Filename to save the image.
    """
    has_imagery = True
    filename = f"{str(lat)[:8]}_{str(lon)[:8]}_test.png"
    print(f"Fetching images for coordinates: {lat}, {lon}")

    # Get script that will retrieve image bands
    evalscript_true_color = generate_evalscript() if bands is None else generate_evalscript([bands]) if len(bands) < 2  else generate_evalscript(bands)  
    has_imagery = download_sentinel_image(lat, lon, size, zoom, filename, evalscript_true_color)
    if has_imagery:
        download_esri_image(lat, lon, filename, size, zoom)
        print()
    else:
        lat, lon = get_n_random_coordinate_pairs(1)[0]
        print(f"Trying with new coordinates:\n{lat},{lon}")
        download_image_pairs(lat, lon, size, zoom)

def main():
    """
    Downloads n pairs of HR-LR ESRI-Sentinel images.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LR_DIR.mkdir(parents=True, exist_ok=True)
    HR_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Download image pairs from Google Maps and Sentinel.",
        epilog="""Examples:
        python get_image_pairs.py -n 50
            Download 50 random image pairs (default Spain bounding box).

        python get_image_pairs.py -n 10 -s 512 512
            Download 10 pairs, each 512x512 pixels.

        python get_image_pairs.py -n 20 -bz 43.5 36.0 -9.5 -1.5
            Download 20 pairs restricted to northern Spain.
    """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-n", "--number",
        type=int,
        default=500,
        metavar="amount",
        help="Number of image pairs to download (default: 30)."
    )
    parser.add_argument(
        "-s", "--size",
        type=int,
        nargs=2,
        metavar=("size_x", "size_y"),
        default=[SIZE, SIZE],
        help="Image X and Y size."
    )

    parser.add_argument(
        "-bb", "--bounded-box",
        type=float,
        nargs=4,
        metavar=("lat_max", "lat_min", "lon_max", "lon_min"),
        default=[LAT_MAX, LAT_MIN, LON_MAX, LON_MIN],
        help="Coordinates of the bounded box to take the coordinates from (default: most of Spain)."
    )

    args = parser.parse_args()
    n = args.number
    bounded_zone = args.bounded_box

    zoom = ZOOM
    size = (SIZE,SIZE)
    coordinates = get_n_random_coordinate_pairs(n, bounded_zone)

    for pair in coordinates:
        lat, lon = pair
        download_image_pairs(lat, lon, size, zoom)

if __name__ == "__main__":
    start_time = time.time()
    main()
    finish_time = time.time()
    print(f"Total time:\t{(finish_time - start_time)/60:.1f} minutes")
