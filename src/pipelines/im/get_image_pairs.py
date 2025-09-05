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
# SENTINEL REQUEST
# ----------------------------------
def download_sentinel_image(lat, lon, size, zoom, filename, evalscript_true_color):
    """
    Fetches a Sentinel image for the given lat, lon, size, and zoom level,
    and saves it to the specified filename.
    
    Arguments:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        size (tuple): Size of the image in pixels (width, height).
        zoom (int): Zoom level for the image.
        filename (str): Filename to save the image.
    """
    resolution = 5  # meters per pixel for Sentinel
    bbox = get_bbox_from_zoom(lat, lon, size, zoom)

    print(f"Image shape at {resolution} m resolution: {size} pixels")

    # Get a range of dates to ensure cloud-free scenes
    # Get a range of dates to ensure cloud-free scenes
    now = datetime.now()
    delta = DELTA_DAYS
    look_from = now - timedelta(days=delta)
    initial_date = f"{look_from.year}-{look_from.month:02d}-{look_from.day:02d}"
    final_date = f"{now.year}-{now.month:02d}-{now.day:02d}"

    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url),
                time_interval=(initial_date, final_date),
                # maxcc=0.2  # maximum cloud coverage (20%)
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]

    sentinel = Image.fromarray(image)
    # Enhance brightness (1.0 = original, >1.0 = brighter, <1.0 = darker)
    enhancer = ImageEnhance.Brightness(sentinel)
    sentinel = enhancer.enhance(4.0)
    filepath = LR_DIR / filename
    sentinel.save(filepath)

    print(f"Sentinel LR image saved to {filepath}")

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
    is_image_downloaded = True

    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size[0]}x{size[1]}&maptype=satellite&style=feature:all|element:labels|visibility:off&key={GOOGLE_MAPS_STATIC_API_KEY}"
    resp = requests.get(url)
    print("Google response status:", resp.status_code, resp.headers.get("Content-Type"))
    if resp.status_code != 200:
        print(f"\nERROR {resp.status_code}:\nResponse text:", resp.text[:500] + "...\n")  # print first 500 chars

    google = Image.open(io.BytesIO(resp.content))

    # Quick sanity check: if file is too small, probably "no imagery"
    if len(resp.content) < 15_000:  # tweak threshold as needed
        print(f"No imagery (small file) at {lat},{lon}")
        return False

    # Check if the image is mainly white (no imagery available)
    google_converted = Image.open(io.BytesIO(resp.content)).convert("L")
    arr = np.array(google_converted)
    std_val = np.std(arr)

    # If almost all pixels are very bright (white background) and there's little variation, it's probably "no imagery"
    if np.mean(arr) > 230 and std_val < 15:
        print(f"No imagery available for {lat},{lon}")
        is_image_downloaded = False
    else:
        filepath = HR_DIR / filename
        google.save(filepath)
        print(f"Google HR image saved to {filepath}")

    return is_image_downloaded

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

def num_to_deg(xtile, ytile, zoom):
    """Convert XYZ tile numbers back to lat/lon (NW corner)."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def fetch_tile(z, x, y):
    """Fetch a single ESRI World Imagery tile."""
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

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

    # # Find center tile's geographic bounds
    # lat_ul, lon_ul = num_to_deg(x, y, zoom)  # NW corner of center tile
    # lat_lr, lon_lr = num_to_deg(x + 1, y + 1, zoom)  # SE corner of center tile

    # Approximate resolution
    mpp = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)  # meters per pixel
    print(f"Approximate resolution: {mpp:.2f} m/px at zoom {zoom}")

    # Crop mosaic to requested size centered at lat/lon
    cx = mosaic.width // 2
    cy = mosaic.height // 2
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
    # is_hr_image_downloaded = True
    filename = f"{str(lat)[:8]}_{str(lon)[:8]}_test.png"
    print(f"Fetching images for coordinates: {lat}, {lon}")
    # is_hr_image_downloaded = download_google_image(lat, lon, size, zoom, filename)
    # if is_hr_image_downloaded:
    #     # Get script that will retrieve image bands
    #     evalscript_true_color = generate_evalscript(bands)
    #     download_sentinel_image(lat, lon, size, zoom, filename, evalscript_true_color)
    #     print()
    # else:
    #     lat, lon = get_n_random_coordinate_pairs(1)[0]
    #     print(f"Trying with new coordinates:\n{lat},{lon}")
    #     download_image_pairs(lat, lon, size, zoom)
    download_esri_image(lat, lon, filename, size, zoom)
    evalscript_true_color = generate_evalscript() if bands is None else generate_evalscript([bands]) if len([bands]) == 1  else generate_evalscript(bands)  
    download_sentinel_image(lat, lon, size, zoom, filename, evalscript_true_color)
    print()


def main():
    """
    Downloads n pairs of HR-LR Google-Sentinel images.
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
