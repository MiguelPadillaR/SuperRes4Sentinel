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

# ----------------------------
# SENTINEL REQUEST
# ----------------------------
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
                # Pick a recent interval (e.g., last week) to ensure cloud-free scenes
                time_interval=(initial_date, final_date),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
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

# ----------------------------
# GOOGLE MAPS REQUEST
# ----------------------------
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
    is_google_image_downloaded = True
    filename = f"{str(lat)[:8]}_{str(lon)[:8]}_test.png"
    print(f"Fetching images for coordinates: {lat}, {lon}")
    is_google_image_downloaded = download_google_image(lat, lon, size, zoom, filename)
    if is_google_image_downloaded:
        # Get script that will retrieve image bands
        evalscript_true_color = generate_evalscript(bands)
        download_sentinel_image(lat, lon, size, zoom, filename, evalscript_true_color)
        print()
    else:
        lat, lon = get_n_random_coordinate_pairs(1)[0]
        print(f"Trying with new coordinates:\n{lat},{lon}")
        download_image_pairs(lat, lon, size, zoom)

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
        "-bz", "--bounded-zone",
        type=float,
        nargs=4,
        metavar=("lat_max", "lat_min", "lon_max", "lon_min"),
        default=[LAT_MAX, LAT_MIN, LON_MAX, LON_MIN],
        help="Coordinates of the bounded zone box to take the coordinates from (default: entire Spain)."
    )

    args = parser.parse_args()
    n = args.number
    bounded_zone = args.bounded_zone

    zoom = ZOOM
    size = (int(SIZE),int(SIZE))
    coordinates = get_n_random_coordinate_pairs(n, bounded_zone)

    for pair in coordinates:
        lat, lon = pair
        download_image_pairs(lat, lon, size, zoom)

if __name__ == "__main__":
    start_time = time.time()
    main()
    finish_time = time.time()
    print(f"Total time:\t{(finish_time - start_time)/60:.1f} minutes")
