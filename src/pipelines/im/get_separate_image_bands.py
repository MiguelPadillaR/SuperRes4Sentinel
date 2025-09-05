import argparse

import time
from datetime import datetime, timedelta

from sentinelhub import SHConfig, DataCollection, MimeType, SentinelHubRequest
from .sh_config import CONFIG_NAME
from .utils import *
from src.utils.constants import *

config = SHConfig(CONFIG_NAME)

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

# ----------------------------
# SENTINEL REQUEST
# ----------------------------
def download_sentinel_image(lat, lon, size, zoom, filename, evalscript_true_color):
    resolution = 10  # meters per pixel for Sentinel
    bbox = get_bbox_from_zoom(lat, lon, size, zoom)

    print(f"Image shape at {resolution} m resolution: {size} pixels")

    # Get a range of dates to ensure cloud-free scenes
    now = datetime.now()
    delta = DELTA_DAYS
    look_from = now - timedelta(days=delta)
    initial_date = f"{look_from.year}-{look_from.month:02d}-{look_from.day:02d}"
    final_date = f"{now.year}-{now.month:02d}-{now.day:02d}"

    sh_request = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=config.sh_base_url),
                time_interval=(initial_date, final_date),
                # maxcc=0.2  # maximum cloud coverage (20%)
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    # Retieve imagen band and save it
    image = sh_request.get_data()[0]
    filepath = BAND_DIR / filename
    save_tiff(image, filepath, bbox, crs="EPSG:4326")

    print(f"Sentinel band image saved to {filepath}")

def download_image_bands(lat, lon, size, zoom, bands=None):
    """
    Downloads separate Sentinel image bandsfor the given latitude and longitude.
    
    Arguments:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        size (tuple): Size of the image in pixels (width, height).
        filename (str): Filename to save the image.
    """
    for band in bands:
        filename = f"{str(lat)[:8]}_{str(lon)[:8]}-{band}.tiff"
        print(f"Fetching images for coordinates: {lat}, {lon}")
        # Get script that will retrieve  image bands
        evalscript_band = generate_evalscript(
            bands=[band],
            units="DN",
            bit_scale="UINT16",
        )

        download_sentinel_image(lat, lon, size, zoom, filename, evalscript_band)
        print()

def main():
    """
    Downloads n pairs of HR-LR Google-Sentinel images.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BAND_DIR.mkdir(parents=True, exist_ok=True)
    HR_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Download image pairs from Google Maps and Sentinel.")
    parser.add_argument("-n", "--number", type=int, default=30, help="Number of image pairs to download (default: 30)")
    args = parser.parse_args()
    n = args.number
    bands=['B02', 'B03', 'B04', 'B08']
    zoom = 17
    size = (SIZE,SIZE)
    coordinates = get_n_random_coordinate_pairs(n)

    for pair in coordinates:
        lat, lon = pair
        download_image_bands(lat, lon, size, zoom, bands)

if __name__ == "__main__":
    start_time = time.time()
    main()
    finish_time = time.time()
    print(f"Total time:\t{(finish_time - start_time)/60:.1f} minutes")