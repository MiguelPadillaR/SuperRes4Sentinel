import argparse

import time
from datetime import datetime, timedelta

from sentinelhub import SHConfig, DataCollection, MimeType, SentinelHubRequest, bbox_to_dimensions
from .sh_config import CONFIG_NAME
from .utils import *
from src.utils.constants import *

config = SHConfig(CONFIG_NAME)

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

# ----------------------------
# SENTINEL REQUEST
# ----------------------------
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
    resolution = 10
    bbox = get_bbox_from_center(lat, lon, size[0], size[-1], resolution)
    width, height = bbox_to_dimensions(bbox, resolution=resolution)

    # Get a range of dates to ensure cloud-free scenes
    now = datetime.now()
    delta = DELTA_DAYS
    look_from = now - timedelta(days=delta)
    initial_date = f"{look_from.year}-{look_from.month:02d}-{look_from.day:02d}"
    final_date = f"{now.year}-{now.month:02d}-{now.day:02d}"

    sh_request = SentinelHubRequest(
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
        size=(width, height),
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
            # units="DN",
            # data_type="UINT16",
            # bit_scale="UNINT8",
            # resampling= "BICUBIC"
        )

        download_sentinel_image(lat, lon, size, zoom, filename, evalscript_band)
        print()

def main():
    """
    Downloads n pairs of Sentinel band images, specifically, B02, B03, B04 and B08.
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