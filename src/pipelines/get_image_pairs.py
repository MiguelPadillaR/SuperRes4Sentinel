import io
from PIL import Image, ImageEnhance
import requests
from sentinelhub import SHConfig, DataCollection, MimeType, SentinelHubRequest
from config import CONFIG_NAME, GOOGLE_MAPS_STATIC_API_KEY
from utils import *

config = SHConfig(CONFIG_NAME)

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

# Example location
lat, lon = 36.627058, -6.051960
zoom = 17
size = (255,255)
coordinates = get_random_coordinate_pairs(40)

# ----------------------------
# SENTINEL REQUEST
# ----------------------------
def download_sentinel_image(lat, lon, size, zoom, filename):
    """
    Fetches a Sentinel image for the given lat, lon, size, and zoom level,
    and saves it to the specified filename.
    """
    resolution = 5  # meters per pixel for Sentinel
    bbox = get_bbox_from_zoom(lat, lon, size, zoom)

    print(f"Image shape at {resolution} m resolution: {size} pixels")

    evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """

    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url),
                # Pick a recent interval (e.g., last week) to ensure cloud-free scenes
                time_interval=("2025-08-15", "2025-08-25"),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config,
    )

    true_color_imgs = request_true_color.get_data()
    print(f"Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.")
    print(f"Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}")

    image = true_color_imgs[0]
    print(f"Image type: {image.dtype}")

    sentinel = Image.fromarray(image)
    # Enhance brightness (1.0 = original, >1.0 = brighter, <1.0 = darker)
    enhancer = ImageEnhance.Brightness(sentinel)
    sentinel = enhancer.enhance(3.5)  # try 1.5x brightness
    sentinel.save("data/HR/" + filename)

# ----------------------------
# GOOGLE MAPS REQUEST
# ----------------------------
def download_google_image(lat, lon, size, zoom, filename):
    """
    Fetches a Google Maps satellite image for the given lat, lon, size, and zoom level,
    and saves it to the specified filename.
    """
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size[0]}x{size[1]}&maptype=satellite&style=feature:all|element:labels|visibility:off&key={GOOGLE_MAPS_STATIC_API_KEY}"
    resp = requests.get(url)
    google = Image.open(io.BytesIO(resp.content))
    google.save("data/LR/" + filename)

for pair in coordinates:
    lat, lon = pair
    filename = f"{lat[:8]}_{lon[:8]}_test.png"
    print(f"Fetching images for coordinates: {lat}, {lon}")
    download_sentinel_image(lat, lon, size, zoom, filename)
    download_google_image(lat, lon, size, zoom, filename)

print('SIZE', size)
print('TYPE', type(size))
