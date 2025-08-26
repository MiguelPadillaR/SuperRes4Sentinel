import datetime
import io
import os
from PIL import Image, ImageEnhance
import requests

import matplotlib.pyplot as plt
import numpy as np

from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)

from config import CONFIG_NAME, GOOGLE_MAPS_STATIC_API_KEY

config = SHConfig(CONFIG_NAME)

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

# Example location
lat, lon = 36.627058, -6.051960

# ----------------------------
# SENTINEL REQUEST
# ----------------------------
resolution = 10  # meters per pixel for Sentinel
offset = 0.0015
bbox = BBox(bbox=[lon - offset, lat - offset, lon + offset, lat + offset], crs=CRS.WGS84)
# size = bbox_to_dimensions(bbox, resolution=resolution)
size = (512, 512)
filename = f"lat{lat}_lon{lon}_test.png"

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

filename = 'lol.png'

sentinel = Image.fromarray(image)
# Enhance brightness (1.0 = original, >1.0 = brighter, <1.0 = darker)
enhancer = ImageEnhance.Brightness(sentinel)
sentinel = enhancer.enhance(3.5)  # try 1.5x brightness
sentinel.save("data/LR/"+ filename)

# ----------------------------
# GOOGLE MAPS REQUEST
# ----------------------------
zoom = 18
url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size[0]}x{size[1]}&maptype=satellite&key={GOOGLE_MAPS_STATIC_API_KEY}"
resp = requests.get(url)
google = Image.open(io.BytesIO(resp.content))
google.save("data/HR/" + filename)

print('SIZE', size)
print('TYPE', type(size))
