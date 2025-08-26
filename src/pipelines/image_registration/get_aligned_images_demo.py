import requests
import io
from PIL import Image
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, CRS, BBox, bbox_to_dimensions
from config import CONFIG_NAME, GOOGLE_MAPS_STATIC_API_KEY

# ----------------------------
# CONFIG
# ----------------------------

# SentinelHub credentials
config = SHConfig(CONFIG_NAME)

# Example location (lat, lon, size in meters)
lat, lon = 36.627058, -6.051960
bbox = BBox(bbox=[lon-0.002, lat-0.002, lon+0.002, lat+0.002], crs=CRS.WGS84)
resolution = 10  # meters per pixel for Sentinel
filename = f"lat{lat}_lon{lon}_test.png"

# ----------------------------
# SENTINEL REQUEST
# ----------------------------
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3 }
  };
}
function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02];
}
"""

request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=bbox,
    size=bbox_to_dimensions(bbox, resolution=resolution),
    config=config
)
sentinel_img = request.get_data()[0]
sentinel = Image.fromarray(sentinel_img)
sentinel.save("data/LR/"+ filename)

# ----------------------------
# GOOGLE MAPS REQUEST
# ----------------------------
url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=18&size=512x512&maptype=satellite&key={GOOGLE_MAPS_STATIC_API_KEY}"
resp = requests.get(url)
google = Image.open(io.BytesIO(resp.content))
google.save("data/HR/" + filename)
