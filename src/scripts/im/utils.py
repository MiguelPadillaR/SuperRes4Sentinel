import io
import math
import numpy as np
import random
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from sentinelhub import BBox, CRS
from src.utils.constants import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

def get_bbox_from_zoom(lat, lon, size, zoom):
    """
    Compute a bounding box in WGS84 matching Google Maps Static API tile.
    
    lat, lon: center coordinates
    size: (width_px, height_px)
    zoom: Google zoom level
    """
    # meters per pixel at this latitude and zoom
    mpp = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    
    # total size in meters
    width_m = size[0] * mpp
    height_m = size[1] * mpp
    
    # convert meters to degrees (approx)
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(lat))
    
    width_deg = width_m / meters_per_degree_lon
    height_deg = height_m / meters_per_degree_lat
    
    min_lon = lon - width_deg / 2
    max_lon = lon + width_deg / 2
    min_lat = lat - height_deg / 2
    max_lat = lat + height_deg / 2
    
    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)

def get_zoom_from_bbox(bbox: BBox, size: tuple):
    """
    Given a SentinelHub bbox + image size, compute the Google Maps zoom level
    and the equivalent Google bbox that matches coverage.
    
    bbox: BBox in WGS84
    size: (width_px, height_px)
    
    Returns: (zoom, google_bbox)
    """
    # Extract bbox coordinates
    min_lon, min_lat, max_lon, max_lat = bbox
    width_deg = max_lon - min_lon
    height_deg = max_lat - min_lat
    lat_center = (min_lat + max_lat) / 2
    
    # Approx conversion: degrees → meters
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(lat_center))
    
    width_m = width_deg * meters_per_degree_lon
    height_m = height_deg * meters_per_degree_lat
    
    # Sentinel resolution (meters per pixel)
    mpp_x = width_m / size[0]
    mpp_y = height_m / size[1]
    mpp = (mpp_x + mpp_y) / 2   # average
    
    # Compute Google zoom
    zoom_float = math.log2((156543.03392 * math.cos(math.radians(lat_center))) / mpp)
    zoom = max(0, min(21, round(zoom_float)))  # clamp to [0,21]
    
    # # Now recompute bbox that Google would actually cover at this zoom
    # mpp_google = 156543.03392 * math.cos(math.radians(lat_center)) / (2 ** zoom)
    # width_m_google = size[0] * mpp_google
    # height_m_google = size[1] * mpp_google
    
    # width_deg_google = width_m_google / meters_per_degree_lon
    # height_deg_google = height_m_google / meters_per_degree_lat
    
    # google_bbox = BBox(
    #     bbox=[
    #         (min_lon + max_lon) / 2 - width_deg_google / 2,
    #         (min_lat + max_lat) / 2 - height_deg_google / 2,
    #         (min_lon + max_lon) / 2 + width_deg_google / 2,
    #         (min_lat + max_lat) / 2 + height_deg_google / 2
    #     ],
    #     crs=CRS.WGS84
    # )
    
    return zoom, # google_bbox

def get_n_random_coordinate_pairs(amount:int, bounded_zone = [LAT_MIN, LAT_MAX, LON_MIN, LON_MAX]):
    """
    Generate random coordinates from a bounded zone.
    
    amount: number of coordinate pairs to generate
    
    Returns: list of (lat, lon) tuples
    """    
    coordinates = []
    lat_min, lat_max, lon_min, lon_max = bounded_zone
    for _ in range(amount):
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        coordinates.append((lat, lon))
    
    return coordinates

def generate_evalscript(
        bands=["B02", "B03", "B04"], 
        units=None, 
        data_type="UINT8", 
        mosaicking_type=None, 
        bit_scale=None, 
        id=None, 
        rendering=None, 
        mask=None 
    ):
    """
    Generate an evalscript for SentinelHub requests. The script is dinamically generated with the values provided.

    DOC: https://docs.sentinel-hub.com/api/latest/evalscript/v3/
    
    Arguments:
        bands (list | None): Bands to include, e.g. `"B02"` or `["B02", "B03", "B04"]`.
        units (str | None): Units of the input bands (e.g. "DN", "REFLECTANCE"). If None, omitted.
        data_type (str | None): Data type for input bands. If None, omitted.
        mosaicking_type (str | None): Type of mosaicking. If None, omitted.
        bit_scale (str | None): Bit scale of output bands. If None, omitted.
        id (str): Response ID. If None, omitted.
        rendering (bool): Whether to apply rendering/visualization. If None, omitted.
        mask (bool): Whether to output mask. If None, omitted.

    Returns:
        evalscript (str): The generated evalscript for SentinelHub image request.
    """
    # print(bands)
    output_bands = list(reversed(bands))
    # print(output_bands)
    # print(', '.join([f'sample.{band}' for band in output_bands]))
    bands_str = ", ".join([f'"{band}"' for band in bands])

    # Build optional parts dynamically
    input_options = [f"bands: [{bands_str}]"]
    if units is not None:
        input_options.append(f'units: "{units}"')
    if data_type is not None:
        input_options.append(f'dataType: "{data_type}"')
    if mosaicking_type is not None:
        input_options.append(f'mosaicking: "{mosaicking_type}"')

    output_options = [f"bands: {len(output_bands)}"]

    if id is not None:
        output_options.append(f'id: "{id}"')
    if bit_scale is not None:
        output_options.append(f'sampleType: "{bit_scale}"')
    if rendering is not None and type(rendering) == bool:
        output_options.append(f"rendering: {str(rendering).lower()}")
    if mask is not None and type(mask) == bool:
        output_options.append(f"mask: {str(mask).lower()}")

    output = ", ".join([f"sample.{band}" for band in output_bands])
    
    evalscript = f"""
    //VERSION=3

    function setup() {{
        return {{
            input: [{{
                {', '.join(input_options)}
            }}],
            output: {{
                {', '.join(output_options)}
            }}
        }};
    }}

    function evaluatePixel(sample) {{
        return [{output},];
    }}
    """
    return evalscript

def save_tiff(image: np.ndarray, filename: str, bbox, crs="EPSG:4326"):
    """
    Save a numpy array as a GeoTIFF with georeferencing info.
    
    Args:
        image (np.ndarray): Image array (H, W) or (H, W, C).
        filename (str): Output file path (.tiff).
        bbox (sentinelhub.BBox): Bounding box used in the request.
        crs (str): Coordinate reference system (default WGS84).
    """
    height, width = image.shape[:2]
    count = 1 if image.ndim == 2 else image.shape[2]

    # Create an affine transform (maps pixel <-> geo coordinates)
    transform = from_bounds(*bbox, width=width, height=height)

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=image.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        if count == 1:
            dst.write(image, 1)
        else:
            for i in range(count):
                dst.write(image[:, :, i], i + 1)

def perform_image_sanity_check(lat, lon, image_bytes):
    """
    Perform a sanity check over the response's image
    Arguments:
        image_bytes (bytes): Sentinel Hub request image in bytes format.
        lat (float): Latitude.
        lon (float): Longitude.
    Return:
        has_imagery (bool): If True, there is a valid image in the response.
    """
    has_imagery = True
    
    # # Quick sanity check: if file is too small, probably "no imagery"
    # if len(image_bytes) < 15_000:  # tweak threshold as needed
    #     print(f"No imagery (small file) at {lat},{lon}")
    #     return False

    # Check if the image is mainly white (no imagery available)
    iamge_converted = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(iamge_converted)
    std_val = np.std(arr)

    # If almost all pixels are very bright (white background) and there's little variation, it's probably "no imagery"
    if np.mean(arr) > 230 and std_val < 15:
        print(f"No imagery available for {lat},{lon}")
        has_imagery = False
    
    return has_imagery
