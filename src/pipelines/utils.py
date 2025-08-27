import math
import random
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

def get_n_random_coordinate_pairs(amount:int):
    """
    Generate random coordinates from a bounded zone.
    
    amount: number of coordinate pairs to generate
    
    Returns: list of (lat, lon) tuples
    """    
    coordinates = []
    for _ in range(amount):
        lat = random.uniform(LAT_MIN, LAT_MAX)
        lon = random.uniform(LON_MIN, LON_MAX)
        coordinates.append((lat, lon))
    
    return coordinates
