import re

# [south, west, north, east] — Chicagoland / upper Midwest (covers lory_vision_lake demo points)
DEFAULT_BBOX_SELECTION = [40.3800, -91.0986, 42.6986, -84.4189]


def extract_point_coords(point_str):
    """Extract lat, lon from POINT(lon lat) or similar formats."""
    if not isinstance(point_str, str):
        return None
    match = re.search(r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)", point_str, re.IGNORECASE)
    if match:
        lon, lat = float(match.group(1)), float(match.group(2))
        return (lat, lon)
    return None


def bbox_from_geojson_feature(feature):
    """Extract [south, west, north, east] from a GeoJSON-like feature."""
    if not feature:
        return None
    geom = feature.get("geometry") if isinstance(feature, dict) else None
    if not geom:
        return None
    coords = geom.get("coordinates")
    if not coords:
        return None

    flat = []

    def walk(node):
        if isinstance(node, (list, tuple)):
            if node and all(isinstance(v, (int, float)) for v in node):
                if len(node) >= 2:
                    flat.append((node[0], node[1]))
            else:
                for child in node:
                    walk(child)

    walk(coords)
    if not flat:
        return None
    lons = [p[0] for p in flat]
    lats = [p[1] for p in flat]
    return [min(lats), min(lons), max(lats), max(lons)]
