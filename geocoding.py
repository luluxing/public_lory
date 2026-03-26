from __future__ import annotations

import hashlib
import json
import re

import pandas as pd

import ddg_spatial_joins
import pandas_helper


GOOGLE_GEOCODING_API_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_MAPS_API_KEY"

POINT_COLUMN = "coordinates"
POLYGON_COLUMN = "location_polygon"
POINT_SUFFIX = "_coor"
POLYGON_SUFFIX = "_polygon"
GEO_ASSOCIATION_METADATA_SUFFIX = ".geo_association.json"
ZIPCODE_COLUMN = "zipcode"
ADDRESS_COLUMN_CANDIDATES = ("address", "street_address", "hq_address", "report_address")
AREA_COLUMN_CANDIDATES = ("neighborhood", "precinct", "Precinct", "region")
GEOCODE_REQUIRED_COLUMNS = ("city", "state", "country")
MAPPING_TABLE_NAMES = frozenset({"accidents_geo", "area_polygons"})

ZIPCODE_CENTROIDS = {
    "60201": (42.0464, -87.6949),
    "60202": (42.0308, -87.6877),
    "60301": (41.8869, -87.7935),
    "60302": (41.8888, -87.7848),
    "60601": (41.8864, -87.6225),
    "60602": (41.8820, -87.6280),
    "60603": (41.8807, -87.6287),
    "60605": (41.8673, -87.6246),
    "60607": (41.8810, -87.6525),
    "60608": (41.8498, -87.6690),
    "60611": (41.8948, -87.6203),
    "60612": (41.8804, -87.6877),
    "60615": (41.8019, -87.6004),
    "60616": (41.8454, -87.6256),
    "60618": (41.9461, -87.7032),
    "60621": (41.7760, -87.6404),
    "60622": (41.9012, -87.6763),
    "60623": (41.8490, -87.7179),
    "60625": (41.9719, -87.7020),
    "60628": (41.6947, -87.6203),
    "60637": (41.7807, -87.6038),
    "60640": (41.9718, -87.6580),
    "60642": (41.8992, -87.6585),
    "60647": (41.9216, -87.7012),
    "60653": (41.8197, -87.6115),
    "60654": (41.8927, -87.6353),
    "60660": (41.9901, -87.6626),
}

CITY_DEFAULT_CENTROIDS = {
    ("chicago", "il", "us"): (41.8781, -87.6298),
    ("evanston", "il", "us"): (42.0451, -87.6877),
    ("oak park", "il", "us"): (41.8850, -87.7845),
}

STATE_ALIASES = {
    "ia": "ia",
    "iowa": "ia",
    "il": "il",
    "illinois": "il",
    "in": "in",
    "indiana": "in",
    "mn": "mn",
    "minnesota": "mn",
    "wi": "wi",
    "wisconsin": "wi",
}

COUNTRY_ALIASES = {
    "us": "us",
    "usa": "us",
    "united states": "us",
}

STATE_BBOXES = {
    "ia": (40.3754, -96.6395, 43.5011, -90.1401),
    "il": (36.9703, -91.5131, 42.5085, -87.4948),
    "in": (37.7717, -88.0978, 41.7614, -84.7846),
    "mn": (43.4994, -97.2392, 49.3844, -89.4917),
    "wi": (42.4919, -92.8894, 47.3098, -86.8054),
}

COUNTRY_BBOXES = {
    "us": (24.3963, -124.8489, 49.3844, -66.8854),
}

HARDCODED_ADDRESS_COORDINATES = {
    ("120 n wabash ave", "60602", "chicago"): (41.8857, -87.6194),
    ("401 n state st", "60654", "chicago"): (41.8929, -87.6337),
    ("1460 s michigan ave", "60605", "chicago"): (41.8679, -87.6274),
    ("950 w lake st", "60607", "chicago"): (41.8862, -87.6467),
    ("1821 s halsted st", "60608", "chicago"): (41.8532, -87.6609),
    ("2200 s wentworth ave", "60616", "chicago"): (41.8513, -87.6248),
    ("3501 s martin luther king dr", "60653", "chicago"): (41.8239, -87.6105),
    ("5300 s hyde park blvd", "60615", "chicago"): (41.7955, -87.5993),
    ("2800 s kedzie ave", "60623", "chicago"): (41.8434, -87.7087),
    ("2650 n milwaukee ave", "60647", "chicago"): (41.9211, -87.7044),
    ("1560 n damen ave", "60622", "chicago"): (41.9009, -87.6844),
    ("1100 n ashland ave", "60622", "chicago"): (41.9074, -87.6800),
    ("4700 n broadway", "60640", "chicago"): (41.9635, -87.6558),
    ("5900 n broadway", "60660", "chicago"): (41.9972, -87.6591),
    ("3100 n elston ave", "60618", "chicago"): (41.9460, -87.7042),
    ("11100 s cottage grove ave", "60628", "chicago"): (41.6969, -87.6255),
    ("6300 s halsted st", "60621", "chicago"): (41.7807, -87.6398),
    ("1010 lake st", "60301", "oak park"): (41.8886, -87.7889),
    ("816 dempster st", "60202", "evanston"): (42.0214, -87.6908),
    ("33 w monroe st", "60603", "chicago"): (41.8877, -87.6320),
    ("401 n wabash ave", "60611", "chicago"): (41.8955, -87.6135),
    ("901 w randolph st", "60607", "chicago"): (41.8854, -87.6509),
    ("2750 s kedzie ave", "60623", "chicago"): (41.8535, -87.7111),
    ("4753 n broadway", "60640", "chicago"): (41.9718, -87.6572),
    ("11101 s cottage grove ave", "60628", "chicago"): (41.6890, -87.6231),
    ("2525 s michigan ave", "60616", "chicago"): (41.8382, -87.6340),
    ("251 e huron st", "60611", "chicago"): (41.8927, -87.6122),
    ("1653 w congress pkwy", "60612", "chicago"): (41.8705, -87.6860),
    ("2233 w division st", "60622", "chicago"): (41.8944, -87.6722),
    ("5841 s maryland ave", "60637", "chicago"): (41.7890, -87.6022),
    ("1500 s fairfield ave", "60608", "chicago"): (41.8587, -87.6725),
    ("3 erie ct", "60302", "oak park"): (41.8856, -87.7871),
    ("5145 n california ave", "60625", "chicago"): (41.9796, -87.7080),
    ("500 e 51st st", "60615", "chicago"): (41.7982, -87.6054),
    ("355 ridge ave", "60202", "evanston"): (42.0300, -87.6942),
    ("55 e monroe st", "60603", "chicago"): (41.8708, -87.6230),
    ("405 n wabash ave", "60611", "chicago"): (41.8992, -87.6132),
    ("933 w van buren st", "60607", "chicago"): (41.8809, -87.6479),
    ("1830 s racine ave", "60608", "chicago"): (41.8520, -87.6722),
    ("200 n dearborn st", "60601", "chicago"): (41.8885, -87.6143),
    ("757 n orleans st", "60654", "chicago"): (41.8892, -87.6264),
    ("1250 s michigan ave", "60605", "chicago"): (41.8739, -87.6158),
    ("2754 s millard ave", "60623", "chicago"): (41.8508, -87.7215),
    ("1509 n milwaukee ave", "60622", "chicago"): (41.9080, -87.6693),
    ("1020 n wood st", "60622", "chicago"): (41.9077, -87.6775),
    ("4625 n magnolia ave", "60640", "chicago"): (41.9692, -87.6633),
    ("6030 n kenmore ave", "60660", "chicago"): (41.9919, -87.6632),
    ("3111 n central park ave", "60618", "chicago"): (41.9539, -87.6983),
    ("11206 s st lawrence ave", "60628", "chicago"): (41.7002, -87.6277),
    ("820 mulford st", "60202", "evanston"): (42.0252, -87.6940),
    ("130 n michigan ave", "60602", "chicago"): (41.8753, -87.6296),
    ("410 n state st", "60654", "chicago"): (41.8968, -87.6339),
    ("1500 s wabash ave", "60605", "chicago"): (41.8731, -87.6237),
    ("1000 w lake st", "60607", "chicago"): (41.8775, -87.6555),
    ("1858 s halsted st", "60608", "chicago"): (41.8542, -87.6655),
    ("2210 s wentworth ave", "60616", "chicago"): (41.8479, -87.6283),
    ("3510 s martin luther king dr", "60653", "chicago"): (41.8186, -87.6102),
    ("5307 s hyde park blvd", "60615", "chicago"): (41.8031, -87.5916),
    ("2815 s kedzie ave", "60623", "chicago"): (41.8501, -87.7105),
    ("2658 n milwaukee ave", "60647", "chicago"): (41.9117, -87.6919),
    ("1575 n damen ave", "60622", "chicago"): (41.9042, -87.6861),
    ("1112 n ashland ave", "60622", "chicago"): (41.9102, -87.6850),
    ("4712 n broadway", "60640", "chicago"): (41.9730, -87.6669),
    ("5915 n broadway", "60660", "chicago"): (41.9947, -87.6571),
    ("3120 n elston ave", "60618", "chicago"): (41.9396, -87.7100),
    ("11120 s cottage grove ave", "60628", "chicago"): (41.6915, -87.6212),
    ("6310 s halsted st", "60621", "chicago"): (41.7787, -87.6452),
    ("1024 lake st", "60301", "oak park"): (41.8904, -87.7845),
    ("830 dempster st", "60202", "evanston"): (42.0246, -87.6861),
    ("201 e randolph st", "60601", "chicago"): (41.8919, -87.6229),
    ("325 w chicago ave", "60654", "chicago"): (41.9014, -87.6439),
    ("1850 s blue island ave", "60608", "chicago"): (41.8495, -87.6779),
    ("6220 s stony island ave", "60637", "chicago"): (41.7809, -87.5970),
    ("1440 w hubbard st", "60642", "chicago"): (41.9043, -87.6528),
    ("4707 n broadway", "60640", "chicago"): (41.9758, -87.6560),
    ("6201 s stewart ave", "60621", "chicago"): (41.7752, -87.6450),
    ("1600 dodge ave", "60201", "evanston"): (42.0397, -87.6858),
    ("2410 n sawyer ave", "60647", "chicago"): (41.9281, -87.7057),
    ("211 s ridgeland ave", "60302", "oak park"): (41.8945, -87.7903),
}

HARDCODED_AREA_POLYGONS = {
    "loop": "POLYGON((-87.6330 41.8760, -87.6200 41.8760, -87.6200 41.8840, -87.6330 41.8840, -87.6330 41.8760))",
    "river north": "POLYGON((-87.6400 41.8900, -87.6250 41.8900, -87.6250 41.8970, -87.6400 41.8970, -87.6400 41.8900))",
    "pilsen": "POLYGON((-87.6680 41.8520, -87.6500 41.8520, -87.6500 41.8620, -87.6680 41.8620, -87.6680 41.8520))",
    "west loop": "POLYGON((-87.6670 41.8780, -87.6500 41.8780, -87.6500 41.8860, -87.6670 41.8860, -87.6670 41.8780))",
    "uptown": "POLYGON((-87.6650 41.9680, -87.6500 41.9680, -87.6500 41.9800, -87.6650 41.9800, -87.6650 41.9680))",
    "englewood": "POLYGON((-87.6500 41.7700, -87.6250 41.7700, -87.6250 41.7850, -87.6500 41.7850, -87.6500 41.7700))",
}

HARDCODED_AREA_CENTERS = {
    "avondale": (41.9500, -87.7013),
    "bronzeville": (41.8239, -87.6105),
    "chinatown": (41.8513, -87.6248),
    "edgewater": (41.9946, -87.6611),
    "englewood": (41.7780, -87.6424),
    "evanston": (42.0288, -87.6902),
    "hyde park": (41.7882, -87.5982),
    "little village": (41.8471, -87.7151),
    "logan square": (41.9246, -87.7050),
    "loop": (41.8888, -87.6212),
    "oak park": (41.8915, -87.7896),
    "pilsen": (41.8516, -87.6703),
    "pullman": (41.6985, -87.6266),
    "river north": (41.8971, -87.6388),
    "south loop": (41.8679, -87.6274),
    "uptown": (41.9695, -87.6584),
    "west loop": (41.8862, -87.6467),
    "west town": (41.9065, -87.6701),
    "wicker park": (41.9044, -87.6769),
}

AREA_POLYGON_RADII = {
    "default": (0.0060, 0.0085),
    "evanston": (0.0085, 0.0115),
    "oak park": (0.0085, 0.0115),
    "pullman": (0.0075, 0.0100),
}

_HARDCODED_ADDRESS_BY_ADDRESS = {
    key[0]: coords for key, coords in HARDCODED_ADDRESS_COORDINATES.items()
}


def google_geocode_address(*_args, **_kwargs):
    raise NotImplementedError(
        f"Google geocoding is not wired yet. Set {GOOGLE_API_KEY_ENV_VAR} once the API client is implemented."
    )


def normalize_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _tokenize_column_name(name: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(name).lower())
    return {part for part in normalized.split() if part}


def _normalize_key(value) -> str:
    return normalize_text(value).lower()


def _canonicalize_state(value) -> str:
    state = _normalize_key(value)
    return STATE_ALIASES.get(state, state)


def _canonicalize_country(value) -> str:
    country = _normalize_key(value)
    return COUNTRY_ALIASES.get(country, country)


def _normalize_address_key(address, zipcode="", city="") -> tuple[str, str, str]:
    return (_normalize_key(address), normalize_text(zipcode), _normalize_key(city))


def find_address_column(df: pd.DataFrame) -> str | None:
    for column_name in df.columns:
        column_label = str(column_name)
        if column_label in ADDRESS_COLUMN_CANDIDATES or "address" in _tokenize_column_name(column_label):
            return column_label
    return None


def find_zipcode_column(df: pd.DataFrame) -> str | None:
    for column_name in df.columns:
        column_label = str(column_name)
        tokens = _tokenize_column_name(column_label)
        lowered = column_label.strip().lower()
        if (
            lowered == ZIPCODE_COLUMN
            or "zipcode" in lowered
            or "zip" in tokens
            or "postal" in tokens
            or "postcode" in tokens
        ):
            return column_label
    return None


def _find_city_column(df: pd.DataFrame) -> str | None:
    for column_name in df.columns:
        tokens = _tokenize_column_name(str(column_name))
        if "city" in tokens:
            return str(column_name)
    return None


def _find_state_column(df: pd.DataFrame) -> str | None:
    for column_name in df.columns:
        tokens = _tokenize_column_name(str(column_name))
        if "state" in tokens or "province" in tokens:
            return str(column_name)
    return None


def _find_country_column(df: pd.DataFrame) -> str | None:
    for column_name in df.columns:
        tokens = _tokenize_column_name(str(column_name))
        if "country" in tokens or "nation" in tokens:
            return str(column_name)
    return None


def table_supports_geocoding(df: pd.DataFrame) -> bool:
    return (
        not table_has_explicit_geo(df)
        and bool(
            find_address_column(df)
            or find_zipcode_column(df)
            or _find_area_column(df)
            or _find_state_column(df)
            or _find_country_column(df)
        )
    )


def table_has_explicit_geo(df: pd.DataFrame) -> bool:
    return bool(
        ddg_spatial_joins.geometry_attributes_from_dataframe(
            "__preview__",
            df,
            origin="native",
            source_kind="physical",
        )
    )


def find_geocoding_source_columns(df: pd.DataFrame) -> list[str]:
    source_columns: list[str] = []
    seen: set[str] = set()
    for finder in (
        find_address_column,
        find_zipcode_column,
        _find_city_column,
        _find_state_column,
        _find_country_column,
        _find_area_column,
    ):
        column_name = finder(df)
        if column_name and column_name not in seen:
            source_columns.append(column_name)
            seen.add(column_name)
    return source_columns


def point_output_column_name(source_column: str) -> str:
    return f"{source_column}{POINT_SUFFIX}"


def polygon_output_column_name(source_column: str) -> str:
    return f"{source_column}{POLYGON_SUFFIX}"


def is_generated_geocode_column(column_name: str) -> bool:
    lowered = str(column_name).strip().lower()
    return (
        lowered == POINT_COLUMN
        or lowered == POLYGON_COLUMN
        or lowered.endswith(POINT_SUFFIX)
        or lowered.endswith(POLYGON_SUFFIX)
    )


def generated_geo_columns(
    original_df: pd.DataFrame,
    geocoded_df: pd.DataFrame,
) -> list[str]:
    return [
        str(column_name)
        for column_name in geocoded_df.columns
        if column_name not in original_df.columns and is_generated_geocode_column(str(column_name))
    ]


def geo_association_metadata_path(table_name: str):
    return pandas_helper.temp_tables_dir() / f"{table_name}{GEO_ASSOCIATION_METADATA_SUFFIX}"


def _normalize_origin_attributes(values: object) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        attr = str(value).strip()
        if not attr or attr in seen:
            continue
        seen.add(attr)
        normalized.append(attr)
    return normalized


def _normalize_attribute_scores(values: object) -> dict[str, float]:
    normalized: dict[str, float] = {}
    if not isinstance(values, dict):
        return normalized
    for raw_attr, raw_score in values.items():
        attr = str(raw_attr).strip()
        if not attr:
            continue
        try:
            score = round(float(raw_score), 3)
        except (TypeError, ValueError):
            continue
        normalized[attr] = min(max(score, 0.0), 1.0)
    return normalized


def normalize_geo_association_details(
    details_map: object,
) -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    if not isinstance(details_map, dict):
        return normalized

    for raw_geo_attr, raw_detail in details_map.items():
        geo_attr = str(raw_geo_attr).strip()
        if not geo_attr or not isinstance(raw_detail, dict):
            continue
        detail: dict[str, object] = {}
        geometry_type = str(raw_detail.get("geometry_type") or "").strip().lower()
        if geometry_type:
            detail["geometry_type"] = geometry_type
        source_table = str(raw_detail.get("source_table") or "").strip()
        if source_table:
            detail["source_table"] = source_table
        source_origin = str(raw_detail.get("source_origin") or "").strip()
        if source_origin:
            detail["source_origin"] = source_origin
        source_attribute = str(raw_detail.get("source_attribute") or "").strip()
        if source_attribute:
            detail["source_attribute"] = source_attribute
        target_attribute = str(raw_detail.get("target_attribute") or "").strip()
        if target_attribute:
            detail["target_attribute"] = target_attribute

        origin_attributes = _normalize_origin_attributes(
            raw_detail.get("origin_attributes")
        )
        if origin_attributes:
            detail["origin_attributes"] = origin_attributes

        attribute_scores = _normalize_attribute_scores(
            raw_detail.get("attribute_scores")
        )
        if attribute_scores:
            detail["attribute_scores"] = attribute_scores

        if detail:
            normalized[geo_attr] = detail

    return normalized


def read_geo_association_details(table_name: str) -> dict[str, dict[str, object]]:
    path = geo_association_metadata_path(table_name)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return normalize_geo_association_details(payload)


def write_geo_association_details(
    table_name: str,
    details_map: object,
) -> None:
    path = geo_association_metadata_path(table_name)
    normalized = normalize_geo_association_details(details_map)
    if not normalized:
        path.unlink(missing_ok=True)
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)


def _point_geocoding_source_columns(df: pd.DataFrame) -> list[str]:
    source_columns: list[str] = []
    address_col = find_address_column(df)
    if address_col:
        source_columns.append(address_col)
    for finder in (
        find_zipcode_column,
        _find_city_column,
        _find_state_column,
        _find_country_column,
    ):
        column_name = finder(df)
        if column_name and column_name not in source_columns:
            source_columns.append(column_name)
    return source_columns


def _polygon_geocoding_source_columns(df: pd.DataFrame) -> list[str]:
    source_columns: list[str] = []
    for finder in (
        _find_area_column,
        find_zipcode_column,
        _find_state_column,
        _find_country_column,
    ):
        column_name = finder(df)
        if column_name and column_name not in source_columns:
            source_columns.append(column_name)
    return source_columns


def generated_geo_association_details(
    original_df: pd.DataFrame,
    geocoded_df: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    details: dict[str, dict[str, object]] = {}

    address_col = find_address_column(original_df)
    if address_col:
        point_column = point_output_column_name(address_col)
        if point_column in geocoded_df.columns and point_column not in original_df.columns:
            point_sources = _point_geocoding_source_columns(original_df)
            details[point_column] = {
                "geometry_type": "point",
                "origin_attributes": point_sources,
                "attribute_scores": {attr: 1.0 for attr in point_sources},
            }

    polygon_source_col = _preferred_polygon_source_column(original_df)
    if polygon_source_col:
        polygon_column = polygon_output_column_name(polygon_source_col)
        if polygon_column in geocoded_df.columns and polygon_column not in original_df.columns:
            polygon_sources = _polygon_geocoding_source_columns(original_df)
            details[polygon_column] = {
                "geometry_type": "polygon",
                "origin_attributes": polygon_sources,
                "attribute_scores": {attr: 1.0 for attr in polygon_sources},
            }

    return normalize_geo_association_details(details)


def _base_coords_for_row(row: pd.Series) -> tuple[float, float]:
    zipcode = _matching_row_value(row, _is_zipcode_column)
    if zipcode in ZIPCODE_CENTROIDS:
        return ZIPCODE_CENTROIDS[zipcode]

    city_key = (
        _normalize_key(_matching_row_value(row, _is_city_column)),
        _canonicalize_state(_matching_row_value(row, _is_state_column)),
        _canonicalize_country(_matching_row_value(row, _is_country_column)),
    )
    if city_key in CITY_DEFAULT_CENTROIDS:
        return CITY_DEFAULT_CENTROIDS[city_key]

    return CITY_DEFAULT_CENTROIDS[("chicago", "il", "us")]


def _lookup_hardcoded_coords(row: pd.Series, address_col: str) -> tuple[float, float] | None:
    address = normalize_text(row.get(address_col))
    if not address:
        return None

    candidates = [
        _normalize_address_key(
            address,
            _matching_row_value(row, _is_zipcode_column),
            _matching_row_value(row, _is_city_column),
        ),
        _normalize_address_key(address, _matching_row_value(row, _is_zipcode_column)),
        _normalize_address_key(address, city=_matching_row_value(row, _is_city_column)),
    ]
    for key in candidates:
        if key in HARDCODED_ADDRESS_COORDINATES:
            return HARDCODED_ADDRESS_COORDINATES[key]

    return _HARDCODED_ADDRESS_BY_ADDRESS.get(_normalize_key(address))


def _fallback_coords_for_row(row: pd.Series, address_col: str) -> tuple[float, float]:
    base_lat, base_lon = _base_coords_for_row(row)
    seed = "|".join(
        [
            normalize_text(row.get(address_col)),
            _matching_row_value(row, _is_zipcode_column),
            _matching_row_value(row, _is_city_column),
            _matching_row_value(row, _is_state_column),
            _matching_row_value(row, _is_country_column),
        ]
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    lat_offset = ((int(digest[:8], 16) / 0xFFFFFFFF) - 0.5) * 0.02
    lon_offset = ((int(digest[8:16], 16) / 0xFFFFFFFF) - 0.5) * 0.02
    return (base_lat + lat_offset, base_lon + lon_offset)


def geocode_row_to_point(row: pd.Series, address_col: str) -> str | None:
    if not normalize_text(row.get(address_col)):
        return None
    lat, lon = _lookup_hardcoded_coords(row, address_col) or _fallback_coords_for_row(row, address_col)
    return f"POINT({lon:.4f} {lat:.4f})"


def build_preview_geocoded_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    preview_df = augment_table_with_hardcoded_geo(df)
    return preview_df, find_geocoding_source_columns(preview_df)


def build_intermediate_geocoded_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    preview_df, marked_cols = build_preview_geocoded_table(df)
    if not generated_geo_columns(df, preview_df):
        return preview_df, df.copy(), marked_cols
    # Persist the original table plus appended geo columns.
    intermediate_df = preview_df.copy()
    return preview_df, intermediate_df, marked_cols


def _find_area_column(df: pd.DataFrame) -> str | None:
    for column_name in df.columns:
        if str(column_name) in AREA_COLUMN_CANDIDATES:
            return str(column_name)
    return None


def _preferred_polygon_source_column(df: pd.DataFrame) -> str | None:
    for finder in (
        _find_area_column,
        find_zipcode_column,
        _find_state_column,
        _find_country_column,
    ):
        column_name = finder(df)
        if column_name:
            return column_name
    return None


def _is_zipcode_column(column_name: str) -> bool:
    tokens = _tokenize_column_name(column_name)
    lowered = str(column_name).strip().lower()
    return (
        lowered == ZIPCODE_COLUMN
        or "zipcode" in lowered
        or "zip" in tokens
        or "postal" in tokens
        or "postcode" in tokens
    )


def _is_city_column(column_name: str) -> bool:
    return "city" in _tokenize_column_name(column_name)


def _is_state_column(column_name: str) -> bool:
    tokens = _tokenize_column_name(column_name)
    return "state" in tokens or "province" in tokens


def _is_country_column(column_name: str) -> bool:
    tokens = _tokenize_column_name(column_name)
    return "country" in tokens or "nation" in tokens


def _matching_row_value(row: pd.Series, matcher) -> str:
    for column_name in row.index:
        if matcher(str(column_name)):
            value = normalize_text(row.get(column_name))
            if value:
                return value
    return ""


def _bbox_polygon_wkt(south: float, west: float, north: float, east: float) -> str:
    return (
        f"POLYGON(({west:.4f} {south:.4f}, {east:.4f} {south:.4f}, {east:.4f} {north:.4f}, "
        f"{west:.4f} {north:.4f}, {west:.4f} {south:.4f}))"
    )


def _square_polygon_wkt(
    center: tuple[float, float],
    lat_radius: float,
    lon_radius: float,
) -> str:
    lat, lon = center
    south = lat - lat_radius
    north = lat + lat_radius
    west = lon - lon_radius
    east = lon + lon_radius
    return (
        f"POLYGON(({west:.4f} {south:.4f}, {east:.4f} {south:.4f}, {east:.4f} {north:.4f}, "
        f"{west:.4f} {north:.4f}, {west:.4f} {south:.4f}))"
    )


def _generated_polygon_for_area(area_key: str) -> str | None:
    if area_key in HARDCODED_AREA_POLYGONS:
        return HARDCODED_AREA_POLYGONS[area_key]

    center = HARDCODED_AREA_CENTERS.get(area_key)
    if center is None:
        return None

    lat_radius, lon_radius = AREA_POLYGON_RADII.get(area_key, AREA_POLYGON_RADII["default"])
    return _square_polygon_wkt(center, lat_radius, lon_radius)


def polygon_for_row(row: pd.Series) -> str | None:
    for area_col in AREA_COLUMN_CANDIDATES:
        area_key = _normalize_key(row.get(area_col))
        if not area_key:
            continue
        polygon = _generated_polygon_for_area(area_key)
        if polygon is not None:
            return polygon

    zipcode = _matching_row_value(row, _is_zipcode_column)
    if zipcode in ZIPCODE_CENTROIDS:
        return _square_polygon_wkt(ZIPCODE_CENTROIDS[zipcode], 0.0050, 0.0070)

    state_key = _canonicalize_state(_matching_row_value(row, _is_state_column))
    if state_key in STATE_BBOXES:
        return _bbox_polygon_wkt(*STATE_BBOXES[state_key])

    country_key = _canonicalize_country(_matching_row_value(row, _is_country_column))
    if country_key in COUNTRY_BBOXES:
        return _bbox_polygon_wkt(*COUNTRY_BBOXES[country_key])

    return None


def augment_table_with_hardcoded_geo(df: pd.DataFrame) -> pd.DataFrame:
    augmented_df = df.copy()
    address_col = find_address_column(augmented_df)
    polygon_source_col = _preferred_polygon_source_column(augmented_df)

    if address_col:
        point_column = point_output_column_name(address_col)
        if point_column in augmented_df.columns:
            return augmented_df
        geocoded = augmented_df.apply(lambda row: geocode_row_to_point(row, address_col), axis=1)
        if geocoded.notna().any():
            augmented_df[point_column] = geocoded
        return augmented_df

    if polygon_source_col:
        polygon_column = polygon_output_column_name(polygon_source_col)
        if polygon_column in augmented_df.columns:
            return augmented_df
        polygons = augmented_df.apply(polygon_for_row, axis=1)
        if polygons.notna().any():
            augmented_df[polygon_column] = polygons

    return augmented_df


def augment_data_lake_with_hardcoded_geo(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        name: augment_table_with_hardcoded_geo(df)
        for name, df in tables.items()
        if name not in MAPPING_TABLE_NAMES
    }
