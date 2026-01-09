import math
import re
from collections import deque
from difflib import get_close_matches

import pandas as pd


def fuzzy_match_attribute(user_input, data_lake):
    """
    Fuzzy match user input to actual table.column in data lake.
    Returns a list of (table_name, column_name) tuples - can return multiple matches
    if the same column appears in multiple tables.
    """
    if not user_input or not user_input.strip():
        return []
    
    user_input = user_input.strip()
    matches = []
    
    # Collect all table.column pairs
    all_attributes = []
    for table_name, df in data_lake.items():
        for col_name in df.columns:
            full_name = f"{table_name}.{col_name}"
            all_attributes.append((table_name, col_name, full_name))
    
    if not all_attributes:
        return []
    
    # If user input contains a dot, try to match table.column format
    if "." in user_input:
        parts = user_input.split(".", 1)
        user_table = parts[0].strip().lower()
        user_col = parts[1].strip().lower()
        
        # First try exact match (case insensitive)
        for table_name, col_name, full_name in all_attributes:
            if table_name.lower() == user_table and col_name.lower() == user_col:
                matches.append((table_name, col_name))
        
        if matches:
            return matches
        
        # Then try fuzzy match on both parts
        table_matches = get_close_matches(user_table, [t.lower() for t, _, _ in all_attributes], n=1, cutoff=0.6)
        if table_matches:
            matched_table_lower = table_matches[0]
            # Find columns in the matched table
            table_columns = [(t, c) for t, c, _ in all_attributes if t.lower() == matched_table_lower]
            if table_columns:
                col_matches = get_close_matches(user_col, [c.lower() for _, c in table_columns], n=1, cutoff=0.6)
                if col_matches:
                    matched_col_lower = col_matches[0]
                    for t, c in table_columns:
                        if c.lower() == matched_col_lower:
                            matches.append((t, c))
        return matches
    else:
        # User input is just a column name, search across all tables
        # First try exact match (case insensitive) - return all matches
        for table_name, col_name, full_name in all_attributes:
            if col_name.lower() == user_input.lower():
                matches.append((table_name, col_name))
        
        if matches:
            return matches
        
        # Then try fuzzy match on column names - return all tables with matching column
        col_matches = get_close_matches(user_input.lower(), [c.lower() for _, c, _ in all_attributes], n=1, cutoff=0.6)
        if col_matches:
            matched_col_lower = col_matches[0]
            for table_name, col_name, full_name in all_attributes:
                if col_name.lower() == matched_col_lower:
                    matches.append((table_name, col_name))
        return matches


def _find_lat_lon_columns(df):
    lat_col = None
    lon_col = None
    for col in df.columns:
        col_lower = col.lower()
        if lat_col is None and "lat" in col_lower:
            sample = pd.to_numeric(df[col].dropna().head(20), errors="coerce").dropna()
            if len(sample) > 0 and sample.between(-90, 90).all():
                lat_col = col
        if lon_col is None and ("lon" in col_lower or "lng" in col_lower):
            sample = pd.to_numeric(df[col].dropna().head(20), errors="coerce").dropna()
            if len(sample) > 0 and sample.between(-180, 180).all():
                lon_col = col
    return lat_col, lon_col


def _extract_point_coords_series(series):
    coords = series.astype(str).str.extract(
        r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)", flags=re.IGNORECASE
    )
    if coords.empty:
        return None, None
    lons = pd.to_numeric(coords[0], errors="coerce")
    lats = pd.to_numeric(coords[1], errors="coerce")
    return lats, lons


def _get_lat_lon(df):
    lat_col, lon_col = _find_lat_lon_columns(df)
    if lat_col and lon_col:
        lats = pd.to_numeric(df[lat_col], errors="coerce")
        lons = pd.to_numeric(df[lon_col], errors="coerce")
        return lats, lons

    for col in df.columns:
        if "point" in col.lower() or "location" in col.lower() or "coord" in col.lower():
            lats, lons = _extract_point_coords_series(df[col])
            if lats is not None and lons is not None:
                return lats, lons
    return None, None


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return r * c


def _spatial_knn_join(left_df, right_df, left_cols, right_cols, max_rows):
    left_sample = left_df.head(max_rows).reset_index(drop=True)
    right_sample = right_df.reset_index(drop=True)
    left_lats, left_lons = _get_lat_lon(left_sample)
    right_lats, right_lons = _get_lat_lon(right_sample)
    if left_lats is None or right_lats is None:
        return None

    rows = []
    for idx, row in left_sample.iterrows():
        lat = left_lats.iloc[idx]
        lon = left_lons.iloc[idx]
        if pd.isna(lat) or pd.isna(lon):
            rows.append([None] * len(right_cols))
            continue
        best_idx = None
        best_dist = None
        for r_idx in range(len(right_sample)):
            r_lat = right_lats.iloc[r_idx]
            r_lon = right_lons.iloc[r_idx]
            if pd.isna(r_lat) or pd.isna(r_lon):
                continue
            dist = _haversine_km(lat, lon, r_lat, r_lon)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = r_idx
        if best_idx is None:
            rows.append([None] * len(right_cols))
        else:
            rows.append(right_sample.loc[best_idx, right_cols].tolist())

    right_joined = pd.DataFrame(rows, columns=right_cols)
    merged = pd.concat([left_sample[left_cols].reset_index(drop=True), right_joined], axis=1)
    return merged


def generate_joined_tuples(
    matched_attributes,
    data_lake,
    max_rows=100,
    join_path=None,
    join_edges=None,
    spatial_mode=None,
    spatial_distance_km=None,
):
    """
    Generate joined results as tuples from matched attributes.
    matched_attributes is a dict mapping user_input to a list of (table_name, col_name) tuples.
    Returns a list of tuples where each tuple contains values from all matched attributes.
    Also returns the column order for display (only column names, not table.column).
    """
    if not matched_attributes:
        return [], []
    
    # Preserve user input order
    user_input_order = list(matched_attributes.keys())
    
    # Flatten all matches - collect all (table, column) pairs
    all_matches = []
    for user_input in user_input_order:
        matches = matched_attributes[user_input]
        all_matches.extend(matches)
    
    # Group attributes by table
    by_table = {}
    for table_name, col_name in all_matches:
        if table_name not in by_table:
            by_table[table_name] = []
        if col_name not in by_table[table_name]:
            by_table[table_name].append(col_name)
    
    # If all attributes are from the same table, simple case
    if len(by_table) == 1:
        table_name = list(by_table.keys())[0]
        df = data_lake[table_name]
        # Get unique columns in order (preserve first occurrence order)
        col_order = []
        seen_cols = set()
        for user_input in user_input_order:
            for table, col in matched_attributes[user_input]:
                if table == table_name and col not in seen_cols:
                    col_order.append(col)
                    seen_cols.add(col)
        
        # Get sample rows
        sample_df = df[col_order].head(max_rows)
        # Convert to list of tuples
        return sample_df.values.tolist(), col_order
    
    def _unique_cols(cols):
        seen = set()
        out = []
        for col in cols:
            if col not in seen:
                out.append(col)
                seen.add(col)
        return out

    # Multiple tables - need to join
    tables_list = join_path or list(by_table.keys())
    if len(tables_list) >= 2:
        merged = data_lake[tables_list[0]]

        for i in range(1, len(tables_list)):
            next_table = tables_list[i]
            next_df = data_lake[next_table]
            edge_type = None
            if join_edges:
                for edge in join_edges:
                    if edge["from"] == tables_list[i - 1] and edge["to"] == next_table:
                        edge_type = edge["type"]
                        break
                    if edge["from"] == next_table and edge["to"] == tables_list[i - 1]:
                        edge_type = edge["type"]
                        break

            if edge_type == "spatial" and spatial_mode == "distance":
                spatial_joined = _spatial_knn_join(
                    merged, next_df, list(merged.columns), list(next_df.columns), max_rows
                )
                if spatial_joined is not None:
                    merged = spatial_joined
                    continue

            common_cols = list(set(merged.columns) & set(next_df.columns))
            if not common_cols:
                continue
            join_col = common_cols[0]
            merged = pd.merge(
                merged[_unique_cols(list(merged.columns))],
                next_df[_unique_cols(list(next_df.columns))],
                on=join_col,
                how='inner',
                suffixes=('', '_dup')
            )
            
            # Get column order based on user input order (only column names)
            col_order = []
            seen_cols = set()
            for user_input in user_input_order:
                for table_name, col_name in matched_attributes[user_input]:
                    # Check if column exists in merged (might have suffix)
                    if col_name in merged.columns:
                        if col_name not in seen_cols:
                            col_order.append(col_name)
                            seen_cols.add(col_name)
                    else:
                        # Check for suffixed version
                        for col in merged.columns:
                            if (col.startswith(col_name + '_') or col == col_name) and col not in seen_cols:
                                col_order.append(col)
                                seen_cols.add(col)
                                break
            
            if col_order:
                # Filter to only columns that exist
                col_order = [col for col in col_order if col in merged.columns]
                sample_df = merged[col_order].head(max_rows)
                return sample_df.values.tolist(), col_order
    
    # Fallback: return individual table results (concatenate)
    # This is not ideal but handles cases where tables can't be joined
    all_results = []
    col_order = []
    seen_cols = set()
    for user_input in user_input_order:
        for table_name, col_name in matched_attributes[user_input]:
            if col_name not in seen_cols:
                df = data_lake[table_name]
                if col_name in df.columns:
                    sample_vals = df[col_name].head(max_rows).tolist()
                    all_results.append(sample_vals)
                    col_order.append(col_name)
                    seen_cols.add(col_name)
    
    # Transpose to get tuples
    if all_results:
        max_len = max(len(r) for r in all_results)
        tuples = []
        for i in range(min(max_len, max_rows)):
            tuple_row = tuple(r[i] if i < len(r) else None for r in all_results)
            tuples.append(tuple_row)
        return tuples, col_order
    
    return [], []


def infer_spatial_preferences(user_inputs):
    """
    Infer spatial predicate from natural-language inputs.
    Returns dict with mode ("distance", "contain", "intersect" or None) and distance_km.
    """
    if not user_inputs:
        return {"mode": None, "distance_km": None}

    text = " ".join(str(val) for val in user_inputs).lower()

    # Prefer explicit distance/nearest signals.
    distance_km = None
    distance_match = re.search(
        r'(\d+(?:\.\d+)?)\s*(km|kilometers|kilometres|miles|mi)\b', text
    )
    if distance_match:
        distance_val = float(distance_match.group(1))
        unit = distance_match.group(2)
        if unit in {"miles", "mi"}:
            distance_val *= 1.60934
        distance_km = distance_val
        return {"mode": "distance", "distance_km": distance_km}

    if re.search(r'\b(closest|nearest|nearby|near)\b', text):
        return {"mode": "distance", "distance_km": None}

    if re.search(r'\b(within|inside|contained|contain)\b', text):
        return {"mode": "contain", "distance_km": None}

    if re.search(r'\b(intersect|intersection|overlap|overlapping|cross)\b', text):
        return {"mode": "intersect", "distance_km": None}

    return {"mode": None, "distance_km": None}


def find_min_join_path(
    matched_attributes,
    rel_edges,
    spatial_edges,
    allow_spatial,
    require_spatial,
    max_tables,
):
    """
    Find a minimal-table join path covering all matched inputs.
    If allow_spatial is False, only relational edges are allowed.
    If require_spatial is True, at least one spatial edge must appear in the path.
    """
    if not matched_attributes:
        return None

    inputs = list(matched_attributes.keys())
    all_mask = (1 << len(inputs)) - 1

    table_input_mask = {}
    input_tables = set()
    for idx, user_input in enumerate(inputs):
        for table_name, _ in matched_attributes[user_input]:
            input_tables.add(table_name)
            table_input_mask[table_name] = table_input_mask.get(table_name, 0) | (1 << idx)

    if not input_tables:
        return None

    adjacency = {}
    edge_labels = {}

    def _add_edge(left, right, label, edge_type):
        adjacency.setdefault(left, []).append((right, edge_type))
        adjacency.setdefault(right, []).append((left, edge_type))
        key = (min(left, right), max(left, right), edge_type)
        edge_labels[key] = label or ""

    for left, right, label in rel_edges:
        _add_edge(left, right, label, "relation")
    for left, right, label in spatial_edges:
        _add_edge(left, right, label, "spatial")

    def _search_with_bridge_limit(max_bridge):
        queue = deque()
        visited = set()
        prev = {}

        for table in sorted(input_tables):
            mask = table_input_mask.get(table, 0)
            state = (table, mask, 0, False)
            queue.append((state, 1))
            visited.add(state)

        while queue:
            (table, mask, bridge_count, has_spatial), depth = queue.popleft()
            if mask == all_mask and (not require_spatial or has_spatial):
                return (table, mask, bridge_count, has_spatial), prev
            if depth >= max_tables:
                continue

            for neighbor, edge_type in adjacency.get(table, []):
                if edge_type == "spatial" and not allow_spatial:
                    continue
                next_bridge = bridge_count + (0 if neighbor in input_tables else 1)
                if next_bridge > max_bridge:
                    continue
                next_has_spatial = has_spatial or (edge_type == "spatial")
                next_mask = mask | table_input_mask.get(neighbor, 0)
                next_state = (neighbor, next_mask, next_bridge, next_has_spatial)
                if next_state in visited:
                    continue
                visited.add(next_state)
                prev[next_state] = (table, mask, bridge_count, has_spatial, edge_type)
                queue.append((next_state, depth + 1))

        return None, None

    max_bridge_tables = max(0, max_tables - 1)
    final_state = None
    final_prev = None
    for max_bridge in range(max_bridge_tables + 1):
        final_state, final_prev = _search_with_bridge_limit(max_bridge)
        if final_state:
            break

    if not final_state:
        return None

    path_tables = []
    path_edges = []
    state = final_state
    while True:
        table, mask, bridge_count, has_spatial = state
        path_tables.append(table)
        if state not in final_prev:
            break
        prev_table, prev_mask, prev_bridge_count, prev_has_spatial, edge_type = final_prev[state]
        label_key = (min(prev_table, table), max(prev_table, table), edge_type)
        path_edges.append(
            {
                "from": prev_table,
                "to": table,
                "type": edge_type,
                "attributes": edge_labels.get(label_key, ""),
            }
        )
        state = (prev_table, prev_mask, prev_bridge_count, prev_has_spatial)

    path_tables.reverse()
    path_edges.reverse()

    bridge_tables = [t for t in path_tables if t not in input_tables]
    return {
        "tables": path_tables,
        "edges": path_edges,
        "bridge_tables": bridge_tables,
        "used_spatial": any(edge["type"] == "spatial" for edge in path_edges),
    }
