import json
import random
import re
from itertools import combinations

import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False

import graph_joins


def _build_graphviz(nodes, rel_edges, spatial_edges) -> str:
    lines = ["graph G {", "  node [shape=box];", "  layout=neato;"]
    for node in sorted(nodes):
        lines.append(f'  "{node}";')

    for left, right, label in rel_edges:
        label_txt = f' [label="{label}"]' if label else ""
        lines.append(f'  "{left}" -- "{right}"{label_txt};')

    for left, right, label in spatial_edges:
        label_txt = (
            f' [label="{label}", color="#1e88e5", fontcolor="#1e88e5", style="dashed"]'
        )
        lines.append(f'  "{left}" -- "{right}"{label_txt};')

    lines.append("}")
    return "\n".join(lines)


def render_data_discovery_graph_section(
    west_lafayette_bbox, lafayette_default_bbox
):
    # Inject CSS once at the top (no spacing)
    st.markdown('<style>div[data-testid="stHtml"] iframe[src*="folium"]{height:1400px!important;min-height:1400px!important;max-height:1400px!important;}</style>', unsafe_allow_html=True)
    st.markdown('<div id="data-discovery-graph"></div>', unsafe_allow_html=True)
    st.subheader("Data Discovery Graph")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Show nodes"):
            st.session_state.graph_nodes = list(st.session_state.data_lake.keys())

    with col2:
        if st.button("Show relational joins"):
            st.session_state.graph_rel_edges = graph_joins.find_relational_joins(
                st.session_state.data_lake
            )

    with col3:
        if st.button("Show spatial joins"):
            st.session_state.graph_spatial_edges = graph_joins.find_spatial_joins(
                st.session_state.data_lake
            )

    nodes_data = [
        {
            "id": name,
            "rows": len(df),
            "columns": len(df.columns),
        }
        for name, df in st.session_state.data_lake.items()
        if name in st.session_state.graph_nodes
    ]

    edges_data = []
    # Track relational join pairs to avoid duplicate spatial joins
    rel_join_pairs = set()
    for left, right, label in st.session_state.graph_rel_edges:
        edges_data.append(
            {"source": left, "target": right, "label": label, "type": "relation"}
        )
        # Store both orderings since joins are bidirectional
        rel_join_pairs.add((left, right))
        rel_join_pairs.add((right, left))
    
    for left, right, label in st.session_state.graph_spatial_edges:
        # Skip if relational join already exists between this pair
        if (left, right) in rel_join_pairs or (right, left) in rel_join_pairs:
            continue
        
        # Randomly assign spatial join predicate type
        predicate = random.choice(["contain", "overlap", "distance"])
        
        # Format attributes as "table.column"
        # Parse the label (which contains column names) and format them
        # Remove "…" if present and split by comma
        label_clean = label.replace("…", "").strip()
        columns = [col.strip() for col in label_clean.split(",") if col.strip()]
        
        formatted_attrs = []
        df_left = st.session_state.data_lake.get(left, pd.DataFrame())
        df_right = st.session_state.data_lake.get(right, pd.DataFrame())
        
        for col in columns[:3]:  # Limit to first 3 columns
            # Check which table(s) have this column and add both if present in both
            attrs_for_col = []
            if col in df_left.columns:
                attrs_for_col.append(f"{left}.{col}")
            if col in df_right.columns:
                attrs_for_col.append(f"{right}.{col}")
            # If column exists in at least one table, add it
            if attrs_for_col:
                formatted_attrs.extend(attrs_for_col)
        
        attr_label = ", ".join(formatted_attrs)
        # Add "…" if original label had it or if there were more columns
        if "…" in label or len(columns) > 3:
            attr_label = f"{attr_label}…"
        
        edges_data.append(
            {"source": left, "target": right, "label": attr_label, "type": "spatial", "predicate": predicate}
        )

    d3_data = {
        "nodes": nodes_data,
        "links": edges_data,
    }

    html = f"""
    <div id="graph" style="width:100%;height:520px;border:1px solid #e0e0e0;border-radius:8px;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <script>
    const data = {json.dumps(d3_data)};
    const width = document.getElementById('graph').clientWidth;
    const height = 500;

    // Start nodes at random positions inside the viewport
    const padding = 40;
    data.nodes.forEach((d) => {{
      d.x = padding + Math.random() * (width - 2 * padding);
      d.y = padding + Math.random() * (height - 2 * padding);
    }});

    const svg = d3.select('#graph')
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    const color = d3.scaleOrdinal()
      .domain(['relation', 'spatial'])
      .range(['#4caf50', '#1e88e5']);

    // Create invisible wider paths for spatial joins to enable hover detection
    const linkHover = svg.append('g')
      .attr('stroke', 'transparent')
      .attr('stroke-width', 20)
      .selectAll('line')
      .data(data.links.filter(d => d.type === 'spatial'))
      .join('line')
      .style('cursor', 'pointer');

    const link = svg.append('g')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.7)
      .selectAll('line')
      .data(data.links)
      .join('line')
      .attr('stroke-width', 2)
      .attr('stroke', d => color(d.type))
      .attr('stroke-dasharray', d => d.type === 'spatial' ? '6,4' : '0');

    // Adjust link distance and charge strength based on whether there are links
    const hasLinks = data.links.length > 0;
    const linkDistance = hasLinks ? 200 : 120;
    const chargeStrength = hasLinks ? -300 : -140;
    
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d => d.id).distance(linkDistance))
      .force('charge', d3.forceManyBody().strength(chargeStrength))
      .force('x', d3.forceX(width / 2).strength(0.05))
      .force('y', d3.forceY(height / 2).strength(0.05))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .on('tick', ticked);

    const node = svg.append('g')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .selectAll('circle')
      .data(data.nodes)
      .join('circle')
      .attr('r', 26)
      .attr('fill', '#fdd835')
      .call(drag(simulation));

    const label = svg.append('g')
      .style('font', '16px sans-serif')
      .selectAll('text')
      .data(data.nodes)
      .join('text')
      .attr('dy', 5)
      .attr('text-anchor', 'middle')
      .text(d => d.id);

    const tooltip = d3.select('#graph').append('div')
      .style('position', 'absolute')
      .style('pointer-events', 'none')
      .style('background', 'rgba(0,0,0,0.75)')
      .style('color', '#fff')
      .style('padding', '6px 8px')
      .style('border-radius', '4px')
      .style('font', '12px sans-serif')
      .style('opacity', 0);

    const linkTooltip = d3.select('#graph').append('div')
      .style('position', 'absolute')
      .style('pointer-events', 'none')
      .style('background', 'rgba(0,0,0,0.75)')
      .style('color', '#fff')
      .style('padding', '6px 8px')
      .style('border-radius', '4px')
      .style('font', '12px sans-serif')
      .style('opacity', 0);

    node.on('mouseover', (event, d) => {{
      tooltip.style('opacity', 1)
        .html(`<strong>${{d.id}}</strong><br/>rows: ${{d.rows}}<br/>columns: ${{d.columns}}`);
    }}).on('mousemove', (event) => {{
      tooltip
        .style('left', (event.offsetX + 12) + 'px')
        .style('top', (event.offsetY + 12) + 'px');
    }}).on('mouseout', () => tooltip.style('opacity', 0));

    // Add hover handlers for spatial join edges using the invisible wider paths
    linkHover.on('mouseover', function(event, d) {{
      const graphRect = document.getElementById('graph').getBoundingClientRect();
      const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
      const targetId = typeof d.target === 'object' ? d.target.id : d.target;
      linkTooltip.style('opacity', 1)
        .html(`<strong>Spatial Join</strong><br/>${{sourceId}} & ${{targetId}}<br/>Spatial Predicate: <strong>${{d.predicate}}</strong><br/>Attributes involved: ${{d.label}}`);
      linkTooltip
        .style('left', (event.clientX - graphRect.left + 12) + 'px')
        .style('top', (event.clientY - graphRect.top + 12) + 'px');
      // Highlight the corresponding visible link
      link.filter(linkData => {{
        const linkSourceId = typeof linkData.source === 'object' ? linkData.source.id : linkData.source;
        const linkTargetId = typeof linkData.target === 'object' ? linkData.target.id : linkData.target;
        return (linkSourceId === sourceId && linkTargetId === targetId) || 
               (linkSourceId === targetId && linkTargetId === sourceId);
      }})
        .attr('stroke-width', 4)
        .attr('stroke-opacity', 1);
    }}).on('mousemove', function(event, d) {{
      const graphRect = document.getElementById('graph').getBoundingClientRect();
      linkTooltip
        .style('left', (event.clientX - graphRect.left + 12) + 'px')
        .style('top', (event.clientY - graphRect.top + 12) + 'px');
    }}).on('mouseout', function(event, d) {{
      linkTooltip.style('opacity', 0);
      // Reset the corresponding visible link
      const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
      const targetId = typeof d.target === 'object' ? d.target.id : d.target;
      link.filter(linkData => {{
        const linkSourceId = typeof linkData.source === 'object' ? linkData.source.id : linkData.source;
        const linkTargetId = typeof linkData.target === 'object' ? linkData.target.id : linkData.target;
        return (linkSourceId === sourceId && linkTargetId === targetId) || 
               (linkSourceId === targetId && linkTargetId === sourceId);
      }})
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.7);
    }});

    function ticked() {{
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      linkHover
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node
        .attr('cx', d => d.x = Math.max(26, Math.min(width - 26, d.x)))
        .attr('cy', d => d.y = Math.max(26, Math.min(height - 26, d.y)));

      label
        .attr('x', d => d.x)
        .attr('y', d => d.y);
    }}

    function drag(simulation) {{
      function dragstarted(event, d) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }}

      function dragged(event, d) {{
        d.fx = event.x;
        d.fy = event.y;
      }}

      function dragended(event, d) {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }}

      return d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
    }}
    </script>
    """
    components.html(html, height=540)

    # Add button to show join details
    if st.button("Show join details"):
        st.session_state.show_join_details = True
    
    # Display join details table if requested
    if st.session_state.get("show_join_details", False):
        join_details_data = []
        def _pair_join_attributes(label, table_left, table_right):
            if not label:
                return [("", "")]
            label_clean = label.replace("…", "").strip()
            parts = [part.strip() for part in label_clean.split(",") if part.strip()]
            left_attrs = []
            right_attrs = []
            for part in parts:
                if part.startswith(f"{table_left}."):
                    left_attrs.append(part.split(".", 1)[1])
                elif part.startswith(f"{table_right}."):
                    right_attrs.append(part.split(".", 1)[1])
                else:
                    left_attrs.append(part)
                    right_attrs.append(part)
            max_len = max(len(left_attrs), len(right_attrs), 1)
            pairs = []
            for i in range(max_len):
                left_attr = left_attrs[i] if i < len(left_attrs) else ""
                right_attr = right_attrs[i] if i < len(right_attrs) else ""
                pairs.append((left_attr, right_attr))
            return pairs

        for edge in edges_data:
            base_table_1 = edge["source"]
            base_table_2 = edge["target"]
            attributes = edge.get("label", "")
            if edge["type"] == "relation":
                join_type = "relational"
            elif edge["type"] == "spatial":
                predicate = edge.get("predicate", "unknown")
                join_type = f"spatial ({predicate})"
            else:
                join_type = edge["type"]
            
            attribute_pairs = _pair_join_attributes(attributes, base_table_1, base_table_2)
            for attr_left, attr_right in attribute_pairs:
                join_details_data.append({
                    "Base Table #1": base_table_1,
                    "Attribute (Table 1)": attr_left,
                    "Base Table #2": base_table_2,
                    "Attribute (Table 2)": attr_right,
                    "Type of join": join_type
                })
        
        if join_details_data:
            join_details_df = pd.DataFrame(join_details_data)
            join_details_df.index = range(1, len(join_details_df) + 1)
            # Render as HTML so font sizing reliably applies to the table.
            join_details_html = (
                join_details_df.style
                .set_properties(**{"font-size": "100%"})
                .set_table_styles([{"selector": "th", "props": [("font-size", "100%")]}])
                .to_html()
            )
            st.markdown(join_details_html, unsafe_allow_html=True)
            
            # Add "Next step" button below the table
            if st.button("Next step"):
                st.session_state.show_join_paths = True
        else:
            st.info("No joins are currently displayed in the graph.")

    # Only show Join Paths section if "Next step" has been clicked
    if st.session_state.get("show_join_paths", False):
        _render_interesting_paths(west_lafayette_bbox, lafayette_default_bbox)


def _extract_point_coords(point_str):
    """Extract lat, lon from POINT(lon lat) or similar formats."""
    if not isinstance(point_str, str):
        return None
    match = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', point_str, re.IGNORECASE)
    if match:
        lon, lat = float(match.group(1)), float(match.group(2))
        return (lat, lon)
    return None


def _extract_points_from_path(path_tables, data_lake, max_points_per_table=200):
    """
    Extract all point coordinates from all tables in a path.
    Returns: list of (lat, lon) tuples
    """
    all_coords = []
    
    for table_name in path_tables:
        if table_name not in data_lake:
            continue
        df = data_lake[table_name]
        
        # Try to find lat/lon columns
        lat_col = None
        lon_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'lat' in col_lower and lat_col is None:
                sample = df[col].dropna()
                if len(sample) > 0:
                    try:
                        sample_vals = pd.to_numeric(sample.head(10), errors='coerce').dropna()
                        if len(sample_vals) > 0 and all(-90 <= v <= 90 for v in sample_vals):
                            lat_col = col
                    except Exception:
                        pass
            if ('lon' in col_lower or 'lng' in col_lower) and lon_col is None:
                sample = df[col].dropna()
                if len(sample) > 0:
                    try:
                        sample_vals = pd.to_numeric(sample.head(10), errors='coerce').dropna()
                        if len(sample_vals) > 0 and all(-180 <= v <= 180 for v in sample_vals):
                            lon_col = col
                    except Exception:
                        pass
        
        # Extract coordinates from lat/lon columns
        if lat_col and lon_col:
            try:
                coords_df = df[[lat_col, lon_col]].dropna()
                for _, row in coords_df.head(max_points_per_table).iterrows():
                    try:
                        lat = float(row[lat_col])
                        lon = float(row[lon_col])
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            all_coords.append((lat, lon, table_name))
                    except Exception:
                        continue
            except Exception:
                pass
        
        # Extract coordinates from spatial columns (POINT format)
        spatial_cols = graph_joins._find_spatial_columns(df)
        for col in spatial_cols:
            sample = df[col].dropna().astype(str).head(max_points_per_table)
            for val in sample:
                coords = _extract_point_coords(val)
                if coords:
                    all_coords.append((coords[0], coords[1], table_name))
    
    return all_coords


def _compute_bbox_from_path(path_tables, data_lake):
    """
    Extract spatial coordinates from all tables in a path and compute bounding box.
    Returns: [south, west, north, east] or None if no coordinates found
    """
    all_coords = _extract_points_from_path(path_tables, data_lake)
    
    # Compute bounding box from all coordinates
    if all_coords:
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        bbox = [min(lats), min(lons), max(lats), max(lons)]
        return bbox
    
    return None


def _render_interesting_paths(west_lafayette_bbox, lafayette_default_bbox):
    st.markdown('<div id="interesting-paths"></div>', unsafe_allow_html=True)
    st.subheader("Join paths")

    XA = 40.0  # south (minimum latitude)
    XB = -87.0  # west (minimum longitude)
    XC = 41.0  # north (maximum latitude)
    XD = -86.0  # east (maximum longitude)
    row_1_col_1, row_1_col_2 = st.columns([1, 1])
    max_len_checked = row_1_col_1.checkbox("Path length at most 5", value=False)
    require_relational_checked = row_1_col_2.checkbox("At least one relational join", value=False)
    row_2_col_1, row_2_col_2 = st.columns([1, 1])
    min_len_checked = row_2_col_1.checkbox("Path length at least 3", value=False)
    require_spatial_checked = row_2_col_2.checkbox("At least one spatial join", value=False)
    max_len_input = st.text_input("Max path length (default 10)", value="")

    if st.button("Generate path"):
        # Build graph from edges_data (joinability information)
        edges_data = []
        rel_join_pairs = set()
        for left, right, label in st.session_state.graph_rel_edges:
            edges_data.append({"source": left, "target": right, "type": "relation"})
            rel_join_pairs.add((left, right))
            rel_join_pairs.add((right, left))
        
        for left, right, label in st.session_state.graph_spatial_edges:
            if (left, right) not in rel_join_pairs and (right, left) not in rel_join_pairs:
                predicate = random.choice(["contain", "overlap", "distance"])
                edges_data.append({"source": left, "target": right, "type": "spatial", "predicate": predicate})
        
        # Build adjacency list from edges
        graph = {}
        edge_types = {}
        for edge in edges_data:
            source = edge["source"]
            target = edge["target"]
            if source not in graph:
                graph[source] = []
            if target not in graph:
                graph[target] = []
            graph[source].append(target)
            graph[target].append(source)
            edge_key = frozenset((source, target))
            edge_types.setdefault(edge_key, set()).add(edge["type"])
        
        # Find all valid paths through the graph
        def find_paths(start, max_length, visited=None, current_path=None):
            if visited is None:
                visited = set()
            if current_path is None:
                current_path = []
            
            # Create copies to avoid mutation issues
            new_visited = visited.copy()
            new_path = current_path.copy()
            
            # Add current node to path
            new_path.append(start)
            new_visited.add(start)
            
            paths = []
            # If path has at least 2 nodes, add it as a valid path
            if len(new_path) >= 2:
                paths.append(tuple(new_path))
            
            # Continue exploring neighbors if we haven't reached max length
            if len(new_path) < max_length:
                if start in graph:
                    for neighbor in graph[start]:
                        if neighbor not in new_visited:
                            paths.extend(find_paths(neighbor, max_length, new_visited, new_path))
            
            return paths
        
        # Generate paths from all nodes
        candidate_paths = []
        max_path_length = 10
        if max_len_input.strip():
            try:
                max_path_length = max(2, int(max_len_input))
            except ValueError:
                max_path_length = 10
        if max_len_checked:
            max_path_length = min(max_path_length, 5)
        all_nodes = sorted(st.session_state.data_lake.keys())
        
        for start_node in all_nodes:
            paths_from_node = find_paths(start_node, max_path_length)
            candidate_paths.extend(paths_from_node)
        
        # Remove duplicates
        candidate_paths = list(set(candidate_paths))
        
        if max_len_checked:
            candidate_paths = [path for path in candidate_paths if len(path) <= 5]

        if min_len_checked:
            candidate_paths = [path for path in candidate_paths if len(path) >= 3]

        if require_spatial_checked or require_relational_checked:
            def _path_has_type(path, required_type):
                for i in range(len(path) - 1):
                    edge_key = frozenset((path[i], path[i + 1]))
                    if required_type in edge_types.get(edge_key, set()):
                        return True
                return False

            if require_spatial_checked:
                candidate_paths = [
                    path for path in candidate_paths if _path_has_type(path, "spatial")
                ]
            if require_relational_checked:
                candidate_paths = [
                    path for path in candidate_paths if _path_has_type(path, "relation")
                ]
        
        # Sort by length descending (longer paths first), then alphabetically
        candidate_paths = sorted(candidate_paths, key=lambda x: (-len(x), x))
        
        # Filter to only keep paths of length 5 (or longest available if none are length 5)
        filtered_paths = []
        if candidate_paths:
            max_path_len = max(len(p) for p in candidate_paths)
            # Prioritize paths of length 5, or use the maximum available length
            target_length = min(5, max_path_len)
            long_paths = [p for p in candidate_paths if len(p) == target_length]
            if long_paths:
                filtered_paths = long_paths
            else:
                # If no paths of target length, take longest available
                filtered_paths = [p for p in candidate_paths if len(p) == max_path_len]

        st.session_state["generated_path_count"] = len(filtered_paths)
        st.session_state["all_join_paths"] = filtered_paths
        st.session_state["interesting_paths"] = filtered_paths
        st.session_state["paths_page"] = 0
        st.session_state["selected_path_for_map"] = None

    if "path_map_selections" not in st.session_state:
        st.session_state["path_map_selections"] = {}

    if "selected_path_for_map" not in st.session_state:
        st.session_state["selected_path_for_map"] = None

    if "paths_page" not in st.session_state:
        st.session_state["paths_page"] = 0

    paths = st.session_state.get("interesting_paths", [])
    if "generated_path_count" in st.session_state:
        st.markdown(f"{st.session_state['generated_path_count']} join paths generated")

    if paths:
        page_size = 8
        total_paths = len(paths)
        page_count = (total_paths + page_size - 1) // page_size
        current_page = min(st.session_state["paths_page"], max(page_count - 1, 0))
        st.session_state["paths_page"] = current_page
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total_paths)

        col_paths, col_map = st.columns([1, 1.2])

        with col_paths:
            st.markdown("**Join Paths**")
            
            def get_join_info_discovery(table1, table2):
                """Get join information between two tables for data discovery graph."""
                for left, right, label in st.session_state.graph_rel_edges:
                    if (left == table1 and right == table2) or (left == table2 and right == table1):
                        return {"type": "relation", "attributes": label}
                for left, right, label in st.session_state.graph_spatial_edges:
                    if (left == table1 and right == table2) or (left == table2 and right == table1):
                        return {"type": "spatial", "attributes": label}
                return {"type": "relation", "attributes": ""}
            
            table_metadata = {}
            for table_name in st.session_state.data_lake.keys():
                df = st.session_state.data_lake[table_name]
                table_metadata[table_name] = {
                    "columns": list(df.columns),
                    "cardinality": len(df),
                    "num_columns": len(df.columns)
                }
            
            # Display paths with interactive visualization
            selected_path_idx = st.session_state.get("selected_path_for_map")
            for idx, path in enumerate(paths[start_idx:end_idx], start=start_idx + 1):
                is_selected = selected_path_idx == idx
                st.markdown(f"**Path {idx}:**" + (" ✓ Selected" if is_selected else ""))

                path_data = {
                    "tables": list(path),
                    "joins": []
                }

                for i in range(len(path) - 1):
                    join_info = get_join_info_discovery(path[i], path[i+1])
                    path_data["joins"].append({
                        "from": path[i],
                        "to": path[i+1],
                        "attributes": join_info["attributes"],
                        "type": join_info["type"]
                    })

                highlight_style = "border: 2px solid #4caf50; background: #f1f8e9;" if is_selected else "border: 1px solid #e0e0e0; background: #f9f9f9;"
                path_html = f"""
                <div id="discovery_path_{idx}_container" style="margin: 10px 0; padding: 15px; {highlight_style} border-radius: 8px; position: relative; overflow: visible; min-height: 120px;">
                    <div id="discovery_path_{idx}_visualization" style="display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
                    </div>
                </div>
                <script>
                (function() {{
                    const pathData = {json.dumps(path_data)};
                    const tableMetadata = {json.dumps(table_metadata)};
                    const container = document.getElementById('discovery_path_{idx}_visualization');
                    
                    pathData.tables.forEach((table, i) => {{
                        const tableNode = document.createElement('span');
                        tableNode.className = 'table-node';
                        tableNode.textContent = table;
                        tableNode.style.cssText = 'padding: 8px 12px; background: #fff; border: 2px solid #1e88e5; border-radius: 6px; cursor: pointer; font-weight: 600; transition: all 0.2s;';
                        
                        const tableInfo = tableMetadata[table];
                        const tooltip = document.createElement('div');
                        tooltip.style.cssText = 'position: absolute; background: rgba(0,0,0,0.9); color: white; padding: 10px; border-radius: 6px; font-size: 12px; z-index: 1000; pointer-events: none; opacity: 0; transition: opacity 0.2s; max-width: 300px;';
                        tooltip.innerHTML = `<strong>${{table}}</strong><br/>Cardinality: ${{tableInfo.cardinality}}<br/>Columns: ${{tableInfo.num_columns}}<br/><small>${{tableInfo.columns.slice(0, 5).join(', ')}}${{tableInfo.columns.length > 5 ? '...' : ''}}</small>`;
                        document.body.appendChild(tooltip);
                        
                        tableNode.addEventListener('mouseenter', (e) => {{
                            tooltip.style.opacity = '1';
                            tooltip.style.left = (e.pageX + 10) + 'px';
                            tooltip.style.top = (e.pageY + 10) + 'px';
                            tableNode.style.background = '#e3f2fd';
                        }});
                        tableNode.addEventListener('mouseleave', () => {{
                            tooltip.style.opacity = '0';
                            tableNode.style.background = '#fff';
                        }});
                        tableNode.addEventListener('mousemove', (e) => {{
                            tooltip.style.left = (e.pageX + 10) + 'px';
                            tooltip.style.top = (e.pageY + 10) + 'px';
                        }});
                        
                        container.appendChild(tableNode);
                        
                        if (i < pathData.tables.length - 1) {{
                            const arrow = document.createElement('span');
                            arrow.textContent = '--';
                            arrow.style.cssText = 'font-size: 20px; color: #666; margin: 0 4px;';
                            
                            const edgeInfo = pathData.joins[i];
                            const edgeTooltip = document.createElement('div');
                            edgeTooltip.style.cssText = 'position: absolute; background: rgba(0,0,0,0.9); color: white; padding: 10px; border-radius: 6px; font-size: 12px; z-index: 1000; pointer-events: none; opacity: 0; transition: opacity 0.2s;';
                            edgeTooltip.innerHTML = `<strong>Join Attributes:</strong><br/>${{edgeInfo.attributes}}<br/><small>Type: ${{edgeInfo.type}}</small>`;
                            document.body.appendChild(edgeTooltip);
                            
                            arrow.addEventListener('mouseenter', (e) => {{
                                edgeTooltip.style.opacity = '1';
                                edgeTooltip.style.left = (e.pageX + 10) + 'px';
                                edgeTooltip.style.top = (e.pageY + 10) + 'px';
                                arrow.style.color = '#1e88e5';
                                arrow.style.fontWeight = 'bold';
                            }});
                            arrow.addEventListener('mouseleave', () => {{
                                edgeTooltip.style.opacity = '0';
                                arrow.style.color = '#666';
                                arrow.style.fontWeight = 'normal';
                            }});
                            arrow.addEventListener('mousemove', (e) => {{
                                edgeTooltip.style.left = (e.pageX + 10) + 'px';
                                edgeTooltip.style.top = (e.pageY + 10) + 'px';
                            }});
                            
                            container.appendChild(arrow);
                        }}
                    }});
                }})();
                </script>
                """
                components.html(path_html, height=180)

                if st.button("Show on the map", key=f"show_path_{idx}"):
                    st.session_state["selected_path_for_map"] = idx
                    # Compute bounding box from spatial information in all tables in the path
                    computed_bbox = _compute_bbox_from_path(path, st.session_state.data_lake)
                    if computed_bbox:
                        st.session_state.path_map_selections[f"path_{idx}"] = {
                            "bbox": computed_bbox,
                            "region_name": "Computed from path tables",
                            "method": "spatial_coordinates"
                        }
                    else:
                        # Fallback to default if no spatial data found
                        st.session_state.path_map_selections[f"path_{idx}"] = {
                            "bbox": [XA, XB, XC, XD],
                            "region_name": "Path Region",
                            "method": "default_coordinates"
                        }
                    st.rerun()

            if page_count > 1:
                remaining_paths = max(0, total_paths - end_idx)
                next_count = min(page_size, remaining_paths)
                next_label = f"Next {next_count} paths" if next_count else "Next paths"
                if st.button(
                    next_label,
                    key="paths_next_page",
                    disabled=current_page >= page_count - 1,
                ):
                    st.session_state["paths_page"] = current_page + 1
                    st.rerun()

        with col_map:
            st.markdown("**Map View**")

            selected_path_idx = st.session_state.get("selected_path_for_map")

            path_region_info = None
            path_map_bbox = None

            if selected_path_idx:
                stored_info = st.session_state.path_map_selections.get(f"path_{selected_path_idx}")
                if stored_info:
                    if isinstance(stored_info, dict):
                        path_region_info = stored_info
                        path_map_bbox = stored_info.get("bbox")
                    else:
                        path_map_bbox = stored_info
                else:
                    path_map_bbox = None
                # Use stored bbox if available, otherwise compute from path
                if path_map_bbox is None and paths:
                    path = paths[selected_path_idx - 1]
                    path_map_bbox = _compute_bbox_from_path(path, st.session_state.data_lake)
                # Fallback to default if still no bbox
                if path_map_bbox is None:
                    path_map_bbox = west_lafayette_bbox
            else:
                path_map_bbox = None

            # Calculate center and zoom based on the bounding box
            if path_map_bbox:
                center_lat = (path_map_bbox[0] + path_map_bbox[2]) / 2
                center_lon = (path_map_bbox[1] + path_map_bbox[3]) / 2
                # Calculate appropriate zoom level based on bbox size
                lat_range = path_map_bbox[2] - path_map_bbox[0]
                lon_range = path_map_bbox[3] - path_map_bbox[1]
                max_range = max(lat_range, lon_range)
                # Adjust zoom based on bbox size (smaller bbox = higher zoom)
                if max_range < 0.1:
                    zoom_level = 13
                elif max_range < 0.5:
                    zoom_level = 11
                elif max_range < 1.0:
                    zoom_level = 9
                elif max_range < 5.0:
                    zoom_level = 7
                else:
                    zoom_level = 5
            else:
                center_lat = (lafayette_default_bbox[0] + lafayette_default_bbox[2]) / 2
                center_lon = (lafayette_default_bbox[1] + lafayette_default_bbox[3]) / 2
                zoom_level = 12

            m_path = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom_level,
                tiles='OpenStreetMap'
            )

            if path_map_bbox is not None:
                folium.Rectangle(
                    bounds=[[path_map_bbox[0], path_map_bbox[1]], [path_map_bbox[2], path_map_bbox[3]]],
                    color="#0066cc",
                    fillColor="#0066cc",
                    fillOpacity=0.3,
                    weight=2,
                    opacity=0.8,
                    popup=f"Path {selected_path_idx}",
                    tooltip=f"Path {selected_path_idx} bounding box"
                ).add_to(m_path)
                
                # Plot points from tables in the path
                if selected_path_idx and paths:
                    path = paths[selected_path_idx - 1]
                    points = _extract_points_from_path(path, st.session_state.data_lake, max_points_per_table=200)
                    
                    # Group points by table for color coding
                    table_colors = {}
                    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
                             'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 
                             'lightgreen', 'gray', 'black', 'lightgray']
                    for idx, table_name in enumerate(path):
                        table_colors[table_name] = colors[idx % len(colors)]
                    
                    # Plot points with different colors for each table
                    for lat, lon, table_name in points:
                        # Check if point is within bounding box (with small margin)
                        margin = 0.01
                        if (path_map_bbox[0] - margin <= lat <= path_map_bbox[2] + margin and 
                            path_map_bbox[1] - margin <= lon <= path_map_bbox[3] + margin):
                            color = table_colors.get(table_name, 'blue')
                            folium.CircleMarker(
                                location=[lat, lon],
                                radius=4,
                                popup=f"{table_name}<br>({lat:.4f}, {lon:.4f})",
                                tooltip=table_name,
                                color=color,
                                fillColor=color,
                                fillOpacity=0.7,
                                weight=1
                            ).add_to(m_path)

            # Use st_folium if available for better height control, otherwise use components.html
            if HAS_ST_FOLIUM:
                st_folium(m_path, width=None, height=1400, returned_objects=[])
            else:
                map_html = m_path._repr_html_()
                # Modify folium HTML to set map div height
                import re
                # Replace height in map div style
                map_html = re.sub(r'(<div[^>]*id="[^"]*map[^"]*"[^>]*style="[^"]*height:\s*)\d+px', r'\11400px', map_html)
                # Add height to style if not present
                if 'id="map' in map_html:
                    if 'style=' in map_html:
                        map_html = re.sub(r'(<div[^>]*id="[^"]*map[^"]*"[^>]*style="[^"]*)"', r'\1 height: 1400px !important;"', map_html)
                    else:
                        map_html = re.sub(r'(<div[^>]*id="[^"]*map[^"]*")', r'\1 style="height: 1400px !important;"', map_html)
                
                components.html(map_html, height=1400)
            # Add JavaScript to force resize after component renders (inline, no spacing)
            st.markdown('<script>(function(){{function forceResize(){{var iframes=document.querySelectorAll("iframe");iframes.forEach(function(iframe){{iframe.style.setProperty("height","1400px","important");iframe.style.setProperty("min-height","1400px","important");iframe.style.setProperty("max-height","1400px","important");iframe.setAttribute("height","1400");}});}}forceResize();setTimeout(forceResize,100);setTimeout(forceResize,500);setTimeout(forceResize,1000);setTimeout(forceResize,2000);var observer=new MutationObserver(forceResize);observer.observe(document.body,{{childList:true,subtree:true}});}})();</script>', unsafe_allow_html=True)

            if selected_path_idx and path_region_info:
                region_name = path_region_info.get("region_name", "Unknown") if isinstance(path_region_info, dict) else "Unknown"
                method = path_region_info.get("method", "unknown") if isinstance(path_region_info, dict) else "unknown"
                st.success(f"**Path {selected_path_idx} Region:** {region_name} (detected via {method})")
            elif selected_path_idx and path_map_bbox:
                bbox_label_path = (
                    f"{path_map_bbox[0]:.4f}, {path_map_bbox[1]:.4f}, "
                    f"{path_map_bbox[2]:.4f}, {path_map_bbox[3]:.4f}"
                )
                st.info(f"**Path {selected_path_idx}:** Region highlighted on map (bbox: {bbox_label_path})")
