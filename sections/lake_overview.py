from __future__ import annotations

import json
import math
import re
from html import escape
from typing import Iterable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import detect_geo
import ddg_spatial_joins


EXPLICIT_GEO_COLOR = "#0d47a1"
IMPLICIT_GEO_COLOR = "#bfdbfe"
NO_GEO_COLOR = "#ffffff"
NO_GEO_BORDER = "#cbd5e1"

LANE_CONFIG = {
    "explicit": {
        "title": "Explicit Geo",
        "subtitle": "Map-ready coordinates, points, or polygons",
        "color": EXPLICIT_GEO_COLOR,
        "accent": "#dbeafe",
        "empty": "No tables with map-ready geometry are loaded yet.",
        "badge": "Explicit Geo",
    },
    "implicit": {
        "title": "Implicit Geo",
        "subtitle": "Place names, addresses, postal areas, or regions",
        "color": IMPLICIT_GEO_COLOR,
        "accent": "#eff6ff",
        "empty": "No tables with place-based geo context are loaded yet.",
        "badge": "Implicit Geo",
    },
    "none": {
        "title": "No Geo",
        "subtitle": "No geo signals detected from current columns",
        "color": NO_GEO_COLOR,
        "accent": "#f8fafc",
        "empty": "Every loaded table currently has some geo signal.",
        "badge": "No Geo",
    },
}

def _attribute_coverage(df: pd.DataFrame, attribute_name: str) -> dict[str, object]:
    total_rows = len(df)
    populated = 0

    if " + " in attribute_name:
        left_col, right_col = [part.strip() for part in attribute_name.split(" + ", 1)]
        if left_col in df.columns and right_col in df.columns and total_rows:
            populated = int((df[left_col].notna() & df[right_col].notna()).sum())
    elif attribute_name in df.columns and total_rows:
        populated = int(df[attribute_name].notna().sum())

    percentage = (populated / total_rows) if total_rows else 0.0
    return {
        "populated": populated,
        "percentage": percentage,
        "label": f"{populated:,}/{total_rows:,} rows ({percentage:.0%})" if total_rows else "0 rows",
    }


def _estimate_node_box(label: str) -> tuple[int, int]:
    target_chars_per_line = 12
    char_width_px = 6
    line_height_px = 16

    normalized = re.sub(r"[_-]+", " ", str(label)).strip()
    if not normalized:
        return 54, 36

    words = normalized.split()
    lines: list[str] = []
    current_line = ""

    for raw_word in words:
        chunks = [
            raw_word[index:index + target_chars_per_line]
            for index in range(0, len(raw_word), target_chars_per_line)
        ] or [raw_word]
        for chunk in chunks:
            if not current_line:
                current_line = chunk
            elif len(current_line) + 1 + len(chunk) <= target_chars_per_line:
                current_line = f"{current_line} {chunk}"
            else:
                lines.append(current_line)
                current_line = chunk

    if current_line:
        lines.append(current_line)

    if not lines:
        lines = [normalized[:target_chars_per_line]]

    longest_line_chars = max(len(line) for line in lines)
    text_height_px = max(1, len(lines)) * line_height_px
    width_px = max(52, min(126, int((longest_line_chars * char_width_px) + 8)))
    height_px = max(36, int(text_height_px * 2))
    return width_px, height_px


def _lane_for_table(
    explicit_geo_attributes: list[dict[str, object]],
    implicit_geo_attributes: list[dict[str, object]],
) -> str:
    if explicit_geo_attributes:
        return "explicit"
    if implicit_geo_attributes:
        return "implicit"
    return "none"


def _build_table_records(
    *,
    chosen_lake: str | None,
    data_lake: dict[str, pd.DataFrame],
    uploaded_table_names: Iterable[str],
) -> list[dict[str, object]]:
    uploaded_name_set = {str(name) for name in uploaded_table_names}
    manual_geo_details = st.session_state.get("ddg_manual_geo_augmentation_details", {})
    geometry_map = ddg_spatial_joins.discover_table_geometries(
        chosen_lake=chosen_lake,
        in_memory_tables=data_lake,
        manual_geo_details=manual_geo_details,
    )

    table_records: list[dict[str, object]] = []
    for table_name, df in sorted(data_lake.items()):
        raw_geometry_attrs = geometry_map.get(table_name, [])
        explicit_geo_attributes = []
        explicit_column_kinds: dict[str, str] = {}
        for attr in raw_geometry_attrs:
            coverage = _attribute_coverage(df, attr.attribute_name)
            explicit_geo_attributes.append(
                {
                    "name": attr.attribute_name,
                    "kind": attr.geometry_type.title(),
                    "origin": attr.origin.title(),
                    "coverage": coverage["label"],
                    "coveragePct": coverage["percentage"],
                }
            )
            attr_column_names = (
                [part.strip() for part in str(attr.attribute_name).split(" + ")]
                if " + " in str(attr.attribute_name)
                else [str(attr.attribute_name)]
            )
            for attr_column_name in attr_column_names:
                if attr_column_name in df.columns and attr_column_name not in explicit_column_kinds:
                    explicit_column_kinds[attr_column_name] = attr.geometry_type.title()

        implicit_geo_attributes = []
        implicit_geo_kinds: dict[str, str] = {}
        for item in detect_geo.implicit_geo_columns(
            df,
            excluded_columns=explicit_column_kinds,
        ):
            column_label = item["name"]
            kind = item["kind"]
            coverage = _attribute_coverage(df, column_label)
            implicit_geo_attributes.append(
                {
                    "name": column_label,
                    "kind": kind,
                    "coverage": coverage["label"],
                    "coveragePct": coverage["percentage"],
                }
            )
            implicit_geo_kinds[column_label] = kind

        all_attributes = []
        for column_name in df.columns:
            column_label = str(column_name)
            if column_label in explicit_column_kinds:
                all_attributes.append(
                    {
                        "name": column_label,
                        "geoRole": "Explicit Geo",
                        "geoClass": "explicit",
                        "detail": explicit_column_kinds[column_label],
                    }
                )
            elif column_label in implicit_geo_kinds:
                all_attributes.append(
                    {
                        "name": column_label,
                        "geoRole": "Implicit Geo",
                        "geoClass": "implicit",
                        "detail": implicit_geo_kinds[column_label],
                    }
                )
            else:
                all_attributes.append(
                    {
                        "name": column_label,
                        "geoRole": "",
                        "geoClass": "none",
                        "detail": "",
                    }
                )

        lane = _lane_for_table(explicit_geo_attributes, implicit_geo_attributes)
        row_count = len(df)
        width_px, height_px = _estimate_node_box(table_name)
        table_records.append(
            {
                "id": table_name,
                "label": table_name,
                "lane": lane,
                "laneLabel": LANE_CONFIG[lane]["badge"],
                "rowCount": row_count,
                "width": width_px,
                "height": height_px,
                "source": "Uploaded" if table_name in uploaded_name_set else "Lake",
                "allAttributes": all_attributes,
                "explicitGeoAttributes": explicit_geo_attributes,
                "implicitGeoAttributes": implicit_geo_attributes,
            }
        )

    sorted_records = sorted(
        table_records,
        key=lambda item: (
            {"explicit": 0, "implicit": 1, "none": 2}.get(str(item["lane"]), 3),
            -int(item["rowCount"]),
            str(item["label"]),
        ),
    )
    for item in sorted_records:
        item.pop("rowCount", None)
    return sorted_records


def _component_height(table_records: list[dict[str, object]]) -> int:
    lane_counts = {
        lane_key: sum(1 for table in table_records if table["lane"] == lane_key)
        for lane_key in ("explicit", "implicit", "none")
    }
    estimated_rows = sum(max(1, math.ceil(count / 4)) for count in lane_counts.values())
    return max(820, min(1600, 260 + (estimated_rows * 120)))


def _build_overview_html(table_records: list[dict[str, object]]) -> str:
    lane_blocks: list[str] = []

    for lane_key in ("explicit", "implicit", "none"):
        config = LANE_CONFIG[lane_key]
        lane_tables = [table for table in table_records if table["lane"] == lane_key]
        node_html = "".join(
            f"""
            <button
              class="atlas-node atlas-node--{lane_key}"
              type="button"
              data-table-id="{escape(str(table['id']), quote=True)}"
              title="{escape(f"{table['label']} · {table['laneLabel']}", quote=True)}"
              style="height:{table['height']}px;"
            >
              <span class="atlas-node__name">{escape(str(table['label']))}</span>
            </button>
            """
            for table in lane_tables
        )
        if not node_html:
            node_html = f'<div class="atlas-lane__empty">{config["empty"]}</div>'

        lane_blocks.append(
            f"""
            <section class="atlas-lane atlas-lane--{lane_key}">
              <div class="atlas-lane__header">
                <div>
                  <h3>{config['title']}</h3>
                  <p>{config['subtitle']}</p>
                </div>
                <div class="atlas-lane__count">{len(lane_tables)}</div>
              </div>
              <div class="atlas-lane__nodes">
                {node_html}
              </div>
            </section>
            """
        )

    return f"""
    <div class="lake-atlas">
      <div class="lake-atlas__summary">
        <div class="lake-atlas__summary-copy">
          <h2>Lake Overview</h2>
          <p>All tables are shown at once. Click any node to open a popup window to show details of the table. The color highlights whether the table contains any implicit or explicit geographical attributes or not.</p>
        </div>
        <div class="lake-atlas__legend">
          <div class="legend-chip"><span class="legend-swatch legend-swatch--explicit"></span>Has Explicit Geo Attr</div>
          <div class="legend-chip"><span class="legend-swatch legend-swatch--implicit"></span>Has Implicit Geo Attr</div>
          <div class="legend-chip"><span class="legend-swatch legend-swatch--none"></span>Has No Geo Attr</div>
        </div>
      </div>
      <div class="lake-atlas__lanes">
        {''.join(lane_blocks)}
      </div>
    </div>
    <div class="atlas-popup" id="lake-atlas-popup" hidden>
      <button class="atlas-popup__close" id="lake-atlas-popup-close" type="button" aria-label="Close details">×</button>
      <div class="atlas-popup__body" id="lake-atlas-popup-body"></div>
    </div>
    <script>
      const atlasData = {json.dumps(table_records)};
      const tableById = Object.fromEntries(atlasData.map((table) => [table.id, table]));
      const popup = document.getElementById("lake-atlas-popup");
      const popupBody = document.getElementById("lake-atlas-popup-body");
      const popupCloseButton = document.getElementById("lake-atlas-popup-close");
      const nodes = Array.from(document.querySelectorAll(".atlas-node"));

      function escapeHtml(value) {{
        return String(value)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }}

      function renderAttributeRows(items, emptyMessage) {{
        if (!items || !items.length) {{
          return `<div class="atlas-popup__empty">${{escapeHtml(emptyMessage)}}</div>`;
        }}
        return items.map((item) => {{
          const tagLabel = item.geoRole ? (item.detail || item.geoRole) : "";
          const tagHtml = tagLabel
            ? `<span class="atlas-popup__tag atlas-popup__tag--${{escapeHtml(item.geoClass || "none")}}">${{escapeHtml(tagLabel)}}</span>`
            : "";
          return `
            <div class="atlas-popup__attr">
              <span class="atlas-popup__attr-name">${{escapeHtml(item.name)}}</span>
              ${{tagHtml}}
            </div>
          `;
        }}).join("");
      }}

      function hidePopup() {{
        popup.hidden = true;
        popupBody.innerHTML = "";
        nodes.forEach((node) => node.classList.remove("is-selected"));
      }}

      function showPopupHtml(event, html) {{
        popupBody.innerHTML = html;
        popup.hidden = false;

        const popupNode = popup;
        let left = event.clientX + 12;
        let top = event.clientY + 12;
        const maxLeft = Math.max(12, window.innerWidth - popupNode.offsetWidth - 12);
        const maxTop = Math.max(12, window.innerHeight - popupNode.offsetHeight - 12);
        left = Math.max(12, Math.min(left, maxLeft));
        top = Math.max(12, Math.min(top, maxTop));
        popup.style.left = `${{left}}px`;
        popup.style.top = `${{top}}px`;
      }}

      function showNodePopup(event, tableId) {{
        const table = tableById[tableId] || atlasData[0];
        if (!table) {{
          return;
        }}

        nodes.forEach((node) => {{
          node.classList.toggle("is-selected", node.dataset.tableId === table.id);
        }});

        const explicitCount = table.explicitGeoAttributes.length;
        const implicitCount = table.implicitGeoAttributes.length;
        const geoSummary = explicitCount
          ? `${{explicitCount}} explicit geo attribute${{explicitCount === 1 ? "" : "s"}}`
          : implicitCount
            ? `${{implicitCount}} implicit geo attribute${{implicitCount === 1 ? "" : "s"}}`
            : "No geo attributes detected";

        const popupHtml = `
          <div class="atlas-popup__eyebrow">${{escapeHtml(table.source)}} table</div>
          <div class="atlas-popup__title-row">
            <div class="atlas-popup__title">${{escapeHtml(table.label)}}</div>
            <span class="atlas-popup__badge atlas-popup__badge--${{escapeHtml(table.lane)}}">${{escapeHtml(table.laneLabel)}}</span>
          </div>
          <div class="atlas-popup__summary">${{escapeHtml(geoSummary)}}</div>
          <div class="atlas-popup__section-label">Attributes</div>
          ${{renderAttributeRows(table.allAttributes, "No attributes found for this table.")}}
        `;
        showPopupHtml(event, popupHtml);
      }}

      nodes.forEach((node) => {{
        node.addEventListener("click", (event) => {{
          event.stopPropagation();
          showNodePopup(event, node.dataset.tableId);
        }});
      }});

      popup.addEventListener("click", (event) => {{
        event.stopPropagation();
      }});
      popupCloseButton.addEventListener("click", (event) => {{
        event.stopPropagation();
        hidePopup();
      }});
      document.addEventListener("click", hidePopup);
      document.addEventListener("keydown", (event) => {{
        if (event.key === "Escape" && !popup.hidden) {{
          hidePopup();
        }}
      }});
    </script>
    <style>
      :root {{
        color-scheme: light;
      }}

      body {{
        margin: 0;
        font-family: "SF Pro Text", "Segoe UI", sans-serif;
        color: #0f172a;
        background:
          radial-gradient(circle at top right, rgba(191, 219, 254, 0.48), transparent 36%),
          linear-gradient(180deg, #f8fbff 0%, #f7fafc 100%);
      }}

      .lake-atlas {{
        min-height: 100%;
        padding: 18px;
        box-sizing: border-box;
      }}

      .lake-atlas__summary {{
        display: flex;
        justify-content: space-between;
        gap: 18px;
        align-items: flex-start;
        margin-bottom: 18px;
        padding: 18px 20px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.82);
        backdrop-filter: blur(8px);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
      }}

      .lake-atlas__summary-copy h2 {{
        margin: 0 0 6px 0;
        font-size: 26px;
        line-height: 1.1;
      }}

      .lake-atlas__summary-copy p {{
        margin: 0;
        max-width: 760px;
        color: #475569;
        line-height: 1.5;
      }}

      .lake-atlas__legend {{
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 10px;
      }}

      .legend-chip {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: #ffffff;
        border: 1px solid rgba(148, 163, 184, 0.28);
        color: #334155;
        font-size: 13px;
        font-weight: 600;
        white-space: nowrap;
      }}

      .legend-chip--text {{
        background: transparent;
      }}

      .legend-swatch {{
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 999px;
      }}

      .legend-swatch--explicit {{
        background: {EXPLICIT_GEO_COLOR};
      }}

      .legend-swatch--implicit {{
        background: {IMPLICIT_GEO_COLOR};
      }}

      .legend-swatch--none {{
        background: {NO_GEO_COLOR};
        border: 1px solid {NO_GEO_BORDER};
        box-sizing: border-box;
      }}

      .lake-atlas__lanes {{
        display: grid;
        gap: 14px;
      }}

      .atlas-lane {{
        border-radius: 22px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
        overflow: hidden;
      }}

      .atlas-lane--explicit {{
        background:
          linear-gradient(180deg, rgba(219, 234, 254, 0.72), rgba(255, 255, 255, 0.92)),
          #ffffff;
      }}

      .atlas-lane--implicit {{
        background:
          linear-gradient(180deg, rgba(239, 246, 255, 0.92), rgba(255, 255, 255, 0.92)),
          #ffffff;
      }}

      .atlas-lane--none {{
        background:
          linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(255, 255, 255, 0.92)),
          #ffffff;
      }}

      .atlas-lane__header {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: center;
        padding: 18px 20px 12px 20px;
      }}

      .atlas-lane__header h3 {{
        margin: 0;
        font-size: 19px;
      }}

      .atlas-lane__header p {{
        margin: 4px 0 0 0;
        color: #475569;
        line-height: 1.4;
        font-size: 14px;
      }}

      .atlas-lane__count {{
        min-width: 40px;
        height: 40px;
        padding: 0 10px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.28);
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
      }}

      .atlas-lane__nodes {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        padding: 6px 20px 20px 20px;
      }}

      .atlas-lane__empty {{
        width: 100%;
        padding: 14px 16px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.74);
        color: #64748b;
        border: 1px dashed rgba(148, 163, 184, 0.4);
        font-size: 14px;
      }}

      .atlas-node {{
        border: 1px solid transparent;
        border-radius: 18px;
        box-sizing: border-box;
        display: flex;
        width: 100%;
        justify-content: center;
        align-items: center;
        padding: 4px 6px;
        text-align: center;
        cursor: pointer;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.10);
        transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
      }}

      .atlas-node:hover,
      .atlas-node.is-selected {{
        transform: translateY(-2px);
        box-shadow: 0 18px 34px rgba(15, 23, 42, 0.14);
      }}

      .atlas-node--explicit {{
        background: {EXPLICIT_GEO_COLOR};
        color: #ffffff;
      }}

      .atlas-node--explicit.is-selected {{
        border-color: #93c5fd;
      }}

      .atlas-node--implicit {{
        background: {IMPLICIT_GEO_COLOR};
        color: #0f172a;
      }}

      .atlas-node--implicit.is-selected {{
        border-color: #3b82f6;
      }}

      .atlas-node--none {{
        background: {NO_GEO_COLOR};
        color: #0f172a;
        border-color: {NO_GEO_BORDER};
      }}

      .atlas-node--none.is-selected {{
        border-color: #64748b;
      }}

      .atlas-node__name {{
        display: block;
        white-space: normal;
        overflow-wrap: anywhere;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.2;
      }}

      .atlas-popup[hidden] {{
        display: none !important;
      }}

      .atlas-popup {{
        position: fixed;
        z-index: 60;
        min-width: 240px;
        max-width: 340px;
        max-height: 420px;
        overflow-y: auto;
        padding: 12px 14px 10px 14px;
        background: #ffffff;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
        font: 12px "SF Pro Text", "Segoe UI", sans-serif;
      }}

      .atlas-popup__close {{
        position: absolute;
        top: 6px;
        right: 8px;
        width: 24px;
        height: 24px;
        border: none;
        border-radius: 999px;
        background: transparent;
        color: #475569;
        font-size: 18px;
        line-height: 1;
        cursor: pointer;
      }}

      .atlas-popup__body {{
        padding-right: 10px;
      }}

      .atlas-popup__eyebrow {{
        color: #64748b;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}

      .atlas-popup__title-row {{
        display: flex;
        gap: 10px;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 8px;
      }}

      .atlas-popup__title {{
        font-weight: 600;
        font-size: 14px;
        color: #0f172a;
      }}

      .atlas-popup__badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        white-space: nowrap;
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 700;
      }}

      .atlas-popup__badge--explicit {{
        background: #dbeafe;
        color: #0d47a1;
      }}

      .atlas-popup__badge--implicit {{
        background: #eff6ff;
        color: #1d4ed8;
      }}

      .atlas-popup__badge--none {{
        background: #f8fafc;
        color: #334155;
      }}

      .atlas-popup__summary {{
        margin-bottom: 10px;
        color: #475569;
        line-height: 1.45;
      }}

      .atlas-popup__section-label {{
        font-weight: 600;
        color: #0f172a;
        margin: 8px 0 4px 0;
      }}

      .atlas-popup__attr {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-top: 6px;
      }}

      .atlas-popup__attr-name {{
        color: #0f172a;
        font-weight: 500;
      }}

      .atlas-popup__tag {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 700;
      }}

      .atlas-popup__tag--explicit {{
        background: #dbeafe;
        color: #0d47a1;
      }}

      .atlas-popup__tag--implicit {{
        background: #eff6ff;
        color: #1d4ed8;
      }}

      .atlas-popup__empty {{
        margin-top: 6px;
        color: #64748b;
      }}

      @media (max-width: 980px) {{
        .lake-atlas__summary {{
          display: grid;
        }}

        .lake-atlas__legend {{
          justify-content: flex-start;
        }}

        .atlas-lane__nodes {{
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }}

        .atlas-node {{
          width: 100% !important;
          height: auto !important;
          min-height: 44px !important;
        }}
      }}

      @media (max-width: 640px) {{
        .atlas-lane__nodes {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
    """


def render_loaded_lake_overview(
    *,
    chosen_lake: str | None,
    data_lake: dict[str, pd.DataFrame],
    uploaded_table_names: Iterable[str],
) -> None:
    total_tables = len(data_lake)
    uploaded_table_names = list(uploaded_table_names)
    uploaded_count = len(uploaded_table_names)
    lake_tables = max(total_tables - uploaded_count, 0)

    lake_label = chosen_lake if chosen_lake and chosen_lake != "<no datalakes found>" else None
    if lake_label:
        st.success(
            f"✅ Loaded data lake \"{lake_label}\" "
            f"({lake_tables} lake tables, {uploaded_count} uploaded tables)."
        )
    else:
        st.success(
            f"✅ Loaded uploaded data lake ({lake_tables} lake tables, {uploaded_count} uploaded tables)."
        )

    table_records = _build_table_records(
        chosen_lake=chosen_lake,
        data_lake=data_lake,
        uploaded_table_names=uploaded_table_names,
    )
    if not table_records:
        st.info("Load or upload at least one table to see the lake atlas.")
        return

    st.caption("Click any table node to open a popup with all attributes and geo highlights.")
    components.html(
        _build_overview_html(table_records),
        height=_component_height(table_records),
        scrolling=True,
    )
