from __future__ import annotations

from html import escape
from typing import Iterable, Mapping, Sequence

import streamlit as st
import streamlit.components.v1 as components


RELATION_COLOR = "#4caf50"
SEMANTIC_COLOR = "#ff9800"
SPATIAL_COLOR = "#1e88e5"

FOCUS_TABLE_BG = "#dff4b4"
FOCUS_TABLE_BORDER = "#96b63a"
FOCUS_TABLE_TEXT = "#496222"

BRIDGE_TABLE_BG = "#f8fafc"
BRIDGE_TABLE_BORDER = "#cbd5e1"
BRIDGE_TABLE_TEXT = "#334155"

CARD_BG = "#ffffff"
CARD_BORDER = "#e2e8f0"
CARD_SELECTED_BG = "#f6fde8"
CARD_SELECTED_BORDER = "#93c54b"

STAT_BG = "#f8fafc"
STAT_BORDER = "#dbe4f0"
STAT_LABEL = "#64748b"
STAT_VALUE = "#111827"


def build_table_metadata(data_lake: Mapping[str, object]) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for table_name, df in data_lake.items():
        metadata[table_name] = {
            "columns": list(df.columns),
            "cardinality": len(df),
            "num_columns": len(df.columns),
        }
    return metadata


def _join_style(join_type: str) -> tuple[str, str]:
    normalized = (join_type or "relation").lower()
    if normalized == "semantic":
        return SEMANTIC_COLOR, "solid"
    if normalized == "spatial":
        return SPATIAL_COLOR, "dashed"
    return RELATION_COLOR, "solid"


def _format_stat_value(value: object) -> str:
    if value is None:
        return "--"
    return str(value)


def _table_tooltip(table_name: str, table_info: Mapping[str, object]) -> str:
    columns = table_info.get("columns") or []
    preview = ", ".join(str(col) for col in list(columns)[:6])
    if len(columns) > 6:
        preview += ", ..."
    if not preview:
        preview = "No columns available"
    return (
        f"{table_name}\n"
        f"Rows: {table_info.get('cardinality', 0)}\n"
        f"Columns: {table_info.get('num_columns', 0)}\n"
        f"{preview}"
    )


def render_join_path_legend(
    *,
    focus_table_label: str = "Requested attribute table",
    bridge_table_label: str = "Bridge table",
    rows_hint: str | None = None,
    show_table_legend: bool = True,
    show_join_legend: bool = True,
) -> None:
    legend_parts: list[str] = []
    if show_table_legend:
        legend_parts.extend(
            [
                '<span style="font-weight:700;color:#0f172a;">Tables</span>',
                (
                    '<span style="display:inline-flex;align-items:center;gap:8px;">'
                    f'<span style="display:inline-block;padding:5px 12px;background:{FOCUS_TABLE_BG};'
                    f'border:2px solid {FOCUS_TABLE_BORDER};border-radius:999px;color:{FOCUS_TABLE_TEXT};font-weight:700;">'
                    "sample_table</span>"
                    f'<span>{escape(focus_table_label, quote=True)}</span>'
                    "</span>"
                ),
                (
                    '<span style="display:inline-flex;align-items:center;gap:8px;">'
                    f'<span style="display:inline-block;padding:5px 12px;background:{BRIDGE_TABLE_BG};'
                    f'border:2px solid {BRIDGE_TABLE_BORDER};border-radius:999px;color:{BRIDGE_TABLE_TEXT};font-weight:700;">'
                    "bridge_table</span>"
                    f'<span>{escape(bridge_table_label, quote=True)}</span>'
                    "</span>"
                ),
            ]
        )
    if show_join_legend:
        legend_parts.extend(
            [
                '<span style="font-weight:700;color:#0f172a;margin-left:8px;">Joins</span>',
                (
                    '<span style="display:inline-flex;align-items:center;gap:8px;">'
                    f'<span style="display:inline-block;width:34px;height:0;border-bottom:3px solid {RELATION_COLOR};"></span>'
                    "<span>Equi-join</span></span>"
                ),
                (
                    '<span style="display:inline-flex;align-items:center;gap:8px;">'
                    f'<span style="display:inline-block;width:34px;height:0;border-bottom:3px solid {SEMANTIC_COLOR};"></span>'
                    "<span>Semantic join</span></span>"
                ),
                (
                    '<span style="display:inline-flex;align-items:center;gap:8px;">'
                    f'<svg width="34" height="8" aria-hidden="true">'
                    f'<line x1="0" y1="4" x2="34" y2="4" stroke="{SPATIAL_COLOR}" stroke-width="3" stroke-dasharray="6 4"/>'
                    "</svg>"
                    "<span>Spatial join</span></span>"
                ),
            ]
        )

    if legend_parts:
        legend_html = (
            '<div style="display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center;'
            'margin:0.15rem 0 0.55rem 0;font-size:13px;color:#334155;">'
            + "".join(legend_parts)
            + "</div>"
        )
        st.markdown(legend_html, unsafe_allow_html=True)
    if rows_hint:
        st.caption(rows_hint)


def render_join_path_card(
    *,
    component_id: str,
    title: str,
    path_data: Mapping[str, object],
    table_metadata: Mapping[str, Mapping[str, object]],
    highlighted_tables: Iterable[str] | None = None,
    stats: Mapping[str, object] | None = None,
    stat_items: Sequence[tuple[str, str]] | None = None,
    selected: bool = False,
    height: int = 240,
) -> None:
    highlighted = set(highlighted_tables or [])
    tables = list(path_data.get("tables") or [])
    joins = list(path_data.get("joins") or [])

    stats = stats or {}
    stat_order = stat_items or [("rows", "ROWS"), ("cols", "COLS"), ("len", "LEN"), ("hops", "HOPS")]
    stats_html = "".join(
        (
            f'<span class="jp-stat">'
            f'<span class="jp-stat-label">{label}</span>'
            f'<span class="jp-stat-value">{escape(_format_stat_value(stats.get(key)), quote=True)}</span>'
            f"</span>"
        )
        for key, label in stat_order
    )
    stats_block_html = f'<div class="jp-stats">{stats_html}</div>' if stats_html else ""

    items_html: list[str] = []
    for index, table in enumerate(tables):
        table_info = table_metadata.get(table, {})
        table_class = "jp-node jp-node-focus" if table in highlighted else "jp-node jp-node-bridge"
        items_html.append(
            f'<span class="{table_class}" title="{escape(_table_tooltip(table, table_info), quote=True)}">'
            f"{escape(str(table), quote=True)}</span>"
        )

        if index >= len(joins):
            continue

        edge_info = joins[index]
        join_type = str(edge_info.get("type") or "relation")
        join_label = str(edge_info.get("attributes") or "").strip()
        if not join_label:
            join_label = f"{join_type.title()} join"
        color, dash_style = _join_style(join_type)
        segment_width = max(160, min(320, 120 + len(join_label) * 5))
        items_html.append(
            f'<span class="jp-edge" style="width:{segment_width}px;" '
            f'title="{escape(f"{join_type.title()} join: {join_label}", quote=True)}">'
            f'<span class="jp-edge-line" style="border-top-color:{color};border-top-style:{dash_style};"></span>'
            f'<span class="jp-edge-label">{escape(join_label, quote=True)}</span>'
            f"</span>"
        )

    selected_class = " jp-card-selected" if selected else ""
    html = f"""
<div id="{escape(component_id, quote=True)}">
  <style>
    #{escape(component_id, quote=True)} * {{
      box-sizing: border-box;
      font-family: "SF Pro Text", "Segoe UI", sans-serif;
    }}
    #{escape(component_id, quote=True)} .jp-card {{
      background: {CARD_BG};
      border: 1px solid {CARD_BORDER};
      border-radius: 14px;
      padding: 14px 14px 12px;
      overflow: hidden;
    }}
    #{escape(component_id, quote=True)} .jp-card.jp-card-selected {{
      background: {CARD_SELECTED_BG};
      border: 2px solid {CARD_SELECTED_BORDER};
      box-shadow: 0 0 0 1px rgba(147, 197, 75, 0.12);
    }}
    #{escape(component_id, quote=True)} .jp-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    #{escape(component_id, quote=True)} .jp-title {{
      font-size: 14px;
      line-height: 1.2;
      font-weight: 700;
      color: #334155;
      white-space: nowrap;
    }}
    #{escape(component_id, quote=True)} .jp-stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-left: auto;
    }}
    #{escape(component_id, quote=True)} .jp-stat {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 4px 10px;
      background: {STAT_BG};
      border: 1px solid {STAT_BORDER};
      border-radius: 999px;
    }}
    #{escape(component_id, quote=True)} .jp-stat-label {{
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.06em;
      color: {STAT_LABEL};
    }}
    #{escape(component_id, quote=True)} .jp-stat-value {{
      font-size: 13px;
      font-weight: 700;
      color: {STAT_VALUE};
    }}
    #{escape(component_id, quote=True)} .jp-track {{
      display: flex;
      align-items: center;
      gap: 0;
      overflow-x: auto;
      overflow-y: hidden;
      padding: 8px 4px 10px;
      scrollbar-width: thin;
    }}
    #{escape(component_id, quote=True)} .jp-track::-webkit-scrollbar {{
      height: 8px;
    }}
    #{escape(component_id, quote=True)} .jp-track::-webkit-scrollbar-thumb {{
      background: #cbd5e1;
      border-radius: 999px;
    }}
    #{escape(component_id, quote=True)} .jp-node {{
      position: relative;
      z-index: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 7px 14px;
      border-radius: 999px;
      border: 2px solid transparent;
      font-size: 14px;
      font-weight: 700;
      line-height: 1;
      white-space: nowrap;
      flex: 0 0 auto;
      cursor: help;
    }}
    #{escape(component_id, quote=True)} .jp-node-focus {{
      background: {FOCUS_TABLE_BG};
      border-color: {FOCUS_TABLE_BORDER};
      color: {FOCUS_TABLE_TEXT};
    }}
    #{escape(component_id, quote=True)} .jp-node-bridge {{
      background: {BRIDGE_TABLE_BG};
      border-color: {BRIDGE_TABLE_BORDER};
      color: {BRIDGE_TABLE_TEXT};
    }}
    #{escape(component_id, quote=True)} .jp-edge {{
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex: 0 0 auto;
      margin: 0 2px;
      height: 34px;
      cursor: help;
    }}
    #{escape(component_id, quote=True)} .jp-edge-line {{
      position: absolute;
      inset: 50% 0 auto 0;
      transform: translateY(-50%);
      border-top-width: 3px;
      border-top-style: solid;
    }}
    #{escape(component_id, quote=True)} .jp-edge-label {{
      position: relative;
      z-index: 1;
      display: inline-block;
      max-width: calc(100% - 16px);
      padding: 0 10px;
      background: {CARD_BG};
      color: #475569;
      font-size: 13px;
      font-weight: 600;
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    #{escape(component_id, quote=True)} .jp-card-selected .jp-edge-label {{
      background: {CARD_SELECTED_BG};
    }}
  </style>
  <div class="jp-card{selected_class}">
    <div class="jp-header">
      <div class="jp-title">{escape(title, quote=True)}</div>
      {stats_block_html}
    </div>
    <div class="jp-track">
      {''.join(items_html)}
    </div>
  </div>
</div>
"""
    components.html(html, height=height)
