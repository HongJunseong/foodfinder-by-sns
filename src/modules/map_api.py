# src/modules/map_api.py

from __future__ import annotations

import os, sys, json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import folium
import pandas as pd
from typing import Tuple, Optional
from utils.map_utils import geocode_by_address

def _coerce_address(x) -> str:
    # 리스트/JSON 문자열이면 첫 원소로 통일
    if isinstance(x, list):
        return x[0] if x else ""
    if isinstance(x, str):
        xs = x.strip()
        if xs.startswith("[") and xs.endswith("]"):
            try:
                arr = json.loads(xs)
                return arr[0] if arr else ""
            except Exception:
                return xs
        return xs
    return ""

def _get_name(row) -> str:
    # restaurant_name / restaurant name 둘 다 대응(+리스트 폼)
    v = row.get("restaurant_name") or row.get("restaurant name") or ""
    if isinstance(v, list):
        v = v[0] if v else ""
    return v.strip() if isinstance(v, str) else ""

def build_map(
    df: pd.DataFrame,
    center: Tuple[float, float] = (37.5665, 126.9780),
    zoom_start: int = 12,
    limit: int = 80
):
    m = folium.Map(location=list(center), zoom_start=zoom_start)
    ok = fail = 0

    if "restaurant name" in df.columns and "restaurant_name" not in df.columns:
        df = df.copy(); df["restaurant_name"] = df["restaurant name"]
    if "address" not in df.columns:
        df = df.copy(); df["address"] = ""

    it = df.head(limit).iterrows() if limit and limit > 0 else df.iterrows()

    for _, row in it:
        name = _get_name(row)
        addr = _coerce_address(row.get("address"))

        # 주소 없으면 무조건 스킵 (환경변수/옵션 없이 고정 정책)
        if not addr:
            fail += 1
            continue

        res = geocode_by_address(name, addr, radius=300, sleep_sec=0.12)
        if not res:
            fail += 1
            continue

        popup = f"{res['place_name']}<br>({res['address_name']})<br><i>{res['source']}</i>"
        folium.Marker(
            [res["lat"], res["lng"]],
            popup=folium.Popup(popup, max_width=300),
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)
        ok += 1

    return m, ok, fail
