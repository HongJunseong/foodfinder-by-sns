# scripts/run_map_api.py

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# --- 경로/환경 ---
CUR = Path(__file__).resolve()
ROOT = CUR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

for _env in (ROOT / ".env", CUR.parents[2] / ".env"):
    if _env.exists():
        load_dotenv(dotenv_path=_env)
        break

# --- 설정 ---
DATA_BACKEND = os.getenv("DATA_BACKEND", "auto").lower()  # auto|csv|mongo
CSV_PATH_ENV = os.getenv("CSV_PATH", "")
MONGO_DB = os.getenv("MONGO_DB", "post")
MONGO_COLL = os.getenv("MONGO_COLL", "all_post")
DROP_DUP_KEY = os.getenv("DROP_DUP_KEY", "post_id")

LIMIT = int(os.getenv("LIMIT", "80"))
CENTER_LAT = float(os.getenv("MAP_CENTER_LAT", "37.5665"))
CENTER_LNG = float(os.getenv("MAP_CENTER_LNG", "126.9780"))
ZOOM_START = int(os.getenv("MAP_ZOOM_START", "12"))

# 비광고만 표시(기본 ON)
MAP_FILTER_NONADS = os.getenv("MAP_FILTER_NONADS", "1") == "1"

if not os.getenv("KAKAO_REST_API_KEY"):
    raise RuntimeError("KAKAO_REST_API_KEY가 없습니다(.env에 설정).")

from src.modules.data_io import load_posts
from src.utils.map_api import build_map


def pick_csv_path() -> Optional[str]:
    cand: list[Path] = []
    if CSV_PATH_ENV.strip():
        cand.append(Path(CSV_PATH_ENV.strip()))
    cand += [
        ROOT / "data" / "clean" / "non_ads.csv",
        ROOT / "data" / "ad_extracted.csv",
        ROOT / "data" / "extracted.csv",
        ROOT / "data" / "extract.csv",
        ROOT / "data" / "region_restaurant.csv",
    ]
    for p in cand:
        if p.exists():
            return p.as_posix()
    return None


def load_data_auto() -> Tuple[pd.DataFrame, str]:
    source = DATA_BACKEND
    csv_path = pick_csv_path()

    if source == "csv":
        if not csv_path:
            raise FileNotFoundError("DATA_BACKEND=csv 이지만 사용 가능한 CSV를 찾지 못했습니다.")
        df = load_posts(source="csv", csv_path=csv_path, drop_duplicates_on=DROP_DUP_KEY)
        return df, f"csv:{csv_path}"

    if source == "mongo":
        df = load_posts(
            source="mongo",
            mongo_db=MONGO_DB,
            mongo_coll=MONGO_COLL,
            drop_duplicates_on=DROP_DUP_KEY,
        )
        return df, f"mongo:{MONGO_DB}/{MONGO_COLL}"

    # auto: CSV 우선, 없으면 Mongo
    df = load_posts(
        source="auto",
        csv_path=csv_path,
        mongo_db=MONGO_DB,
        mongo_coll=MONGO_COLL,
        drop_duplicates_on=DROP_DUP_KEY,
    )
    used = f"csv:{csv_path}" if csv_path else f"mongo:{MONGO_DB}/{MONGO_COLL}"
    return df, used


def main() -> None:
    df, source = load_data_auto()
    print(f"[info] loaded rows={len(df)} from {source}")

    # 비광고만
    if MAP_FILTER_NONADS and "is_ad" in df.columns:
        before = len(df)
        df = df[df["is_ad"].astype(int) == 0].reset_index(drop=True)
        print(f"[info] non-ads only: {before} → {len(df)}")

    m, ok, fail = build_map(
        df,
        center=(CENTER_LAT, CENTER_LNG),
        zoom_start=ZOOM_START,
        limit=LIMIT,
    )
    print(f"[info] geocoding success={ok} / fail={fail}")

    out = ROOT / "matched_places_map.html"
    m.save(out.as_posix())
    print(f"[info] map saved → {out.as_posix()}")


if __name__ == "__main__":
    main()
