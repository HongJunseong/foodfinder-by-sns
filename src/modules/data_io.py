# src/modules/data_io.py

from __future__ import annotations
import os, json, re, sys
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

# sys.path: src 추가 (from modules.* import ...)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Mongo 유틸
from modules.db.db_mongo import get_coll, read_df

# --- small utils ---
def _maybe_parse_json_like(x):
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x

def _to_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        try:
            f = float(str(x).replace(",", ""))
            return int(f)
        except Exception:
            return default

def _parse_dt(x):
    if pd.isna(x) or x == "":
        return None
    try:
        dt = pd.to_datetime(x, errors="coerce", utc=False)
        return None if pd.isna(dt) else dt.to_pydatetime()
    except Exception:
        return None

def _robust_comment_count(x) -> int:
    if isinstance(x, (list, tuple)): return len(x)
    if isinstance(x, (int, float)) and not pd.isna(x): return int(x)
    if isinstance(x, str):
        s = x.strip()
        if not s: return 0
        return len([t for t in re.split(r'[\n,]+', s) if t.strip()])
    return 0

# --- 표준화 ---
def normalize_posts_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # hashtags: 문자열(JSON류) → list
    if "hashtags" in out.columns:
        out["hashtags"] = out["hashtags"].apply(_maybe_parse_json_like)

    # 숫자 필드
    for col in ["likes", "views", "bookmarks"]:
        if col in out.columns:
            out[col] = out[col].apply(_to_int)

    # 댓글수 파생
    if "comment_count" not in out.columns and "comments" in out.columns:
        out["comment_count"] = out["comments"].apply(_robust_comment_count)

    # 날짜
    if "created_at" in out.columns:
        out["created_at"] = out["created_at"].apply(_parse_dt)

    # 라벨
    if "is_ad" in out.columns:
        out["is_ad"] = out["is_ad"].fillna(0).astype(int)

    return out

# --- 로더 ---
def load_posts(
    source: str = "auto",
    csv_path: str | Path | None = None,
    mongo_db: str = "sns",
    mongo_coll: str = "posts_all",
    mongo_query: Optional[Dict[str, Any]] = None,
    mongo_projection: Optional[Dict[str, int]] = None,
    drop_duplicates_on: str | None = None,
) -> pd.DataFrame:
    """
    CSV/Mongo 공용 로더. 반환 전 normalize_posts_df 적용.
    - source='csv'  → csv_path 필요
    - source='mongo'→ mongo_db/mongo_coll 사용
    - source='auto' → csv_path 있으면 CSV, 없으면 Mongo
    """
    # 소스 결정
    if source not in {"auto", "csv", "mongo"}:
        raise ValueError("source must be one of {'auto','csv','mongo'}")

    if source == "auto":
        source = "csv" if (csv_path and Path(csv_path).exists()) else "mongo"

    # 읽기
    if source == "csv":
        if not csv_path:
            raise ValueError("csv_path is required when source='csv'")
        df = pd.read_csv(csv_path)
    else:
        col = get_coll(mongo_db, mongo_coll)
        df = read_df(col, query=mongo_query or {}, projection=mongo_projection or {})

    # 표준화
    df = normalize_posts_df(df)

    # 중복 정리(옵션)
    if drop_duplicates_on and drop_duplicates_on in df.columns:
        df = df.drop_duplicates(subset=[drop_duplicates_on], keep="last").reset_index(drop=True)

    return df
