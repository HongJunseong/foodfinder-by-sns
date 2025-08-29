# scripts/load_to_mongo.py

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# ---------------- .env 로드(있으면) ----------------
try:
    from dotenv import load_dotenv

    CUR = Path(__file__).resolve()
    ROOT = CUR.parents[1]  # 프로젝트 루트(= 리포 루트)
    for cand in (ROOT / ".env", CUR.parents[2] / ".env"):
        if cand.exists():
            load_dotenv(dotenv_path=cand)
            break
except Exception:
    ROOT = Path(__file__).resolve().parents[1]

# 프로젝트 루트 import 경로 추가 (src.* import 용)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ---------------- Mongo 유틸 ----------------
from src.modules.db.db_mongo import (  # noqa: E402
    get_coll,
    save_df_unique,
    remove_duplicates,
    count as mongo_count,
)

# ---------------- 기본 설정(환경변수로 변경 가능) ----------------
DB_NAME = os.getenv("MONGO_DB", "post2")
COL_ALL = os.getenv("MONGO_COL_ALL", "all_post")
COL_NON_ADS = os.getenv("MONGO_COL_NON_ADS", "non_ads")
COL_ADS = os.getenv("MONGO_COL_ADS", "ads")

CSV_AD_ALL = os.getenv("CSV_AD_ALL", (ROOT / "data" / "ad_extracted.csv").as_posix())
CSV_NON_ADS = os.getenv("CSV_NON_ADS", (ROOT / "data" / "clean" / "non_ads.csv").as_posix())
CSV_ADS = os.getenv("CSV_ADS", (ROOT / "data" / "clean" / "ads.csv").as_posix())

# 적재 후 content 기준 중복 제거 여부
DEDUP_CONTENT = os.getenv("DEDUP_CONTENT", "0") == "1"

# CSV 인코딩 시도 순서
ENCODINGS = ["utf-8", "utf-8-sig", "cp949"]


def read_csv_smart(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV 읽기 실패: {path} / 마지막 에러: {last_err}")


def maybe_load(path: str, db: str, col_name: str, key_hint: Optional[str] = None) -> Tuple[bool, Optional[int]]:
    p = Path(path)
    if not p.exists():
        print(f"[skip] 파일 없음: {p.as_posix()}")
        return False, None

    df = read_csv_smart(p)
    if len(df) == 0:
        print(f"[skip] 빈 CSV: {p.as_posix()}")
        return False, 0

    col = get_coll(db, col_name)
    before = mongo_count(col)

    # upsert 키: post_id 우선, 그다음 전달 key_hint, 최종 content (save_df_unique 내부 처리)
    key = "post_id" if "post_id" in df.columns else (key_hint or "content")

    print(f"[load] {p.name} → {db}.{col_name} (rows={len(df)}, key={key})")
    save_df_unique(df, col, key=key)

    if DEDUP_CONTENT:
        try:
            remove_duplicates(col, field="content")
        except Exception as e:
            print(f"[warn] content 중복 제거 실패: {e}")

    after = mongo_count(col)
    print(f"[done] {db}.{col_name} count: {before} → {after} (+{after - before})")
    return True, after


def main() -> None:
    print("=== CSV → MongoDB 적재 ===")

    # 1) 전체(ad_extracted.csv) → all_post
    maybe_load(CSV_AD_ALL, DB_NAME, COL_ALL)

    # 2) 비광고(clean/non_ads.csv) → non_ads
    maybe_load(CSV_NON_ADS, DB_NAME, COL_NON_ADS)

    # 3) 광고(clean/ads.csv) → ads
    maybe_load(CSV_ADS, DB_NAME, COL_ADS)

    # 요약 저장
    col_all = get_coll(DB_NAME, COL_ALL)
    col_na = get_coll(DB_NAME, COL_NON_ADS)
    col_ad = get_coll(DB_NAME, COL_ADS)
    summary = {
        "db": DB_NAME,
        "all_post": mongo_count(col_all),
        "non_ads": mongo_count(col_na),
        "ads": mongo_count(col_ad),
    }
    out = ROOT / "data" / "clean" / "mongo_load_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {summary}")
    print(f"[save] {out.as_posix()}")


if __name__ == "__main__":
    main()
