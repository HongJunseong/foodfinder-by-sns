# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, io, json, ast, math, argparse, hashlib
from pathlib import Path
from typing import Optional, List, Tuple, Set, Dict
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# env / paths
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if (HERE.parent.name == "modules" and HERE.parents[1].name == "src") else HERE.parents[1]
ENV = ROOT / ".env"
load_dotenv(ENV if ENV.exists() else None)

# import from src/modules
MODS = ROOT / "src" / "modules"
if str(MODS) not in sys.path:
    sys.path.insert(0, str(MODS))

from llm_extract import extract_restaurant_info_one

# defaults (env overridable)
IN_CSV   = Path(os.getenv("CRAWLED_CSV", "data/region_restaurant.csv"))
OUT_CSV  = Path(os.getenv("EXTRACTED_CSV", "data/extracted.csv"))
FAIL_LOG = Path(os.getenv("EXTRACT_FAIL_LOG", "data/extracted_failures.csv"))

# ------------ utils ------------
def to_listish(v):
    if v is None: return []
    if isinstance(v, list): return v
    if isinstance(v, tuple): return list(v)
    if isinstance(v, float) and math.isnan(v): return []
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}: return []
    if s in {"[]", "[ ]", "['']", '[""]'}: return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            try: obj = json.loads(s)
            except json.JSONDecodeError: obj = ast.literal_eval(s)
            if isinstance(obj, list): return obj
            if isinstance(obj, tuple): return list(obj)
            return []
        except Exception:
            return []
    return [s]

def read_csv_robust(path: Path) -> pd.DataFrame:
    encs = [os.getenv("CRAWLED_CSV_ENCODING"), "utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]
    tried = set()
    for enc in encs:
        if not enc or enc in tried: continue
        try: return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError: tried.add(enc); continue
    with open(path, "rb") as f: raw = f.read()
    return pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")))

def pick_content_col(df: pd.DataFrame) -> str:
    for cand in ["content", "본문", "text", "caption"]:
        if cand in df.columns: return cand
    return df.columns[0]

def make_key_from_row(row: pd.Series, content_col: str) -> Tuple[str, str]:
    """
    처리 단위 식별 키:
      - post_id가 있으면:  key="id:<post_id>", hash="" (굳이 해시 불필요)
      - 없으면 content 해시: key="c:<sha1_16>", hash="<sha1_16>"
    """
    pid = str(row.get("post_id", "") or "").strip()
    if pid:
        return (f"id:{pid}", "")
    content = str(row.get(content_col, "") or "")
    h = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    return (f"c:{h}", h)

# ------------ fail log I/O ------------
def load_fail_log(path: Path) -> Dict[str, Dict]:
    """
    CSV schema:
      key, post_id, content_hash, attempts, last_ts, sample_tag
    """
    if not path.exists(): return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    out: Dict[str, Dict] = {}
    for _, r in df.iterrows():
        k = str(r.get("key", "") or "")
        if not k: continue
        out[k] = {
            "post_id": str(r.get("post_id", "") or ""),
            "content_hash": str(r.get("content_hash", "") or ""),
            "attempts": int(r.get("attempts", 1) or 1),
            "last_ts": str(r.get("last_ts", "") or ""),
            "sample_tag": str(r.get("sample_tag", "") or ""),
        }
    return out

def save_fail_log(path: Path, mapping: Dict[str, Dict]) -> None:
    if not mapping:
        # 빈 경우는 파일 삭제보단 빈 파일 유지
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["key","post_id","content_hash","attempts","last_ts","sample_tag"]).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
        return
    rows = []
    for k, v in mapping.items():
        rows.append({
            "key": k,
            "post_id": v.get("post_id",""),
            "content_hash": v.get("content_hash",""),
            "attempts": int(v.get("attempts",1) or 1),
            "last_ts": v.get("last_ts",""),
            "sample_tag": v.get("sample_tag",""),
        })
    df = pd.DataFrame(rows, columns=["key","post_id","content_hash","attempts","last_ts","sample_tag"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

# ------------ main ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_csv",  default=str(IN_CSV),  help="입력 CSV (region_restaurant.csv)")
    ap.add_argument("--out", dest="out_csv", default=str(OUT_CSV), help="출력 CSV (extracted.csv)")
    ap.add_argument("--retry-failed", action="store_true", help="실패 로그 무시하고 실패건도 재시도")
    ap.add_argument("--verbose", action="store_true", help="로그 자세히")
    args = ap.parse_args()

    QUIET = not args.verbose and (os.getenv("QUIET", "1").lower() in {"1","true","y","yes"})
    def log(*a, **k):
        if not QUIET: print(*a, **k)

    in_path, out_path, fail_path = Path(args.in_csv), Path(args.out_csv), Path(FAIL_LOG)

    if not in_path.exists():
        raise SystemExit(f"[!] 입력 없음: {in_path}")

    # 1) 입력/출력/실패로그 로드
    src_df = read_csv_robust(in_path)
    content_col = pick_content_col(src_df)

    # 빈/공백 정리 + 내부 중복 제거
    src_df = src_df.dropna(subset=[content_col]).copy()
    src_df[content_col] = src_df[content_col].astype(str).str.strip()
    src_df = src_df[src_df[content_col].str.len() > 0]
    src_df = src_df.drop_duplicates(subset=[content_col], keep="last").reset_index(drop=True)

    # 성공 로그(=extracted.csv)에서 성공 키 생성
    seen_ok_keys: Set[str] = set()
    if out_path.exists():
        try:
            ok_df = pd.read_csv(out_path)
            if len(ok_df):
                # post_id 있으면 id:post_id, 아니면 c:<hash(content)>
                has_pid = "post_id" in ok_df.columns
                for _, r in ok_df.iterrows():
                    if has_pid and str(r.get("post_id","") or "").strip():
                        seen_ok_keys.add(f"id:{str(r['post_id']).strip()}")
                    else:
                        c = str(r.get("content","") or "")
                        if c:
                            h = hashlib.sha1(c.encode("utf-8")).hexdigest()[:16]
                            seen_ok_keys.add(f"c:{h}")
        except Exception as e:
            log(f"[warn] read {out_path} failed: {e}")

    # 실패 로그에서 실패 키 로드
    fail_map = load_fail_log(fail_path)
    seen_fail_keys: Set[str] = set(fail_map.keys())

    log(f"[seen] ok={len(seen_ok_keys)}  fail={len(seen_fail_keys)}")

    # 2) 스킵 규칙: 성공/실패 모두 스킵 (단, --retry-failed면 실패는 다시 시도)
    to_proc_rows: List[pd.Series] = []
    for _, row in src_df.iterrows():
        key, _ = make_key_from_row(row, content_col)
        if key in seen_ok_keys:           # 이미 성공한 건 스킵
            continue
        if (not args.retry_failed) and (key in seen_fail_keys):  # 실패했던 것도 스킵
            continue
        to_proc_rows.append(row)

    log(f"[filter] to process: {len(to_proc_rows)} / {len(src_df)}")

    if not to_proc_rows:
        log("[done] nothing to do.")
        return

    # 3) 처리
    new_rows: List[dict] = []
    now_ts = datetime.now().isoformat(timespec="seconds")
    succeeded_keys: Set[str] = set()
    new_fail_add: Dict[str, Dict] = {}

    for row in to_proc_rows:
        key, chash = make_key_from_row(row, content_col)
        content  = str(row.get(content_col, "") or "").strip()
        hashtags = to_listish(row.get("hashtags")) if "hashtags" in src_df.columns else []
        comments = to_listish(row.get("comments")) if "comments" in src_df.columns else []

        info = extract_restaurant_info_one(content, hashtags=hashtags, comments=comments)

        if info is None:
            # 실패 기록(시도 수 +1)
            prev = fail_map.get(key, {"attempts": 0})
            new_fail_add[key] = {
                "post_id": str(row.get("post_id","") or ""),
                "content_hash": chash,
                "attempts": int(prev.get("attempts", 0)) + 1,
                "last_ts": now_ts,
                "sample_tag": str(row.get("search_tag","") or ""),
            }
            continue

        name, addr = info
        new_rows.append({
            "content": content,
            "likes": row.get("likes", ""),
            "hashtags": hashtags,
            "comments": comments,
            "date": row.get("date", ""),
            "search_tag": row.get("search_tag", ""),
            "post_id": row.get("post_id", ""),
            "restaurant name": [name] if (name or "").strip() else [],
            "address": [addr] if (addr or "").strip() else [],
        })
        succeeded_keys.add(key)

    # 4) 성공 저장(append merge + dedupe by content)
    new_df = pd.DataFrame(new_rows, columns=[
        "content","likes","hashtags","comments","date","search_tag","post_id","restaurant name","address"
    ])
    if out_path.exists():
        try:
            exist_df = pd.read_csv(out_path)
            merged = pd.concat([exist_df, new_df], ignore_index=True)
        except Exception:
            merged = new_df
    else:
        merged = new_df
    merged = merged.drop_duplicates(subset=["content"], keep="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"[save] {out_path}  (added={len(new_df)}, total={len(merged)})")

    # 5) 실패로그 업데이트: 성공한 키는 제거, 새 실패는 반영
    #    (= 성공/실패 모두 '시도 이력'으로 남기되 다음 실행에 스킵되도록 유지)
    #    성공키 제거
    for k in list(fail_map.keys()):
        if k in succeeded_keys:
            del fail_map[k]
    #    새 실패 merge
    fail_map.update(new_fail_add)
    save_fail_log(fail_path, fail_map)
    log(f"[fail-log] wrote {fail_path}  (fail-keys={len(fail_map)}, added-fails={len(new_fail_add)})")

    log("[done] enrich finished.")

if __name__ == "__main__":
    main()
