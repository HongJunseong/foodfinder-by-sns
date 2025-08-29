# src/modules/crawling_instagram.py

import os
import re
import json
import time
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By

# 프로젝트 루트 경로(sys.path) 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# .env 로드
_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_ENV if _ENV.exists() else None)

# 로그인 정보
_INSTAGRAM_ID = os.getenv("IG_ID")
_INSTAGRAM_PW = os.getenv("IG_PW")

# 검색/수집 설정(.env)
_SEARCH_TAG = (os.getenv("IG_SEARCH_TAG") or "").strip()  # '#서울맛집' 또는 '서울맛집' 둘 다 OK

if not _SEARCH_TAG:
    raise SystemExit("[CONFIG] IG_SEARCH_TAG가 없습니다. .env에 예: IG_SEARCH_TAG=서울맛집")

_CRAWL_COUNT = int(os.getenv("IG_CRAWL_COUNT", "10")) # 크롤링할 게시글 개수

# 드라이버 준비
options = webdriver.ChromeOptions()
options.add_argument(
    "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.4.1.6 Safari/537.36"
)
dr = webdriver.Chrome(options=options)
dr.set_window_size(820, 1000)

# 유틸 임포트
from utils.crawling_utils import select_first, move_next, get_content, clear_emoji  # noqa: E402

# post_id 폴백 추출
_POST_ID_RX = re.compile(r"/p/([^/?#]+)/")


def _extract_post_id_fallback(driver) -> str:
    try:
        m = _POST_ID_RX.search(driver.current_url or "")
        return m.group(1) if m else ""
    except Exception:
        return ""


# CSV 경로
CSV_PATH = Path("./data/region_restaurant.csv")


def _load_seen_post_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        df_old = pd.read_csv(csv_path, dtype=str)
        if "post_id" in df_old.columns:
            return set(df_old["post_id"].dropna().astype(str).tolist())
        return set()
    except Exception:
        return set()


def _serialize_list_cols(df: pd.DataFrame, cols=("hashtags", "comments")) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, list)
                else (x if isinstance(x, str) else "")
            )
    return df

def _build_search_url(tag: str) -> str:
    t = (tag or "").strip()
    if not t:
        # 명시적으로 실패시켜서 사용자에게 .env 설정을 요구
        raise ValueError("IG_SEARCH_TAG 가 비어 있습니다. .env에 예: IG_SEARCH_TAG=서울맛집 를 넣어주세요.")
    if not t.startswith("#"):
        t = "#" + t
    return f"https://www.instagram.com/explore/search/keyword/?q={quote(t)}"

def _append_and_dedupe(csv_path: Path, df_new: pd.DataFrame) -> None:
    if csv_path.exists():
        try:
            df_old = pd.read_csv(csv_path, dtype=str)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new.copy()
    else:
        df_all = df_new.copy()

    # 같은 post_id 그룹에서 date 비어있으면 보정
    if "post_id" in df_all.columns and "date" in df_all.columns:
        s = df_all["date"].astype(str)
        s = s.where(s.str.strip().ne(""), pd.NA)
        df_all["date"] = s
        df_all["date"] = df_all.groupby("post_id", dropna=False)["date"].transform(
            lambda x: x.ffill().bfill()
        )
        df_all["date"] = df_all["date"].fillna("")

    # post_id 있는 건 post_id 기준, 없는 건 (content, date) 기준으로 중복 제거
    if "post_id" in df_all.columns:
        has_id = df_all["post_id"].astype(str).str.strip().ne("")
        df_id = df_all[has_id].drop_duplicates(subset=["post_id"], keep="last")
        df_no = df_all[~has_id].drop_duplicates(subset=["content", "date"], keep="last")
        df_all = pd.concat([df_id, df_no], ignore_index=True)
    else:
        df_all = df_all.drop_duplicates(subset=["content", "date"], keep="last")

    # 날짜 기준 정렬(가능한 경우만)
    if "date" in df_all.columns:
        try:
            _dt = pd.to_datetime(df_all["date"], errors="coerce")
            df_all = df_all.assign(_dt=_dt).sort_values("_dt", ascending=False).drop(columns="_dt")
        except Exception:
            pass

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")


# 로그인
dr.get("https://www.instagram.com/")
time.sleep(3)

input_id = dr.find_element(By.CSS_SELECTOR, 'input[name="username"]')
input_pw = dr.find_element(By.CSS_SELECTOR, 'input[name="password"]')
button_login = dr.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
input_id.send_keys(_INSTAGRAM_ID)
input_pw.send_keys(_INSTAGRAM_PW)
button_login.click()
time.sleep(5)

# 팝업 처리
btn = dr.find_elements(By.XPATH, "//*[contains(text(), '나중에 하기')]")
if btn:
    btn[0].click()

time.sleep(5)
btn = dr.find_elements(By.XPATH, "//*[contains(text(), '취소')]")
if btn:
    btn[0].click()

# 해시태그 검색 페이지 이동
dr.get(_build_search_url(_SEARCH_TAG))
time.sleep(5)

# 첫 게시글 열기
select_first(dr)

# 중복 체크
seen_post_ids = _load_seen_post_ids(CSV_PATH)

# 결과 누적
results = []


def crawling_instagram(target: int):
    cnt = 0
    while cnt < target:
        try:
            data = get_content(dr) or {}
            pid = str(data.get("post_id") or "") or _extract_post_id_fallback(dr)
            if pid:
                data["post_id"] = pid

            if pid and pid in seen_post_ids:
                print(f"[skip] duplicate post: {pid}")
                move_next(dr)
                continue

            data["search_tag"] = _SEARCH_TAG

            results.append(data)
            if pid:
                seen_post_ids.add(pid)

            cnt += 1
            print(f"[ok] {cnt} / {target}")
            move_next(dr)

        except Exception as e:
            print(f"[warn] {e}")
            time.sleep(3)
            try:
                move_next(dr)
            except Exception:
                pass


# 수집 실행(필요 개수 조정 가능)
crawling_instagram(_CRAWL_COUNT)

# 후처리: 이모지 제거 → 리스트 컬럼 직렬화 → CSV 합치고 중복 제거 저장
results_df = pd.DataFrame(results)
try:
    results_df = clear_emoji(results_df)
except Exception as e:
    print(f"[warn] clear_emoji skipped: {e}")

results_df = _serialize_list_cols(results_df, cols=("hashtags", "comments"))
_append_and_dedupe(CSV_PATH, results_df)

# 종료
try:
    dr.quit()
except Exception:
    pass

print(f"[done] saved → {CSV_PATH.as_posix()}")