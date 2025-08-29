# src/utils/map_utils.py

from __future__ import annotations

import os
import re
import time
from typing import Optional, Tuple, Dict, List
import requests

# Kakao REST API 키 (필수)
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
if not KAKAO_REST_API_KEY:
    raise RuntimeError("KAKAO_REST_API_KEY가 없습니다. .env 또는 환경변수에 설정하세요.")
HEADERS = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}

# 주소 중 도로명만 뽑아 쓰기(있으면)
_ROAD_RX = re.compile(r'([가-힣]+\s[가-힣]+\s[가-힣0-9\-]+(?:로|길)\s*\d+(?:-\d+)?)')
def extract_road_address(address: str) -> str:
    if not address:
        return ""
    m = _ROAD_RX.search(address)
    return m.group(0) if m else address.strip()

# 이름 단순 포함 매칭(정규화: 소문자/공백제거)
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    return s
def _name_contains(query: str, cand: str) -> bool:
    q = _norm(query); c = _norm(cand)
    return bool(q and c and (q in c))

# Kakao API: 주소 → 좌표
def coord_info(address: str, timeout: int = 8) -> Tuple[Optional[float], Optional[float]]:
    if not address:
        return (None, None)
    r = requests.get(
        "https://dapi.kakao.com/v2/local/search/address.json",
        headers=HEADERS, params={"query": address}, timeout=timeout
    )
    if r.status_code != 200:
        return (None, None)
    docs = (r.json() or {}).get("documents") or []
    if not docs:
        return (None, None)
    d = docs[0]
    # Kakao: x=lng, y=lat
    return float(d["y"]), float(d["x"])

# Kakao API: 키워드 검색(좌표 반경 기준)
def _keyword_search(place_name: str, x: float, y: float, radius: int = 300, timeout: int = 8) -> List[Dict]:
    if not place_name:
        return []
    r = requests.get(
        "https://dapi.kakao.com/v2/local/search/keyword.json",
        headers=HEADERS, params={"query": place_name, "x": x, "y": y, "radius": radius},
        timeout=timeout
    )
    if r.status_code != 200:
        return []
    return (r.json() or {}).get("documents", []) or []

def _pick_by_contains(docs: List[Dict], place_name: str) -> Optional[Dict]:
    if not docs:
        return None
    # 1) 음식점(FD6) 우선
    fd6 = [d for d in docs if (d.get("category_group_code") or "") == "FD6"]
    pool = fd6 or docs
    # 2) 이름 포함 우선
    for d in pool:
        if _name_contains(place_name, d.get("place_name", "") or ""):
            return d
    # 3) 그래도 없으면 1순위
    return pool[0]


# 메인: "주소가 있으면" 좌표화. 필요 시 동일 좌표 주변에서 이름으로 정밀화.
def geocode_by_address(place_name: str, address: str, *, radius: int = 300, sleep_sec: float = 0.1) -> Optional[Dict]:
    """
    반환:
      {"place_name": str, "address_name": str, "lat": float, "lng": float, "source": "address"|"keyword+around-addr"}
    - address 필수: 없으면 None
    - 주소 좌표 성공 시, 그 좌표 반경에서 place_name 키워드로 재탐색(있으면 그 결과, 없으면 주소 좌표 그대로)
    """
    name = (place_name or "").strip()
    addr = extract_road_address(address or "")

    lat, lng = coord_info(addr)
    time.sleep(sleep_sec)
    if lat is None or lng is None:
        return None  # 주소 없거나 실패면 끝

    # 주소 좌표 주변에서 이름 교차확인(선택적 정밀화)
    docs = _keyword_search(name, x=lng, y=lat, radius=radius) if name else []
    time.sleep(sleep_sec)
    pick = _pick_by_contains(docs, name) if docs else None
    if pick:
        return {
            "place_name": pick.get("place_name") or name,
            "address_name": pick.get("road_address_name") or pick.get("address_name") or addr,
            "lat": float(pick["y"]), "lng": float(pick["x"]),
            "source": "keyword+around-addr",
        }

    # 이름 매칭이 없으면 주소 좌표만 사용
    return {"place_name": name, "address_name": addr, "lat": lat, "lng": lng, "source": "address"}
