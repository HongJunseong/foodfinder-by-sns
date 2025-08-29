# src/modules/llm/llm_extract.py
from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# env
ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / ".env"
load_dotenv(ENV if ENV.exists() else None)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DISABLE_LLM    = os.getenv("DISABLE_LLM", "0").strip().lower() in {"1","true","y","yes"}
QUIET          = os.getenv("QUIET", "0").strip().lower() in {"1","true","y","yes"}

def _log(*a, **k):
    if not QUIET:
        print(*a, **k)

if DISABLE_LLM:
    _log("[llm] disabled: DISABLE_LLM=1")
elif not OPENAI_API_KEY:
    _log("[llm] missing OPENAI_API_KEY (skip)")
else:
    _log(f"[llm] ready: model={OPENAI_MODEL}")

# 프롬프트
SYSTEM_PROMPT = """인스타 본문만 보고 ‘실존 식당/매장명’만 추출하세요.
- 해시태그(#...)나 @핸들(@acc)은 이름 후보로 쓰지 말 것.
- ‘..맛집/레전드맛집/종결판/가이드/리스트/핫플/먹킷’ 등 설명/가이드명은 식당명이 아님.
- 본문에 한글 고유 상호가 또렷이 있으면 그 이름을 채택. 모호하면 place_count=0.
- 주소는 있으면 도로명, 없으면 "".
- 광고/협찬/체험단/무료제공/AD/링크/문의/쿠폰 표현이 있으면 is_ad=true.
- 출력은 아래 JSON 단일 객체 한 개만.
{
  "place_count": 0,
  "places": [{"restaurant_name": "", "address": ""}],
  "is_ad": false,
  "confidence": 0.0
}
"""

USER_TMPL = """다음 본문만 보고 판단하세요. 해시태그로 이름 만들지 말고, 가이드/모음글이면 place_count=0.

[본문]
{content}

위 스키마의 JSON '한 객체'만 출력하세요.
"""

def _call_llm_json(content: str) -> Dict[str, Any]:
    """LLM 호출 → 최상위 JSON 하나 파싱."""
    if DISABLE_LLM or not OPENAI_API_KEY:
        return {}
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.2)
    prompt = USER_TMPL.format(content=(content or "")[:6000])
    msg = llm.invoke([("system", SYSTEM_PROMPT), ("user", prompt)])
    text = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg)

    # 본문에서 바깥쪽 JSON만 잘라 파싱
    try:
        start = text.index("{"); end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 통째로 파싱 시도
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}

def _postprocess(obj: Dict[str, Any]) -> Dict[str, Any]:
    """필드 보정 + 안전 범위 클리핑."""
    if not isinstance(obj, dict):
        obj = {}
    try:
        pc = int(obj.get("place_count", 0))
    except Exception:
        pc = 0

    places = obj.get("places") or []
    if not isinstance(places, list):
        places = []
    norm: List[Dict[str, str]] = []
    for p in places:
        if not isinstance(p, dict):
            continue
        name = str(p.get("restaurant_name", "") or "").strip()
        addr = str(p.get("address", "") or "").strip()
        norm.append({"restaurant_name": name, "address": addr})

    is_ad = bool(obj.get("is_ad", False))
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    return {"place_count": pc, "places": norm, "is_ad": is_ad, "confidence": conf}

# 공개 API
def extract_one_with_meta(content: str, allow_consolidation: bool = False) -> Optional[Dict[str, Any]]:
    """
    단일 가게만 받는다. place_count==1일 때만 반환.
    반환: {"restaurant_name","address","is_ad","confidence","source_place_count":1}
    """
    content = (content or "").strip()
    if not content or DISABLE_LLM or not OPENAI_API_KEY:
        return None

    data = _postprocess(_call_llm_json(content))
    if int(data.get("place_count", 0)) != 1:
        time.sleep(float(os.getenv("LLM_THROTTLE_SEC", "0.2")))
        return None

    places = data.get("places") or []
    if not places:
        time.sleep(float(os.getenv("LLM_THROTTLE_SEC", "0.2")))
        return None

    pick = places[0]
    time.sleep(float(os.getenv("LLM_THROTTLE_SEC", "0.2")))
    return {
        "restaurant_name": pick.get("restaurant_name", ""),
        "address": pick.get("address", ""),
        "is_ad": bool(data.get("is_ad", False)),
        "confidence": float(data.get("confidence", 0.0)),
        "source_place_count": 1,
    }

def extract_restaurant_info_one(content: str, **_) -> Optional[Tuple[str, str]]:
    """(호환) 단일이면 (name, address), 아니면 None."""
    meta = extract_one_with_meta(content)
    if not meta:
        return None
    return (meta["restaurant_name"], meta["address"])
