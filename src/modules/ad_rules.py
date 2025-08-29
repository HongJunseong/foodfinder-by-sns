# src/modules/ad_rules.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re, unicodedata
from typing import List, Tuple, Dict, Any

# --- small utils ---
def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)  # 제로폭 제거
    return s

def _fuzz(word: str) -> str:
    # 글자 사이 공백/구두점 최대 2개 허용 (예: 유  료, 광.고)
    sep = r"[ \t\n\r._\-·]{0,2}"
    return sep.join(map(re.escape, word))

def _extract_hashtags_from_text(text: str) -> List[str]:
    return re.findall(r"#\S+", text or "")

def _to_tag_list(x) -> List[str]:
    # 리스트면 그대로, 문자열이면 본문에서 #태그 뽑기, 아니면 []
    if isinstance(x, list):
        return [str(t) for t in x]
    if isinstance(x, str):
        return _extract_hashtags_from_text(x)
    return []

# --- dict/patterns ---
AD_HASH = {
    "#광고", "#유료광고", "#협찬", "#무료협찬", "#제품협찬", "#체험단",
    "#원고료", "#원고지원", "#제품제공", "#제공받음", "#제공받아",
    "#서포터즈", "#앰버서더", "#브랜드앰버서더", "#브랜디드콘텐츠", "#제작지원",
    "#후원", "#홍보", "#공구", "#공동구매",
    "#PPL", "#ppl", "#sponsored", "#ad", "#AD", "#paidpartnership"
}
AD_HASH_NORM = {h.lower().lstrip("#") for h in AD_HASH}

# 퍼지 키워드(본문)
_KW_PATTERNS = [
    rf"{_fuzz('유료')}\s*{_fuzz('광고')}",
    rf"{_fuzz('광고')}",
    rf"{_fuzz('협찬')}",
    rf"{_fuzz('제품')}\s*{_fuzz('제공')}",
    rf"{_fuzz('제공받')}",
    rf"{_fuzz('무상')}\s*{_fuzz('제공')}",
    rf"{_fuzz('원고')}\s*{_fuzz('료')}",
    rf"{_fuzz('제작')}\s*{_fuzz('지원')}",
    rf"{_fuzz('서포터즈')}",
    rf"{_fuzz('앰버서더')}",
    rf"{_fuzz('브랜드')}\s*{_fuzz('앰버서더')}",
    rf"{_fuzz('브랜디드')}\s*{_fuzz('콘텐츠')}",
    rf"{_fuzz('제휴')}",
    rf"{_fuzz('콜라보')}",
    rf"{_fuzz('위탁')}",
    rf"{_fuzz('대행')}",
    rf"{_fuzz('공동')}\s*{_fuzz('구매')}",
    rf"{_fuzz('공구')}",
    # 영어
    r"\bpaid\s*partnership\b",
    r"\bsponsored\b",
    r"\bsponsorship\b",
    r"\bgifted\b",
    r"\bPR\b",
    r"\bPPL\b",
    r"\bAD\b",
    r"\brepost\b",
    r"\bregram\b",
]
KW_RX = [re.compile(p, re.I) for p in _KW_PATTERNS]

# 심플 키워드(부분포함 허용) — 리콜↑
KW_SIMPLE = {
    "부업","광고","협찬","재테크","출금","공짜","수익","카톡","원금","댓글",
    "할인","이벤트","부자","repost","regram","구매","문의전화","오시는",
    "구매링크","예약문의","상담문의","문의주세요","디엠주세요","dm주세요",
    "링크클릭","프로필 링크","공구","공동구매", "먹킷리스트", "먹킷"
}

# CTA / 링크 / 전화
CTA_WORDS = [
    "예약","문의","상담","상담문의","예약문의","문의주세요","주문","구매",
    "구매링크","링크","프로필 링크","링크클릭","dm","디엠","카톡","카카오톡","톡주세요","카톡주세요"
]
CTA_RX = re.compile("|".join(map(re.escape, CTA_WORDS)), re.I)

LINK_RX = re.compile(
    r"(https?://|linktr\.ee|smartstore(\.naver\.com)?|shopping\.naver\.com|naver\.me|"
    r"coupa\.ng|forms\.gle|forms\.office\.com|tally\.so|instabio|link\.inbio|linkin\.bio|"
    r"pf\.kakao\.com|open\.kakao\.com|page\.stibee\.com|link\.inpock|urlr\.me|kko\.to|"
    r"bit\.ly|t\.co|goo\.gl|notion\.site|instagram\.com/p/|youtu\.be|youtube\.com)",
    re.I
)
PHONE_RX = re.compile(r"(?<!\d)(01[016789])[-\s\.]?\d{3,4}[-\s\.]?\d{4}(?!\d)")

# --- scoring ---
def ad_score(text: str, tags: List[str], extra_tags_from_text: bool = True) -> Tuple[int, Dict[str, Any]]:
    """
    규칙 점수(0~100) + 이유 dict 반환
    - extra_tags_from_text: 본문 내 #태그까지 합쳐서 판단
    """
    t = _normalize(text)

    tags_norm = {str(x).lower().lstrip("#") for x in (tags or [])}
    if extra_tags_from_text:
        in_text = {h.lower().lstrip("#") for h in _extract_hashtags_from_text(t)}
        tags_norm |= in_text

    score, why = 0, {}

    # 1) 광고성 해시태그
    if AD_HASH_NORM & tags_norm:
        score += 70; why["hash"] = True

    # 2) 본문 키워드
    kw_hit = any(rx.search(t) for rx in KW_RX)
    if kw_hit:
        score += 30; why["kw"] = True
    else:
        t_low = t.lower()
        if any(k in t_low for k in (w.lower() for w in KW_SIMPLE)):
            score += 20; why["kw_simple"] = True

    # 3) CTA
    if CTA_RX.search(t):
        score += 10; why["cta"] = True

    # 4) 링크
    if LINK_RX.search(t):
        score += 15; why["link"] = True

    # 5) 전화
    if PHONE_RX.search(t):
        score += 10; why["phone"] = True

    return min(score, 100), why

# --- DF helper: 라벨 생성 ---
def apply_ad_rules_df(
    df,
    content_col: str = "content",
    hashtags_col: str = "hashtags",
    thr: int = 40,
    mode: str = "balanced",  # "balanced" | "recall"
):
    """
    rule_score / rule_why / is_ad 추가
    - balanced: (해시태그) OR (키워드 AND (링크/CTA/전화)) OR (점수>=thr)
    - recall  : (해시태그) OR (키워드) OR (링크 AND CTA) OR 전화 OR (점수>=thr)
    """
    import pandas as pd

    contents = df.get(content_col, "").fillna("")
    tags_src = df[hashtags_col] if hashtags_col in df.columns else pd.Series([""] * len(df), index=df.index)

    rule_scores, whys, labels = [], [], []

    for txt, h in zip(contents, tags_src):
        tag_list = _to_tag_list(h)
        s, why = ad_score(txt, tag_list, extra_tags_from_text=True)
        rule_scores.append(s)
        whys.append(why)

        has_hash  = bool(why.get("hash"))
        has_kw    = bool(why.get("kw") or why.get("kw_simple"))
        has_cta   = bool(why.get("cta"))
        has_link  = bool(why.get("link"))
        has_phone = bool(why.get("phone"))

        if mode == "recall":
            is_ad = 1 if (has_hash or has_kw or (has_link and has_cta) or has_phone or s >= thr) else 0
        else:
            is_ad = 1 if (has_hash or (has_kw and (has_link or has_cta or has_phone)) or s >= thr) else 0

        labels.append(is_ad)

    out = df.copy()
    out["rule_score"] = rule_scores
    out["rule_why"]   = whys
    out["is_ad"]      = labels
    return out
