# src/utils/crawling_utils.py

import time
import random
import re
import unicodedata
import emoji
from selenium.webdriver.common.by import By
from typing import Optional

# --- 포스트 선택/이동/정보 추출 ---
def select_first(driver):
    el = driver.find_element(By.CSS_SELECTOR, "div._aagw")
    el.click()
    time.sleep(5)

def move_next(driver):
    right = driver.find_element(By.CSS_SELECTOR, "div._aaqg._aaqh")
    right.click()
    time.sleep(3)

def get_post_id(driver):
    try:
        href = driver.current_url or ""
        if "/p/" not in href:
            a = driver.find_element(By.CSS_SELECTOR, "a[href*='/p/']")
            href = a.get_attribute("href") or ""
        m = re.search(r"/p/([^/?#]+)/", href)
        return m.group(1) if m else None
    except Exception:
        return None

def _get_post_date(driver) -> str:
    sels = ['div[role="dialog"] time[datetime]', 'article time[datetime]', 'time[datetime]']
    for sel in sels:
        try:
            el = driver.find_element(By.CSS_SELECTOR, sel)
            dt = (el.get_attribute("datetime") or "").strip()
            if dt:
                return dt
            txt = (el.text or "").strip()
            if txt:
                return txt
        except Exception:
            pass
    try:
        return driver.execute_script("""
            const els = Array.from(document.querySelectorAll('div[role="dialog"] time, article time, time'));
            for (let i = els.length - 1; i >= 0; i--) {
                const d = els[i].getAttribute('datetime');
                if (d) return d;
            }
            return els.length ? (els[els.length - 1].textContent || '').trim() : '';
        """) or ""
    except Exception:
        return ""

# header의 프로필 링크에서 username 추출
def _get_author_username(driver):
    sels = [
        'div[role="dialog"] header a[href^="/"][href$="/"]',
        'article header a[href^="/"][href$="/"]',
        'header a[href^="/"][href$="/"]',
    ]
    for sel in sels:
        try:
            for a in driver.find_elements(By.CSS_SELECTOR, sel):
                href = (a.get_attribute("href") or "")
                m = re.search(r"/([^/]+)/$", href)
                if not m:
                    continue
                cand = (m.group(1) or "").lower()
                if cand in {"p", "reel", "explore", "stories"}:
                    continue
                return cand
        except Exception:
            pass
    return ""

def _strip_leading_username(text, username):
    s = (text or "").lstrip()
    if not s:
        return s
    if username and s.lower().startswith(username.lower()):
        s = s[len(username):]
        s = re.sub(r'^[\s:·•\-\|]+', '', s)
        return s
    m = re.match(r'^([A-Za-z0-9._]{2,30})\b([\s:·•\-\|]+)', s)
    if m and not re.search(r'[가-힣ㄱ-ㅎㅏ-ㅣ]', m.group(1)):
        s = s[m.end():]
    return s

def _strip_trailing_relativetime(text):
    s = (text or "").rstrip()
    if not s:
        return s
    s = re.sub(
        r'\s*(?:\d+\s*(?:초|분|시간|일|주|개월|년)\s*전?|\d+\s*[smhdwSMHDW]|'
        r'\d+\s*(?:hour|hours|hr|hrs|minute|minutes|min|mins|day|days|week|weeks))\s*$',
        '',
        s,
        flags=re.I
    )
    return s.strip()

# --- 댓글과 해쉬태그로 부터 좋아요 추정 ---
def estimate_likes_from_comments_and_tags(comments_count, tags_count):
    return (comments_count * 5) + (tags_count * 2) + random.randint(1, 10)

def _like_text_to_int(s: str) -> Optional[int]:
    """좋아요 텍스트 → 정수. 실패 시 None."""
    if not s:
        return None
    s = s.strip()
    if not any(ch.isdigit() for ch in s):
        return None
    m = re.search(r'(\d+(?:\.\d+)?)\s*([kKmM])', s)
    if m:
        v = float(m.group(1))
        return int(v * (1000 if m.group(2).lower() == 'k' else 1_000_000))
    m = re.search(r'(\d+(?:\.\d+)?)\s*(천|만)', s)
    if m:
        v = float(m.group(1)); unit = m.group(2)
        return int(v * (1_000 if unit == '천' else 10_000))
    m = re.search(r'(\d{1,3}(?:,\d{3})+|\d+)', s)
    if m:
        return int(m.group(1).replace(',', ''))
    return None

# --- 본문/댓글/태그/날짜만 수집 ---
def get_content(driver):
    try:
        content_element = driver.find_element(By.CSS_SELECTOR, "div._a9zr")
        raw = (content_element.text or "").replace("\n", " ").strip()
        username = _get_author_username(driver)
        content = _strip_trailing_relativetime(_strip_leading_username(raw, username))
        content = unicodedata.normalize("NFC", content)
    except Exception as e:
        print(f"[get_content] Error extracting content: {e}")
        content = ""

    tags = re.findall(r"#[^\s#,\\]+", content) if content else []

    try:
        comments_elements = driver.find_elements(By.CSS_SELECTOR, "div._a9zr span._ap3a")
        comments = [
            unicodedata.normalize("NFC", c.text)
            for c in comments_elements
            if (c.text or "").strip()
        ]
    except Exception as e:
        print(f"[get_content] Error extracting comments: {e}")
        comments = []

    try:
        likes_element = driver.find_element(By.CSS_SELECTOR, 'section.x12nagc span.xdj266r')
        likes_text = (likes_element.text or "").strip()
        parsed = _like_text_to_int(likes_text)
        likes = parsed if parsed is not None else estimate_likes_from_comments_and_tags(len(comments), len(tags))
    except Exception:
        likes = estimate_likes_from_comments_and_tags(len(comments), len(tags))
    likes = str(likes)

    try:
        date = _get_post_date(driver)
    except Exception:
        date = ""

    post_id = get_post_id(driver)

    return {
        "post_id": post_id,
        "content": content,
        "likes": likes,
        "hashtags": tags,
        "comments": comments,
        "date": date,
    }

# --- 이모지 제거 ---
def remove_emoji(text: str) -> str:
    return emoji.replace_emoji(text, replace="")

def clear_emoji(df):
    if "content" in df.columns:
        df["content"] = df["content"].apply(remove_emoji)
    if "hashtags" in df.columns:
        df["hashtags"] = df["hashtags"].apply(
            lambda tags: [remove_emoji(tag) for tag in tags] if isinstance(tags, list) else []
        )
    if "comments" in df.columns:
        df["comments"] = df["comments"].apply(
            lambda cs: [remove_emoji(c) for c in cs] if isinstance(cs, list) else []
        )
        df["comments"] = df["comments"].apply(lambda x: "" if x == [] else x)
    if "restaurant name" in df.columns:
        df["restaurant name"] = df["restaurant name"].apply(
            lambda names: [remove_emoji(n) for n in names] if isinstance(names, list) else names
        )
    if "address" in df.columns:
        df["address"] = df["address"].apply(
            lambda addrs: [remove_emoji(a) for a in addrs] if isinstance(addrs, list) else addrs
        )
    return df
