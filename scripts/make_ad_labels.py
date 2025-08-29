# scripts/make_ad_labels.py

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# sys.path: 프로젝트 루트 + src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.modules.data_io import load_posts
from src.modules.ad_rules import apply_ad_rules_df


def _maybe_parse_json_like(x):
    """문자열로 직렬화된 리스트/딕셔너리를 실제 객체로 파싱 시도."""
    if not isinstance(x, str):
        return x
    s = x.strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            return x
    return x


def main():
    ap = argparse.ArgumentParser(description="규칙 기반 광고 라벨 생성기 (rule_score/why/is_ad 추가)")
    ap.add_argument("--inp", type=str, default="data/extracted.csv", help="입력 CSV 경로(라벨 없는 원본)")
    ap.add_argument("--out", type=str, default="data/ad_extracted.csv", help="출력 CSV 경로(is_ad 포함)")
    ap.add_argument("--thr", type=int, default=30, help="점수 임계값(기본 30; 보수적이면 50~60)")
    ap.add_argument("--mode", choices=["balanced", "recall"], default="recall", help="라벨링 모드")
    ap.add_argument("--content_col", type=str, default="content", help="본문 텍스트 컬럼명")
    ap.add_argument("--hashtags_col", type=str, default="hashtags", help="해시태그 컬럼명(없으면 본문만 사용)")
    ap.add_argument("--encoding", type=str, default=None, help="입력 CSV 인코딩(미사용: load_posts 내부 처리)")
    ap.add_argument("--debug_top", type=int, default=0, help="점수 상위 N개 샘플을 data/debug_rule_hits.csv로 저장")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise SystemExit(f"[ERROR] 입력 없음: {inp.as_posix()}")

    # 1) 로드
    df = load_posts(source="csv", csv_path=args.inp)

    # 2) 필수 컬럼 확인
    if args.content_col not in df.columns:
        raise SystemExit(
            f"[ERROR] 본문 컬럼 '{args.content_col}' 이 없습니다. "
            f"--content_col로 지정하거나 컬럼명을 확인하세요. "
            f"현재 컬럼: {list(df.columns)[:20]} ..."
        )

    # 3) hashtags 문자열(JSON)일 경우 파싱
    if args.hashtags_col in df.columns:
        df[args.hashtags_col] = df[args.hashtags_col].apply(_maybe_parse_json_like)

    # 4) 규칙 적용 → rule_score / rule_why / is_ad
    df2 = apply_ad_rules_df(
        df,
        content_col=args.content_col,
        hashtags_col=args.hashtags_col if args.hashtags_col in df.columns else "___NO_COL___",
        thr=args.thr,
        mode=args.mode,
    )

    # 5) 통계
    vc = df2["is_ad"].value_counts().to_dict()
    pos, neg = int(vc.get(1, 0)), int(vc.get(0, 0))
    total = len(df2)
    ratio = (pos / total * 100) if total else 0.0
    print(f"[INFO] is_ad 분포: 1(광고)={pos}, 0(비광고)={neg}, total={total}, ratio={ratio:.1f}%")
    print(
        f"[INFO] thr={args.thr}, mode={args.mode}, "
        f"content_col={args.content_col}, "
        f"hashtags_col={'(없음)' if args.hashtags_col not in df.columns else args.hashtags_col}"
    )

    # 6) 디버그 출력(선택)
    if args.debug_top and args.debug_top > 0:
        debug_path = Path("data/debug_rule_hits.csv")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [args.content_col, "rule_score", "rule_why", "is_ad"]
        extra = [args.hashtags_col] if args.hashtags_col in df.columns else []
        df2.sort_values("rule_score", ascending=False)[cols + extra].head(args.debug_top).to_csv(
            debug_path, index=False, encoding="utf-8-sig"
        )
        print(f"[INFO] top-{args.debug_top} 저장 → {debug_path.as_posix()}")

    # 7) 저장
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] 저장 완료 → {out.as_posix()}")


if __name__ == "__main__":
    main()
