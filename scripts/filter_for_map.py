# scripts/filter_for_map.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

CONFIG = {
    "INPUT_CSV": "data/ad_extracted.csv",  # make_ad_labels.py --out 기본값
    "OUTPUT_DIR": "data/clean",
}


def main(cfg: dict = CONFIG) -> None:
    inp = Path(cfg["INPUT_CSV"])
    if not inp.exists():
        raise SystemExit(f"[ERROR] 입력 없음: {inp}")

    df = pd.read_csv(inp)

    if "is_ad" not in df.columns:
        raise SystemExit("[ERROR] 'is_ad' 컬럼이 없습니다. 먼저 make_ad_labels.py를 실행하세요.")

    out_dir = Path(cfg["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 비광고(지도용)
    non_ads = df[df["is_ad"] == 0].copy()
    non_path = out_dir / "non_ads.csv"
    non_ads.to_csv(non_path, index=False, encoding="utf-8-sig")

    # 광고(참고용)
    ads = df[df["is_ad"] == 1]
    ads_path = out_dir / "ads.csv"
    ads.to_csv(ads_path, index=False, encoding="utf-8-sig")

    print(f"[OK] 비광고 저장 → {non_path} (rows={len(non_ads)})")
    print(f"[OK] 광고 저장 → {ads_path} (rows={len(ads)})")


if __name__ == "__main__":
    main()
