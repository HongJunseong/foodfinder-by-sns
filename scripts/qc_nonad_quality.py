# scripts/qc_nonad_quality.py

from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None

CONFIG = {
    "MODE": "TRAIN",               # "TRAIN" | "CHECK" | "BOTH"
    "TRAIN_CSV": "data/ad_extracted.csv",
    "CHECK_CSV": "data/clean/non_ads.csv",
    "OUT_DIR": "data/qc",
    "ART_DIR": "artifacts/qc",
    "SAVE_ARTIFACTS": True,       # ← 아티팩트 저장 토글 (False면 저장만 생략)
    "MODEL": "lgbm",               # "lgbm" | "rf" | "xgb"
    "USE_TEXT": True,
    "MASK_RULE_SIGNALS": True,
    "TFIDF_MIN_DF": 3,
    "TFIDF_MAX_FEATURES": 40000,
    "RANDOM_STATE": 42,
    "N_SPLITS": 5,
    "QC_THR": 0.35,
    "BORDER_DELTA": 0.15,
    "CONTENT_COL": "content",
    "HASHTAGS_COL": "hashtags",
    "LABEL_COL": "is_ad",
}

AD_WORDS = ["광고","유료광고","협찬","제품제공","제공받","원고료","서포터즈","앰배서더","sponsored","paid partnership","gifted","AD","PPL","PR"]
CTA_WORDS = ["예약","문의","상담","구매","구매링크","디엠","DM","카톡","카카오톡","전화","프로필 링크","링크"]
HASHTAG_RX = re.compile(r"#\S+")
KW_RX  = re.compile("|".join(map(re.escape, AD_WORDS)), re.I)
CTA_RX = re.compile("|".join(map(re.escape, CTA_WORDS)), re.I)
URL_RX = re.compile(r"(https?://\S+|linktr\.ee|smartstore|naver\.me|bit\.ly|pf\.kakao\.com)", re.I)
PHONE_RX = re.compile(r"(?<!\d)(01[016789])[-\s\.]?\d{3,4}[-\s\.]?\d{4}(?!\d)")
EMOJI_RX = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]")

def clean_text(series: pd.Series, mask: bool = True) -> pd.Series:
    s = series.fillna("").astype(str)
    if mask:
        s = s.apply(lambda t: HASHTAG_RX.sub(" ", t))
        s = s.apply(lambda t: KW_RX.sub(" ", t))
        s = s.apply(lambda t: CTA_RX.sub(" ", t))
        s = s.apply(lambda t: URL_RX.sub(" ", t))
        s = s.apply(lambda t: PHONE_RX.sub(" ", t))
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def robust_comment_count(x):
    if isinstance(x, (list, tuple)): return len(x)
    if isinstance(x, (int, float)):  return int(x) if pd.notnull(x) else 0
    if isinstance(x, str):
        s = x.strip()
        if not s: return 0
        return len([t for t in re.split(r'[\n,]+', s) if t.strip()])
    return 0

def extra_num_feats(df: pd.DataFrame, cfg=CONFIG) -> pd.DataFrame:
    out = {}
    if "likes" in df.columns:     out["likes_log1p"] = np.log1p(pd.to_numeric(df["likes"], errors="coerce").fillna(0))
    if "views" in df.columns:     out["views_log1p"] = np.log1p(pd.to_numeric(df["views"], errors="coerce").fillna(0))
    if "bookmarks" in df.columns: out["bookmarks_log1p"] = np.log1p(pd.to_numeric(df["bookmarks"], errors="coerce").fillna(0))
    if "comments" in df.columns:  out["comment_count"] = df["comments"].apply(robust_comment_count)
    if "hashtags" in df.columns:
        h = df["hashtags"]
        out["hashtag_count"] = np.where(
            h.apply(lambda v: isinstance(v, (list, tuple))),
            h.apply(lambda v: len(v) if isinstance(v, (list,tuple)) else 0),
            h.astype(str).str.count("#")
        )
    for col in ["image_count", "video_count", "media_count"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.DataFrame(out) if out else pd.DataFrame(index=df.index)

def basic_num_feats(series: pd.Series) -> pd.DataFrame:
    s = series.fillna("").astype(str)
    return pd.DataFrame({
        "len": s.str.len(),
        "words": s.str.split().apply(len),
        "exc": s.str.count("!"),
        "q": s.str.count(r"\?"),
        "digits": s.str.count(r"\d"),
        "emoji": s.apply(lambda t: len(EMOJI_RX.findall(t))),
    })

def build_X_fit(df: pd.DataFrame, cfg=CONFIG):
    Xn_basic = csr_matrix(basic_num_feats(df[cfg["CONTENT_COL"]]).values)
    Xn_extra = extra_num_feats(df, cfg)
    Xn = Xn_basic if Xn_extra.empty else hstack([Xn_basic, csr_matrix(Xn_extra.values)], format="csr")

    if not cfg["USE_TEXT"]:
        return Xn, None

    from sklearn.feature_extraction.text import TfidfVectorizer
    txt = clean_text(df[cfg["CONTENT_COL"]], cfg["MASK_RULE_SIGNALS"])
    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,5),
        min_df=cfg["TFIDF_MIN_DF"], max_features=cfg["TFIDF_MAX_FEATURES"],
        dtype=np.float32
    )
    Xt = tfidf.fit_transform(txt)
    X = hstack([Xn, Xt]).tocsr()
    return X, tfidf

def build_X_transform(df: pd.DataFrame, tfidf, cfg=CONFIG):
    Xn_basic = csr_matrix(basic_num_feats(df[cfg["CONTENT_COL"]]).values)
    Xn_extra = extra_num_feats(df, cfg)
    Xn = Xn_basic if Xn_extra.empty else hstack([Xn_basic, csr_matrix(Xn_extra.values)], format="csr")
    if not cfg["USE_TEXT"] or tfidf is None:
        return Xn
    txt = clean_text(df[cfg["CONTENT_COL"]], cfg["MASK_RULE_SIGNALS"])
    Xt = tfidf.transform(txt)
    return hstack([Xn, Xt]).tocsr()

def build_model(kind: str, seed: int):
    kind = (kind or "").lower()
    if kind == "lgbm" and lgb is not None:
        return Pipeline([("scaler", StandardScaler(with_mean=False)),
            ("clf", lgb.LGBMClassifier(
                n_estimators=600, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
                subsample=0.9, colsample_bytree=0.8, class_weight="balanced",
                random_state=seed, n_jobs=-1, verbosity=-1))])
    if kind == "xgb":
        if xgb is None:
            raise SystemExit("[ERROR] xgboost 미설치. conda/pip로 설치하세요.")
        return Pipeline([("scaler", StandardScaler(with_mean=False)),
            ("clf", xgb.XGBClassifier(
                n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                eval_metric="logloss", tree_method="hist", n_jobs=-1))])
    return Pipeline([("scaler", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(n_estimators=400, random_state=seed, class_weight="balanced", n_jobs=-1))])

def choose_thr(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    thr = np.r_[0.0, thr]
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    i = int(np.nanargmax(f1))
    return float(thr[i]), {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}

def train(cfg=CONFIG):
    art_dir = Path(cfg["ART_DIR"]); art_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["OUT_DIR"]); out_dir.mkdir(parents=True, exist_ok=True)

    print("=== TRAIN QC ===")
    print(f"[INFO] QC model = {cfg['MODEL']} | text={cfg['USE_TEXT']} | mask_rules={cfg['MASK_RULE_SIGNALS']}")

    df = pd.read_csv(cfg["TRAIN_CSV"])
    if cfg["LABEL_COL"] not in df.columns:
        raise SystemExit(f"[ERROR] '{cfg['LABEL_COL']}' 컬럼이 없습니다. 먼저 make_ad_labels.py 실행.")

    X, tfidf = build_X_fit(df, cfg)
    y = df[cfg["LABEL_COL"]].astype(int).values

    vc = pd.Series(y).value_counts()
    if len(vc) < 2:
        raise SystemExit("[ERROR] 한 클래스만 존재. TRAIN_CSV에 0/1 모두 필요.")
    minority = int(vc.min())
    if minority < 2:
        raise SystemExit(f"[ERROR] 소수 클래스 표본 부족(min={minority}). 표본을 늘리거나 규칙 임계 완화.")

    n_splits_eff = max(2, min(cfg["N_SPLITS"], minority))
    print(f"[INFO] label dist = {vc.to_dict()} | n_splits={n_splits_eff}")

    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=cfg["RANDOM_STATE"])
    oof = np.zeros(len(df), dtype=float)

    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        pipe = build_model(cfg["MODEL"], cfg["RANDOM_STATE"] + k)
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[va])[:, 1]
        oof[va] = proba
        thr_fold, _ = choose_thr(y[va], proba)
        pred_fold = (proba >= thr_fold).astype(int)
        ap = average_precision_score(y[va], proba)
        roc = roc_auc_score(y[va], proba)
        f1f = f1_score(y[va], pred_fold)
        prec = precision_score(y[va], pred_fold, zero_division=0)
        rec = recall_score(y[va], pred_fold, zero_division=0)
        acc = accuracy_score(y[va], pred_fold)
        print(f"[Fold {k}] AP={ap:.4f} ROC-AUC={roc:.4f} F1={f1f:.4f} P={prec:.4f} R={rec:.4f} ACC={acc:.4f} thr={thr_fold:.3f}")

    thr_all, stat_all = choose_thr(y, oof)
    pred_oof = (oof >= thr_all).astype(int)
    print(f"\n[OOF] AP={average_precision_score(y, oof):.4f} ROC-AUC={roc_auc_score(y, oof):.4f} "
          f"F1@thr={f1_score(y, pred_oof):.4f} P={precision_score(y, pred_oof, zero_division=0):.4f} "
          f"R={recall_score(y, pred_oof, zero_division=0):.4f} ACC={accuracy_score(y, pred_oof):.4f} thr={thr_all:.3f}")

    # 전체 재학습
    pipe = build_model(cfg["MODEL"], cfg["RANDOM_STATE"])
    pipe.fit(X, y)

    # 저장 토글
    if cfg.get("SAVE_ARTIFACTS", True):
        joblib.dump(pipe, art_dir / "qc_model.joblib")
        if tfidf is not None:
            joblib.dump(tfidf, art_dir / "qc_tfidf.joblib")
        meta = {
            "threshold": float(thr_all),
            "stat": stat_all,
            "model": cfg["MODEL"],
            "use_text": bool(cfg["USE_TEXT"]),
            "mask_rule_signals": bool(cfg["MASK_RULE_SIGNALS"]),
            "tfidf": {"min_df": cfg["TFIDF_MIN_DF"], "max_features": cfg["TFIDF_MAX_FEATURES"]},
        }
        (art_dir / "qc_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVE] {art_dir/'qc_model.joblib'}")
        if tfidf is not None: print(f"[SAVE] {art_dir/'qc_tfidf.joblib'}")
        print(f"[SAVE] {art_dir/'qc_meta.json'}")
    else:
        print("[SKIP] SAVE_ARTIFACTS=False → 파일 저장 생략")

def check(cfg=CONFIG):
    print("\n=== CHECK QC ===")
    out_dir = Path(cfg["OUT_DIR"]); out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = Path(cfg["ART_DIR"])

    model_path = art_dir / "qc_model.joblib"
    tfidf_path = art_dir / "qc_tfidf.joblib"
    meta_path  = art_dir / "qc_meta.json"

    if not model_path.exists():
        raise SystemExit(f"[ERROR] QC 모델이 없습니다: {model_path}  (먼저 TRAIN 하거나 SAVE_ARTIFACTS=True로 생성)")

    pipe = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path) if tfidf_path.exists() else None

    thr = float(cfg.get("QC_THR", 0.5))
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        thr = float(cfg.get("QC_THR", meta.get("threshold", 0.5)))

    inp = Path(cfg["CHECK_CSV"])
    if not inp.exists():
        raise SystemExit(f"[ERROR] 점검 입력 파일이 없습니다: {inp}")

    df = pd.read_csv(inp)
    if len(df) == 0:
        print("[INFO] 점검할 비광고가 없습니다.")
        pd.DataFrame().to_csv(out_dir/"suspect_nonads.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(out_dir/"borderline_nonads.csv", index=False, encoding="utf-8-sig")
        (out_dir/"qc_summary.json").write_text(json.dumps({"nonads":0,"suspects":0,"borderline":0,"thr":thr}, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    X = build_X_transform(df, tfidf, cfg)
    proba = pipe.predict_proba(X)[:, 1]
    df = df.copy()
    df["qc_proba_ad"] = proba

    suspects = df[df["qc_proba_ad"] >= thr].copy()
    delta = cfg["BORDER_DELTA"]
    borderline = df[df["qc_proba_ad"].between(0.5 - delta, 0.5 + delta)].copy()

    suspects.to_csv(out_dir / "suspect_nonads.csv", index=False, encoding="utf-8-sig")
    borderline.to_csv(out_dir / "borderline_nonads.csv", index=False, encoding="utf-8-sig")

    summary = {
        "nonads": int(len(df)),
        "suspects": int(len(suspects)),
        "borderline": int(len(borderline)),
        "thr": float(thr),
        "avg_proba": float(df["qc_proba_ad"].mean()),
        "p90_proba": float(df["qc_proba_ad"].quantile(0.9)),
    }
    (out_dir / "qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[QC] nonads={summary['nonads']}  suspects={summary['suspects']}  borderline={summary['borderline']}")
    print(f"[SAVE] {out_dir/'suspect_nonads.csv'}")
    print(f"[SAVE] {out_dir/'borderline_nonads.csv'}")
    print(f"[SAVE] {out_dir/'qc_summary.json'}")

def main():
    mode = CONFIG["MODE"].upper()
    if mode in ("TRAIN", "BOTH"):
        train(CONFIG)
    if mode in ("CHECK", "BOTH"):
        check(CONFIG)

if __name__ == "__main__":
    main()
