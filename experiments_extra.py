"""
AdShield-X :: experiments_extra.py
----------------------------------
Two additions to the main study.

  LOFO   Leave-one-family-out. The zero-day result in run_pipeline.py holds out a
         single bot family. That answers "can the cascade catch THIS unseen
         family", which is weaker than the claim we want to make. Here every
         family takes its turn as the unknown: for each family f we train on
         legitimate clicks plus the other three families, then measure recall on
         f alone. Feature selection is refitted inside each fold, so the held-out
         family contributes nothing at any stage.

  CURVES ROC curves and confusion matrices for the strongest models, which the
         accuracy table cannot show.

Usage
    python experiments_extra.py --fold 0     # one family at a time (each ~2 min)
    python experiments_extra.py --fold 1
    python experiments_extra.py --fold 2
    python experiments_extra.py --fold 3
    python experiments_extra.py --curves     # ROC + confusion, needs stage-1 ckpt
    python experiments_extra.py --collect    # merge folds into results_extra.json
"""

from __future__ import annotations
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

import data as D
from features import Preprocessor, add_entity_velocity_features, rfe_select
from models import evaluate
from novel import ZeroDayCascade

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
SEED = 42
FAMILIES = ["crude", "jittered", "mimicry", "proxy_farm"]
PRETTY = {"crude": "Crude headless automation",
          "jittered": "Jittered automation",
          "mimicry": "Behavioural mimicry",
          "proxy_farm": "Residential proxy farm"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_corpus(n=20000, seed=SEED):
    """One corpus containing all four bot families in equal proportion."""
    mix = {f: 0.25 for f in FAMILIES}
    df = D.generate_clickstream(n=n, fraud_ratio=0.40, family_mix=mix, seed=seed)
    return add_entity_velocity_features(df)


# ---------------------------------------------------------------------------
def run_fold(k, args):
    held = FAMILIES[k]
    log(f"Fold {k}: holding out '{held}'")
    df = build_corpus(n=args.n, seed=SEED)

    is_held = (df["bot_family"] == held).to_numpy()
    log(f"  corpus {len(df):,} clicks, {int(is_held.sum()):,} of the held-out family")

    # ---- training pool: everything except the held-out family --------------
    pool = df.loc[~is_held].reset_index(drop=True)
    pre = Preprocessor(use_evgf=True)
    Xp, yp = pre.fit_transform(pool)
    idx, sel = rfe_select(Xp, yp, pre.feature_names_,
                          n_features=min(18, Xp.shape[1]), seed=SEED)
    Xs = Xp[:, idx]

    X_tr, X_te, y_tr, y_te, Xf_tr, Xf_te = train_test_split(
        Xs, yp, Xp, test_size=0.2, stratify=yp, random_state=SEED)

    ops = RandomForestClassifier(n_estimators=250, max_depth=18, n_jobs=-1,
                                 random_state=SEED).fit(X_tr, y_tr)
    cas = ZeroDayCascade(ops, ae_epochs=args.ae_epochs, human_fp_budget=0.03,
                         seed=SEED).fit(X_tr, y_tr, X_full=Xf_tr)

    r_in_sup = evaluate("in-dist supervised", ops, X_te, y_te)
    r_in_cas = evaluate("in-dist cascade", cas.bind(Xf_te), X_te, y_te)

    # ---- unknown set: held-out family bots + fresh legitimate clicks -------
    unk = pd.concat([df.loc[is_held],
                     df.loc[df["bot_family"] == "human"].sample(
                         n=min(6000, int((df["bot_family"] == "human").sum())),
                         random_state=SEED)],
                    ignore_index=True)
    Xu_full = pre.transform(unk)
    Xu, yu = Xu_full[:, idx], pre.transform_y(unk)

    r_zd_sup = evaluate("zero-day supervised", ops, Xu, yu)
    r_zd_cas = evaluate("zero-day cascade", cas.bind(Xu_full), Xu, yu)

    out = {
        "family": held, "family_label": PRETTY[held],
        "held_out_clicks": int(is_held.sum()),
        "supervised_recall": round(r_zd_sup["Recall"], 2),
        "cascade_recall": round(r_zd_cas["Recall"], 2),
        "supervised_precision": round(r_zd_sup["Precision"], 2),
        "cascade_precision": round(r_zd_cas["Precision"], 2),
        "supervised_f1": round(r_zd_sup["F1-Score"], 2),
        "cascade_f1": round(r_zd_cas["F1-Score"], 2),
        "in_dist_f1_supervised": round(r_in_sup["F1-Score"], 2),
        "in_dist_f1_cascade": round(r_in_cas["F1-Score"], 2),
        "in_dist_fpr_supervised": round(r_in_sup["FPR"], 2),
        "in_dist_fpr_cascade": round(r_in_cas["FPR"], 2),
    }
    log(f"  supervised recall {out['supervised_recall']:.2f}%  ->  "
        f"cascade {out['cascade_recall']:.2f}%  "
        f"(in-dist F1 {out['in_dist_f1_supervised']:.2f} -> {out['in_dist_f1_cascade']:.2f})")
    json.dump(out, open(os.path.join(OUT, f"_lofo_{k}.json"), "w"), indent=2)
    return out


# ---------------------------------------------------------------------------
def run_curves(args):
    """ROC curves and confusion matrices from the stage-1 checkpoint."""
    st = joblib.load(os.path.join(OUT, "_state.joblib"))
    X_tr, X_te, y_tr, y_te = st["X_tr"], st["X_te"], st["y_tr"], st["y_te"]
    ops = st["ops"]

    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    models = {
        "XGBoost": XGBClassifier(n_estimators=350, learning_rate=0.08, max_depth=6,
                                 subsample=0.9, colsample_bytree=0.9,
                                 eval_metric="logloss", random_state=SEED,
                                 n_jobs=-1, tree_method="hist").fit(X_tr, y_tr),
        "Random Forest": ops,
        "Logistic Regression": LogisticRegression(max_iter=1500).fit(X_tr, y_tr),
        "CNN+Attention": st["ext"],
    }
    curves, cms = {}, {}
    for name, m in models.items():
        p = m.predict_proba(X_te)
        p = p[:, 1] if p.ndim == 2 else p
        fpr, tpr, _ = roc_curve(y_te, p)
        step = max(1, len(fpr) // 120)
        curves[name] = {"fpr": [round(float(v), 5) for v in fpr[::step]] + [1.0],
                        "tpr": [round(float(v), 5) for v in tpr[::step]] + [1.0],
                        "auc": round(float(roc_auc_score(y_te, p)) * 100, 2)}
        thr = 0.63 if name != "CNN+Attention" else 0.5
        cm = confusion_matrix(y_te, (p >= thr).astype(int), labels=[0, 1])
        cms[name] = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                     "fn": int(cm[1, 0]), "tp": int(cm[1, 1]), "threshold": thr}
        log(f"  {name}: AUC {curves[name]['auc']}  "
            f"TP {cms[name]['tp']} FP {cms[name]['fp']} "
            f"FN {cms[name]['fn']} TN {cms[name]['tn']}")
    json.dump({"roc": curves, "confusion": cms},
              open(os.path.join(OUT, "_curves.json"), "w"), indent=2)
    log("Curves written.")


def collect():
    lofo = []
    for k in range(len(FAMILIES)):
        p = os.path.join(OUT, f"_lofo_{k}.json")
        if os.path.exists(p):
            lofo.append(json.load(open(p)))
    extra = {"table8_lofo": lofo}
    cp = os.path.join(OUT, "_curves.json")
    if os.path.exists(cp):
        extra.update(json.load(open(cp)))
    json.dump(extra, open(os.path.join(OUT, "results_extra.json"), "w"), indent=2)
    if lofo:
        sr = np.mean([r["supervised_recall"] for r in lofo])
        cr = np.mean([r["cascade_recall"] for r in lofo])
        log(f"LOFO mean recall: supervised {sr:.2f}%  cascade {cr:.2f}%  "
            f"(+{cr-sr:.2f} points across {len(lofo)} families)")
    return extra


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ae-epochs", type=int, default=60)
    a = ap.parse_args()
    if a.fold is not None:
        run_fold(a.fold, a)
    if a.curves:
        run_curves(a)
    if a.collect:
        collect()
