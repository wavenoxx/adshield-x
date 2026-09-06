"""
AdShield-X :: run_pipeline.py
-----------------------------
End-to-end experimental driver.

    python run_pipeline.py                       # everything, synthetic data
    python run_pipeline.py --csv data/click.csv  # everything, real dataset
    python run_pipeline.py --quick               # small/fast smoke run
    python run_pipeline.py --stage 1             # run one stage at a time
    python run_pipeline.py --stage 2             #   (resumes from checkpoint)
    python run_pipeline.py --stage 3

Stages
  1  data -> EVGF -> preprocess -> RFE -> 10 ML + 4 DL baselines -> EVGF ablation
  2  zero-day cascade (C2), adversarial mimicry + hardening (C3)
  3  cost-optimal threshold (C4), drift + auto-retrain (C5),
     reason codes (C6), latency (C7), artefact export
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

import data as D
from features import Preprocessor, add_entity_velocity_features, rfe_select, EVGF_COLS
from models import build_ml_models, evaluate, results_table
from dl import DNNClassifier, CNN1DClassifier, RNNClassifier, CNNAttentionClassifier
from novel import (ZeroDayCascade, MimicryAttack, adversarial_augment,
                   CostOptimalThreshold, DriftMonitor, ReasonCodeExplainer,
                   latency_benchmark)

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, "_state.joblib")
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _save(state):
    joblib.dump(state, CKPT, compress=3)


def _load():
    return joblib.load(CKPT)


# ===========================================================================
# STAGE 1 - baselines
# ===========================================================================
def stage1(args):
    np.random.seed(SEED)
    report = {}

    if args.csv:
        log(f"Loading real dataset: {args.csv}")
        df, _ = D.load_real_csv(args.csv)
    else:
        n = 12_000 if args.quick else args.n
        log(f"Generating synthetic clickstream (n={n:,})")
        df = D.generate_clickstream(n=n, fraud_ratio=args.fraud_ratio, seed=SEED)
    log(f"Raw shape {df.shape} | {df[D.LABEL_COL].value_counts().to_dict()}")
    report["dataset"] = {
        "rows": int(len(df)), "raw_columns": int(df.shape[1]),
        "source": args.csv or "behavioural simulator (4 bot families, 2% label noise)",
        "class_counts": {k: int(v) for k, v in df[D.LABEL_COL].value_counts().items()},
    }
    if "bot_family" in df.columns:
        report["dataset"]["family_counts"] = {
            k: int(v) for k, v in df["bot_family"].value_counts().items()}

    # -- C1: entity-velocity graph features ---------------------------------
    log("Contribution 1: entity-velocity graph features (EVGF)")
    t0 = time.perf_counter()
    df = add_entity_velocity_features(df)
    ev_s = time.perf_counter() - t0
    log(f"  +{len(EVGF_COLS)} features in {ev_s:.1f}s "
        f"({len(df)/max(ev_s,1e-9):,.0f} clicks/s single-threaded)")
    report["evgf_build_rate_clicks_per_s"] = int(len(df) / max(ev_s, 1e-9))

    pre = Preprocessor(use_evgf=True)
    X, y = pre.fit_transform(df)
    log(f"Feature matrix {X.shape}; fraud rate {y.mean():.3f}")

    log("RFE feature selection")
    idx, sel_names = rfe_select(X, y, pre.feature_names_,
                                n_features=min(args.rfe_k, X.shape[1]), seed=SEED)
    log(f"  kept {len(sel_names)}/{X.shape[1]}")
    report["rfe_selected_features"] = sel_names
    report["all_features"] = pre.feature_names_

    Xs = X[:, idx]
    X_tr, X_te, y_tr, y_te, Xf_tr, Xf_te = train_test_split(
        Xs, y, X, test_size=0.2, stratify=y, random_state=SEED)
    log(f"Split: train {X_tr.shape}, test {X_te.shape}")

    rows = []
    for name, m in build_ml_models(SEED).items():
        t0 = time.perf_counter()
        if name == "SVM (RBF)" and len(X_tr) > 10000:
            sub = np.random.RandomState(SEED).choice(len(X_tr), 10000, False)
            m.fit(X_tr[sub], y_tr[sub])          # kernel SVM is O(n^2)-O(n^3)
        else:
            m.fit(X_tr, y_tr)
        tt = time.perf_counter() - t0
        r = evaluate(name, m, X_te, y_te, train_time=tt)
        rows.append(r)
        log(f"  {name:<20} acc={r['Accuracy']:.2f} f1={r['F1-Score']:.2f} "
            f"auc={r['ROC-AUC']:.2f} ({tt:.1f}s)")

    ep = 8 if args.quick else args.epochs
    dl_specs = [
        ("DNN", DNNClassifier(epochs=ep, lr=2e-3, seed=SEED)),
        ("CNN", CNN1DClassifier(epochs=ep, lr=2e-3, seed=SEED)),
        ("RNN", RNNClassifier(epochs=max(4, ep // 2), lr=3e-3, seed=SEED)),
        ("CNN+Attention (Extension)", CNNAttentionClassifier(epochs=ep, lr=2e-3, seed=SEED)),
    ]
    ext = None
    for name, mdl in dl_specs:
        t0 = time.perf_counter()
        mdl.fit(X_tr, y_tr)
        tt = time.perf_counter() - t0
        r = evaluate(name, mdl, X_te, y_te, train_time=tt)
        rows.append(r)
        if name.startswith("CNN+Attention"):
            ext = mdl
        log(f"  {name:<26} acc={r['Accuracy']:.2f} f1={r['F1-Score']:.2f} "
            f"auc={r['ROC-AUC']:.2f} ({tt:.1f}s)")

    table = results_table(rows)
    table.to_csv(os.path.join(OUT, "table1_model_comparison.csv"), index=False)
    report["table1_model_comparison"] = table.to_dict(orient="records")
    log(f"Best by F1: {table.iloc[0]['Algorithm']} ({table.iloc[0]['F1-Score']:.2f})")

    ops = RandomForestClassifier(n_estimators=300, max_depth=18, n_jobs=-1,
                                 random_state=SEED).fit(X_tr, y_tr)

    # -- ablation: does EVGF earn its place? --------------------------------
    log("Ablation: with vs. without EVGF")
    abl, views = [], {}
    for tag, fs in [("behaviour only", "behaviour"),
                    ("behaviour + network", "no_evgf"),
                    ("full (+EVGF)", "all")]:
        pv = Preprocessor(use_evgf=(fs == "all"), feature_set=fs)
        Xv, yv = pv.fit_transform(df)
        iv, nv = rfe_select(Xv, yv, pv.feature_names_,
                            n_features=min(args.rfe_k, Xv.shape[1]), seed=SEED)
        Xv = Xv[:, iv]
        a_tr, a_te, b_tr, b_te = train_test_split(Xv, yv, test_size=0.2,
                                                  stratify=yv, random_state=SEED)
        mm = RandomForestClassifier(n_estimators=300, max_depth=18, n_jobs=-1,
                                    random_state=SEED).fit(a_tr, b_tr)
        r = evaluate(f"Random Forest ({tag})", mm, a_te, b_te)
        abl.append(r)
        views[fs] = {"names": nv, "X_tr": a_tr, "X_te": a_te,
                     "y_tr": b_tr, "y_te": b_te, "model": mm}
        log(f"  {tag:<20} acc={r['Accuracy']:.2f} f1={r['F1-Score']:.2f} "
            f"recall={r['Recall']:.2f} FPR={r['FPR']:.2f}")
    report["table2_evgf_ablation"] = results_table(abl).to_dict(orient="records")
    pd.DataFrame(abl).to_csv(os.path.join(OUT, "table2_evgf_ablation.csv"), index=False)

    _save({"args": vars(args), "df": df, "pre": pre, "idx": idx,
           "sel_names": sel_names, "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr,
           "y_te": y_te, "Xf_tr": Xf_tr, "Xf_te": Xf_te, "ops": ops, "ext": ext,
           "views": views, "report": report})
    log("Stage 1 complete -> checkpoint saved.")
    return report


# ===========================================================================
# STAGE 2 - zero-day cascade + adversarial robustness
# ===========================================================================
def stage2(args):
    st = _load()
    report = st["report"]
    df, pre, idx = st["df"], st["pre"], st["idx"]
    X_tr, X_te, y_tr, y_te = st["X_tr"], st["X_te"], st["y_tr"], st["y_te"]
    Xf_tr, Xf_te = st["Xf_tr"], st["Xf_te"]
    ops, sel_names, views = st["ops"], st["sel_names"], st["views"]

    # -- C2 : zero-day cascade ----------------------------------------------
    log("Contribution 2: zero-day cascade against an unseen bot family")
    zd = D.generate_clickstream(n=6000 if args.quick else 10000,
                                fraud_ratio=args.fraud_ratio,
                                family_mix=D.ZERO_DAY_MIX, seed=777)
    zd = add_entity_velocity_features(zd)
    Xz_full = pre.transform(zd)
    Xz, yz = Xz_full[:, idx], pre.transform_y(zd)

    cas = ZeroDayCascade(ops, ae_epochs=30 if args.quick else 80,
                         human_fp_budget=0.03, seed=SEED)
    cas.fit(X_tr, y_tr, X_full=Xf_tr)
    zd_rows = [
        evaluate("Supervised only - in-distribution", ops, X_te, y_te),
        evaluate("Supervised only - ZERO-DAY family", ops, Xz, yz),
        evaluate("AdShield-X cascade - ZERO-DAY family", cas.bind(Xz_full), Xz, yz),
        evaluate("AdShield-X cascade - in-distribution", cas.bind(Xf_te), X_te, y_te),
    ]
    for r in zd_rows:
        log(f"  {r['Algorithm']:<42} recall={r['Recall']:.2f} "
            f"prec={r['Precision']:.2f} f1={r['F1-Score']:.2f}")
    report["table3_zero_day"] = results_table(zd_rows).to_dict(orient="records")
    report["cascade_config"] = {"human_fp_budget": cas.budget,
                                "supervised_gate_high": cas.high}
    pd.DataFrame(zd_rows).to_csv(os.path.join(OUT, "table3_zero_day.csv"), index=False)

    # -- C3 : adversarial mimicry -------------------------------------------
    log("Contribution 3: adversarial mimicry attack, three defences")
    vb = views["behaviour"]
    atk_b = MimicryAttack(vb["names"], seed=SEED).fit(vb["X_tr"], vb["y_tr"])
    r_beh = atk_b.evaluate(vb["model"], vb["X_te"], vb["y_te"])

    atk = MimicryAttack(sel_names, seed=SEED).fit(X_tr, y_tr)
    r_evgf = atk.evaluate(ops, X_te, y_te)

    Xa, ya = adversarial_augment(X_tr, y_tr, atk, seed=SEED)
    ops_adv = RandomForestClassifier(n_estimators=300, max_depth=18, n_jobs=-1,
                                     random_state=SEED).fit(Xa, ya)
    r_hard = atk.evaluate(ops_adv, X_te, y_te)

    adv = []
    for a, b, c in zip(r_beh, r_evgf, r_hard):
        adv.append({"attack_budget": a["budget"],
                    "ASR_behaviour_only_%": round(a["attack_success_rate"] * 100, 2),
                    "ASR_plus_EVGF_%": round(b["attack_success_rate"] * 100, 2),
                    "ASR_plus_EVGF_advtrain_%": round(c["attack_success_rate"] * 100, 2)})
        log(f"  budget={a['budget']:.1f}  behaviour-only "
            f"{adv[-1]['ASR_behaviour_only_%']:5.2f}%  +EVGF "
            f"{adv[-1]['ASR_plus_EVGF_%']:5.2f}%  +adv-train "
            f"{adv[-1]['ASR_plus_EVGF_advtrain_%']:5.2f}%")
    cb = evaluate("RF +EVGF", ops, X_te, y_te)
    ca = evaluate("RF +EVGF +adv", ops_adv, X_te, y_te)
    report["table4_adversarial"] = adv
    report["adversarial_clean_cost"] = {
        "clean_F1_plus_EVGF": round(cb["F1-Score"], 2),
        "clean_F1_plus_EVGF_advtrain": round(ca["F1-Score"], 2),
        "clean_FPR_plus_EVGF": round(cb["FPR"], 2),
        "clean_FPR_plus_EVGF_advtrain": round(ca["FPR"], 2)}
    pd.DataFrame(adv).to_csv(os.path.join(OUT, "table4_adversarial.csv"), index=False)

    st["report"] = report
    st["cascade"] = cas
    st["ops_adv"] = ops_adv
    _save(st)
    log("Stage 2 complete -> checkpoint saved.")
    return report


# ===========================================================================
# STAGE 3 - cost, drift, explanations, latency, export
# ===========================================================================
def stage3(args):
    st = _load()
    report = st["report"]
    df, pre, idx = st["df"], st["pre"], st["idx"]
    X_tr, X_te, y_tr, y_te = st["X_tr"], st["X_te"], st["y_tr"], st["y_te"]
    ops, ext, sel_names = st["ops"], st["ext"], st["sel_names"]
    cas = st.get("cascade")

    # -- C4 : revenue-aware operating point ---------------------------------
    log("Contribution 4: revenue-aware operating point")
    proba = ops.predict_proba(X_te)[:, 1]
    cost_rows, curves = [], {}
    for cpc, mult in [(18.0, 6.0), (18.0, 3.0), (45.0, 6.0)]:
        co = CostOptimalThreshold(cpc=cpc, lost_value_mult=mult).fit(y_te, proba)
        rep = co.report(); rep.update({"CPC_INR": cpc, "false_block_multiplier": mult})
        cost_rows.append(rep)
        curves[f"cpc{cpc}_m{mult}"] = {"t": co.curve_[0].tolist(),
                                       "cost": co.curve_[1].tolist()}
        log(f"  CPC=Rs{cpc}, x{mult}: t*={rep['optimal_threshold']} "
            f"saves Rs{rep['saving_per_million_clicks_INR']:,.0f}/1M clicks")
    report["table5_cost"] = cost_rows
    report["cost_curves"] = curves
    pd.DataFrame(cost_rows).to_csv(os.path.join(OUT, "table5_cost.csv"), index=False)

    # -- C5 : drift ----------------------------------------------------------
    log("Contribution 5: drifting stream with automatic retraining")
    stream = D.make_drift_stream(n_per_block=2500 if args.quick else 5000, seed=7)
    stream = add_entity_velocity_features(stream)
    Xst = pre.transform(stream)[:, idx]
    yst = pre.transform_y(stream)

    def run_stream(adaptive, micro=250):
        mdl = RandomForestClassifier(n_estimators=150, max_depth=18, n_jobs=-1,
                                     random_state=SEED).fit(X_tr, y_tr)
        mon, errs, retr, i = DriftMonitor(), np.zeros(len(Xst), int), [], 0
        while i < len(Xst):
            j = min(i + micro, len(Xst))
            errs[i:j] = (mdl.predict(Xst[i:j]) != yst[i:j]).astype(int)
            fired = False
            if adaptive:
                for t in range(i, j):
                    if mon.update(int(errs[t]), t):
                        fired = True
            if fired:
                retr.append(j)
                lo = max(0, j - 4000)
                mdl = RandomForestClassifier(
                    n_estimators=150, max_depth=18, n_jobs=-1, random_state=SEED
                ).fit(np.vstack([X_tr, Xst[lo:j]]),
                      np.concatenate([y_tr, yst[lo:j]]))
            i = j
        return errs, retr

    e_static, _ = run_stream(False)
    e_adapt, retr = run_stream(True)
    bs = len(Xst) // 4
    drift = []
    for b in range(4):
        sl = slice(b * bs, (b + 1) * bs)
        drift.append({"block": b,
                      "accuracy_static_%": round((1 - e_static[sl].mean()) * 100, 2),
                      "accuracy_adaptive_%": round((1 - e_adapt[sl].mean()) * 100, 2)})
        log(f"  block {b}: static {drift[-1]['accuracy_static_%']}% "
            f"adaptive {drift[-1]['accuracy_adaptive_%']}%")
    report["table6_drift"] = drift
    report["drift_retrain_points"] = retr
    report["drift_curve"] = {
        "static": pd.Series(e_static).rolling(400, min_periods=50).mean()
                    .bfill().round(4).tolist()[::25],
        "adaptive": pd.Series(e_adapt).rolling(400, min_periods=50).mean()
                    .bfill().round(4).tolist()[::25]}
    pd.DataFrame(drift).to_csv(os.path.join(OUT, "table6_drift.csv"), index=False)

    # -- C6 : reason codes ---------------------------------------------------
    log("Contribution 6: SHAP reason codes")
    expl = ReasonCodeExplainer(ops, sel_names, kind="tree")
    picks = list(np.where((proba >= 0.9) & (y_te == 1))[0][:3]) + \
            list(np.where((proba <= 0.05) & (y_te == 0))[0][:1])
    samples = []
    for i in picks:
        samples.append({"fraud_probability": round(float(proba[i]), 4),
                        "true_label": "Bot" if y_te[i] == 1 else "Human",
                        "reasons": expl.reasons(X_te[i:i + 1], k=4)[0]})
        log(f"  p={samples[-1]['fraud_probability']}: " +
            "; ".join(r["explanation"] for r in samples[-1]["reasons"][:3]))
    report["reason_code_examples"] = samples

    # -- C7 : latency --------------------------------------------------------
    log("Contribution 7: inference latency")
    lat = {"RandomForest (+EVGF)": latency_benchmark(ops, X_te),
           "CNN+Attention": latency_benchmark(ext, X_te)}
    report["table7_latency"] = lat
    for k, v in lat.items():
        log(f"  {k}: {v[0]['batch_latency_ms']} ms single, "
            f"{v[-1]['throughput_clicks_per_s']:,}/s at batch 1024")

    # -- export --------------------------------------------------------------
    joblib.dump({"preprocessor": pre, "rfe_idx": idx, "feature_names": sel_names,
                 "model": ops, "cascade": cas,
                 "threshold": cost_rows[0]["optimal_threshold"]},
                os.path.join(OUT, "adshield_model.joblib"), compress=3)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    keep = [c for c in df.columns if c != "bot_family"]
    df[keep].head(500).to_csv(os.path.join(OUT, "sample_test_clicks.csv"), index=False)
    log(f"All artefacts written to {OUT}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--fraud-ratio", type=float, default=0.35)
    ap.add_argument("--rfe-k", type=int, default=18)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", type=int, default=0)
    a = ap.parse_args()
    if a.stage in (0, 1):
        stage1(a)
    if a.stage in (0, 2):
        stage2(a)
    if a.stage in (0, 3):
        stage3(a)
