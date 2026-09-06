"""
AdShield-X :: app/scoring.py
----------------------------
The service layer between the web tier and the trained artefacts.

Loads the model bundle once per process, turns a raw click frame into verdicts
with reason codes, and knows how to assemble a single click typed into a form.
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import data as D                                     # noqa: E402
from features import add_entity_velocity_features, EVGF_COLS   # noqa: E402
from novel import ReasonCodeExplainer                # noqa: E402

MODEL_PATH = os.path.join(ROOT, "outputs", "adshield_model.joblib")
CPC = float(os.environ.get("ADSHIELD_CPC", 18.0))

#: SHAP on a 300-tree forest costs roughly 80 ms per click on a dedicated core
#: and several times that on a shared one, so explaining a whole batch would
#: blow any sensible request budget. We explain the clicks an analyst actually
#: has to justify -- the escalated ones, then the blocked ones, then whatever
#: else is riskiest -- and leave the obviously-clean tail unexplained.
REASON_LIMIT = int(os.environ.get("ADSHIELD_REASON_LIMIT", 60))

_B = None


class ModelMissing(RuntimeError):
    pass


def bundle():
    global _B
    if _B is None:
        if not os.path.exists(MODEL_PATH):
            raise ModelMissing(
                "outputs/adshield_model.joblib not found. Train the model first "
                "with: python run_pipeline.py")
        b = joblib.load(MODEL_PATH)
        b["explainer"] = ReasonCodeExplainer(b["model"], b["feature_names"],
                                             kind="tree")
        _B = b
    return _B


def model_card():
    b = bundle()
    return {
        "features_used": len(b["feature_names"]),
        "feature_names": b["feature_names"],
        "threshold": round(float(b["threshold"]), 3),
        "supervised_model": type(b["model"]).__name__,
        "novelty_head": "denoising autoencoder on legitimate clicks"
                        if b.get("cascade") else "not loaded",
        "cpc": CPC,
        "reason_limit": REASON_LIMIT,
    }


def score_frame(df: pd.DataFrame):
    """Score a raw click frame. Returns (records, summary)."""
    b = bundle()
    # Compute the entity-graph layer only when it is absent. A log that already
    # carries these columns had them computed over its own full stream, and
    # recomputing over whatever subset we happen to be holding would replace
    # real neighbourhood counts with the counts of this batch alone.
    if not any(c in df.columns for c in EVGF_COLS):
        if all(c in df.columns for c in D.ENTITY_COLS) and D.TIME_COL in df.columns:
            df = add_entity_velocity_features(df)
    Xf = b["preprocessor"].transform(df)
    imputed = list(getattr(b["preprocessor"], "last_imputed_", []))
    X = Xf[:, b["rfe_idx"]]
    proba = b["model"].predict_proba(X)[:, 1]
    thr = float(b["threshold"])

    cas = b.get("cascade")
    escalated = np.zeros(len(X), dtype=bool)
    if cas is not None:
        try:
            nov = cas.novelty_score(Xf)
            escalated = (nov > cas.novelty_cut_) & (proba < cas.high)
        except Exception:
            pass
    # The cascade's own escalation floor (0.55) was set against the 0.50
    # threshold the offline experiments score at. This deployment runs at the
    # cost-optimal threshold instead, which is higher, so an escalated click
    # would land below it and quietly pass -- the second head would be doing
    # work nobody acts on. Escalation means "block this", so lift it just over
    # whatever threshold is actually in force. The false-positive cost of doing
    # so is the novelty gate's 3% budget, chosen in advance.
    final = np.where(escalated, np.maximum(proba, thr + 0.02), proba)
    blocked = final >= thr

    # rank by how much the decision needs defending, then explain the top slice
    priority = np.lexsort((-final, ~blocked, ~escalated))
    explain_idx = priority[:min(REASON_LIMIT, len(X))]
    reasons = {}
    if len(explain_idx):
        computed = b["explainer"].reasons(X[explain_idx], k=4, ignore=imputed)
        reasons = {int(j): r for j, r in zip(explain_idx, computed)}

    has_truth = D.LABEL_COL in df.columns
    recs = []
    for i in range(len(X)):
        recs.append({
            "row": i + 1,
            "probability": round(float(final[i]) * 100, 1),
            "supervised": round(float(proba[i]) * 100, 1),
            "escalated": bool(escalated[i]),
            "blocked": bool(blocked[i]),
            "verdict": "Invalid click" if blocked[i] else "Legitimate",
            "truth": str(df[D.LABEL_COL].iloc[i]) if has_truth else None,
            "reasons": reasons.get(i, []),
        })

    correct = None
    if has_truth:
        truth = (df[D.LABEL_COL].astype(str) == "Bot").to_numpy()
        correct = int((blocked == truth).sum())

    summary = {
        "scanned": int(len(X)),
        "blocked": int(blocked.sum()),
        "allowed": int((~blocked).sum()),
        "escalated": int(escalated.sum()),
        "threshold": round(thr, 3),
        "budget_held": round(float(blocked.sum()) * CPC, 2),
        "cpc": CPC,
        "correct": correct,
        "explained": int(len(explain_idx)),
        "imputed_features": imputed,
    }
    return recs, summary


#: A held-out stream, generated once with the same parameters as the training
#: corpus and never used to fit anything. Sampling from it is the only honest way
#: to hand the console a small batch: the entity-graph features count what an
#: address, device or publisher did in the hour before each click, so they only
#: mean anything when they were computed over a full stream. Recomputing them on
#: twenty-five isolated clicks yields zeros, which is not a quiet, small error --
#: it is a batch that looks nothing like training, and the model rightly treats
#: the whole thing as strange.
SAMPLE_POOL = os.path.join(ROOT, "outputs", "sample_pool.csv")
_POOL = None


def sample_frame(n=25, seed=None):
    """Draw n clicks from the held-out pool, labels included."""
    global _POOL
    n = max(5, min(int(n), 500))
    if _POOL is None:
        if not os.path.exists(SAMPLE_POOL):
            raise ModelMissing(
                "outputs/sample_pool.csv not found. Rebuild it with: "
                "python -c \"import sys;sys.path.insert(0,'src');import data as D;"
                "from features import add_entity_velocity_features as f;"
                "f(D.generate_clickstream(n=24000,seed=20260906)).sample(8000,"
                "random_state=1).to_csv('outputs/sample_pool.csv',index=False)\"")
        _POOL = pd.read_csv(SAMPLE_POOL)
    rs = np.random.randint(1 << 30) if seed is None else int(seed)
    return _POOL.sample(n=n, random_state=rs).reset_index(drop=True)


# ---------------------------------------------------------------------------
#: fields the manual-entry form exposes, with sane defaults for a real visitor
MANUAL_FIELDS = [
    ("session_duration",       "Session length (seconds)",            60.0),
    ("pages_viewed",           "Pages viewed",                        4),
    ("clicks_in_session",      "Clicks in the session",               3),
    ("time_on_ad",             "Dwell time on the creative (seconds)", 6.0),
    ("mouse_move_events",      "Pointer-move events",                 180),
    ("touch_events",           "Touch events",                        0),
    ("scroll_depth_pct",       "Scroll depth (%)",                    55.0),
    ("keystrokes",             "Keystrokes",                          4),
    ("avg_click_interval_ms",  "Mean gap between clicks (ms)",        2800.0),
    ("click_interval_std_ms",  "Spread of that gap (ms)",             1400.0),
    ("time_to_first_click_ms", "Time to first click (ms)",            4200.0),
    ("ua_entropy",             "User-agent entropy",                  4.3),
    ("hour_of_day",            "Hour of day (0-23)",                  19),
    ("day_of_week",            "Day of week (0-6)",                   3),
    ("ip_clicks_1h",           "Clicks from this IP in the last hour", 2),
    ("conversion_rate_pub",    "Publisher conversion rate",           0.08),
    ("js_enabled",             "JavaScript enabled (1/0)",            1),
    ("cookie_enabled",         "Cookies accepted (1/0)",              1),
    ("bounce",                 "Bounced immediately (1/0)",           0),
]
MANUAL_CATEGORICAL = [
    ("browser",     "Browser",          D.BROWSERS,  "Chrome"),
    ("os",          "Operating system", D.OSES,      "Windows"),
    ("device_type", "Device",           D.DEVICES,   "desktop"),
    ("country",     "Country",          D.COUNTRIES, "IN"),
    ("referrer",    "Referrer",         D.REFERRERS, "search"),
    ("ad_position", "Ad slot",          D.AD_POS,    "top_banner"),
]


def manual_defaults():
    d = {k: v for k, _, v in MANUAL_FIELDS}
    d.update({k: v for k, _, _, v in MANUAL_CATEGORICAL})
    return d


def frame_from_form(form):
    """Build a one-row frame from the manual-entry form."""
    row = {}
    for k, _, default in MANUAL_FIELDS:
        raw = form.get(k, "")
        try:
            row[k] = float(raw) if raw not in ("", None) else float(default)
        except ValueError:
            row[k] = float(default)
    for k, _, _, default in MANUAL_CATEGORICAL:
        row[k] = form.get(k) or default
    return pd.DataFrame([row])
