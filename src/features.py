"""
AdShield-X :: features.py
-------------------------
Preprocessing + feature engineering.

Contains the FIRST novel contribution:  EVGF - Entity-Velocity Graph Features.

Baseline click-fraud papers score every click in isolation (a row = a click).
But click fraud is a *coordinated* activity: a bot farm reuses a small pool of
IPs/devices and concentrates on a handful of publishers. EVGF projects each
click onto a tripartite graph  IP -- DEVICE -- PUBLISHER  and attaches the local
neighbourhood statistics of that click to the row, so a per-row classifier can
still see collusion structure.  Crucially, all aggregates are computed with a
BACKWARD-ONLY time window, so there is no label leakage and the features can be
maintained in production with O(1) streaming counters.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

from data import CATEGORICAL_COLS, LABEL_COL, ENTITY_COLS, TIME_COL

EVGF_COLS = [
    "ip_click_velocity_1h", "ip_unique_pub_24h", "ip_burstiness",
    "dev_click_velocity_1h", "dev_unique_ip_24h",
    "pub_click_velocity_1h", "pub_unique_ip_ratio_24h",
    "pub_ip_concentration", "triad_repeat_count", "entity_degree_score",
]


# ---------------------------------------------------------------------------
# NOVELTY 1 : Entity-Velocity Graph Features
# ---------------------------------------------------------------------------
class EVGFStream:
    """Streaming, backward-only computation of the entity-velocity features.

    State is a set of per-entity event deques with a sliding time horizon, so
    `observe()` costs O(events in window) rather than O(history) and the same
    code path serves both offline training and online scoring. That equivalence
    matters: it is what stops the training features and the serving features
    from silently diverging, which is the usual way a velocity-feature system
    fails in production.

    Every value returned by `observe()` is computed from events STRICTLY BEFORE
    the current one, so the features cannot leak the click being scored.
    """

    def __init__(self, short_window="1h", long_window="24h"):
        from collections import defaultdict, deque
        self.sw = int(pd.Timedelta(short_window).total_seconds())
        self.lw = int(pd.Timedelta(long_window).total_seconds())
        self._dd, self._dq = defaultdict, deque
        self.ip_hist = defaultdict(deque)
        self.ip_pub = defaultdict(deque)
        self.dev_hist = defaultdict(deque)
        self.dev_ip = defaultdict(deque)
        self.pub_hist = defaultdict(deque)
        self.pub_ip = defaultdict(deque)
        self.triad = defaultdict(deque)

    @staticmethod
    def _evict(dq, now, win, pair=False):
        while dq and (now - (dq[0][0] if pair else dq[0])) > win:
            dq.popleft()

    def observe(self, ts: int, ip, dev, pub) -> dict:
        now, lw, sw = int(ts), self.lw, self.sw
        self._evict(self.ip_hist[ip], now, lw)
        self._evict(self.ip_pub[ip], now, lw, True)
        self._evict(self.dev_hist[dev], now, lw)
        self._evict(self.dev_ip[dev], now, lw, True)
        self._evict(self.pub_hist[pub], now, lw)
        self._evict(self.pub_ip[pub], now, lw, True)
        self._evict(self.triad[(ip, dev, pub)], now, lw)

        ih, dh, ph = self.ip_hist[ip], self.dev_hist[dev], self.pub_hist[pub]
        f = {}
        f["ip_click_velocity_1h"] = float(sum(1 for x in ih if now - x <= sw))
        f["ip_unique_pub_24h"] = float(len({q for _, q in self.ip_pub[ip]}))
        if len(ih) >= 3:
            gaps = np.diff(np.array(ih, dtype=float))
            f["ip_burstiness"] = float(1.0 / (np.median(gaps) + 1.0))
        else:
            f["ip_burstiness"] = 0.0
        f["dev_click_velocity_1h"] = float(sum(1 for x in dh if now - x <= sw))
        f["dev_unique_ip_24h"] = float(len({q for _, q in self.dev_ip[dev]}))
        f["pub_click_velocity_1h"] = float(sum(1 for x in ph if now - x <= sw))
        pub_ips = [q for _, q in self.pub_ip[pub]]
        f["pub_unique_ip_ratio_24h"] = (len(set(pub_ips)) / len(ph)) if ph else 0.0
        if pub_ips:
            cnt = pd.Series(pub_ips).value_counts(normalize=True).to_numpy()
            f["pub_ip_concentration"] = float((cnt ** 2).sum())
        else:
            f["pub_ip_concentration"] = 0.0
        f["triad_repeat_count"] = float(len(self.triad[(ip, dev, pub)]))
        f["entity_degree_score"] = float(np.log1p(
            f["ip_click_velocity_1h"] * f["dev_click_velocity_1h"] + 1))

        self.ip_hist[ip].append(now); self.ip_pub[ip].append((now, pub))
        self.dev_hist[dev].append(now); self.dev_ip[dev].append((now, ip))
        self.pub_hist[pub].append(now); self.pub_ip[pub].append((now, ip))
        self.triad[(ip, dev, pub)].append(now)
        return f


def add_entity_velocity_features(df: pd.DataFrame, short_window="1h",
                                 long_window="24h", stream=None) -> pd.DataFrame:
    """Attach EVGF columns by replaying the frame through an EVGFStream.

    Degrades gracefully: if the entity or timestamp columns are absent (a
    stripped public dataset, for instance) the frame is returned unchanged and
    the rest of the pipeline runs on behavioural features alone.
    """
    if not all(c in df.columns for c in ENTITY_COLS) or TIME_COL not in df.columns:
        return df

    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL])
    d = d.sort_values(TIME_COL).reset_index(drop=False).rename(columns={"index": "_orig"})

    st = stream or EVGFStream(short_window, long_window)
    t = d[TIME_COL].astype("int64").to_numpy() // 10**9
    ip = d["ip_hash"].to_numpy(); dev = d["device_hash"].to_numpy()
    pub = d["publisher_id"].to_numpy()

    acc = {c: np.zeros(len(d)) for c in EVGF_COLS}
    for i in range(len(d)):
        f = st.observe(t[i], ip[i], dev[i], pub[i])
        for c in EVGF_COLS:
            acc[c][i] = f[c]
    for c in EVGF_COLS:
        d[c] = acc[c]

    return d.sort_values("_orig").drop(columns=["_orig"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Standard preprocessing (mirrors the base paper: LabelEncoder + mean-impute)
# ---------------------------------------------------------------------------
#: signals that come from the network/economic layer rather than the client.
#: A fraudster controls the browser, not the ad server's counters.
NETWORK_COLS = ["ip_clicks_1h", "conversion_rate_pub"]


class Preprocessor:
    """feature_set: 'all' | 'behaviour' (client-side signals only) | 'no_evgf'"""

    def __init__(self, use_evgf=True, feature_set="all"):
        self.use_evgf = use_evgf and feature_set == "all"
        self.feature_set = feature_set
        self.encoders, self.means, self.scaler = {}, {}, StandardScaler()
        self.feature_names_, self.label_encoder_ = None, LabelEncoder()
        #: columns the most recent transform() had to impute because the caller
        #: never supplied them. Downstream code uses this to avoid explaining a
        #: decision with a signal it never actually observed.
        self.last_imputed_ = []

    def _feature_frame(self, df):
        drop = set([LABEL_COL, TIME_COL] + ENTITY_COLS + ["block", "bot_family", "bot_generation"])
        cols = [c for c in df.columns if c not in drop]
        if not self.use_evgf:
            cols = [c for c in cols if c not in EVGF_COLS]
        if self.feature_set == "behaviour":
            cols = [c for c in cols if c not in NETWORK_COLS]
        return df[cols]

    def fit_transform(self, df):
        X = self._feature_frame(df).copy()
        for c in X.columns:
            if X[c].dtype == object or str(X[c].dtype) == "str":
                le = LabelEncoder()
                X[c] = le.fit_transform(X[c].astype(str))
                self.encoders[c] = le
            else:
                m = float(np.nanmean(X[c].to_numpy(dtype=float)))
                self.means[c] = m
                X[c] = X[c].astype(float).fillna(m)
        self.feature_names_ = list(X.columns)
        Xs = self.scaler.fit_transform(X.to_numpy(dtype=float))
        y = self.label_encoder_.fit_transform(df[LABEL_COL].astype(str))
        # ensure Bot == 1 (positive class = fraud)
        if list(self.label_encoder_.classes_) == ["Bot", "Human"]:
            y = 1 - y
            self.classes_ = ["Human", "Bot"]
        else:
            self.classes_ = list(self.label_encoder_.classes_)
        return Xs, y

    def transform(self, df):
        X = self._feature_frame(df).copy()
        # A caller may hand us a record that simply does not carry every column
        # -- a single click typed into a form has no entity graph behind it, so
        # none of the EVGF aggregates exist. Filling those with zero would be
        # wrong twice over: zero is a *meaningful* value for a velocity counter,
        # and after scaling it lands several standard deviations from anything
        # in training, which drags the novelty head into escalating every such
        # record. Marking them missing lets the ordinary imputation below put
        # them at the training mean, which is the honest answer to "we do not
        # know what this click's neighbourhood looked like".
        self.last_imputed_ = [c for c in self.feature_names_ if c not in X.columns]
        for c in self.last_imputed_:
            X[c] = (self.encoders[c].classes_[0] if c in self.encoders
                    else np.nan)
        X = X[self.feature_names_]
        for c in X.columns:
            if c in self.encoders:
                le = self.encoders[c]
                known = set(le.classes_)
                vals = X[c].astype(str).map(lambda v: v if v in known else le.classes_[0])
                X[c] = le.transform(vals)
            else:
                X[c] = X[c].astype(float).fillna(self.means.get(c, 0.0))
        return self.scaler.transform(X.to_numpy(dtype=float))

    def transform_y(self, df):
        return (df[LABEL_COL].astype(str) == "Bot").astype(int).to_numpy()


def rfe_select(X, y, feature_names, n_features=16, seed=42):
    """Recursive Feature Elimination, as used in the base paper."""
    est = RandomForestClassifier(n_estimators=60, max_depth=12,
                                 n_jobs=-1, random_state=seed)
    sel = RFE(estimator=est, n_features_to_select=min(n_features, X.shape[1]), step=2)
    sel.fit(X, y)
    idx = np.where(sel.support_)[0]
    return idx, [feature_names[i] for i in idx]


# ---------------------------------------------------------------------------
# Rank normaliser (used by the novelty head)
# ---------------------------------------------------------------------------
class RankNormalizer:
    """Map each feature to its rank within the reference population, then to a
    standard normal.

    Equivalent in spirit to sklearn's QuantileTransformer, but deliberately
    written as a small, explicit knot table: the exported model has to run
    unchanged inside the browser demo, and a 200-knot table with two-sided
    interpolation is something ~20 lines of JavaScript can reproduce bit-for-bit.
    Two-sided interpolation is what keeps tied values (zeros, binary flags) from
    collapsing onto one end of the range.
    """

    EPS = 1e-7

    def __init__(self, n_knots=200):
        self.n_knots = n_knots

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        q = np.linspace(0.0, 1.0, self.n_knots)
        self.grid_ = q
        self.knots_ = np.quantile(X, q, axis=0).T          # (d, n_knots)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        g = self.grid_
        for j in range(X.shape[1]):
            k = self.knots_[j]
            lo = np.interp(X[:, j], k, g)
            hi = -np.interp(-X[:, j], -k[::-1], -g[::-1])
            r = 0.5 * (lo + hi)
            # A value at or beyond the reference extremes is pushed into the
            # tail rather than averaged back toward the body of the
            # distribution. This is what makes "this session typed exactly zero
            # keys" or "this desktop reported touch events" read as an outlier
            # instead of as an ordinary low value.
            r = np.where(X[:, j] <= k[0], 0.0, r)
            r = np.where(X[:, j] >= k[-1], 1.0, r)
            out[:, j] = _ppf(np.clip(r, self.EPS, 1 - self.EPS))
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def _ppf(p):
    """Acklam's rational approximation to the inverse normal CDF.

    Used instead of scipy so that the Python model and the JavaScript demo
    compute identical values.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)

    q = np.sqrt(-2 * np.log(np.where(lo, p, 0.5)))
    out = np.where(lo, (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1), out)
    q = np.sqrt(-2 * np.log(np.where(hi, 1 - p, 0.5)))
    out = np.where(hi, -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1), out)
    q = np.where(mid, p, 0.5) - 0.5
    r = q * q
    out = np.where(mid, (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1), out)
    return out
