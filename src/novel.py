"""
AdShield-X :: novel.py
----------------------
The contributions that take this project past a "compare 13 classifiers"
reproduction.  Each class below is self-contained and evaluated separately in
run_pipeline.py.

  C2  ZeroDayCascade        - supervised head + human-only anomaly head, so bot
                              families that never appeared in training are still
                              caught.
  C3  MimicryAttack /       - a realistic adaptive adversary that only perturbs
      adversarial_augment     features a fraudster can actually control, plus
                              adversarial training as the defence.
  C4  CostOptimalThreshold  - converts probabilities into money: picks the
                              operating point that minimises expected loss for a
                              given CPC and false-block cost.
  C5  DriftMonitor          - ADWIN-style change detector on the streaming
                              score distribution with automatic retraining.
  C6  ReasonCodeExplainer   - per-click, dispute-ready explanations.
  C7  latency_benchmark     - per-click inference latency, the constraint every
                              real ad exchange has and almost no paper reports.
"""

from __future__ import annotations
import time
import numpy as np
from collections import deque
from sklearn.metrics import recall_score, precision_score, f1_score


# ===========================================================================
# C2 :: Zero-day hybrid cascade
# ===========================================================================
class ZeroDayCascade:
    """Supervised detector + a density model of legitimate behaviour.

    Rationale. A supervised model can only recognise fraud that resembles the
    fraud in its labels; a new bot family is by definition out of distribution.
    The second head is a denoising autoencoder fitted on LEGITIMATE clicks only
    (`HumanBehaviourAutoencoder`), which learns the joint structure of a real
    session rather than its marginals. A click is escalated to fraud when

        (a) the supervised head is already confident it is fraud, OR
        (b) the supervised head is NOT confident it is fraud (p < `high`) AND
            the click reconstructs badly under the legitimate-behaviour model.

    Rule (b) is what recovers an unseen family. Its cost is bounded explicitly:
    the novelty gate is calibrated to a fixed false-positive budget measured on
    known-legitimate traffic (`human_fp_budget`), so the extra recall is bought
    at a price the operator chooses in advance rather than discovers later.

    The novelty head is deliberately given the UNREDUCED feature matrix. RFE is
    optimised against known fraud, so it discards precisely the columns an
    unseen family betrays itself on.
    """

    def __init__(self, supervised, high=0.55, human_fp_budget=0.03,
                 ae_epochs=80, ae_hidden=(64, 20), seed=42):
        self.sup = supervised
        self.high = high
        self.budget = human_fp_budget
        self.ae_epochs, self.ae_hidden, self.seed = ae_epochs, ae_hidden, seed
        self.qt_ = None
        self.ae_ = None
        self.novelty_cut_ = None
        self.full_ = False

    def fit(self, X, y, X_full=None):
        from features import RankNormalizer
        from dl import HumanBehaviourAutoencoder
        self.full_ = X_full is not None
        Z = (X_full if self.full_ else X)
        Zh = Z[y == 0]
        # rank-normalise first: the raw features are heavy-tailed lognormals and
        # a plain MSE would be dominated by their outliers instead of by the
        # cross-feature inconsistencies we are looking for
        self.qt_ = RankNormalizer(n_knots=200).fit(Zh)
        self.ae_ = HumanBehaviourAutoencoder(hidden=self.ae_hidden,
                                             epochs=self.ae_epochs, noise=0.15,
                                             lr=3e-3, seed=self.seed)
        self.ae_.fit(self.qt_.transform(Zh))
        e = self.ae_.reconstruction_error(self.qt_.transform(Zh))
        self.novelty_cut_ = float(np.quantile(e, 1.0 - self.budget))
        return self

    def novelty_score(self, X_full):
        return self.ae_.reconstruction_error(self.qt_.transform(X_full))

    def predict_proba(self, X, X_full=None):
        p = self.sup.predict_proba(X)[:, 1]
        Z = X_full if (self.full_ and X_full is not None) else X
        e = self.novelty_score(Z)
        escalate = (e > self.novelty_cut_) & (p < self.high)
        # map the novelty margin into a probability just above the boundary, so
        # ranking, thresholding and cost optimisation all still work downstream
        margin = np.clip((e - self.novelty_cut_) /
                         (abs(self.novelty_cut_) + 1e-9), 0, 1)
        p2 = np.where(escalate, np.maximum(p, 0.55 + 0.4 * margin), p)
        return np.column_stack([1 - p2, p2])

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)

    def bind(self, X_full):
        """Thin view feeding X_full to the novelty head, so the cascade can be
        handed to the ordinary evaluate() helper."""
        outer = self

        class _Bound:
            def predict_proba(self, X):
                return outer.predict_proba(X, X_full)
        return _Bound()


# ===========================================================================
# C3 :: Adversarial mimicry attack + defence
# ===========================================================================
#: features a real fraudster can cheaply forge in the browser/automation layer
CONTROLLABLE = [
    "session_duration", "pages_viewed", "time_on_ad", "mouse_move_events",
    "touch_events", "clicks_in_session",
    "scroll_depth_pct", "keystrokes", "avg_click_interval_ms",
    "click_interval_std_ms", "time_to_first_click_ms", "ua_entropy",
    "js_enabled", "cookie_enabled", "bounce", "browser", "os",
    "device_type", "referrer", "hour_of_day",
]
#: features that cost the fraudster money/throughput to fake -> treated immutable
IMMUTABLE_HINT = [
    "ip_click_velocity_1h", "ip_unique_pub_24h", "ip_burstiness",
    "dev_click_velocity_1h", "dev_unique_ip_24h", "pub_click_velocity_1h",
    "pub_unique_ip_ratio_24h", "pub_ip_concentration", "triad_repeat_count",
    "entity_degree_score", "ip_clicks_1h",
]


class MimicryAttack:
    """Exemplar-replay mimicry: blend each fraudulent click toward a real human.

    A fraudster does not aim at the *average* human -- averaging heavy-tailed
    behavioural features produces a profile no real visitor has. What an
    operator actually does is capture genuine sessions and replay their
    profiles. So for every bot row we draw a random legitimate exemplar and
    interpolate the controllable features toward it:

        x' = (1 - b) * x_bot + b * x_human_exemplar  + small jitter

    `budget` b in [0,1] is how far the operator is willing to go. Higher b means
    a more faithful replay but less throughput and more engineering cost, so it
    is an honest axis for "how sophisticated is this bot".

    Only columns in `controllable` move. This is what makes the evaluation
    meaningful: an attack allowed to edit the ad server's own velocity counters
    is not an attack, it is a database breach.
    """

    def __init__(self, feature_names, controllable=None, seed=42):
        self.names = list(feature_names)
        ctrl = set(controllable or CONTROLLABLE)
        self.idx = [i for i, n in enumerate(self.names) if n in ctrl]
        self.rng = np.random.default_rng(seed)
        self.pool_ = None
        self.spread_ = None

    def fit(self, X, y):
        self.pool_ = X[y == 0]                    # legitimate exemplar pool
        self.spread_ = self.pool_.std(axis=0) + 1e-6
        return self

    def attack(self, X, budget=0.6, noise=0.05):
        if budget <= 0:
            return X.copy()
        Xa = X.copy()
        j = self.idx
        pick = self.rng.integers(0, len(self.pool_), len(Xa))
        target = self.pool_[pick][:, j]
        Xa[:, j] = (1.0 - budget) * Xa[:, j] + budget * target
        Xa[:, j] += noise * self.spread_[j] * self.rng.normal(size=(len(Xa), len(j)))
        return Xa

    def evaluate(self, model, X, y, budgets=(0.0, 0.3, 0.6, 0.9), thr=0.5):
        """Attack Success Rate = fraction of true bots that slip through."""
        rows = []
        Xb, yb = X[y == 1], y[y == 1]
        for bd in budgets:
            Xadv = self.attack(Xb, budget=bd)
            p = _proba(model, Xadv)
            caught = (p >= thr).mean()
            rows.append({"budget": bd, "robust_recall": float(caught),
                         "attack_success_rate": float(1 - caught)})
        return rows


def adversarial_augment(X, y, attacker, budgets=(0.3, 0.6, 0.9), frac=0.6, seed=42):
    """Build an adversarially-augmented training set (the defence)."""
    rng = np.random.default_rng(seed)
    bots = np.where(y == 1)[0]
    parts_X, parts_y = [X], [y]
    for bd in budgets:
        pick = rng.choice(bots, size=int(len(bots) * frac), replace=False)
        parts_X.append(attacker.attack(X[pick], budget=bd))
        parts_y.append(np.ones(len(pick), dtype=int))
    return np.vstack(parts_X), np.concatenate(parts_y)


# ===========================================================================
# C4 :: Cost-optimal (revenue-aware) threshold
# ===========================================================================
class CostOptimalThreshold:
    """Turn a probability into a business decision.

    Expected loss per click at threshold t:
        L(t) = FP(t) * c_fp + FN(t) * c_fn
    where
        c_fn = cpc                     (advertiser pays for a fraudulent click)
        c_fp = cpc * lost_value_mult   (a real customer is blocked: you lose the
                                        click spend *and* the expected margin)
    Reported alongside: money saved per 1,000,000 clicks vs. the naive t=0.5
    that every accuracy-optimising paper implicitly uses.
    """

    def __init__(self, cpc=18.0, lost_value_mult=6.0, currency="INR"):
        self.cpc, self.mult, self.currency = cpc, lost_value_mult, currency
        self.c_fn = cpc
        self.c_fp = cpc * lost_value_mult
        self.threshold_ = 0.5
        self.curve_ = None

    def fit(self, y_true, proba):
        ts = np.linspace(0.01, 0.99, 197)
        n = len(y_true)
        costs = []
        for t in ts:
            pred = (proba >= t).astype(int)
            fp = int(((pred == 1) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            costs.append((fp * self.c_fp + fn * self.c_fn) / n)
        costs = np.array(costs)
        self.curve_ = (ts, costs)
        self.threshold_ = float(ts[int(np.argmin(costs))])
        self.cost_at_opt_ = float(costs.min())
        self.cost_at_half_ = float(costs[np.argmin(np.abs(ts - 0.5))])
        self.saving_per_million_ = (self.cost_at_half_ - self.cost_at_opt_) * 1e6
        return self

    def report(self):
        return {
            "optimal_threshold": round(self.threshold_, 3),
            "cost_per_click_at_0.5": round(self.cost_at_half_, 4),
            "cost_per_click_at_opt": round(self.cost_at_opt_, 4),
            f"saving_per_million_clicks_{self.currency}": round(self.saving_per_million_, 2),
        }


# ===========================================================================
# C5 :: Drift monitor with automatic retraining
# ===========================================================================
class DriftMonitor:
    """ADWIN-style two-window change detector on the streaming error signal.

    Keeps a reference window and a recent window; when the difference in mean
    error exceeds a Hoeffding-style bound, drift is declared. Unlike a fixed
    schedule, this retrains only when the adversary actually changes.
    """

    def __init__(self, ref_size=1500, recent_size=600, delta=0.01, cooldown=900,
                 sensitivity=0.35):
        self.ref = deque(maxlen=ref_size)
        self.recent = deque(maxlen=recent_size)
        self.delta = delta
        self.cooldown = cooldown
        self.sensitivity = sensitivity
        self._since = 0
        self.events_ = []

    def _bound(self):
        n1, n2 = len(self.ref), len(self.recent)
        if n1 < 100 or n2 < 100:
            return np.inf
        m = 1.0 / (1.0 / n1 + 1.0 / n2)
        return np.sqrt(2.0 / m * np.log(4.0 / self.delta)) * self.sensitivity

    def update(self, err, i=None):
        """err: 0/1 (or a loss in [0,1]) for one observation. Returns True on drift."""
        self.recent.append(err)
        self._since += 1
        if len(self.recent) == self.recent.maxlen:
            self.ref.append(self.recent[0])
        if self._since < self.cooldown or len(self.ref) < 100:
            return False
        d = abs(np.mean(self.ref) - np.mean(self.recent))
        if d > self._bound():
            self.events_.append({"index": i, "gap": float(d),
                                 "ref_err": float(np.mean(self.ref)),
                                 "recent_err": float(np.mean(self.recent))})
            self.ref.clear(); self.recent.clear(); self._since = 0
            return True
        return False


# ===========================================================================
# C6 :: Reason codes
# ===========================================================================
class ReasonCodeExplainer:
    """Per-click, human-readable justification for a fraud verdict.

    Advertisers cannot dispute an invalid-click charge with "the model said 0.97".
    This produces the top-k contributing signals with direction and plain-English
    phrasing, which is what an invalid-traffic credit request actually needs.
    """

    PHRASES = {
        "mouse_move_events": ("pointer activity", "almost no pointer movement recorded"),
        "click_interval_std_ms": ("click-timing regularity", "clicks arrive at machine-regular intervals"),
        "avg_click_interval_ms": ("click cadence", "unnaturally fast click cadence"),
        "scroll_depth_pct": ("scroll behaviour", "page never scrolled below the fold"),
        "session_duration": ("session length", "session ended almost immediately"),
        "keystrokes": ("keyboard activity", "no keyboard interaction at all"),
        "ua_entropy": ("user-agent profile", "low-entropy / templated user agent"),
        "js_enabled": ("client capability", "JavaScript disabled or stubbed"),
        "cookie_enabled": ("client capability", "cookies rejected"),
        "time_on_ad": ("dwell time", "near-zero dwell time on the creative"),
        "ip_clicks_1h": ("IP velocity", "abnormal click volume from this IP in one hour"),
        "ip_click_velocity_1h": ("IP velocity", "abnormal click volume from this IP in one hour"),
        "ip_unique_pub_24h": ("IP fan-out", "same IP clicking across many publishers"),
        "ip_burstiness": ("burst pattern", "clicks arrive in tight machine bursts"),
        "dev_click_velocity_1h": ("device velocity", "single device generating excessive clicks"),
        "dev_unique_ip_24h": ("device/IP churn", "one device rotating through many IPs"),
        "pub_click_velocity_1h": ("publisher volume", "publisher traffic spike"),
        "pub_ip_concentration": ("traffic concentration", "publisher traffic concentrated on very few IPs"),
        "pub_unique_ip_ratio_24h": ("audience diversity", "publisher shows implausibly low audience diversity"),
        "triad_repeat_count": ("repeat triad", "identical IP+device+publisher combination repeating"),
        "entity_degree_score": ("network density", "click sits in a dense IP/device cluster"),
        "browser": ("browser", "browser signature associated with automation"),
        "os": ("operating system", "server-class OS behind a consumer ad slot"),
        "bounce": ("engagement", "immediate bounce after the click"),
        "conversion_rate_pub": ("publisher quality", "publisher converts far below network norm"),
    }

    def __init__(self, model, feature_names, background=None, kind="tree"):
        self.model, self.names, self.kind = model, list(feature_names), kind
        self.bg = background
        self._explainer = None

    def _ensure(self):
        if self._explainer is not None:
            return
        import shap
        if self.kind == "tree":
            self._explainer = shap.TreeExplainer(self.model)
        else:
            f = lambda z: _proba(self.model, z)
            self._explainer = shap.KernelExplainer(f, self.bg)

    def shap_values(self, X):
        self._ensure()
        v = self._explainer.shap_values(X)
        v = np.asarray(v)
        if v.ndim == 3:                      # (n, features, classes)
            v = v[..., -1]
        return v

    def reasons(self, X, k=4, ignore=None):
        """ignore: feature names whose values were imputed rather than observed.
        A decision must not be explained with a signal nobody measured, so those
        contributions are dropped before the top-k are chosen."""
        v = self.shap_values(X)
        drop = {self.names.index(n) for n in (ignore or []) if n in self.names}
        out = []
        for i in range(len(X)):
            w = np.abs(v[i]).copy()
            for j in drop:
                w[j] = -1.0
            order = [j for j in np.argsort(-w) if w[j] >= 0][:k]
            items = []
            for j in order:
                n = self.names[j]
                label, phrase = self.PHRASES.get(n, (n.replace("_", " "), f"unusual {n.replace('_',' ')}"))
                items.append({
                    "feature": n, "signal": label,
                    "direction": "raises fraud risk" if v[i][j] > 0 else "lowers fraud risk",
                    "contribution": round(float(v[i][j]), 4),
                    "explanation": phrase,
                })
            out.append(items)
        return out


# ===========================================================================
# C7 :: Latency
# ===========================================================================
def latency_benchmark(model, X, batch_sizes=(1, 32, 256, 1024), repeats=30):
    res = []
    for bs in batch_sizes:
        xb = X[:bs]
        _proba(model, xb)                    # warm-up
        t0 = time.perf_counter()
        for _ in range(repeats):
            _proba(model, xb)
        dt = (time.perf_counter() - t0) / repeats
        res.append({"batch_size": bs,
                    "batch_latency_ms": round(dt * 1e3, 3),
                    "per_click_us": round(dt / bs * 1e6, 2),
                    "throughput_clicks_per_s": int(bs / dt)})
    return res


# ---------------------------------------------------------------------------
def _proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    return model.decision_function(X)
