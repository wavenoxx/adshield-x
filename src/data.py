"""
AdShield-X :: data.py
---------------------
Data layer for ad click-fraud detection.

Two modes:
  1. REAL   -> load_real_csv(path)  : auto-adapts to the Kaggle
              'fraud-detection-dataset' (ziya07) or any CSV carrying a
              Human/Bot (or 0/1) label column.
  2. SYNTH  -> generate_clickstream(): a behaviourally-grounded simulator.

Design notes for the simulator (this is what makes the benchmark non-trivial):

  * FOUR BOT FAMILIES, not one. Real invalid traffic is a mixture of crude
    headless scripts, jittered automation, human-mimicking bots and
    residential-proxy click farms. Each family hides in a different place.
  * GENUINE CLASS OVERLAP. Mobile and tablet humans generate almost no
    pointer-move events and short sessions -- exactly the signature naive
    detectors use to flag bots. This is the main source of false positives in
    production and it is reproduced here on purpose.
  * LABEL NOISE. Ground-truth click labels come from delayed conversion
    signals and are themselves ~2% wrong.

Because of these three properties the baseline models land in a realistic
band instead of a saturated 100%, and the operational contributions
(cost threshold, cascade, drift) have something to actually improve.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

BROWSERS = ["Chrome", "Firefox", "Safari", "Edge", "Opera", "HeadlessChrome", "Other"]
OSES = ["Windows", "Android", "iOS", "macOS", "Linux", "Other"]
DEVICES = ["desktop", "mobile", "tablet", "smart_tv"]
REFERRERS = ["search", "social", "direct", "display_network", "email"]
AD_POS = ["top_banner", "sidebar", "in_feed", "footer", "interstitial"]
COUNTRIES = ["IN", "US", "GB", "DE", "BR", "NG", "VN", "RU", "ID", "PH"]

CATEGORICAL_COLS = ["browser", "os", "device_type", "country", "referrer", "ad_position"]
LABEL_COL = "Clicked"          # 'Human' = legitimate, 'Bot' = fraudulent
ENTITY_COLS = ["ip_hash", "device_hash", "publisher_id"]
TIME_COL = "click_ts"

#: bot family mixture seen in ordinary traffic
DEFAULT_MIX = {"crude": 0.30, "jittered": 0.38, "mimicry": 0.32}
#: the family held out of training for the zero-day experiment
ZERO_DAY_MIX = {"proxy_farm": 1.0}


def _lognorm(rng, mean, sigma, n, lo=None, hi=None):
    x = rng.lognormal(mean=mean, sigma=sigma, size=n)
    if lo is not None:
        x = np.maximum(x, lo)
    if hi is not None:
        x = np.minimum(x, hi)
    return x


def _human_hours(rng, n):
    w = np.array([2, 1.2, .7, .5, .5, .8, 1.6, 3.0, 4.2, 4.6, 4.8, 5.0,
                  5.2, 5.0, 4.8, 4.9, 5.3, 6.0, 6.8, 7.4, 7.2, 6.2, 4.4, 3.0])
    return rng.choice(np.arange(24), size=n, p=w / w.sum())


# ---------------------------------------------------------------------------
# Humans
# ---------------------------------------------------------------------------
def _humans(rng, n):
    dev = rng.choice(DEVICES, n, p=[.34, .52, .12, .02])
    is_touch = np.isin(dev, ["mobile", "tablet"])

    h = {}
    h["session_duration"] = _lognorm(rng, 3.9, 1.15, n, 2, 5400)
    h["pages_viewed"] = np.clip(rng.poisson(3.4, n) + 1, 1, 60)
    h["clicks_in_session"] = np.clip(rng.poisson(2.9, n) + 1, 1, 40)
    h["time_on_ad"] = _lognorm(rng, 1.55, 1.05, n, 0.2, 400)

    # KEY OVERLAP: touch users barely generate pointer-move events
    mouse = rng.negative_binomial(5, 0.035, n).astype(float)
    mouse[is_touch] = rng.negative_binomial(1, 0.45, int(is_touch.sum()))
    h["mouse_move_events"] = np.clip(mouse, 0, 4000)
    touch = rng.negative_binomial(4, 0.10, n).astype(float)
    touch[~is_touch] = 0
    h["touch_events"] = np.clip(touch, 0, 900)

    h["scroll_depth_pct"] = np.clip(rng.beta(2.0, 2.0, n) * 100, 0, 100)
    h["keystrokes"] = np.clip(rng.poisson(4.0, n), 0, 300)
    h["avg_click_interval_ms"] = _lognorm(rng, 7.7, 1.15, n, 150, 400_000)
    h["click_interval_std_ms"] = h["avg_click_interval_ms"] * rng.uniform(0.18, 1.20, n)
    h["time_to_first_click_ms"] = _lognorm(rng, 7.6, 1.20, n, 200, 300_000)
    h["ua_entropy"] = rng.normal(4.25, 0.60, n).clip(1.5, 6.5)
    h["hour_of_day"] = _human_hours(rng, n)
    h["day_of_week"] = rng.integers(0, 7, n)
    h["ip_clicks_1h"] = np.clip(rng.poisson(2.2, n) + 1, 1, 60)
    h["conversion_rate_pub"] = rng.beta(2.2, 24, n)
    h["js_enabled"] = (rng.random(n) > 0.03).astype(int)
    h["cookie_enabled"] = (rng.random(n) > 0.09).astype(int)
    h["bounce"] = (rng.random(n) < 0.40).astype(int)
    h["browser"] = rng.choice(BROWSERS, n, p=[.50, .11, .17, .13, .04, .0, .05])
    h["os"] = rng.choice(OSES, n, p=[.31, .34, .20, .07, .04, .04])
    h["device_type"] = dev
    h["country"] = rng.choice(COUNTRIES, n, p=[.24, .21, .09, .08, .09, .04, .05, .05, .08, .07])
    h["referrer"] = rng.choice(REFERRERS, n, p=[.36, .24, .19, .15, .06])
    h["ad_position"] = rng.choice(AD_POS, n, p=[.30, .22, .30, .12, .06])
    return h


# ---------------------------------------------------------------------------
# Bot families
# ---------------------------------------------------------------------------
def _bots(rng, n, family):
    """Each family hides in a different place.

    crude       - obvious headless automation
    jittered    - randomised timings, some fake pointer noise
    mimicry     - copies human behavioural marginals; still betrayed by the
                  velocity/graph layer (it must click fast to be profitable)
    proxy_farm  - residential-proxy click farm: per-IP velocity looks NORMAL
                  because the pool rotates, but publisher-side concentration
                  and post-click value collapse. Held out for the zero-day test.
    """
    b = {}
    if family == "proxy_farm":
        # A residential-proxy click farm running a scripted browser template.
        # Every MARGINAL distribution is drawn from the human population, and
        # the entity layer is deliberately unremarkable (huge rotating IP pool,
        # traffic spread across many publishers). What betrays it is the JOINT
        # structure: the template produces combinations of signals that no real
        # session produces. A supervised model trained on the other three
        # families has never seen this family and has no rule for it; a
        # density model of legitimate behaviour still finds it off-manifold.
        b = _humans(rng, n)

        # (i) desktop containers emulating a touchscreen -> impossible pairing
        emul = rng.random(n) < 0.55
        b["device_type"] = np.where(emul, "desktop", b["device_type"])
        b["touch_events"] = np.where(
            emul, np.clip(rng.negative_binomial(4, 0.10, n), 1, 900),
            b["touch_events"])

        # (ii) "engaged" sessions that never scroll: long dwell, many pages,
        #      viewport pinned at the top
        deep = rng.random(n) < 0.60
        b["scroll_depth_pct"] = np.where(deep, rng.uniform(0, 3.5, n),
                                         b["scroll_depth_pct"])
        b["time_on_ad"] = np.where(deep, _lognorm(rng, 3.5, 0.35, n, 20, 400),
                                   b["time_on_ad"])
        b["pages_viewed"] = np.where(deep, np.clip(rng.poisson(7, n) + 3, 1, 60),
                                     b["pages_viewed"])

        # (iii) the template never types, and always ships the same UA build
        b["keystrokes"] = np.zeros(n)
        b["ua_entropy"] = rng.normal(4.66, 0.045, n)
        b["js_enabled"] = np.ones(n, dtype=int)

        # (iv) weak, non-decisive economic signal
        b["conversion_rate_pub"] = rng.beta(1.4, 60, n)
        # per-IP velocity is intentionally NORMAL - the proxy pool rotates
        b["ip_clicks_1h"] = np.clip(rng.poisson(2.4, n) + 1, 1, 60)
        return b

    jit = {"crude": 0.10, "jittered": 0.50, "mimicry": 0.88}[family]
    dev = rng.choice(DEVICES, n, p=[.44, .36, .17, .03])
    is_touch = np.isin(dev, ["mobile", "tablet"])

    b["session_duration"] = _lognorm(rng, 2.2 + 1.7 * jit, 0.6 + 0.55 * jit, n, 1, 5400)
    b["pages_viewed"] = np.clip(rng.poisson(1.2 + 2.4 * jit, n) + 1, 1, 60)
    b["clicks_in_session"] = np.clip(rng.poisson(6.5 - 3.4 * jit, n) + 1, 1, 60)
    b["time_on_ad"] = _lognorm(rng, 0.35 + 1.25 * jit, 0.5 + 0.55 * jit, n, 0.05, 400)
    mouse = rng.negative_binomial(max(1, int(1 + 5 * jit)),
                                  max(0.03, 0.40 - 0.36 * jit), n).astype(float)
    mouse[is_touch] = mouse[is_touch] * rng.uniform(0.0, 0.35, int(is_touch.sum()))
    b["mouse_move_events"] = np.clip(mouse, 0, 4000)
    t = rng.negative_binomial(max(1, int(1 + 3 * jit)), 0.35 - 0.25 * jit, n).astype(float)
    t[~is_touch] = 0
    b["touch_events"] = np.clip(t, 0, 900)
    b["scroll_depth_pct"] = np.clip(rng.beta(0.9 + 1.3 * jit, 5.0 - 3.1 * jit, n) * 100, 0, 100)
    b["keystrokes"] = np.clip(rng.poisson(0.15 + 4.2 * jit, n), 0, 300)
    b["avg_click_interval_ms"] = _lognorm(rng, 6.2 + 1.6 * jit, 0.30 + 0.85 * jit, n, 80, 400_000)
    b["click_interval_std_ms"] = b["avg_click_interval_ms"] * rng.uniform(
        0.02 + 0.16 * jit, 0.10 + 1.05 * jit, n)
    b["time_to_first_click_ms"] = _lognorm(rng, 5.7 + 1.9 * jit, 0.45 + 0.7 * jit, n, 40, 300_000)
    b["ua_entropy"] = rng.normal(2.9 + 1.35 * jit, 0.35 + 0.28 * jit, n).clip(1.0, 6.5)
    b["hour_of_day"] = np.where(rng.random(n) < (1 - jit),
                                rng.integers(0, 24, n), _human_hours(rng, n))
    b["day_of_week"] = rng.integers(0, 7, n)
    b["ip_clicks_1h"] = np.clip(rng.poisson(20 - 13 * jit, n) + 1, 1, 400)
    b["conversion_rate_pub"] = rng.beta(1.0 + 1.1 * jit, 110 - 76 * jit, n)
    b["js_enabled"] = (rng.random(n) > (0.42 - 0.40 * jit)).astype(int)
    b["cookie_enabled"] = (rng.random(n) > (0.50 - 0.44 * jit)).astype(int)
    b["bounce"] = (rng.random(n) < (0.86 - 0.42 * jit)).astype(int)
    headless_p = max(0.01, 0.30 - 0.29 * jit)
    pb = np.array([.32, .10, .07, .09, .05, headless_p, .05]); pb = pb / pb.sum()
    b["browser"] = rng.choice(BROWSERS, n, p=pb)
    b["os"] = rng.choice(OSES, n, p=[.24, .27, .07, .05, .32, .05])
    b["device_type"] = dev
    b["country"] = rng.choice(COUNTRIES, n, p=[.12, .08, .04, .04, .07, .15, .18, .13, .11, .08])
    b["referrer"] = rng.choice(REFERRERS, n, p=[.12, .13, .43, .28, .04])
    b["ad_position"] = rng.choice(AD_POS, n, p=[.33, .29, .13, .19, .06])
    return b


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_clickstream(n=40_000, fraud_ratio=0.35, family_mix=None,
                         n_publishers=60, n_ips=6000, seed=42,
                         start_ts="2025-01-01", label_noise=0.02,
                         span_hours=72):
    rng = np.random.default_rng(seed)
    mix = dict(family_mix or DEFAULT_MIX)
    tot = sum(mix.values())
    mix = {k: v / tot for k, v in mix.items()}

    n_fraud = int(n * fraud_ratio)
    n_human = n - n_fraud

    h = _humans(rng, n_human)

    keys, counts, fam_names, cursor = list(mix.keys()), [], [], 0
    for i, k in enumerate(keys):
        c = n_fraud - cursor if i == len(keys) - 1 else int(round(n_fraud * mix[k]))
        counts.append(c); cursor += c
        fam_names += [k] * c
    bot_blocks = [(k, _bots(rng, c, k)) for k, c in zip(keys, counts) if c > 0]

    rows = {}
    for c in list(h.keys()):
        rows[c] = np.concatenate([np.asarray(h[c])] +
                                 [np.asarray(bb[c]) for _, bb in bot_blocks])

    y = np.array(["Human"] * n_human + ["Bot"] * n_fraud, dtype=object)
    fam = np.array(["human"] * n_human + fam_names, dtype=object)

    # ---- entities ---------------------------------------------------------
    pub_h = rng.integers(0, n_publishers, n_human)
    ip_h = rng.integers(0, n_ips, n_human)
    dev_h = rng.integers(0, int(n_ips * 1.5), n_human)

    pub_b, ip_b, dev_b = [], [], []
    n_bot_pub = max(3, int(n_publishers * 0.13))
    for k, c in zip(keys, counts):
        if c == 0:
            continue
        if k == "proxy_farm":
            # spread wide on purpose: normal publisher mix, normal IP velocity
            pub_b.append(rng.integers(0, n_publishers, c))
            ip_b.append(rng.integers(0, int(n_ips * 2.5), c) + n_ips)
            dev_b.append(rng.integers(0, int(n_ips * 1.2), c) + int(n_ips * 1.5))
        else:
            share = {"crude": 0.85, "jittered": 0.72, "mimicry": 0.60}[k]
            pub_b.append(np.where(rng.random(c) < share,
                                  rng.integers(0, n_bot_pub, c),
                                  rng.integers(0, n_publishers, c)))
            pool = {"crude": 0.02, "jittered": 0.06, "mimicry": 0.16}[k]
            ip_b.append(rng.integers(0, max(40, int(n_ips * pool)), c) + n_ips)
            dev_b.append(rng.integers(0, max(25, int(n_ips * pool * 0.7)), c)
                         + int(n_ips * 1.5))

    pub = np.concatenate([pub_h] + pub_b)
    ip = np.concatenate([ip_h] + ip_b)
    dvc = np.concatenate([dev_h] + dev_b)

    # Timestamps are drawn INDEPENDENTLY of the row order. Sorting them here
    # would hand every human an early timestamp and every bot a late one, which
    # silently turns the velocity features into a label leak.
    base = pd.Timestamp(start_ts).value // 10**9
    secs = rng.integers(0, span_hours * 3600, n)
    ts = pd.to_datetime(base + secs, unit="s")

    df = pd.DataFrame(rows)
    df["ip_hash"] = [f"ip_{v}" for v in ip]
    df["device_hash"] = [f"dev_{v}" for v in dvc]
    df["publisher_id"] = [f"pub_{v}" for v in pub]
    df[LABEL_COL] = y
    df["bot_family"] = fam
    df[TIME_COL] = ts
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # ---- realistic imperfections -----------------------------------------
    for c in ["scroll_depth_pct", "keystrokes", "conversion_rate_pub", "touch_events"]:
        df.loc[rng.random(len(df)) < 0.02, c] = np.nan
    if label_noise > 0:
        flip = rng.random(len(df)) < label_noise
        df.loc[flip, LABEL_COL] = df.loc[flip, LABEL_COL].map(
            {"Human": "Bot", "Bot": "Human"})

    df["hour_of_day"] = df["hour_of_day"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Real dataset loader
# ---------------------------------------------------------------------------
_LABEL_CANDIDATES = ["clicked", "class", "label", "target", "is_fraud",
                     "fraud", "isfraud", "click_type", "type", "attributed"]
_POSITIVE_TOKENS = {"bot", "fraud", "fraudulent", "1", "yes", "true", "invalid"}


def load_real_csv(path, label_col=None):
    """Load a real click-fraud CSV, normalising the label to Human/Bot."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if label_col is None:
        low = {c.lower(): c for c in df.columns}
        for cand in _LABEL_CANDIDATES:
            if cand in low:
                label_col = low[cand]; break
        if label_col is None:
            for c in reversed(df.columns):
                if df[c].nunique(dropna=True) == 2:
                    label_col = c; break
    if label_col is None:
        raise ValueError("Could not identify a label column; pass label_col=...")

    out = df.copy()
    out[LABEL_COL] = out[label_col].map(
        lambda v: "Bot" if str(v).strip().lower() in _POSITIVE_TOKENS else "Human")
    if label_col != LABEL_COL:
        out = out.drop(columns=[label_col])
    return out, LABEL_COL


def make_drift_stream(n_per_block=6000, seed=7):
    """Traffic whose bot mixture shifts toward evasive families over time."""
    mixes = [
        {"crude": .40, "jittered": .40, "mimicry": .20},
        {"crude": .30, "jittered": .42, "mimicry": .28},
        {"crude": .10, "jittered": .30, "mimicry": .45, "proxy_farm": .15},
        {"crude": .02, "jittered": .10, "mimicry": .28, "proxy_farm": .60},
    ]
    frames = []
    for i, m in enumerate(mixes):
        d = generate_clickstream(n=n_per_block, family_mix=m, seed=seed + 17 * i)
        d["block"] = i
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
