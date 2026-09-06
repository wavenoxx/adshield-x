# AdShield-X

[![tests](https://github.com/wavenoxx/adshield-x/actions/workflows/ci.yml/badge.svg)](https://github.com/wavenoxx/adshield-x/actions/workflows/ci.yml)
[![licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

### Try it

| | |
|---|---|
| **Live console** | **https://adshield-x.onrender.com** — sign in with `admin` / `admin` |
| **Results page** | https://wavenoxx.github.io/adshield-x/ |

The console runs the real trained model: upload a click log, generate a labelled
sample, or type a single click into a form and watch the verdict move. It is on
a free instance, so it sleeps after fifteen idle minutes and takes about a
minute to wake — open it once before you need it.

The results page is static and always instant. It runs a logistic surrogate
distilled from the deployed forest, which agrees with the full model on 98.6% of
test clicks, so it can score in the browser with no server behind it.

---

Ad click-fraud detection that measures the things a click-fraud paper usually
doesn't: recall against a bot family nobody trained on, attack-success rate
against an adversary that fights back, accuracy on traffic that drifts, and the
threshold that actually minimises money lost.

It reproduces the fourteen-model comparison from
*Alzahrani, Aljabri & Mohammad, "Ad Click Fraud Detection Using Machine Learning
and Deep Learning Algorithms", IEEE Access 13:12746–12763, 2025* — and then
tests the four assumptions that comparison rests on.

---

## What is in this submission

| Deliverable | File |
|---|---|
| **Source code** | this repository |
| **Working demo** | the Flask application in `app/` — backend, database, web UI, JSON API |
| **Research paper** | `paper/AdShield-X_Research_Paper.docx` |
| Speaker guide | `paper/AdShield-X_Explanation_Guide.docx` — how to present the work |
| Results page | `docs/index.html` — a standalone page for *showing* the results in a talk. It is a supplement, not the demo. |

The results page is a single HTML file with the trained model distilled into a
small in-page surrogate, so it opens anywhere with no server. Useful on a
projector. The thing that is actually evaluated as the demo is the Flask app
below, which loads the real model.

## Quick start

```bash
pip install -r requirements.txt

python run_pipeline.py            # trains the model, ~15 min on one CPU core
python app/app.py                 # console at http://127.0.0.1:5000
```

Sign in with `admin` / `admin`, or register a new analyst account.
The database is created on first run at `outputs/adshield.db`.

If the pipeline gets cut off, run it in pieces — each stage checkpoints:

```bash
python run_pipeline.py --stage 1  # baselines + feature ablation
python run_pipeline.py --stage 2  # zero-day cascade + adversarial
python run_pipeline.py --stage 3  # cost, drift, explanations, latency
python run_pipeline.py --quick    # small fast version of everything
```

To run on a real labelled click log instead of the simulator:

```bash
python run_pipeline.py --csv path/to/clicks.csv
```

The loader finds the label column itself (`clicked`, `class`, `label`,
`is_fraud`, `attributed`, …) and normalises it to Human/Bot. Entity and
timestamp columns are used for the graph features when present and skipped
cleanly when absent.

---

## What's here

```
run_pipeline.py         the whole study, in three checkpointed stages
src/data.py             click simulator (4 bot families) + real-CSV loader
src/features.py         preprocessing, RFE, and the EVGF graph features
src/autodiff.py         200-line reverse-mode autodiff engine (NumPy only)
src/dl.py               DNN / CNN / RNN / CNN+Attention + the autoencoder
src/models.py           the ten classical models and the metric harness
src/novel.py            cascade, mimicry attack, cost threshold, drift, SHAP
app/app.py              Flask routes: auth, scanning, history, reports, API
app/db.py               SQLite schema and queries
app/scoring.py          service layer between the web tier and the model
app/templates/          nine Jinja templates
paper/                  the research paper and the speaker guide
app/static/             stylesheet
tests/                  gradient checks and console end-to-end tests
docs/                   the explainer, served by GitHub Pages
Dockerfile              container for Spaces, Railway, Fly
render.yaml, Procfile   one-click deploy on Render
outputs/                tables, results.json, trained model
```

No TensorFlow, no PyTorch, no GPU. The deep models run on `src/autodiff.py`,
which is gradient-checked against finite differences in the test below:

```bash
python -c "
import sys; sys.path.insert(0,'src')
import numpy as np; from autodiff import *
rng=np.random.default_rng(0); x=Tensor(rng.normal(size=(4,10,1)).astype(np.float32))
c=Conv1D(1,3,3,rng); a=AdditiveAttention(3,4,rng); d=Dense(3,1,rng)
f=lambda: d(a(c(x).relu())[0]); y=np.array([[0],[1],[1],[0]])
l=f().bce_with_logits(y); l.backward(); g=d.W.grad[0,0]
o=d.W.data[0,0]; e=1e-3
d.W.data[0,0]=o+e; l1=f().bce_with_logits(y).data
d.W.data[0,0]=o-e; l2=f().bce_with_logits(y).data
print('analytic',g,'numeric',(l1-l2)/(2*e))"
```

---

## Putting it on the web

Two things can be published, and they are published differently.

**The results page** is static — GitHub Pages serves it with no build step.
Settings → Pages, source `main`, folder `/docs`, save. A minute later
https://wavenoxx.github.io/adshield-x/ is live. This is the page to open on a
projector.

**The console** is a Flask app that loads a 7.5 MB model, so it needs a real
host. Three that work:

| Host | How |
|---|---|
| Render | What this deployment uses. Render detects the `Dockerfile` and builds from it; `render.yaml` is here too if you prefer the native Python runtime. Free tier sleeps after 15 idle minutes and takes about a minute to wake, because the model loads on the first request. |
| Hugging Face Spaces | Create a Space with the Docker SDK and push this repo. `Dockerfile` already listens on 7860, which is what Spaces expects. |
| Railway / Fly.io | Both read the `Dockerfile` directly. |

The console also serves the results page at
[`/explainer`](https://adshield-x.onrender.com/explainer), so a single
deployment covers both if you would rather hand out one link.

### Why not a serverless platform

Vercel, Netlify Functions and the like will host the static page happily, but
they are the wrong shape for the console, for two reasons that have nothing to
do with the framework. The dependency tree is about 600 MB unpacked — llvmlite
alone is 180 MB, scipy 109 MB, xgboost 88 MB — which is at or over the Python
bundle ceiling on most serverless tiers. More decisively, serverless instances
get an ephemeral, effectively read-only filesystem, and this application writes
accounts, scan history, verdicts and an audit trail to SQLite. On a serverless
host those tables silently reset between requests and half the console stops
meaning anything. A container platform keeps one process, one disk and one
loaded model, which is what this workload wants.

Two environment variables matter:

| Variable | Why |
|---|---|
| `ADSHIELD_SECRET` | Session key. Without it a fresh one is generated per process, which signs everyone out on every restart. |
| `ADSHIELD_REASON_LIMIT` | How many clicks per batch get a SHAP explanation. Default 60; this deployment uses 25 because a free instance gets 0.1 CPU. |
| `ADSHIELD_CPC` | Cost per click used for the money column. Default ₹18. |

Measured on one core, the console peaks at about 365 MB resident and answers a
500-click scan in roughly 5 s, so it fits a 512 MB free instance. Shared-CPU
tiers are slower; lower `ADSHIELD_REASON_LIMIT` if a scan ever approaches the
gunicorn timeout.

One caveat worth knowing before a demo: free tiers give you an ephemeral
filesystem, so `outputs/adshield.db` is wiped on redeploy and scan history
resets. That is fine for a demonstration. Mount a volume and point
`ADSHIELD_DB` at it if history has to survive.

A session cookie can outlive the row it points at, because the cookie is signed
with `ADSHIELD_SECRET` and survives the restart while the database does not.
`login_required` therefore checks that the account still exists rather than only
that the cookie carries an id; without that check the first write of the next
request fails on a foreign key somewhere deep in the call stack, which is a poor
way to tell someone to sign in again.

## Tests

```bash
python tests/test_autodiff.py   # finite-difference check on every layer
python tests/test_app.py        # registration, all three scan modes, API, access control
```

Both run in CI on every push. The autodiff check is the interesting one: it
verifies the hand-written backward passes for Dense, Conv1D, SimpleRNN and the
additive attention layer against central differences, which is what makes the
"no TensorFlow" claim checkable rather than asserted.

## The backend

`app/` is a three-layer Flask application: `app.py` holds the routes and
session handling, `scoring.py` loads the trained bundle and turns a click frame
into verdicts, and `db.py` owns persistence. Passwords are hashed with
Werkzeug's PBKDF2 helper; every account gets an API key on creation.

**Storage.** SQLite, four tables. `users`, `scans` (one row per job with its
totals and the threshold in force), `verdicts` (one row per click, carrying the
reason codes that justified it), and an append-only `audit` log. Verdicts are
stored *with* their reasons on purpose: an invalid-traffic charge cannot be
disputed with a probability, so a verdict without its justification is not
evidence.

**Web routes.**

| Route | What it does |
|---|---|
| `/` · `/register` · `/signout` | account handling |
| `/dashboard` | running totals and the last eight scans |
| `/scan` | three ways in: upload a CSV, draw a labelled sample from the held-out pool, or type one click into a form |
| `/scan/<id>` | stored verdicts with reason codes, escalations highlighted |
| `/scan/<id>.csv` | the same scan as a downloadable report |
| `/history` | every scan this account has run |
| `/model` | model card — what is deployed, at what threshold, and your API key |
| `/audit` | activity log, admin accounts only |

**JSON API.** `GET /api/v1/health` needs no key. `GET /api/v1/model` and
`POST /api/v1/score` take an `X-API-Key` header. Scores up to 2,000 clicks per
request, stores the job, and returns per-click verdicts with reason codes:

```bash
curl -X POST https://adshield-x.onrender.com/api/v1/score \
  -H "X-API-Key: <your key from /model>" \
  -H "Content-Type: application/json" \
  -d '{"clicks":[{"session_duration":6,"mouse_move_events":0,
       "scroll_depth_pct":1,"avg_click_interval_ms":500,
       "click_interval_std_ms":8,"ua_entropy":2.8,"ip_clicks_1h":40,
       "browser":"HeadlessChrome","os":"Linux","device_type":"desktop"}]}'
```

Missing columns are imputed from the training means, so a partial click record
still scores. Uploads are capped at 16 MB and API batches at 2,000 clicks.

**On samples.** "Generate a sample" draws from `outputs/sample_pool.csv`, a
held-out stream produced with the training parameters and never fitted on. It
does not synthesise twenty-five clicks on the spot, and the reason is the graph
layer: those features count what an address or publisher did in the hour before
each click, so they only carry meaning when computed over a whole stream.
Recomputed over twenty-five isolated clicks they come out at zero, and a batch
of zeros looks nothing like training — the model then treats the entire sample
as anomalous. Sampling a real stream avoids that. For the same reason
`score_frame` computes the graph layer only when a log does not already carry
it.

**On a click typed by hand.** A single form entry has no entity graph behind it
at all. Those columns are marked missing and imputed at the training mean —
"we do not know what this click's neighbourhood looked like" — rather than set
to zero, which would be a claim that the address had no history. Imputed
features are also excluded from the reason codes, because a decision must not be
explained with a signal nobody measured.

**On explanations.** SHAP over a 300-tree forest costs about 80 ms per click on
a dedicated core and several times that on a shared one, so explaining a whole
batch would blow any sensible request budget. The console ranks each batch by
how much the decision needs defending — escalated first, then blocked, then
whatever else is riskiest — and explains the top `ADSHIELD_REASON_LIMIT` clicks,
60 by default and 25 on the deployed free instance. The clean tail is scored but not explained, which is also the
right operational answer: you justify what you block.

**What is deliberately not here.** No password reset, no rate limiting, no TLS
termination, no multi-tenant isolation beyond per-user scan ownership. This is a
research prototype with a real backend, not a production ad exchange, and the
paper says so.

---

## The five contributions

**EVGF — entity-velocity graph features.** Every click carries a hashed IP, a
hashed device and a publisher id, which together form a tripartite graph. For
each click we attach ten summaries of its own neighbourhood as it stood *just
before* that click: click velocity per IP and per device, publisher fan-out,
audience-diversity ratio, Herfindahl concentration of a publisher's traffic
across IPs, burstiness, repeat-triad count. Strictly backward windows, so no
leakage; O(1) streaming counters, so it's deployable. ~1,800 clicks/s
single-threaded.

**Zero-day cascade.** A denoising autoencoder trained on legitimate clicks only,
after a rank-normalising quantile transform. It learns how real signals move
*together* — a desktop client emits no touch events, a visitor who spends a
minute over seven pages has also scrolled. A bot family that copies every
marginal but not the couplings reconstructs badly. The gate sits at a fixed
false-positive budget on known-good traffic, so the cost is a number you choose
rather than one you discover.

**Exemplar-replay mimicry attack.** For each bot, draw a real human click and
interpolate toward it — which is what an operator actually does, versus
perturbing toward a population mean that describes nobody. Only browser-side
columns move; velocity and publisher counters are held fixed, because an attack
allowed to rewrite the defender's own counters isn't an attack.

**Cost-optimal threshold.** Blocking a real customer costs more than paying for
one bad click. Sweep the threshold against expected loss, not accuracy.

**Drift monitor.** Two-window change detector on the streaming error, with a
Hoeffding bound. Retrains when the adversary moves, not on a schedule.

Plus SHAP reason codes in plain English, so a flagged click can be defended in
an invalid-traffic credit request, and a latency benchmark, because a bid path
has a millisecond budget.

---

## Headline results

Measured on a 24,000-click benchmark, held-out 20% split. Full numbers in
`outputs/results.json` and the CSVs beside it.

| | |
|---|---|
| Best baseline | XGBoost, 97.44% accuracy, 96.36 F1 |
| EVGF ablation | 89.92 → 95.83 F1; false positives 3.40% → 1.17% |
| Unseen bot family | supervised recall 0.56% → cascade 83.56% at 91.12% precision |
| Mimicry attack, full fidelity | 89.26% attack success on behavioural features; 9.57% with EVGF; 1.52% with adversarial training |
| Drifting stream | frozen 77.34% → drift-monitored 84.80% |
| Cost threshold | τ* = 0.63 not 0.50, worth ₹1,20,000 per million clicks at ₹18 CPC |

Two findings worth flagging because they cut against the source paper: the
attention-augmented CNN it proposes as an extension is second-*worst* here
(92.56 F1, 4.60% false positives), because a 1-D convolution assumes adjacent
columns are related and click-record column order is arbitrary. And eleven of
fourteen models sit within three F1 points of each other, which is why the
comparison table is a weak instrument for choosing a detector.

---

## On the data

The corpus used by the source paper is licensed from a commercial trust network
and was never published, so that work can't be reproduced as written. The
simulator here is the substitute, and it's built to be hard rather than
flattering: four bot families of increasing evasiveness, genuine class overlap
(mobile visitors generate almost no pointer movement — the exact signature naive
detectors use to flag bots), and 2% label noise.

That means absolute figures are comparisons under a stated model, not production
estimates. The generative assumptions are all in `src/data.py`, in the open,
which is more than the original numbers offer. Point `--csv` at a real log and
the whole pipeline runs unchanged.
