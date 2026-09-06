"""End-to-end check of the Flask console.

Exercises registration, all three scan modes, the CSV report, access control and
the JSON API. Skips gracefully when the trained model is absent, so the test is
safe to run on a fresh checkout before `run_pipeline.py`.
"""
import io, os, re, sys, tempfile, warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, os.path.join(ROOT, "src"))
warnings.filterwarnings("ignore")

os.environ["ADSHIELD_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")

MODEL = os.path.join(ROOT, "outputs", "adshield_model.joblib")
if not os.path.exists(MODEL):
    print("adshield_model.joblib not found — run `python run_pipeline.py` first.")
    print("Skipping the scoring tests; checking routes only.")

from app import app                                    # noqa: E402
import data as D                                       # noqa: E402

c = app.test_client()
ok = lambda cond, msg: print(("  PASS  " if cond else "  FAIL  ") + msg) or (
    None if cond else sys.exit(1))

print("Console checks:")
ok(c.get("/").status_code == 200, "sign-in page renders")
ok(c.post("/register", data={"username": "tester", "password": "secret123"},
          follow_redirects=True).status_code == 200, "account created")
ok(b"clicks scanned" in c.get("/dashboard").data, "dashboard renders")
ok(c.get("/scan").status_code == 200, "scan page renders")

if os.path.exists(MODEL):
    r = c.post("/scan", data={"mode": "sample", "n": "12"}, follow_redirects=True)
    ok(b"Verdicts" in r.data, "generated sample scored")

    r = c.post("/scan", data={"mode": "manual", "mouse_move_events": "0",
                              "ip_clicks_1h": "40", "click_interval_std_ms": "8",
                              "browser": "HeadlessChrome"},
               follow_redirects=True)
    ok(b"Verdicts" in r.data, "single click scored from the form")

    df = D.generate_clickstream(n=10, seed=5).drop(columns=["bot_family"])
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = c.post("/scan", data={"mode": "upload", "csv": (buf, "clicks.csv")},
               content_type="multipart/form-data", follow_redirects=True)
    ok(b"Verdicts" in r.data, "uploaded CSV scored")

    r = c.get("/scan/1.csv")
    ok(r.status_code == 200 and b"fraud_risk_pct" in r.data, "CSV report downloads")

    key = re.search(rb"ak_[0-9a-f]{32}", c.get("/model").data).group(0).decode()
    rows = df.head(3).to_dict("records")
    for x in rows:
        x["click_ts"] = str(x["click_ts"])
    r = c.post("/api/v1/score", json={"clicks": rows}, headers={"X-API-Key": key})
    ok(r.status_code == 200 and "summary" in r.get_json(), "API scores a batch")
    ok(c.post("/api/v1/score", json={"clicks": rows}).status_code == 401,
       "API rejects a missing key")

ok(c.get("/history").status_code == 200, "history renders")
ok(c.get("/audit").status_code == 403, "analyst cannot read the audit log")
ok(c.get("/scan/999").status_code == 404, "unknown scan returns 404")
print("All console checks passed.")
