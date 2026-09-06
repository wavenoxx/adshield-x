"""
AdShield-X :: app/app.py
------------------------
Flask backend for the traffic-quality console.

    python run_pipeline.py     # trains and writes outputs/adshield_model.joblib
    python app/app.py          # then open http://127.0.0.1:5000

A demo account is seeded on first run: admin / admin. Change it, or register a
new analyst from the sign-in page.

Web routes
    /                 sign in
    /register         create an analyst account
    /dashboard        running totals and recent scans
    /scan             run a scan: upload a CSV, generate a sample, or type one click
    /scan/<id>        stored verdicts for one scan, with reason codes
    /scan/<id>.csv    download that scan as a CSV report
    /history          every scan this account has run
    /model            model card: what is deployed and at what threshold
    /audit            append-only activity log (admin only)
    /explainer        the standalone results page, served from docs/

JSON API
    GET  /api/v1/health
    GET  /api/v1/model
    POST /api/v1/score        header: X-API-Key
"""

from __future__ import annotations
import io, os, csv, json, secrets, functools
import pandas as pd
from flask import (Flask, render_template, request, redirect, url_for, session,
                   flash, jsonify, Response, abort, send_from_directory)

import db as store
import scoring
from scoring import ModelMissing

MAX_UPLOAD_MB = 16
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "docs")


def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("ADSHIELD_SECRET", secrets.token_hex(16))
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    store.init_db(app)

    # ---------------- helpers ------------------------------------------
    def current_user():
        uid = session.get("uid")
        return store.user_by_id(uid) if uid else None

    def login_required(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            if not session.get("uid"):
                return redirect(url_for("signin", next=request.path))
            return fn(*a, **kw)
        return wrapper

    @app.context_processor
    def inject_user():
        return {"user": current_user()}

    @app.errorhandler(413)
    def too_big(_):
        flash(f"That file is larger than {MAX_UPLOAD_MB} MB. Split it and try again.")
        return redirect(url_for("scan"))

    # ---------------- auth ---------------------------------------------
    @app.route("/", methods=["GET", "POST"])
    def signin():
        if session.get("uid"):
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            from werkzeug.security import check_password_hash
            u = store.find_user(request.form.get("username", "").strip())
            if u and check_password_hash(u["password_hash"],
                                         request.form.get("password", "")):
                session["uid"] = u["id"]
                store.audit(u["id"], "sign in")
                return redirect(request.args.get("next") or url_for("dashboard"))
            flash("That user name and password do not match an account.")
        return render_template("signin.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            name = request.form.get("username", "").strip()
            pw = request.form.get("password", "")
            if len(name) < 3:
                flash("Pick a user name of at least three characters.")
            elif len(pw) < 6:
                flash("Use a password of at least six characters.")
            elif store.find_user(name):
                flash("That user name is taken.")
            else:
                uid = store.create_user(name, pw)
                store.audit(uid, "register", name)
                session["uid"] = uid
                return redirect(url_for("dashboard"))
        return render_template("register.html")

    @app.route("/signout")
    def signout():
        if session.get("uid"):
            store.audit(session["uid"], "sign out")
        session.clear()
        return redirect(url_for("signin"))

    # ---------------- dashboard ----------------------------------------
    @app.route("/dashboard")
    @login_required
    def dashboard():
        uid = session["uid"]
        warn = None
        try:
            scoring.bundle()
        except ModelMissing as e:
            warn = str(e)
        return render_template("dashboard.html", totals=store.totals(uid),
                               scans=store.list_scans(uid, 8), warn=warn)

    # ---------------- scan ---------------------------------------------
    @app.route("/scan", methods=["GET", "POST"])
    @login_required
    def scan():
        uid = session["uid"]
        error = None
        if request.method == "POST":
            mode = request.form.get("mode", "sample")
            try:
                if mode == "upload":
                    f = request.files.get("csv")
                    if not f or not f.filename:
                        raise ValueError("Choose a CSV file to score.")
                    df = pd.read_csv(io.BytesIO(f.read()))
                    label = f.filename
                elif mode == "manual":
                    df = scoring.frame_from_form(request.form)
                    label = "single click, entered by hand"
                else:
                    df = scoring.sample_frame(request.form.get("n", 25))
                    label = f"generated sample of {len(df)}"
                if df.empty:
                    raise ValueError("That file has no rows.")
                recs, summary = scoring.score_frame(df)
                sid = store.save_scan(uid, mode, label, summary, recs)
                store.audit(uid, "scan", f"#{sid} {mode} {summary['scanned']} clicks")
                return redirect(url_for("scan_detail", scan_id=sid))
            except ModelMissing as e:
                error = str(e)
            except Exception as e:
                error = f"Could not score that input: {e}"
        return render_template("scan.html", error=error,
                               fields=scoring.MANUAL_FIELDS,
                               cats=scoring.MANUAL_CATEGORICAL,
                               defaults=scoring.manual_defaults())

    @app.route("/scan/<int:scan_id>")
    @login_required
    def scan_detail(scan_id):
        s = store.get_scan(scan_id, session["uid"])
        if s is None:
            abort(404)
        return render_template("results.html", scan=s,
                               recs=store.get_verdicts(scan_id))

    @app.route("/scan/<int:scan_id>.csv")
    @login_required
    def scan_csv(scan_id):
        s = store.get_scan(scan_id, session["uid"])
        if s is None:
            abort(404)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["row", "fraud_risk_pct", "supervised_pct", "escalated",
                    "decision", "ground_truth", "reason_1", "reason_2",
                    "reason_3", "reason_4"])
        for r in store.get_verdicts(scan_id):
            phrases = [x["explanation"] for x in r["reasons"]][:4]
            phrases += [""] * (4 - len(phrases))
            w.writerow([r["row_no"], r["probability"], r["supervised"],
                        "yes" if r["escalated"] else "no",
                        "blocked" if r["blocked"] else "allowed",
                        r["truth"] or ""] + phrases)
        store.audit(session["uid"], "export", f"scan #{scan_id}")
        return Response(buf.getvalue(), mimetype="text/csv", headers={
            "Content-Disposition": f'attachment; filename="adshield_scan_{scan_id}.csv"'})

    @app.route("/history")
    @login_required
    def history():
        return render_template("history.html",
                               scans=store.list_scans(session["uid"], 100))

    @app.route("/model")
    @login_required
    def model():
        try:
            card = scoring.model_card()
            err = None
        except ModelMissing as e:
            card, err = None, str(e)
        return render_template("model.html", card=card, err=err,
                               api_key=current_user()["api_key"])

    @app.route("/audit")
    @login_required
    def audit_log():
        if current_user()["role"] != "admin":
            abort(403)
        return render_template("audit.html", rows=store.recent_audit(60))

    # ---------------- explainer -----------------------------------------
    @app.route("/explainer")
    def explainer():
        """The standalone results page, so one deployment serves both the live
        console and the page you would put on a projector. It needs no login:
        it contains no data belonging to any account, only the published
        experimental results."""
        if not os.path.exists(os.path.join(DOCS_DIR, "index.html")):
            abort(404)
        return send_from_directory(DOCS_DIR, "index.html")

    @app.route("/explainer/<path:asset>")
    def explainer_asset(asset):
        return send_from_directory(DOCS_DIR, asset)

    # ---------------- JSON API ------------------------------------------
    def api_user():
        key = request.headers.get("X-API-Key", "")
        return store.user_by_key(key) if key else None

    @app.get("/api/v1/health")
    def api_health():
        try:
            scoring.bundle()
            return jsonify({"status": "ok", "model": "loaded"})
        except ModelMissing as e:
            return jsonify({"status": "degraded", "model": str(e)}), 503

    @app.get("/api/v1/model")
    def api_model():
        if api_user() is None:
            return jsonify({"error": "supply a valid X-API-Key header"}), 401
        try:
            return jsonify(scoring.model_card())
        except ModelMissing as e:
            return jsonify({"error": str(e)}), 503

    @app.post("/api/v1/score")
    def api_score():
        u = api_user()
        if u is None:
            return jsonify({"error": "supply a valid X-API-Key header"}), 401
        payload = request.get_json(force=True, silent=True) or {}
        rows = payload.get("clicks")
        if not isinstance(rows, list) or not rows:
            return jsonify({"error": "send a non-empty 'clicks' array"}), 400
        if len(rows) > 2000:
            return jsonify({"error": "send at most 2000 clicks per request"}), 400
        try:
            recs, summary = scoring.score_frame(pd.DataFrame(rows))
        except ModelMissing as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            return jsonify({"error": f"could not score that payload: {e}"}), 400
        sid = store.save_scan(u["id"], "api", "api request", summary, recs)
        store.audit(u["id"], "api scan", f"#{sid} {summary['scanned']} clicks")
        return jsonify({"scan_id": sid, "summary": summary, "results": recs})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
