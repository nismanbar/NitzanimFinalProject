import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading

# Add parent directory to path so we can import sentinel_logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))
# Allow CORS specifically for WhatsApp Web, Instagram, Discord and Localhost
CORS(app, resources={r"/api/*": {"origins": [
    "https://web.whatsapp.com", 
    "https://www.instagram.com", 
    "https://discord.com", 
    "http://localhost:5000"
]}})

# --- RECOMMENDATION LOGIC FROM UPDATED APP.PY ---
SUPPORTIVE_PATTERNS = {"supportive reassurance", "supportive de-escalation", "repair / apology"}
USER_HARM_PATTERNS = {"explicit self harm intent", "distress / hopelessness"}
PEER_HARM_PATTERNS = {
    "peer escalation",
    "harmful escalation",
    "affirming harm",
    "continuing harm",
    "explicit self harm intent",
}

def normalize_label(value):
    import re
    s = str(value or "").strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def get_recommendation(role: str, score: float, analysis: dict) -> str:
    pattern_label = normalize_label(analysis.get("pattern_label", "neutral casual chat"))
    pattern_risk = float(analysis.get("pattern_risk", 0.0))
    dialog_act_label = normalize_label(analysis.get("dialog_act_label", "neutral reply"))
    dialog_act_risk = float(analysis.get("dialog_act_risk", 0.0))
    dialog_act_conf = float(analysis.get("dialog_act_confidence", 0.0))

    if role == "Peer":
        if pattern_label in SUPPORTIVE_PATTERNS or (
            dialog_act_label in {"inform", "commissive"} and dialog_act_conf >= 0.30 and score < 0.25
        ):
            return "✅ Supportive or repairing message detected."

        if score >= 0.70 or pattern_label in PEER_HARM_PATTERNS or (
            dialog_act_label == "directive" and dialog_act_risk >= 0.30
        ):
            return "🚫 Block / mute peer and report if needed."

        if score >= 0.45 or pattern_risk >= 0.30 or dialog_act_risk >= 0.25:
            return "⚠️ Reduce exposure and monitor the conversation."

        return "✅ No action needed."

    if score >= 0.80 or pattern_label == "explicit self harm intent":
        return "🧠 Check on yourself now. Involve a trusted adult if needed."

    if pattern_label == "distress / hopelessness" and (score >= 0.30 or dialog_act_risk >= 0.18):
        return "🧠 Check on yourself now. Involve a trusted adult if needed."

    if score >= 0.45 or pattern_risk >= 0.18 or dialog_act_risk >= 0.22:
        return "ℹ️ Take a short break and reduce stressors."

    return "✅ No intervention needed."

# --- GLOBAL ERROR HANDLER ---
@app.errorhandler(Exception)
def handle_exception(e):
    if hasattr(e, 'code'):
        return jsonify({"error": str(e)}), e.code
    print(f"ERROR: {str(e)}")
    return jsonify({
        "error": "Internal Server Error",
        "details": str(e),
        "type": type(e).__name__
    }), 500

sentinel = None
sentinel_error = None

def init_sentinel():
    global sentinel, sentinel_error
    try:
        from sentinel_logic import SentinelLogic
        sentinel = SentinelLogic()
        print("SentinelLogic loaded successfully")
    except Exception as e:
        import traceback
        sentinel_error = f"{str(e)}\n{traceback.format_exc()}"
        print(f"CRITICAL: Could not load SentinelLogic: {sentinel_error}")

# Initialize in a thread so the server starts even if models are slow
threading.Thread(target=init_sentinel).start()

DATA_FILE = "sentinel_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"messages": [], "alerts": [], "stats": {"total": 0, "high_risk": 0, "medium_risk": 0}}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

data_lock = threading.Lock()

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/config", methods=["GET"])
def get_config():
    if not sentinel:
        return jsonify({"error": "Sentinel not loaded"}), 503
    return jsonify(sentinel.config)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if not sentinel:
        return jsonify({
            "error": "SentinelLogic not loaded yet",
            "details": sentinel_error or "Initializing..."
        }), 503

    req = request.json
    text = req.get("text", "")
    sender = req.get("sender", "Unknown")
    is_incoming = req.get("is_incoming", True)
    platform = req.get("platform", "whatsapp")

    score = sentinel.get_contextual_score(text)
    analysis = sentinel.last_analysis

    role = "Peer" if is_incoming else "User"
    recommendation = get_recommendation(role, score, analysis)

    risk_level = "low"
    if score >= 0.70:
        risk_level = "high"
    elif score >= 0.45:
        risk_level = "medium"

    alert = {
        "id": len(load_data()["alerts"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "text": text[:100] + "..." if len(text) > 100 else text,
        "full_text": text,
        "sender": sender,
        "role": role,
        "score": score,
        "risk_level": risk_level,
        "pattern_label": analysis.get("pattern_label", "neutral reply"),
        "pattern_risk": float(analysis.get("pattern_risk", 0.0)),
        "dialog_act_label": analysis.get("dialog_act_label", "neutral reply"),
        "recommendation": recommendation,
        "platform": platform,
        "is_incoming": is_incoming
    }

    with data_lock:
        data = load_data()
        data["messages"].insert(0, {
            "timestamp": alert["timestamp"],
            "text": text,
            "sender": sender,
            "role": role,
            "score": score,
            "platform": platform
        })
        data["messages"] = data["messages"][:100]

        if risk_level in ["high", "medium"]:
            data["alerts"].insert(0, alert)
            data["alerts"] = data["alerts"][:50]

        data["stats"]["total"] = len(data["messages"])
        data["stats"]["high_risk"] = len([a for a in data["alerts"] if a["risk_level"] == "high"])
        data["stats"]["medium_risk"] = len([a for a in data["alerts"] if a["risk_level"] == "medium"])

        save_data(data)

    return jsonify({
        "score": score,
        "risk_level": risk_level,
        "pattern_label": analysis.get("pattern_label", "neutral reply"),
        "pattern_risk": float(analysis.get("pattern_risk", 0.0)),
        "recommendation": recommendation,
        "analysis": analysis
    })

@app.route("/api/history", methods=["GET"])
def get_history():
    data = load_data()
    return jsonify(data["messages"][:20])

@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    data = load_data()
    return jsonify(data["alerts"])

@app.route("/api/stats", methods=["GET"])
def get_stats():
    data = load_data()
    return jsonify(data["stats"])

@app.route("/api/clear", methods=["POST"])
def clear_data():
    with data_lock:
        save_data({"messages": [], "alerts": [], "stats": {"total": 0, "high_risk": 0, "medium_risk": 0}})
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    print("Starting Sentinel API Server...")
    print("Dashboard available at: http://localhost:5000/dashboard")
    # Disable debug mode and use threaded=True for better stability on Windows
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)