import streamlit as st
import os, json
from sentinel_logic import SentinelLogic

# 1. Page Config
st.set_page_config(page_title="Sentinel: Autonomous Behavioral Monitor", layout="wide")

# 2. State Initialization
if 'logic' not in st.session_state:
    st.session_state.logic = SentinelLogic()
if 'stress' not in st.session_state:
    st.session_state.stress = 2.0

# --- SIDEBAR: BIO & FORECASTING ---
with st.sidebar:
    st.header("📊 Sentinel Radar")

    # Neural Forecasting Block
    pred, status, conf = st.session_state.logic.get_forecast()

    st.subheader("Predictive Forecast")
    st.metric("Next Message Risk Prob.", f"{pred:.1%}", delta=f"{status}", delta_color="inverse")
    st.write(f"**AI Confidence:** {conf:.1%}")
    st.progress(conf)

    st.divider()

    st.header("🏥 Bio-Analytics")
    sleep = st.slider("Sleep (Hours)", 0.0, 16.0, 8.0)

    # Calculate Live Fragility based on the Asymmetric J-Curve
    fragility = st.session_state.logic.get_fragility(sleep, st.session_state.stress)

    st.metric("Stress Level", f"{st.session_state.stress:.2f}")
    st.metric("Fragility Index", f"{fragility:.1%}")

    st.write("Current Bio-Stress Load:")
    st.progress(st.session_state.stress / 10.0)

    if st.button("Purge Neural Memory"):
        with open("chat_history.json", "w") as f:
            json.dump([], f)
        st.session_state.stress = 2.0
        st.rerun()

# --- MAIN INTERFACE ---
st.title("🛡️ Sentinel Autonomous Monitor")

# --- INTERVENTION ENGINE ---
# This checks the current state and displays alerts at the top of the app
if st.session_state.stress > 8.5 and fragility > 0.70:
    st.error(
        "🚨 **EMERGENCY INTERVENTION REQUIRED** 🚨\n\nHigh-risk crisis state detected. Emergency services and guardians have been alerted.")
elif st.session_state.stress > 7.0:
    st.warning(
        "⚠️ **ADULT NOTIFICATION TRIGGERED**\n\nStress levels have exceeded safe thresholds. Contacting trusted adult for check-in.")
elif fragility > 0.65:
    st.info(
        "ℹ️ **SYSTEM ALERT: HIGH VULNERABILITY**\n\nUser is in a high-fragility state. Entering sensitive monitoring mode.")

# --- INPUT HANDLING ---
role = st.radio("Current Source:", ["User", "Peer"], horizontal=True)


def handle_input():
    val = st.session_state.chat_input
    if val:
        # Get Score using Emotion Model, Toxic BERT, and Conversation Momentum
        score = st.session_state.logic.get_contextual_score(val)

        # Log to JSON for Behavioral Forecasting
        st.session_state.logic.log_message(role, val, score)

        # Stress & Fragility Feedback Loop
        if role == "User":
            if score > 0.85:
                # Direct impact of crisis messages
                st.session_state.stress = min(st.session_state.stress + 5.0, 10.0)
            else:
                # Gradual movement
                change = (score * 3.5) if score > 0.3 else -0.4
                st.session_state.stress = min(max(st.session_state.stress + change, 1.0), 10.0)

        else:  # Peer Impact Logic
            if score > 0.5:
                # Active Advice (The Blocking Recommendation)
                st.toast(f"Peer Aggression Detected ({score:.1%}). Recommendation: Block User.", icon="🚫")

                # Peer stress impact is scaled by the user's current fragility
                impact = score * 4.5 * fragility
                st.session_state.stress = min(st.session_state.stress + impact, 10.0)

        # Clear Input
        st.session_state.chat_input = ""


st.text_input("Enter Chat Data:", key="chat_input", on_change=handle_input)

# --- CHAT HISTORY DISPLAY ---
st.divider()
if os.path.exists("chat_history.json"):
    with open("chat_history.json", "r") as f:
        try:
            history = json.load(f)
            for m in reversed(history):
                # UI Styling based on Risk Score
                if m['score'] > 0.8:
                    box_color, lbl = "rgba(255, 75, 75, 0.2)", "🚨 CRITICAL"
                elif m['score'] > 0.4:
                    box_color, lbl = "rgba(255, 165, 0, 0.2)", "⚠️ WARNING"
                else:
                    box_color, lbl = "rgba(255, 255, 255, 0.05)", "✅ SAFE"

                st.markdown(f"""
                <div style="background-color:{box_color}; padding:15px; border-radius:10px; margin-bottom:10px; border: 1px solid {box_color}">
                    <strong>{m['role']}:</strong> {m['text']}<br>
                    <small>{lbl} | Neural Risk Score: {m['score']:.1%}</small>
                </div>
                """, unsafe_allow_html=True)
        except:
            st.error("Memory file corrupted. Please hit Purge Data.")