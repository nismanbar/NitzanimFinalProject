import streamlit as st
import os, json, copy
from sentinel_logic import SentinelLogic

st.set_page_config(page_title="Sentinel AI V8.5", layout="wide")

DEFAULT_SLEEP = 7.5
DEFAULT_SCREEN = 5.0
DEFAULT_STRESS = 3.0


def risk_tag(role, score):
    if role == "Peer":
        if score >= 0.70:
            return "🚫 HIGH PEER RISK"
        if score >= 0.45:
            return "⚠️ PEER CONCERN"
        return "✅ PEER SAFE"
    else:
        if score >= 0.80:
            return "🚨 HIGH USER RISK"
        if score >= 0.45:
            return "⚠️ USER CHECK-IN"
        return "✅ USER SAFE"


def risk_bar(role, score):
    pct = f"{score:.0%}"
    tag = risk_tag(role, score)
    return f"{tag} | Risk: {pct}"


def build_recommendation(role, score, pattern_label, pattern_risk):
    user_danger = {"self_harm_intent", "affirming_harm", "continuing_harm"}
    peer_danger = {"peer_escalation", "affirming_harm", "continuing_harm", "self_harm_intent"}

    if role == "Peer":
        if score >= 0.70 or (pattern_label in peer_danger and pattern_risk >= 0.55):
            return "🚫 Block / mute peer and report if needed."
        if score >= 0.45 or (pattern_risk >= 0.45 and score >= 0.25):
            return "⚠️ Reduce exposure and monitor the conversation."
        return "✅ No action needed."
    else:
        if score >= 0.80 or pattern_label in user_danger or pattern_risk >= 0.70:
            return "🧠 Check on the user now and involve a trusted adult if needed."
        if score >= 0.45 or pattern_risk >= 0.45:
            return "ℹ️ Send a supportive check-in and reduce stressors."
        return "✅ No intervention needed."


# --- PRE-RENDER RESET HANDLER ---
if "reset_trigger" in st.session_state:
    trigger = st.session_state.reset_trigger

    if trigger == "clear":
        with open("chat_history.json", "w") as f:
            json.dump([], f)
        st.session_state.visual_chat = []
        st.session_state.recommendation_log = []

    elif trigger == "factory":
        if "logic" in st.session_state:
            st.session_state.logic._init_files(force_reset=True)
        st.session_state.visual_chat = []
        st.session_state.recommendation_log = []

    st.session_state.sleep_val = DEFAULT_SLEEP
    st.session_state.screen_val = DEFAULT_SCREEN
    st.session_state.stress = DEFAULT_STRESS
    st.session_state.prev_stress = DEFAULT_STRESS
    st.session_state.role = "User"

    del st.session_state.reset_trigger


# --- INITIALIZATION ---
if "logic" not in st.session_state:
    st.session_state.logic = SentinelLogic()
    st.session_state.initial_brain = copy.deepcopy(st.session_state.logic.state)

if "sleep_val" not in st.session_state:
    st.session_state.sleep_val = DEFAULT_SLEEP

if "screen_val" not in st.session_state:
    st.session_state.screen_val = DEFAULT_SCREEN

if "role" not in st.session_state:
    st.session_state.role = "User"

if "visual_chat" not in st.session_state:
    st.session_state.visual_chat = []

if "recommendation_log" not in st.session_state:
    st.session_state.recommendation_log = []

if "stress" not in st.session_state:
    st.session_state.stress = DEFAULT_STRESS

if "prev_stress" not in st.session_state:
    st.session_state.prev_stress = DEFAULT_STRESS


# --- SIDEBAR ---
with st.sidebar:
    st.header("📱 Simulated OS Data")

    sleep = st.slider("Last Night's Sleep (Hrs)", 0.0, 12.0, key="sleep_val")
    screen = st.slider("Screen Time Today (Hrs)", 0.0, 14.0, key="screen_val")

    st.divider()
    st.header("📊 Autonomous Forecast")

    f_data = st.session_state.logic.get_forecast(
        sleep,
        screen,
        st.session_state.stress,
        st.session_state.prev_stress
    )

    c1, c2 = st.columns(2)
    c1.metric("User Risk", f"{f_data['user']:.0%}")
    c2.metric("Peer Risk", f"{f_data['peer']:.0%}")

    if f_data["user"] >= 0.80:
        st.warning("⚠️ User risk is high. Consider support, rest, and reduced exposure.")
    elif f_data["user"] >= 0.55:
        st.info("ℹ️ User risk is rising. A short break may help.")

    if f_data["peer"] >= 0.70:
        st.error("🚫 Peer aggression risk is high. Blocking or muting is recommended.")
    elif f_data["peer"] >= 0.45:
        st.warning("⚠️ Peer tension is rising.")

    st.divider()
    st.header("🧬 Internal Bio-State")

    fragility = st.session_state.logic.get_fragility(
        sleep,
        st.session_state.stress,
        screen
    )

    st.metric("Biological Fragility", f"{fragility:.1%}")
    st.write(f"Inferred Stress Level: **{st.session_state.stress:.1f}/10**")

    st.divider()

    if st.button("⏪ Undo Learning"):
        st.session_state.reset_trigger = "clear"
        st.session_state.logic.state = copy.deepcopy(st.session_state.initial_brain)
        st.session_state.logic.save_state()
        st.session_state.role = "User"
        st.session_state.recommendation_log = []
        st.rerun()

    if st.button("🧹 Clear Chat"):
        st.session_state.reset_trigger = "clear"
        st.rerun()

    if st.button("🚨 FACTORY RESET", type="primary"):
        st.session_state.reset_trigger = "factory"
        st.rerun()


# --- MAIN ---
st.title("🛡️ Sentinel Autonomous Monitor")

if st.session_state.stress > 8.5:
    st.error("🚨 **CRITICAL STRESS DETECTED**")

if f_data["peer"] > 0.70:
    st.warning("🛡️ **AGGRESSION DETECTED:** Recommendation to block this peer.")

if f_data["user"] > 0.80:
    st.warning("🧠 **HIGH USER VULNERABILITY:** Reduce exposure and take a break.")

chat_box = st.container(height=450)
with chat_box:
    for m in st.session_state.visual_chat:
        border = "red" if (m["role"] == "Peer" and m["score"] >= 0.70) or (m["role"] == "User" and m["score"] >= 0.80) else (
            "orange" if m["score"] >= 0.45 else "green"
        )

        st.markdown(
            f'''
            <div style="
                background-color: rgba(255,255,255,0.05);
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 10px;
                border-left: 5px solid {border};
            ">
                <div><strong>{m["role"]}:</strong> {m["text"]}</div>
                <div style="margin-top:6px; font-size:0.9em; opacity:0.9;">
                    {risk_bar(m["role"], m["score"])}
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

role = st.radio(
    "Active Source:",
    ["User", "Peer"],
    horizontal=True,
    key="role"
)


def handle_input():
    val = st.session_state.chat_input
    if val:
        score = st.session_state.logic.get_contextual_score(val)
        role = st.session_state.role

        pred_check = f_data["user"] if role == "User" else f_data["peer"]
        st.session_state.logic.update_learning(role, score, pred_check)

        st.session_state.logic.log_message(role, val, score)
        st.session_state.visual_chat.append({
            "role": role,
            "text": val,
            "score": score
        })

        analysis = st.session_state.logic.last_analysis
        pattern_label = analysis.get("pattern_label", "neutral_reply")
        pattern_risk = float(analysis.get("pattern_risk", 0.0))

        rec = build_recommendation(role, score, pattern_label, pattern_risk)
        st.session_state.recommendation_log.insert(0, {
            "role": role,
            "message": val,
            "recommendation": rec,
            "score": score
        })
        st.session_state.recommendation_log = st.session_state.recommendation_log[:12]

        # Screen time raises reactivity, but modestly
        fatigue_factor = 1.0 + (max(0, st.session_state.screen_val - 4.0) * 0.08)

        fragility = st.session_state.logic.get_fragility(
            st.session_state.sleep_val,
            st.session_state.stress,
            st.session_state.screen_val
        )

        st.session_state.prev_stress = st.session_state.stress

        if role == "User":
            if score > 0.80:
                jump = 4.2
            elif score >= 0.45:
                jump = (score * 2.8) + 0.15
            elif score >= 0.25:
                jump = score * 1.5
            else:
                jump = -0.10

            st.session_state.stress = min(
                max(st.session_state.stress + (jump * fatigue_factor), 1.0),
                10.0
            )
        else:
            if score > 0.70:
                peer_impact = score * 6.5 * fragility * fatigue_factor
            elif score >= 0.45:
                peer_impact = score * 4.0 * fragility * fatigue_factor
            elif score >= 0.15:
                peer_impact = score * 2.0 * fragility * fatigue_factor
            else:
                peer_impact = -0.15

            st.session_state.stress = min(
                max(st.session_state.stress + peer_impact, 1.0),
                10.0
            )

        st.session_state.chat_input = ""


st.text_input("Enter message:", key="chat_input", on_change=handle_input)

st.subheader("Recommendation log")
log_box = st.container(height=220)
with log_box:
    if st.session_state.recommendation_log:
        for item in st.session_state.recommendation_log:
            st.markdown(
                f"""
                <div style="
                    background-color: rgba(255,255,255,0.05);
                    padding: 10px;
                    border-radius: 10px;
                    margin-bottom: 8px;
                    border-left: 5px solid {'#d97706' if item["role"] == "Peer" else '#2563eb'};
                ">
                    <div><strong>{item["role"]}</strong> • Risk {item["score"]:.0%}</div>
                    <div style="margin-top:4px;">{item["recommendation"]}</div>
                    <div style="margin-top:4px; opacity:0.8; font-size:0.9em;">{item["message"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.caption("No recommendations yet.")