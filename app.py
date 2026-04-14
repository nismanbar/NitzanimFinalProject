import streamlit as st
import os, json, copy
from sentinel_logic import SentinelLogic

st.set_page_config(page_title="Sentinel AI V8.5", layout="wide")

# Baseline constants
DEFAULT_SLEEP = 7.5
DEFAULT_SCREEN = 5.0
DEFAULT_STRESS = 3.0

# --- PRE-RENDER RESET HANDLER ---
if 'reset_trigger' in st.session_state:
    trigger = st.session_state.reset_trigger

    if trigger == "clear":
        with open("chat_history.json", 'w') as f:
            json.dump([], f)
        st.session_state.visual_chat = []

    elif trigger == "factory":
        if 'logic' in st.session_state:
            st.session_state.logic._init_files(force_reset=True)
        st.session_state.visual_chat = []

    # ✅ Global resets
    st.session_state.sleep_val = DEFAULT_SLEEP
    st.session_state.screen_val = DEFAULT_SCREEN
    st.session_state.stress = DEFAULT_STRESS
    st.session_state.prev_stress = DEFAULT_STRESS
    st.session_state.role = "User"   # ✅ FIX: reset radio

    del st.session_state.reset_trigger


# --- INITIALIZATION (BEFORE UI RENDER) ---
if 'logic' not in st.session_state:
    st.session_state.logic = SentinelLogic()
    st.session_state.initial_brain = copy.deepcopy(st.session_state.logic.state)

# ✅ FIX: initialize sliders BEFORE sidebar renders
if "sleep_val" not in st.session_state:
    st.session_state.sleep_val = DEFAULT_SLEEP

if "screen_val" not in st.session_state:
    st.session_state.screen_val = DEFAULT_SCREEN

# ✅ FIX: initialize role state
if "role" not in st.session_state:
    st.session_state.role = "User"

if 'visual_chat' not in st.session_state:
    st.session_state.visual_chat = []

if 'stress' not in st.session_state:
    st.session_state.stress = DEFAULT_STRESS

if 'prev_stress' not in st.session_state:
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

    # Buttons
    if st.button("⏪ Undo Learning"):
        st.session_state.logic.state = copy.deepcopy(st.session_state.initial_brain)
        st.session_state.logic.save_state()
        st.session_state.role = "User"   # ✅ FIX
        st.rerun()

    if st.button("🧹 Clear Chat"):
        st.session_state.reset_trigger = "clear"
        st.rerun()

    if st.button("🚨 FACTORY RESET", type="primary"):
        st.session_state.reset_trigger = "factory"
        st.rerun()


# --- MAIN ---
st.title("🛡️ Sentinel Autonomous Monitor")

# Alerts
if st.session_state.stress > 8.5:
    st.error("🚨 **CRITICAL STRESS DETECTED**")

if f_data['peer'] > 0.70:
    st.warning("🛡️ **AGGRESSION DETECTED:** Recommendation to block this peer.")


# --- CHAT DISPLAY ---
chat_box = st.container(height=450)
with chat_box:
    for m in st.session_state.visual_chat:
        border = "red" if m['score'] > 0.75 else "green" if m['role'] == "Peer" else "blue"
        st.markdown(
            f'<div style="background-color:rgba(255,255,255,0.05); padding:10px; border-radius:8px; margin-bottom:8px; border-left: 5px solid {border}"><strong>{m["role"]}:</strong> {m["text"]}</div>',
            unsafe_allow_html=True
        )


# ✅ FIX: controlled radio (state-driven)
role = st.radio(
    "Active Source:",
    ["User", "Peer"],
    horizontal=True,
    key="role"
)


# --- INPUT HANDLER ---
def handle_input():
    val = st.session_state.chat_input
    if val:
        score = st.session_state.logic.get_contextual_score(val)

        # ✅ Use state-controlled role (not UI snapshot)
        role = st.session_state.role

        # Learning
        pred_check = f_data['user'] if role == "User" else f_data['peer']
        st.session_state.logic.update_learning(role, score, pred_check)

        # Logging
        st.session_state.logic.log_message(role, val, score)
        st.session_state.visual_chat.append({
            "role": role,
            "text": val,
            "score": score
        })

        # Fatigue factor
        fatigue_factor = 1.0 + (
            max(0, st.session_state.screen_val - 4.0) * 0.12
        )

        # Real-time fragility
        fragility = st.session_state.logic.get_fragility(
            st.session_state.sleep_val,
            st.session_state.stress,
            st.session_state.screen_val
        )

        st.session_state.prev_stress = st.session_state.stress

        if role == "User":
            jump = (4.0 if score > 0.7 else (score * 3.5) - 0.5)
            st.session_state.stress = min(
                max(
                    st.session_state.stress + (jump * fatigue_factor),
                    1.0
                ),
                10.0
            )
        else:
            peer_impact = (
                score * 8.0 * fragility * fatigue_factor
            ) if score > 0.5 else -0.5

            st.session_state.stress = min(
                max(
                    st.session_state.stress + peer_impact,
                    1.0
                ),
                10.0
            )

        st.session_state.chat_input = ""


st.text_input("Enter message:", key="chat_input", on_change=handle_input)