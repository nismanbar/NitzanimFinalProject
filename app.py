
import copy
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from sentinel_logic import SentinelLogic

st.set_page_config(page_title="Sentinel AI V8.5", layout="wide")

DEFAULT_SLEEP = 7.5
DEFAULT_SCREEN = 5.0
DEFAULT_STRESS = 3.0

SUPPORTIVE_PATTERNS = {"supportive reassurance", "supportive de-escalation", "repair / apology"}
USER_HARM_PATTERNS = {"explicit self harm intent", "distress / hopelessness"}
PEER_HARM_PATTERNS = {
    "peer escalation",
    "harmful escalation",
    "affirming harm",
    "continuing harm",
    "explicit self harm intent",
}


def normalize_label(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


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


def build_recommendation(role: str, score: float, analysis: Dict[str, Any]) -> str:
    pattern_label = normalize_label(analysis.get("pattern_label", "neutral casual chat"))
    pattern_risk = float(analysis.get("pattern_risk", 0.0))
    dialog_act_label = normalize_label(analysis.get("dialog_act_label", "neutral reply"))
    dialog_act_risk = float(analysis.get("dialog_act_risk", 0.0))
    dialog_act_conf = float(analysis.get("dialog_act_confidence", 0.0))

    if role == "Peer":
        if pattern_label in SUPPORTIVE_PATTERNS or (
            dialog_act_label in {"inform", "commissive"} and dialog_act_conf >= 0.30 and score < 0.25
        ):
            return "✅ Supportive or repairing message detected. Keep the peer talking calmly."

        if score >= 0.70 or pattern_label in PEER_HARM_PATTERNS or (
            dialog_act_label == "directive" and dialog_act_risk >= 0.30
        ):
            return "🚫 Block / mute peer and report if needed."

        if score >= 0.45 or pattern_risk >= 0.30 or dialog_act_risk >= 0.25:
            return "⚠️ Reduce exposure and monitor the conversation."

        if pattern_label in {"mild frustration", "casual frustration"}:
            return "ℹ️ Mild tension. Monitor, but no immediate action."

        return "✅ No action needed."

    if score >= 0.80 or pattern_label == "explicit self harm intent":
        return "🧠 Check on the user now and involve a trusted adult if needed."

    if pattern_label == "distress / hopelessness" and (score >= 0.30 or dialog_act_risk >= 0.18):
        return "🧠 Check on the user now and involve a trusted adult if needed."

    if pattern_label in SUPPORTIVE_PATTERNS or (
        dialog_act_label == "commissive" and dialog_act_risk >= 0.20 and score < 0.40
    ):
        return "✅ Positive self-support or repair detected. Reinforce the healthy direction."

    if score >= 0.45 or pattern_risk >= 0.18 or dialog_act_risk >= 0.22:
        return "ℹ️ Send a supportive check-in and reduce stressors."

    if pattern_label in {"casual exaggeration", "neutral reply", "neutral casual chat"}:
        return "✅ No intervention needed."

    return "✅ No intervention needed."


def build_score_reason(
    role: str,
    score: float,
    analysis: Dict[str, Any],
    forecast_before: Dict[str, float],
    forecast_after: Dict[str, float],
    stress_before: float,
    stress_after: float,
) -> str:
    label = normalize_label(analysis.get("pattern_label", "neutral casual chat"))
    confidence = float(analysis.get("pattern_confidence", 0.0))
    hf = float(analysis.get("hf_toxicity", 0.0))
    local = float(analysis.get("local_toxicity", 0.0))
    emo = float(analysis.get("emotion_risk", 0.0))
    cyber_type = normalize_label(analysis.get("cyber_type", "not_cyberbullying"))
    cyber_conf = float(analysis.get("cyber_confidence", 0.0))
    dialog_act_label = normalize_label(analysis.get("dialog_act_label", "neutral reply"))
    dialog_act_conf = float(analysis.get("dialog_act_confidence", 0.0))
    dialog_act_risk = float(analysis.get("dialog_act_risk", 0.0))

    parts: List[str] = []

    if label in USER_HARM_PATTERNS:
        parts.append(f"pattern signal: {label}")
    elif label in {"peer escalation", "harmful escalation", "affirming harm", "continuing harm"}:
        parts.append(f"pattern signal: {label}")
    elif label in SUPPORTIVE_PATTERNS:
        parts.append(f"pattern signal: {label}")
    elif label in {"mild frustration", "casual frustration"}:
        parts.append(f"pattern signal: {label}")
    else:
        parts.append("pattern signal: mostly neutral")

    if dialog_act_label in {"directive", "commissive"} and dialog_act_conf >= 0.30:
        parts.append(f"dialog act: {dialog_act_label}")

    tox_max = max(hf, local)
    if tox_max >= 0.45:
        parts.append(f"toxicity models added weight ({tox_max:.0%})")
    elif tox_max >= 0.20:
        parts.append(f"minor toxicity signal ({tox_max:.0%})")

    if emo >= 0.35:
        parts.append(f"negative emotion detected ({emo:.0%})")

    if cyber_type != "not_cyberbullying" and cyber_conf >= 0.35:
        parts.append(f"cyberbullying type hint: {cyber_type}")

    role_key = "user" if role == "User" else "peer"
    delta = float(forecast_after[role_key] - forecast_before[role_key])
    if abs(delta) >= 0.05:
        parts.append(f"forecast shifted by {delta:+.0%}")

    stress_delta = float(stress_after - stress_before)
    if stress_delta <= -0.15:
        parts.append("stress eased")
    elif stress_delta >= 0.15:
        parts.append("stress increased")

    if confidence >= 0.50 and label not in {"neutral casual chat", "neutral reply"}:
        parts.append(f"pattern confidence {confidence:.0%}")

    if dialog_act_risk >= 0.20 and dialog_act_label in {"directive", "commissive"}:
        parts.append(f"dialog act risk {dialog_act_risk:.0%}")

    if not parts:
        return "mostly neutral; low signal from the current message"

    return " • ".join(parts[:5])


def parse_bulk_messages(raw_text: str) -> List[Tuple[str, str]]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return []

    role_aliases = {
        "user": "User",
        "peer": "Peer",
        "assistant": "Peer",
        "system": "Peer",
    }

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or item.get("speaker") or item.get("source") or "").strip().lower()
                text = str(item.get("text") or item.get("message") or item.get("content") or "").strip()
                if role in role_aliases and text:
                    out.append((role_aliases[role], text))
            if out:
                return out
    except Exception:
        pass

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    jsonl_out = []
    for line in lines:
        if line.startswith("{") and line.endswith("}"):
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    role = str(item.get("role") or item.get("speaker") or item.get("source") or "").strip().lower()
                    text = str(item.get("text") or item.get("message") or item.get("content") or "").strip()
                    if role in role_aliases and text:
                        jsonl_out.append((role_aliases[role], text))
            except Exception:
                pass
    if jsonl_out:
        return jsonl_out

    plain_out = []
    pattern = re.compile(r"^(user|peer)\s*:\s*(.+)$", re.IGNORECASE)
    for line in lines:
        m = pattern.match(line)
        if m:
            role = m.group(1).strip().lower()
            text = m.group(2).strip()
            if text:
                plain_out.append((role_aliases[role], text))

    return plain_out


def parse_conversation_suite(raw_text: str) -> List[Dict[str, Any]]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return []

    try:
        parsed = json.loads(raw_text)
    except Exception:
        return []

    if isinstance(parsed, dict) and "conversations" in parsed:
        parsed = parsed["conversations"]

    if not isinstance(parsed, list):
        return []

    conversations = []
    for idx, conv in enumerate(parsed, start=1):
        if not isinstance(conv, dict):
            continue

        name = str(conv.get("name") or conv.get("title") or f"Conversation {idx}").strip()
        messages = conv.get("messages") or conv.get("turns") or []
        if not isinstance(messages, list):
            continue

        cleaned_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("speaker") or "").strip()
            text = str(msg.get("text") or msg.get("message") or msg.get("content") or "").strip()
            if role in {"User", "Peer"} and text:
                cleaned_messages.append({"role": role, "text": text})

        if cleaned_messages:
            conversations.append({
                "name": name,
                "messages": cleaned_messages,
            })

    return conversations


def should_apply_supportive_relief(analysis: Dict[str, Any]) -> bool:
    pattern_label = normalize_label(analysis.get("pattern_label", "neutral casual chat"))
    dialog_act_label = normalize_label(analysis.get("dialog_act_label", "neutral reply"))
    return pattern_label in SUPPORTIVE_PATTERNS or (
        pattern_label in {"neutral casual chat", "neutral reply"} and dialog_act_label in {"inform", "commissive"}
    )


def simulate_message(
    logic: SentinelLogic,
    role: str,
    text: str,
    sleep: float,
    screen: float,
    stress: float,
    prev_stress: float,
) -> Tuple[Dict[str, Any], float, float]:
    text = (text or "").strip()
    if not text:
        return {}, stress, prev_stress

    stress_before = float(stress)
    forecast_before = logic.get_forecast(
        sleep,
        screen,
        stress_before,
        prev_stress,
    )

    score = float(logic.get_contextual_score(text))
    pred_check = forecast_before["user"] if role == "User" else forecast_before["peer"]
    logic.update_learning(role, score, pred_check)
    logic.log_message(role, text, score)

    analysis = logic.last_analysis
    pattern_label = normalize_label(analysis.get("pattern_label", "neutral casual chat"))
    pattern_confidence = float(analysis.get("pattern_confidence", 0.0))
    pattern_risk = float(analysis.get("pattern_risk", 0.0))
    dialog_act_label = normalize_label(analysis.get("dialog_act_label", "neutral reply"))
    dialog_act_confidence = float(analysis.get("dialog_act_confidence", 0.0))
    dialog_act_risk = float(analysis.get("dialog_act_risk", 0.0))

    fatigue_factor = 1.0 + (max(0.0, screen - 4.0) * 0.08)
    fragility = float(logic.get_fragility(sleep, stress_before, screen))

    supportive_relief = 0.0
    if should_apply_supportive_relief(analysis):
        supportive_relief = 0.22 + (0.18 * pattern_confidence) + (0.10 * dialog_act_confidence)
        supportive_relief = float(min(supportive_relief, 0.55))

    if supportive_relief > 0:
        stress_after = min(max(stress_before - supportive_relief, 1.0), 10.0)
    elif role == "User":
        if score >= 0.80 or pattern_label == "explicit self harm intent":
            jump = 4.2
        elif pattern_label == "distress / hopelessness" and (score >= 0.30 or dialog_act_risk >= 0.18):
            jump = 2.6 + (score * 1.2)
        elif score >= 0.45:
            jump = (score * 2.8) + 0.15
        elif score >= 0.22:
            jump = score * 1.4
        else:
            jump = -0.08

        jump *= 1.0 + (0.35 * dialog_act_risk)
        stress_after = min(max(stress_before + (jump * fatigue_factor), 1.0), 10.0)
    else:
        if score >= 0.70 or (dialog_act_label == "directive" and dialog_act_risk >= 0.25):
            peer_impact = score * 6.8 * fragility * (1.0 + 0.65 * dialog_act_risk)
        elif score >= 0.45:
            peer_impact = score * 4.2 * fragility * (1.0 + 0.45 * dialog_act_risk)
        elif score >= 0.15:
            peer_impact = score * 2.0 * fragility * (1.0 + 0.25 * dialog_act_risk)
        else:
            peer_impact = -0.15

        stress_after = min(max(stress_before + peer_impact, 1.0), 10.0)

    prev_stress_after_update = stress_before

    forecast_after = logic.get_forecast(
        sleep,
        screen,
        stress_after,
        prev_stress_after_update,
    )

    recommendation = build_recommendation(role, score, analysis)
    reason = build_score_reason(
        role=role,
        score=score,
        analysis=analysis,
        forecast_before=forecast_before,
        forecast_after=forecast_after,
        stress_before=stress_before,
        stress_after=stress_after,
    )

    result = {
        "role": role,
        "text": text,
        "score": float(score),
        "recommendation": recommendation,
        "reason": reason,
        "pattern_label": pattern_label,
        "pattern_confidence": float(pattern_confidence),
        "pattern_risk": float(pattern_risk),
        "dialog_act_label": dialog_act_label,
        "dialog_act_confidence": float(dialog_act_confidence),
        "dialog_act_risk": float(dialog_act_risk),
        "hf_toxicity": float(analysis.get("hf_toxicity", 0.0)),
        "local_toxicity": float(analysis.get("local_toxicity", 0.0)),
        "emotion_risk": float(analysis.get("emotion_risk", 0.0)),
        "cyber_type": str(analysis.get("cyber_type", "not_cyberbullying")),
        "cyber_confidence": float(analysis.get("cyber_confidence", 0.0)),
        "stress_before": float(stress_before),
        "stress_after": float(stress_after),
        "fragility": float(fragility),
        "forecast_before_user": float(forecast_before["user"]),
        "forecast_before_peer": float(forecast_before["peer"]),
        "forecast_after_user": float(forecast_after["user"]),
        "forecast_after_peer": float(forecast_after["peer"]),
        "forecast_user_delta": float(forecast_after["user"] - forecast_before["user"]),
        "forecast_peer_delta": float(forecast_after["peer"] - forecast_before["peer"]),
        "u_sens": float(logic.state.get("u_sens", 1.0)),
        "p_sens": float(logic.state.get("p_sens", 1.0)),
        "final_score": float(logic.last_analysis.get("final_score", score)),
    }
    return result, float(stress_after), float(prev_stress_after_update)


def reset_runtime_state():
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    st.session_state.visual_chat = []
    st.session_state.recommendation_log = []
    st.session_state.stress = DEFAULT_STRESS
    st.session_state.prev_stress = DEFAULT_STRESS


def reset_model_state():
    if "logic" in st.session_state:
        st.session_state.logic._init_files(force_reset=True)
        st.session_state.logic.load_state()
        st.session_state.initial_brain = copy.deepcopy(st.session_state.logic.state)


def run_conversation_suite(
    conversations: List[Dict[str, Any]],
    reset_between_conversations: bool = True,
    factory_reset_before_batch: bool = False,
) -> Dict[str, Any]:
    tmp_dir = tempfile.TemporaryDirectory(prefix="sentinel_batch_")
    history_path = os.path.join(tmp_dir.name, "chat_history.json")
    state_path = os.path.join(tmp_dir.name, "model_state.json")

    batch_logic = SentinelLogic(history_file=history_path, state_file=state_path)
    if "logic" in st.session_state:
        batch_logic.state = copy.deepcopy(st.session_state.logic.state)
    else:
        batch_logic.state = {"u_sens": 1.0, "p_sens": 1.0, "treats": 0}

    batch_logic._init_files(force_reset=False)
    if factory_reset_before_batch:
        batch_logic._init_files(force_reset=True)
        batch_logic.load_state()

    report = {
        "settings": {
            "reset_between_conversations": reset_between_conversations,
            "factory_reset_before_batch": factory_reset_before_batch,
            "default_sleep": DEFAULT_SLEEP,
            "default_screen": DEFAULT_SCREEN,
            "default_stress": DEFAULT_STRESS,
        },
        "conversations": [],
        "turns": [],
        "final_model_state": {},
        "final_history_tail": [],
    }

    stress = DEFAULT_STRESS
    prev_stress = DEFAULT_STRESS

    for conv_index, conv in enumerate(conversations, start=1):
        if reset_between_conversations:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            stress = DEFAULT_STRESS
            prev_stress = DEFAULT_STRESS

        conv_name = conv.get("name") or f"Conversation {conv_index}"
        conv_rows = []

        for turn_index, msg in enumerate(conv.get("messages", []), start=1):
            role = msg["role"]
            text = msg["text"]

            result, stress, prev_stress = simulate_message(
                batch_logic,
                role,
                text,
                DEFAULT_SLEEP,
                DEFAULT_SCREEN,
                stress,
                prev_stress,
            )
            if not result:
                continue

            row = {
                "conversation_index": conv_index,
                "conversation_name": conv_name,
                "turn_index": turn_index,
                **result,
            }
            conv_rows.append(row)
            report["turns"].append(row)

        if conv_rows:
            max_user = max((r["score"] for r in conv_rows if r["role"] == "User"), default=0.0)
            max_peer = max((r["score"] for r in conv_rows if r["role"] == "Peer"), default=0.0)
            peak_turn = max(conv_rows, key=lambda r: (float(r.get("final_score", 0.0)), float(r.get("score", 0.0))))
            summary = {
                "conversation_index": conv_index,
                "conversation_name": conv_name,
                "message_count": len(conv_rows),
                "max_user_risk": float(max_user),
                "max_peer_risk": float(max_peer),
                "final_stress": float(conv_rows[-1]["stress_after"]),
                "final_forecast_user": float(conv_rows[-1]["forecast_after_user"]),
                "final_forecast_peer": float(conv_rows[-1]["forecast_after_peer"]),
                "final_u_sens": float(conv_rows[-1]["u_sens"]),
                "final_p_sens": float(conv_rows[-1]["p_sens"]),
                "dominant_turn_index": int(peak_turn["turn_index"]),
                "dominant_role": str(peak_turn["role"]),
                "dominant_text": str(peak_turn["text"]),
                "dominant_reason": str(peak_turn["reason"]),
            }
        else:
            summary = {
                "conversation_index": conv_index,
                "conversation_name": conv_name,
                "message_count": 0,
                "max_user_risk": 0.0,
                "max_peer_risk": 0.0,
                "final_stress": float(stress),
                "final_forecast_user": float(batch_logic.get_forecast(
                    DEFAULT_SLEEP,
                    DEFAULT_SCREEN,
                    stress,
                    prev_stress,
                )["user"]),
                "final_forecast_peer": float(batch_logic.get_forecast(
                    DEFAULT_SLEEP,
                    DEFAULT_SCREEN,
                    stress,
                    prev_stress,
                )["peer"]),
                "final_u_sens": float(batch_logic.state.get("u_sens", 1.0)),
                "final_p_sens": float(batch_logic.state.get("p_sens", 1.0)),
                "dominant_turn_index": 0,
                "dominant_role": "",
                "dominant_text": "",
                "dominant_reason": "",
            }

        report["conversations"].append(summary)

    report["final_model_state"] = copy.deepcopy(batch_logic.state)
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            tail = json.load(f)
        report["final_history_tail"] = tail[-10:]
    except Exception:
        report["final_history_tail"] = []

    tmp_dir.cleanup()
    return report


# --- PRE-RENDER RESET HANDLER ---
if "reset_trigger" in st.session_state:
    trigger = st.session_state.reset_trigger

    if trigger == "clear":
        with open("chat_history.json", "w", encoding="utf-8") as f:
            json.dump([], f)
        st.session_state.visual_chat = []
        st.session_state.recommendation_log = []

    elif trigger == "factory":
        if "logic" in st.session_state:
            st.session_state.logic._init_files(force_reset=True)
            st.session_state.logic.load_state()
            st.session_state.initial_brain = copy.deepcopy(st.session_state.logic.state)
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

if "batch_report" not in st.session_state:
    st.session_state.batch_report = None


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
        st.session_state.prev_stress,
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
        screen,
    )

    st.metric("Biological Fragility", f"{fragility:.1%}")
    st.write(f"Inferred Stress Level: **{st.session_state.stress:.1f}/10**")

    st.divider()

    if st.button("⏪ Undo Learning"):
        st.session_state.logic.state = copy.deepcopy(st.session_state.initial_brain)
        st.session_state.logic.save_state()
        st.session_state.role = "User"
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
        border = (
            "red"
            if (m["role"] == "Peer" and m["score"] >= 0.70) or (m["role"] == "User" and m["score"] >= 0.80)
            else ("orange" if m["score"] >= 0.45 else "green")
        )

        st.markdown(
            f"""
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
            """,
            unsafe_allow_html=True,
        )

role = st.radio(
    "Active Source:",
    ["User", "Peer"],
    horizontal=True,
    key="role",
)


def handle_input():
    val = st.session_state.chat_input
    if not val:
        return

    result, new_stress, new_prev_stress = simulate_message(
        st.session_state.logic,
        st.session_state.role,
        val,
        st.session_state.sleep_val,
        st.session_state.screen_val,
        st.session_state.stress,
        st.session_state.prev_stress,
    )

    if not result:
        st.session_state.chat_input = ""
        return

    st.session_state.stress = new_stress
    st.session_state.prev_stress = new_prev_stress

    st.session_state.visual_chat.append({
        "role": result["role"],
        "text": result["text"],
        "score": result["score"],
    })

    new_item = {
        "role": result["role"],
        "message": result["text"],
        "recommendation": result["recommendation"],
        "reason": result["reason"],
        "score": result["score"],
    }
    if not st.session_state.recommendation_log or st.session_state.recommendation_log[0] != new_item:
        st.session_state.recommendation_log.insert(0, new_item)
        st.session_state.recommendation_log = st.session_state.recommendation_log[:12]

    st.session_state.chat_input = ""


st.text_input("Enter message:", key="chat_input", on_change=handle_input)

st.subheader("Bulk import conversations")
with st.expander("Paste a suite or upload a file", expanded=False):
    st.caption(
        "Best format: JSON array or JSON object with a `conversations` field. "
        "Each conversation should contain `name` and `messages`."
    )

    suite_text = st.text_area(
        "Paste test suite here",
        height=220,
        placeholder='[{"name":"Conversation 1","messages":[{"role":"User","text":"hello"},{"role":"Peer","text":"hi"}]}]',
    )

    suite_file = st.file_uploader(
        "Or upload a .json file",
        type=["json"],
    )

    col_a, col_b, col_c = st.columns(3)
    run_batch_clicked = col_a.button("Run batch test")
    factory_before_batch = col_b.checkbox("Factory reset before batch", value=True)
    reset_between_conv = col_c.checkbox("Reset chat memory between conversations", value=True)

    if run_batch_clicked:
        raw = suite_text
        if suite_file is not None:
            raw = suite_file.getvalue().decode("utf-8", errors="ignore")

        conversations = parse_conversation_suite(raw)

        if not conversations:
            st.error("No valid conversations found. Use JSON with `name` and `messages`.")
        else:
            st.session_state.batch_report = run_conversation_suite(
                conversations=conversations,
                reset_between_conversations=reset_between_conv,
                factory_reset_before_batch=factory_before_batch,
            )
            st.success(f"Processed {len(conversations)} conversations.")
            st.rerun()

    if st.session_state.batch_report is not None:
        report = st.session_state.batch_report

        json_export = json.dumps(report, indent=2, ensure_ascii=False)
        turns_df = pd.DataFrame(report.get("turns", []))
        conv_df = pd.DataFrame(report.get("conversations", []))
        state_df = pd.DataFrame([report.get("final_model_state", {})])

        st.download_button(
            "Download full JSON report",
            data=json_export,
            file_name="sentinel_batch_report.json",
            mime="application/json",
        )

        if not turns_df.empty:
            st.download_button(
                "Download turn log CSV",
                data=turns_df.to_csv(index=False).encode("utf-8"),
                file_name="sentinel_turn_log.csv",
                mime="text/csv",
            )

        if not conv_df.empty:
            st.download_button(
                "Download conversation summary CSV",
                data=conv_df.to_csv(index=False).encode("utf-8"),
                file_name="sentinel_conversation_summary.csv",
                mime="text/csv",
            )

        st.download_button(
            "Download model state JSON",
            data=state_df.to_json(orient="records", indent=2),
            file_name="sentinel_model_state.json",
            mime="application/json",
        )

        st.markdown("### Batch summary")
        if not conv_df.empty:
            st.dataframe(conv_df, use_container_width=True)

        st.markdown("### Final model state")
        st.json(report.get("final_model_state", {}))

        with st.expander("Last history tail", expanded=False):
            st.json(report.get("final_history_tail", []))

st.subheader("Recommendation log")
log_box = st.container(height=220)
with log_box:
    if st.session_state.recommendation_log:
        for item in st.session_state.recommendation_log:
            border_color = "#d97706" if item["role"] == "Peer" else "#2563eb"
            st.markdown(
                f"""
                <div style="
                    background-color: rgba(255,255,255,0.05);
                    padding: 10px;
                    border-radius: 10px;
                    margin-bottom: 8px;
                    border-left: 5px solid {border_color};
                ">
                    <div><strong>{item["role"]}</strong> • Risk {item["score"]:.0%}</div>
                    <div style="margin-top:4px;">{item["recommendation"]}</div>
                    <div style="margin-top:4px; opacity:0.82; font-size:0.85em;">{item.get("reason", "")}</div>
                    <div style="margin-top:4px; opacity:0.8; font-size:0.9em;">{item["message"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No recommendations yet.")
