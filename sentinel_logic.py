import json
import os
import pickle

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from transformers import pipeline


class SentinelLogic:
    def __init__(self, history_file="chat_history.json"):
        self.history_file = history_file
        self.state_file = "model_state.json"
        self.last_analysis = {
            "hf_toxicity": 0.0,
            "local_toxicity": 0.0,
            "toxicity_severity": 0.0,
            "emotion_risk": 0.0,
            "cyber_type": "not_cyberbullying",
            "cyber_confidence": 0.0,
            "mental_state_risk": 0.0,
            "conversation_risk": 0.0,
            "pattern_label": "neutral_reply",
            "pattern_confidence": 0.0,
            "pattern_risk": 0.0,
            "forecast_pattern_risk": 0.0,
            "final_score": 0.0,
        }

        self.nlp = load_model("sentinel_nlp.keras")
        with open("tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)
        with open("resilience.pkl", "rb") as f:
            self.resilience = pickle.load(f)
        with open("mental_model.pkl", "rb") as f:
            self.mental_model = pickle.load(f)

        self.toxicity_model = self._load_pickle("toxicity_model.pkl")
        self.toxicity_severity_model = self._load_pickle("toxicity_severity_model.pkl")
        self.cyberbully_model = self._load_pickle("cyberbully_model.pkl")

        try:
            self.toxic_pipe = pipeline("text-classification", model="unitary/toxic-bert")
        except Exception:
            self.toxic_pipe = None

        try:
            self.emotion_pipe = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
        except Exception:
            self.emotion_pipe = None

        try:
            self.pattern_pipe = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
        except Exception:
            self.pattern_pipe = None

        self._init_files()
        self.load_state()

    def _load_pickle(self, path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _init_files(self, force_reset=False):
        if force_reset or not os.path.exists(self.history_file):
            with open(self.history_file, "w") as f:
                json.dump([], f)

        if force_reset or not os.path.exists(self.state_file):
            with open(self.state_file, "w") as f:
                json.dump({"u_sens": 1.0, "p_sens": 1.0, "treats": 0}, f)

    def load_state(self):
        with open(self.state_file, "r") as f:
            self.state = json.load(f)

    def save_state(self, custom_state=None):
        target = custom_state if custom_state else self.state
        with open(self.state_file, "w") as f:
            json.dump(target, f)
        self.state = target

    def get_auto_metrics(self):
        with open(self.history_file, "r") as f:
            history = json.load(f)

        chat_volume = min(len(history), 50)
        peer_msgs = [m["score"] for m in history if m["role"] == "Peer"]
        peer_support = 1.0 - np.mean(peer_msgs) if peer_msgs else 0.5
        return chat_volume, peer_support

    def get_forecast(self, sleep, screen, current_stress, prev_stress):
        with open(self.history_file, "r") as f:
            history = json.load(f)

        chat_volume, peer_support = self.get_auto_metrics()

        p_risks = [m["score"] for m in history if m["role"] == "Peer"][-5:]
        if not p_risks:
            pred_p = 0.15 * self.state.get("p_sens", 1.0)
        else:
            p_slope = np.polyfit(np.arange(len(p_risks)), p_risks, 1)[0] if len(p_risks) > 1 else 0
            pred_p = (p_risks[-1] + p_slope) * self.state.get("p_sens", 1.0)

        if p_risks:
            latest_actual = p_risks[-1]
            pred_p = min(pred_p, latest_actual + 0.25)

        df_rf = pd.DataFrame([{
            "sleep_hours": sleep,
            "screen_time_hours": screen,
            "chat_volume": chat_volume,
            "peer_support_score": peer_support,
            "internal_stress": current_stress,
            "stress_trend": current_stress - prev_stress
        }])

        mental_risk_prob = self.mental_model.predict_proba(df_rf)[0][1]

        screen_boost = max(0, screen - 4.0) * 0.008
        sleep_penalty = (max(0, 7.0 - sleep) * 0.018) + (max(0, sleep - 9.0) * 0.004)
        adjusted_mental_risk = float(np.clip(mental_risk_prob + screen_boost + sleep_penalty, 0, 1))

        u_risks = [m["score"] for m in history if m["role"] == "User"][-5:]
        sentiment_momentum = np.mean(u_risks) if u_risks else 0.5

        pattern_summary = self._forecast_pattern_summary(history)

        pattern_boost = (
                0.12 * pattern_summary["peak"] +
                0.18 * pattern_summary["last"] +
                0.08 * pattern_summary["avg"]
        )

        # CHANGE: User risk is influenced by User's actions
        if pattern_summary["user_self_harm"]:
            pattern_boost = max(pattern_boost, 0.86)
        elif pattern_summary["affirming_harm"] or pattern_summary["continuing_harm"]:
            pattern_boost = max(pattern_boost, 0.68)
        elif pattern_summary["user_escalation"] and pattern_summary["peak"] >= 0.60:
            pattern_boost = max(pattern_boost, 0.60)

        pred_u = (
                0.62 * adjusted_mental_risk +
                0.10 * sentiment_momentum +
                0.28 * pattern_boost
        )
        pred_u = pred_u * self.state.get("u_sens", 1.0)

        # CHANGE: Peer forecast is strictly influenced by Peer's actions
        if pattern_summary["peer_escalation"] and pattern_summary["peak"] >= 0.55:
            pred_p = max(pred_p, 0.68 + 0.12 * pattern_summary["peak"])
        if pattern_summary["peer_self_harm"]:
            pred_p = max(pred_p, 0.82)

        self.last_analysis["mental_state_risk"] = float(adjusted_mental_risk)
        self.last_analysis["conversation_risk"] = float(pred_p)
        self.last_analysis["forecast_pattern_risk"] = float(pattern_summary["peak"])
        self.last_analysis["final_score"] = float(max(pred_u, pred_p))

        return {
            "user": float(np.clip(pred_u, 0, 1)),
            "peer": float(np.clip(pred_p, 0, 1)),
            "status": "High Alert" if pred_u > 0.75 or pred_p > 0.75 else "Stable"
        }

    def update_learning(self, role, actual, predicted):
        error = actual - predicted
        key = "u_sens" if role == "User" else "p_sens"
        self.state[key] = np.clip(self.state.get(key, 1.0) + (error * 0.05), 0.7, 1.8)
        self.save_state()

    def _hf_toxicity_score(self, text):
        if self.toxic_pipe is None:
            return 0.0
        try:
            out = self.toxic_pipe(text)
            if isinstance(out, list) and out:
                item = out[0]
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                if "toxic" in label and "non" not in label:
                    return score
                return 0.0
        except Exception:
            return 0.0
        return 0.0

    def _local_toxicity_score(self, text):
        if self.toxicity_model is None:
            return 0.0
        try:
            probs = self.toxicity_model.predict_proba([text])[0]
            if len(probs) == 2:
                return float(probs[1])
            return float(np.max(probs))
        except Exception:
            return 0.0

    def _toxicity_severity_score(self, text):
        if self.toxicity_severity_model is None:
            return 0.0
        try:
            sev = float(self.toxicity_severity_model.predict([text])[0])
            return float(np.clip(sev / 14.0, 0, 1))
        except Exception:
            return 0.0

    def _emotion_risk_score(self, text):
        if self.emotion_pipe is None:
            return 0.0
        try:
            res = self.emotion_pipe(text)
            if isinstance(res, list) and res and isinstance(res[0], list):
                res = res[0]
            emo_scores = {str(e["label"]).lower(): float(e["score"]) for e in res}
            return max(
                emo_scores.get("sadness", 0.0),
                emo_scores.get("fear", 0.0),
                emo_scores.get("anger", 0.0),
            )
        except Exception:
            return 0.0

    def _cyberbully_prediction(self, text):
        if self.cyberbully_model is None:
            return "not_cyberbullying", 0.0
        try:
            label = str(self.cyberbully_model.predict([text])[0]).strip().lower().replace(" ", "_")
            conf = 0.0
            if hasattr(self.cyberbully_model, "predict_proba"):
                probs = self.cyberbully_model.predict_proba([text])[0]
                conf = float(np.max(probs))
            return label, conf
        except Exception:
            return "not_cyberbullying", 0.0

    def _build_pattern_context(self, text, history):
        parts = []
        for m in history[-5:]:
            role = str(m.get("role", "Unknown")).strip()
            msg = str(m.get("text", "")).strip()
            if msg:
                parts.append(f"{role}: {msg}")
        parts.append(f"Current: {text.strip()}")
        return "\n".join(parts)[-1400:]

    def _pattern_fallback(self):
        return "neutral_reply", 0.0, 0.0

    def _pattern_zero_shot(self, text, history):
        candidate_labels = [
            "self_harm_intent",
            "peer_escalation",
            "affirming_harm",
            "continuing_harm",
            "supportive_deescalation",
            "neutral_reply",
        ]

        if self.pattern_pipe is None:
            return self._pattern_fallback()

        context = self._build_pattern_context(text, history)
        solo_text = text.strip()

        def run_zero_shot(sample_text):
            result = self.pattern_pipe(
                sample_text,
                candidate_labels=candidate_labels,
                hypothesis_template="This conversation is {}."
            )
            labels = result.get("labels", [])
            scores = result.get("scores", [])
            if not labels or not scores:
                return None
            return str(labels[0]), float(scores[0])

        try:
            context_out = run_zero_shot(context)
            solo_out = run_zero_shot(solo_text)

            if context_out is None and solo_out is None:
                return self._pattern_fallback()
            if context_out is None:
                label, confidence = solo_out
            elif solo_out is None:
                label, confidence = context_out
            else:
                context_label, context_conf = context_out
                solo_label, solo_conf = solo_out

                # CHANGE: If solo message explicitly de-escalates, break the context chain
                if solo_label in ["supportive_deescalation", "neutral_reply"] and solo_conf > 0.35:
                    label, confidence = solo_label, solo_conf
                else:
                    risk_map = {
                        "self_harm_intent": 1.00,
                        "peer_escalation": 0.88,
                        "affirming_harm": 0.82,
                        "continuing_harm": 0.76,
                        "supportive_deescalation": -0.10,
                        "neutral_reply": 0.00,
                    }
                    context_risk = context_conf * risk_map.get(context_label, 0.0)
                    solo_risk = solo_conf * risk_map.get(solo_label, 0.0)

                    if solo_risk > context_risk:
                        label, confidence = solo_label, solo_conf
                    else:
                        label, confidence = context_label, context_conf

        except Exception:
            return self._pattern_fallback()

        risk_weight_map = {
            "self_harm_intent": 1.00,
            "peer_escalation": 0.92,
            "affirming_harm": 0.82,
            "continuing_harm": 0.74,
            "supportive_deescalation": 0.00,
            "neutral_reply": 0.00,
        }

        label_boost_map = {
            "self_harm_intent": 0.22,
            "peer_escalation": 0.18,
            "affirming_harm": 0.14,
            "continuing_harm": 0.10,
            "supportive_deescalation": 0.00,
            "neutral_reply": 0.00,
        }

        pattern_risk = (confidence * risk_weight_map.get(label, 0.0)) + label_boost_map.get(label, 0.0)
        pattern_risk = float(np.clip(pattern_risk, 0, 1))
        return label, confidence, pattern_risk

    def _forecast_pattern_summary(self, history):
        if not history:
            return {
                "peak": 0.0,
                "avg": 0.0,
                "last": 0.0,
                "user_self_harm": False,
                "peer_self_harm": False,
                "peer_escalation": False,
                "user_escalation": False,
                "affirming_harm": False,
                "continuing_harm": False,
            }

        recent = history[-5:]
        scored_windows = []

        for i in range(len(recent)):
            prev = recent[:i]
            current = recent[i]
            text = str(current.get("text", "")).strip()
            if not text:
                continue
            label, conf, risk = self._pattern_zero_shot(text, prev)
            scored_windows.append({
                "label": label,
                "confidence": conf,
                "risk": risk,
                "role": str(current.get("role", "")).strip(),
                "text": text.lower().strip(),
            })

        for i, window in enumerate(scored_windows):
            if window["label"] in ("supportive_deescalation", "neutral_reply"):
                for j in range(i):
                    scored_windows[j]["risk"] *= 0.5

        if not scored_windows:
            return {
                "peak": 0.0, "avg": 0.0, "last": 0.0,
                "user_self_harm": False, "peer_self_harm": False,
                "peer_escalation": False, "user_escalation": False,
                "affirming_harm": False, "continuing_harm": False,
            }

        risks = [x["risk"] for x in scored_windows]
        labels = [x["label"] for x in scored_windows]

        # CHANGE: Tracking risks by specific roles to prevent false blame
        peer_labels = [x["label"] for x in scored_windows if x["role"] == "Peer"]
        user_labels = [x["label"] for x in scored_windows if x["role"] == "User"]

        return {
            "peak": float(max(risks)),
            "avg": float(np.mean(risks)),
            "last": float(risks[-1]),
            "user_self_harm": any(l == "self_harm_intent" for l in user_labels),
            "peer_self_harm": any(l == "self_harm_intent" for l in peer_labels),
            "peer_escalation": any(l == "peer_escalation" for l in peer_labels),
            "user_escalation": any(l == "peer_escalation" for l in user_labels),
            "affirming_harm": any(l == "affirming_harm" for l in labels),
            "continuing_harm": any(l == "continuing_harm" for l in labels),
        }

    def get_contextual_score(self, text):
        t = (text or "").strip()
        if not t:
            self.last_analysis = {
                "hf_toxicity": 0.0,
                "local_toxicity": 0.0,
                "toxicity_severity": 0.0,
                "emotion_risk": 0.0,
                "cyber_type": "not_cyberbullying",
                "cyber_confidence": 0.0,
                "mental_state_risk": 0.0,
                "conversation_risk": 0.0,
                "pattern_label": "neutral_reply",
                "pattern_confidence": 0.0,
                "pattern_risk": 0.0,
                "forecast_pattern_risk": 0.0,
                "final_score": 0.0,
            }
            return 0.0

        t_low = t.lower()

        hf_toxicity = self._hf_toxicity_score(t_low)
        local_toxicity = self._local_toxicity_score(t)
        severity_score = self._toxicity_severity_score(t)
        emotion_risk = self._emotion_risk_score(t_low)
        cyber_type, cyber_conf = self._cyberbully_prediction(t)

        cyber_boost_map = {
            "age": 0.05,
            "ethnicity": 0.08,
            "gender": 0.08,
            "religion": 0.08,
            "other": 0.04,
            "other_type_of_cyberbullying": 0.04,
            "not_cyberbullying": 0.0,
        }
        cyber_boost = cyber_boost_map.get(cyber_type, 0.03) * cyber_conf

        base_risk = (
                0.46 * severity_score +
                0.24 * local_toxicity +
                0.14 * hf_toxicity +
                0.10 * emotion_risk +
                0.06 * cyber_boost
        )

        # CHANGE: Re-introduced a clean hard-override to ensure explicit threats are never missed
        hard_floor = 0.0
        if any(term in t_low for term in ["kill myself", "suicide", "end my life", "want to die"]):
            hard_floor = 0.92
        elif any(term in t_low for term in ["kill you", "kill yourself", "go die", "you should die"]):
            hard_floor = 0.88

        base_risk = max(base_risk, hard_floor)

        with open(self.history_file, "r") as f:
            history = json.load(f)

        pattern_label, pattern_confidence, pattern_risk = self._pattern_zero_shot(t, history)

        if pattern_label in ("neutral_reply", "supportive_deescalation"):
            final_score = base_risk
        else:
            final_score = max(base_risk, pattern_risk)

        final_score = float(np.clip(final_score, 0, 1))

        self.last_analysis = {
            "hf_toxicity": float(hf_toxicity),
            "local_toxicity": float(local_toxicity),
            "toxicity_severity": float(severity_score),
            "emotion_risk": float(emotion_risk),
            "cyber_type": cyber_type,
            "cyber_confidence": float(cyber_conf),
            "mental_state_risk": 0.0,
            "conversation_risk": float(base_risk),
            "pattern_label": pattern_label,
            "pattern_confidence": float(pattern_confidence),
            "pattern_risk": float(pattern_risk),
            "forecast_pattern_risk": 0.0,
            "final_score": final_score,
        }

        return final_score

    def log_message(self, role, text, score):
        with open(self.history_file, "r") as f:
            history = json.load(f)
        history.append({"role": role, "text": text, "score": score})
        with open(self.history_file, "w") as f:
            json.dump(history[-30:], f)

    def get_fragility(self, sleep, stress, screen_time):
        if sleep < 7:
            sleep_mult = 1.0 + ((7 - sleep) ** 2) * 0.010
        elif 7 <= sleep <= 9:
            sleep_mult = 1.0
        else:
            sleep_mult = 1.0 + (sleep - 9) * 0.025

        screen_impact = max(0, screen_time - 4.0) * 0.015

        df = pd.DataFrame([{
            "sleep_hours": sleep,
            "stress_level": stress,
            "screen_time_hours": screen_time
        }])

        base_fragility = float(self.resilience.predict(df)[0])
        total_fragility = (base_fragility * sleep_mult) + screen_impact
        return np.clip(total_fragility, 0, 0.98)