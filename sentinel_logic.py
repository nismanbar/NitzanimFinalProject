import copy
import json
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class SentinelLogic:
    def __init__(self, history_file: str = "chat_history.json", state_file: str = "model_state.json"):
        self.history_file = history_file
        self.state_file = state_file
        self.config = self._load_config()
        self.weights = self._build_weights()

        self.last_analysis = {
            "hf_toxicity": 0.0,
            "local_toxicity": 0.0,
            "toxicity_severity": 0.0,
            "emotion_risk": 0.0,
            "cyber_type": "not_cyberbullying",
            "cyber_confidence": 0.0,
            "mental_state_risk": 0.0,
            "conversation_risk": 0.0,
            "pattern_label": "neutral casual chat",
            "pattern_confidence": 0.0,
            "pattern_risk": 0.0,
            "dialog_act_label": "neutral reply",
            "dialog_act_confidence": 0.0,
            "dialog_act_risk": 0.0,
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

        self.toxic_pipe = self._load_toxic_pipeline()
        self.emotion_pipe = self._load_emotion_pipeline()
        self.pattern_pipe = self._load_zero_shot_pipeline()
        self.dialog_act_tokenizer, self.dialog_act_model, self.dialog_act_id2label = self._load_dialog_act_model()

        self._init_files()
        self.load_state()

    def _load_config(self) -> dict:
        default = {
            "pattern_models": {
                "zero_shot_model_id": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                "dialog_act_model_id": "diwank/dyda-deberta-pair",
            },
            "risk_weights": {
                "base": {
                    "toxicity_severity": 0.46,
                    "local_toxicity": 0.24,
                    "hf_toxicity": 0.16,
                    "emotion_risk": 0.10,
                    "cyber_boost": 0.06,
                },
                "pattern": {
                    "supportive reassurance": -0.08,
                    "neutral casual chat": 0.00,
                    "casual exaggeration": 0.02,
                    "mild frustration": 0.06,
                    "distress / hopelessness": 0.18,
                    "repair / apology": -0.06,
                    "harmful escalation": 0.74,
                    "explicit self harm intent": 0.95,
                },
                "dialog_act": {
                    "directive": 0.24,
                    "commissive": 0.18,
                    "inform": 0.00,
                    "question": 0.00,
                    "neutral reply": 0.00,
                },
            },
            "thresholds": {
                "user_high": 0.80,
                "peer_high": 0.70,
                "user_check_in": 0.45,
                "peer_concern": 0.45,
            },
        }

        path = "sentinel_config.json"
        if not os.path.exists(path):
            return default

        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            merged = copy.deepcopy(default)
            for key, value in loaded.items():
                if isinstance(value, dict) and key in merged:
                    merged[key].update(value)
                else:
                    merged[key] = value
            return merged
        except Exception:
            return default

    def _build_weights(self) -> dict:
        cfg = self.config.get("risk_weights", {})
        return {
            "base": {
                "toxicity_severity": float(cfg.get("base", {}).get("toxicity_severity", 0.46)),
                "local_toxicity": float(cfg.get("base", {}).get("local_toxicity", 0.24)),
                "hf_toxicity": float(cfg.get("base", {}).get("hf_toxicity", 0.16)),
                "emotion_risk": float(cfg.get("base", {}).get("emotion_risk", 0.10)),
                "cyber_boost": float(cfg.get("base", {}).get("cyber_boost", 0.06)),
            },
            "pattern": cfg.get("pattern", {}),
            "dialog_act": cfg.get("dialog_act", {}),
        }

    def _load_pickle(self, path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_toxic_pipeline(self):
        try:
            return pipeline("text-classification", model="unitary/toxic-bert")
        except Exception:
            return None

    def _load_emotion_pipeline(self):
        try:
            return pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
            )
        except Exception:
            return None

    def _load_zero_shot_pipeline(self):
        model_id = self.config.get("pattern_models", {}).get(
            "zero_shot_model_id",
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        )
        try:
            return pipeline("zero-shot-classification", model=model_id)
        except Exception:
            return None

    def _load_dialog_act_model(self):
        model_id = self.config.get("pattern_models", {}).get(
            "dialog_act_model_id",
            "diwank/dyda-deberta-pair",
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            id2label = getattr(model.config, "id2label", {}) or {}
            return tokenizer, model, id2label
        except Exception:
            return None, None, {}

    def _init_files(self, force_reset: bool = False):
        if force_reset or not os.path.exists(self.history_file):
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump([], f)

        if force_reset or not os.path.exists(self.state_file):
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump({"u_sens": 1.0, "p_sens": 1.0, "treats": 0}, f)

    def load_state(self):
        with open(self.state_file, "r", encoding="utf-8") as f:
            self.state = json.load(f)

    def save_state(self, custom_state=None):
        target = custom_state if custom_state else self.state
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(target, f)
        self.state = target

    def get_auto_metrics(self):
        with open(self.history_file, "r", encoding="utf-8") as f:
            history = json.load(f)

        chat_volume = min(len(history), 50)
        peer_msgs = [m.get("score", 0.0) for m in history if m.get("role") == "Peer"]
        peer_support = 1.0 - float(np.mean(peer_msgs)) if peer_msgs else 0.5
        return chat_volume, float(np.clip(peer_support, 0, 1))

    @staticmethod
    def _normalize_label(label: str) -> str:
        s = str(label or "").strip().lower()
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s)
        s = s.replace("(0)", "").replace("(1)", "").replace("(2)", "").replace("(3)", "").replace("(4)", "")
        return s.strip(" .:-")

    def _run_zero_shot(
        self,
        text: str,
        labels: List[str],
        hypothesis_template: str = "This conversation is {}.",
    ) -> Tuple[str, float, float]:
        if self.pattern_pipe is None or not text.strip():
            return "neutral casual chat", 0.0, 0.0

        try:
            result = self.pattern_pipe(
                text,
                candidate_labels=labels,
                hypothesis_template=hypothesis_template,
            )
            out_labels = result.get("labels", [])
            out_scores = result.get("scores", [])
            if not out_labels or not out_scores:
                return "neutral casual chat", 0.0, 0.0

            top_label = self._normalize_label(out_labels[0])
            top_score = float(out_scores[0])
            second_score = float(out_scores[1]) if len(out_scores) > 1 else 0.0
            margin = max(0.0, top_score - second_score)
            return top_label, top_score, margin
        except Exception:
            return "neutral casual chat", 0.0, 0.0

    def _dialog_act_prediction(self, previous_text: str, current_text: str) -> Tuple[str, float, float]:
        if self.dialog_act_model is None or self.dialog_act_tokenizer is None or not current_text.strip():
            return "neutral reply", 0.0, 0.0

        try:
            inputs = self.dialog_act_tokenizer(
                previous_text or "",
                current_text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self.dialog_act_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]

            best_idx = int(np.argmax(probs))
            best_conf = float(probs[best_idx])
            sorted_probs = np.sort(probs)
            margin = float(best_conf - (sorted_probs[-2] if len(sorted_probs) > 1 else 0.0))

            raw_label = self.dialog_act_id2label.get(best_idx, self.dialog_act_id2label.get(str(best_idx), str(best_idx)))
            label = self._normalize_label(raw_label)

            if "directive" in label:
                label = "directive"
            elif "commissive" in label:
                label = "commissive"
            elif "inform" in label:
                label = "inform"
            elif "question" in label:
                label = "question"
            else:
                label = "neutral reply"

            return label, best_conf, margin
        except Exception:
            return "neutral reply", 0.0, 0.0

    def _score_dialog_act(self, label: str, conf: float, role: str, context: Dict[str, bool]) -> float:
        role = str(role or "").strip().lower()
        label = self._normalize_label(label)

        if label == "directive":
            risk = 0.22 * max(conf, 0.2)
            if role == "peer":
                risk += 0.12 + 0.20 * conf
                if context.get("user_distress_recent", False):
                    risk += 0.12
                if context.get("user_harm_recent", False):
                    risk += 0.08
            else:
                risk += 0.06 + 0.10 * conf
            return float(np.clip(risk, 0, 1))

        if label == "commissive":
            risk = 0.14 * max(conf, 0.2)
            if role == "user":
                risk += 0.08 + 0.12 * conf
                if context.get("peer_harm_recent", False) or context.get("peer_distress_recent", False):
                    risk += 0.14
                if context.get("self_harm_recent", False):
                    risk += 0.10
            else:
                if context.get("user_distress_recent", False) or context.get("supportive_recent", False):
                    risk -= 0.04
                else:
                    risk += 0.02
            return float(np.clip(risk, 0, 1))

        if label == "inform":
            risk = -0.01 if context.get("supportive_recent", False) else 0.0
            return float(np.clip(risk, 0, 1))

        if label == "question":
            return 0.0

        return 0.0

    def _score_stage1(self, label: str, conf: float, margin: float) -> float:
        base_map = {
            "supportive reassurance": -0.08,
            "neutral casual chat": 0.00,
            "casual exaggeration": 0.02,
            "mild frustration": 0.06,
            "distress / hopelessness": 0.12,
            "repair / apology": -0.06,
            "harmful escalation": 0.74,
            "explicit self harm intent": 0.95,
        }

        base = base_map.get(label, 0.0)

        if label in ("harmful escalation", "explicit self harm intent"):
            if conf < 0.32 and margin < 0.08:
                return 0.0
            return max(0.0, base * conf + 0.12)

        if label == "distress / hopelessness":
            if conf < 0.28 and margin < 0.08:
                return 0.0
            return max(0.0, base * conf + 0.02)

        if label == "mild frustration":
            if conf < 0.32:
                return 0.0
            return max(0.0, base * conf)

        if label == "casual exaggeration":
            if conf < 0.33:
                return 0.0
            return max(0.0, base * conf)

        if label in ("supportive reassurance", "repair / apology"):
            if conf < 0.28:
                return 0.0
            return -0.08 if label == "supportive reassurance" else -0.06

        return 0.0

    def _score_stage2(self, label: str, conf: float, margin: float) -> float:
        base_map = {
            "peer escalation": 0.86,
            "affirming harm": 0.78,
            "continuing harm": 0.72,
            "supportive de-escalation": -0.08,
            "repair / apology": -0.06,
            "neutral reply": 0.00,
        }

        base = base_map.get(label, 0.0)

        if label in ("peer escalation", "affirming harm", "continuing harm"):
            if conf < 0.40 and margin < 0.10:
                return 0.0
            return max(0.0, base * conf + 0.12)

        if label in ("supportive de-escalation", "repair / apology"):
            if conf < 0.30:
                return 0.0
            return base

        return 0.0

    def _stage1_pattern(self, text: str, history: List[dict]) -> Tuple[str, float, float]:
        labels = [
            "supportive reassurance",
            "neutral casual chat",
            "casual exaggeration",
            "mild frustration",
            "distress / hopelessness",
            "repair / apology",
            "harmful escalation",
            "explicit self harm intent",
        ]

        solo_label, solo_conf, solo_margin = self._run_zero_shot(text, labels)
        context = self._build_pattern_context(text, history)
        ctx_label, ctx_conf, ctx_margin = self._run_zero_shot(context, labels)

        solo_risk = self._score_stage1(solo_label, solo_conf, solo_margin)
        ctx_risk = self._score_stage1(ctx_label, ctx_conf, ctx_margin)

        benign_labels = {
            "supportive reassurance",
            "neutral casual chat",
            "casual exaggeration",
            "repair / apology",
        }

        if solo_label in benign_labels and solo_conf >= 0.45:
            return solo_label, solo_conf, solo_risk

        if ctx_label in benign_labels and ctx_conf >= 0.45 and ctx_risk <= solo_risk:
            return ctx_label, ctx_conf, ctx_risk

        if solo_risk >= ctx_risk:
            return solo_label, solo_conf, solo_risk
        return ctx_label, ctx_conf, ctx_risk

    def _stage2_pattern(self, text: str, history: List[dict]) -> Tuple[str, float, float]:
        labels = [
            "peer escalation",
            "affirming harm",
            "continuing harm",
            "supportive de-escalation",
            "repair / apology",
            "neutral reply",
        ]

        solo_label, solo_conf, solo_margin = self._run_zero_shot(text, labels)
        context = self._build_pattern_context(text, history)
        ctx_label, ctx_conf, ctx_margin = self._run_zero_shot(context, labels)

        solo_risk = self._score_stage2(solo_label, solo_conf, solo_margin)
        ctx_risk = self._score_stage2(ctx_label, ctx_conf, ctx_margin)

        if solo_risk >= ctx_risk:
            return solo_label, solo_conf, solo_risk
        return ctx_label, ctx_conf, ctx_risk

    def _conversation_context_flags(self, history: List[dict]) -> Dict[str, bool]:
        recent = history[-5:]
        flags = {
            "user_distress_recent": False,
            "peer_distress_recent": False,
            "user_harm_recent": False,
            "peer_harm_recent": False,
            "self_harm_recent": False,
            "supportive_recent": False,
        }

        for item in recent:
            role = str(item.get("role", "")).strip()
            score = float(item.get("score", 0.0))
            pattern_label = self._normalize_label(item.get("pattern_label", "neutral casual chat"))
            dialog_act_label = self._normalize_label(item.get("dialog_act_label", "neutral reply"))

            if pattern_label in ("supportive reassurance", "repair / apology", "supportive de-escalation"):
                flags["supportive_recent"] = True

            if role == "User":
                if score >= 0.40 or pattern_label in ("distress / hopelessness", "harmful escalation", "explicit self harm intent"):
                    flags["user_distress_recent"] = True
                if pattern_label in ("harmful escalation", "explicit self harm intent"):
                    flags["user_harm_recent"] = True
                if pattern_label == "explicit self harm intent" or dialog_act_label == "commissive":
                    flags["self_harm_recent"] = True

            if role == "Peer":
                if score >= 0.35 or pattern_label in ("distress / hopelessness", "harmful escalation", "explicit self harm intent"):
                    flags["peer_distress_recent"] = True
                if pattern_label in ("harmful escalation", "explicit self harm intent"):
                    flags["peer_harm_recent"] = True

            if role == "Peer" and dialog_act_label == "directive" and score >= 0.25:
                flags["peer_harm_recent"] = True

        return flags

    def _history_signal_summary(self, history: List[dict], role_filter: Optional[str] = None) -> Dict[str, float]:
        if role_filter is not None:
            recent = [m for m in history if str(m.get("role", "")).strip() == role_filter][-5:]
        else:
            recent = history[-5:]

        if not recent:
            return {
                "peak": 0.0,
                "avg": 0.0,
                "last": 0.0,
                "weighted_label_signal": 0.0,
                "dialog_signal": 0.0,
                "supportive_recent": False,
                "harmful_seen": False,
                "peer_harm_seen": False,
                "self_harm_seen": False,
                "peer_self_harm_seen": False,
                "directive_seen": False,
                "commissive_seen": False,
                "distress_seen": False,
            }

        label_signal_map = {
            "supportive reassurance": -0.08,
            "neutral casual chat": 0.00,
            "casual exaggeration": 0.02,
            "mild frustration": 0.06,
            "distress / hopelessness": 0.12,
            "repair / apology": -0.06,
            "harmful escalation": 0.46,
            "explicit self harm intent": 0.88,
            "peer escalation": 0.54,
            "affirming harm": 0.40,
            "continuing harm": 0.34,
            "supportive de-escalation": -0.08,
            "neutral reply": 0.00,
        }

        weights = np.linspace(0.6, 1.0, len(recent))
        score_series = []
        label_series = []
        dialog_series = []

        for item in recent:
            score = float(item.get("score", 0.0))
            label = self._normalize_label(item.get("pattern_label", "neutral casual chat"))
            role = str(item.get("role", "")).strip().lower()
            conf = float(item.get("pattern_confidence", 0.0))
            act_label = self._normalize_label(item.get("dialog_act_label", "neutral reply"))
            act_risk = float(item.get("dialog_act_risk", 0.0))
            act_conf = float(item.get("dialog_act_confidence", 0.0))

            signal = label_signal_map.get(label, 0.0)
            if signal != 0:
                signal = signal * max(conf, 0.35)

            dialog_signal = act_risk * max(act_conf, 0.35)

            score_series.append(score)
            label_series.append((label, signal, role))
            dialog_series.append((act_label, dialog_signal, role))

        peak = float(max(score_series))
        avg = float(np.mean(score_series))
        last = float(score_series[-1])
        weighted_label_signal = float(np.average([x[1] for x in label_series], weights=weights))
        dialog_signal = float(np.average([x[1] for x in dialog_series], weights=weights))

        supportive_recent = any(
            lbl in ("supportive reassurance", "supportive de-escalation", "repair / apology")
            for lbl, _, _ in label_series
        )
        harmful_seen = any(
            lbl in ("harmful escalation", "affirming harm", "continuing harm", "explicit self harm intent", "peer escalation")
            for lbl, _, _ in label_series
        )
        peer_harm_seen = any(
            lbl in ("harmful escalation", "peer escalation", "affirming harm", "continuing harm", "explicit self harm intent")
            and role == "peer"
            for lbl, _, role in label_series
        )
        self_harm_seen = any(lbl == "explicit self harm intent" for lbl, _, _ in label_series)
        peer_self_harm_seen = any(lbl == "explicit self harm intent" and role == "peer" for lbl, _, role in label_series)
        directive_seen = any(act_lbl == "directive" and role == "peer" for act_lbl, _, role in dialog_series)
        commissive_seen = any(act_lbl == "commissive" and role == "user" for act_lbl, _, role in dialog_series)
        distress_seen = any(lbl == "distress / hopelessness" for lbl, _, _ in label_series)

        return {
            "peak": peak,
            "avg": avg,
            "last": last,
            "weighted_label_signal": weighted_label_signal,
            "dialog_signal": dialog_signal,
            "supportive_recent": supportive_recent,
            "harmful_seen": harmful_seen,
            "peer_harm_seen": peer_harm_seen,
            "self_harm_seen": self_harm_seen,
            "peer_self_harm_seen": peer_self_harm_seen,
            "directive_seen": directive_seen,
            "commissive_seen": commissive_seen,
            "distress_seen": distress_seen,
        }

    def get_forecast(self, sleep, screen, current_stress, prev_stress):
        with open(self.history_file, "r", encoding="utf-8") as f:
            history = json.load(f)

        chat_volume, peer_support = self.get_auto_metrics()

        peer_scores = [m.get("score", 0.0) for m in history if m.get("role") == "Peer"][-5:]
        if not peer_scores:
            pred_p = 0.08 * self.state.get("p_sens", 1.0)
        else:
            p_slope = np.polyfit(np.arange(len(peer_scores)), peer_scores, 1)[0] if len(peer_scores) > 1 else 0.0
            pred_p = (peer_scores[-1] + p_slope) * self.state.get("p_sens", 1.0)

        if peer_scores:
            latest_actual = peer_scores[-1]
            pred_p = min(pred_p, latest_actual + 0.18)

        df_rf = pd.DataFrame([{
            "sleep_hours": sleep,
            "screen_time_hours": screen,
            "chat_volume": chat_volume,
            "peer_support_score": peer_support,
            "internal_stress": current_stress,
            "stress_trend": current_stress - prev_stress,
        }])

        mental_risk_prob = self.mental_model.predict_proba(df_rf)[0][1]

        screen_boost = max(0, screen - 4.0) * 0.008
        sleep_penalty = (max(0, 7.0 - sleep) * 0.018) + (max(0, sleep - 9.0) * 0.004)
        adjusted_mental_risk = float(np.clip(mental_risk_prob + screen_boost + sleep_penalty, 0, 1))

        user_summary = self._history_signal_summary(history, "User")
        peer_summary = self._history_signal_summary(history, "Peer")
        all_flags = self._conversation_context_flags(history)

        user_recent_scores = [m.get("score", 0.0) for m in history if m.get("role") == "User"][-3:]
        user_momentum = float(np.mean(user_recent_scores)) if user_recent_scores else 0.18

        peer_recent_scores = [m.get("score", 0.0) for m in history if m.get("role") == "Peer"][-3:]
        peer_momentum = float(np.mean(peer_recent_scores)) if peer_recent_scores else 0.08

        user_pattern_boost = (
            0.18 * user_summary["peak"]
            + 0.10 * user_summary["last"]
            + 0.06 * user_summary["avg"]
            + 0.10 * user_summary["weighted_label_signal"]
            + 0.08 * user_summary["dialog_signal"]
        )

        if user_summary["supportive_recent"] or all_flags["supportive_recent"]:
            user_pattern_boost = max(0.0, user_pattern_boost - 0.10)

        if user_summary["self_harm_seen"]:
            user_pattern_boost = max(user_pattern_boost, 0.82)
        elif user_summary["commissive_seen"] and (all_flags["peer_harm_recent"] or all_flags["peer_distress_recent"]):
            user_pattern_boost = max(user_pattern_boost, 0.52)
        elif user_summary["distress_seen"] and all_flags["peer_harm_recent"]:
            user_pattern_boost = max(user_pattern_boost, 0.46)
        elif user_summary["harmful_seen"]:
            user_pattern_boost = max(user_pattern_boost, 0.54)

        peer_pattern_boost = (
            0.18 * peer_summary["peak"]
            + 0.10 * peer_summary["last"]
            + 0.06 * peer_summary["avg"]
            + 0.10 * peer_summary["weighted_label_signal"]
            + 0.10 * peer_summary["dialog_signal"]
        )

        if peer_summary["supportive_recent"]:
            peer_pattern_boost = max(0.0, peer_pattern_boost - 0.08)

        if peer_summary["directive_seen"] and all_flags["user_distress_recent"]:
            peer_pattern_boost = max(peer_pattern_boost, 0.58)
        elif peer_summary["peer_self_harm_seen"]:
            peer_pattern_boost = max(peer_pattern_boost, 0.82)
        elif peer_summary["peer_harm_seen"] or peer_summary["harmful_seen"]:
            peer_pattern_boost = max(peer_pattern_boost, 0.56)

        pred_u = (
            0.66 * adjusted_mental_risk
            + 0.14 * user_momentum
            + 0.20 * user_pattern_boost
        )
        pred_u = pred_u * self.state.get("u_sens", 1.0)

        pred_p = (
            0.52 * pred_p
            + 0.18 * (1.0 - peer_support)
            + 0.30 * peer_pattern_boost
        )
        pred_p = pred_p * self.state.get("p_sens", 1.0)

        self.last_analysis["mental_state_risk"] = float(adjusted_mental_risk)
        self.last_analysis["conversation_risk"] = float(pred_p)
        self.last_analysis["forecast_pattern_risk"] = float(max(user_summary["peak"], peer_summary["peak"]))
        self.last_analysis["final_score"] = float(max(pred_u, pred_p))

        return {
            "user": float(np.clip(pred_u, 0, 1)),
            "peer": float(np.clip(pred_p, 0, 1)),
            "status": "High Alert" if pred_u > 0.75 or pred_p > 0.75 else "Stable",
        }

    def update_learning(self, role, actual, predicted):
        error = actual - predicted
        key = "u_sens" if role == "User" else "p_sens"

        delta = float(np.clip(error * 0.02, -0.04, 0.04))
        self.state[key] = np.clip(self.state.get(key, 1.0) + delta, 0.85, 1.35)
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

    def _safe_pattern_label(self, label: str) -> str:
        return str(label or "neutral casual chat").strip().lower()

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
                "pattern_label": "neutral casual chat",
                "pattern_confidence": 0.0,
                "pattern_risk": 0.0,
                "dialog_act_label": "neutral reply",
                "dialog_act_confidence": 0.0,
                "dialog_act_risk": 0.0,
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
            self.weights["base"]["toxicity_severity"] * severity_score
            + self.weights["base"]["local_toxicity"] * local_toxicity
            + self.weights["base"]["hf_toxicity"] * hf_toxicity
            + self.weights["base"]["emotion_risk"] * emotion_risk
            + self.weights["base"]["cyber_boost"] * cyber_boost
        )

        first_person = bool(re.search(r"\b(i|im|i'm|i am|me|my)\b", t_low))
        distress_hint = 0.0
        if first_person and emotion_risk >= 0.18:
            distress_hint = 0.03 + (0.12 * emotion_risk)

        with open(self.history_file, "r", encoding="utf-8") as f:
            history = json.load(f)

        prev_text = str(history[-1].get("text", "")) if history else ""

        stage1_label, stage1_conf, stage1_risk = self._stage1_pattern(t, history)
        dialog_act_label, dialog_act_conf, _ = self._dialog_act_prediction(prev_text, t)
        context_flags = self._conversation_context_flags(history)
        dialog_act_risk = self._score_dialog_act(
            dialog_act_label,
            dialog_act_conf,
            str(history[-1].get("role", "")) if history else "",
            context_flags,
        )

        benign_labels = {
            "supportive reassurance",
            "neutral casual chat",
            "casual exaggeration",
            "repair / apology",
        }

        supportive_pair = (
            stage1_label in ("supportive reassurance", "repair / apology")
            and dialog_act_label in ("inform", "commissive", "question", "neutral reply")
            and stage1_conf >= 0.35
        )

        pattern_label = stage1_label
        pattern_confidence = stage1_conf
        pattern_risk = stage1_risk

        if stage1_label in benign_labels and stage1_conf >= 0.45:
            final_score = max(base_risk - 0.04, 0.0)
        else:
            if stage1_label in ("harmful escalation", "explicit self harm intent") and stage1_conf >= 0.32:
                stage2_label, stage2_conf, stage2_risk = self._stage2_pattern(t, history)
                candidate_label = stage2_label if stage2_risk >= stage1_risk else stage1_label
                candidate_conf = stage2_conf if stage2_risk >= stage1_risk else stage1_conf
                candidate_risk = max(stage1_risk, stage2_risk)
                pattern_label = candidate_label
                pattern_confidence = candidate_conf
                pattern_risk = candidate_risk

            # Danger should win unless the turn is clearly low-risk.
            final_score = max(
                base_risk + distress_hint + (0.35 * dialog_act_risk),
                pattern_risk,
            )

            # Role-aware dialog-act escalation, especially for coercive peer turns.
            if dialog_act_label == "directive" and dialog_act_conf >= 0.30:
                if str(history[-1].get("role", "")).strip() == "User":
                    if context_flags["user_distress_recent"] or context_flags["peer_harm_recent"]:
                        final_score = max(final_score, 0.46 + 0.18 * dialog_act_conf)
                    else:
                        final_score = max(final_score, 0.34 + 0.14 * dialog_act_conf)
                else:
                    if context_flags["user_distress_recent"]:
                        final_score = max(final_score, 0.56 + 0.22 * dialog_act_conf)
                    else:
                        final_score = max(final_score, 0.44 + 0.18 * dialog_act_conf)

            if dialog_act_label == "commissive" and dialog_act_conf >= 0.30:
                if context_flags["peer_harm_recent"] or context_flags["peer_distress_recent"]:
                    final_score = max(final_score, 0.44 + 0.18 * dialog_act_conf)
                elif supportive_pair:
                    final_score = max(0.0, final_score - 0.06)

            # Pattern-specific minimums.
            if pattern_label == "explicit self harm intent" and pattern_confidence >= 0.30:
                final_score = max(final_score, 0.92)
            elif pattern_label in ("harmful escalation", "peer escalation") and pattern_confidence >= 0.30:
                final_score = max(final_score, 0.82)
            elif pattern_label in ("affirming harm", "continuing harm") and pattern_confidence >= 0.30:
                final_score = max(final_score, 0.70)
            elif pattern_label == "distress / hopelessness":
                final_score = max(final_score, 0.18)
            elif pattern_label == "mild frustration":
                final_score = max(final_score, 0.08)

            # If the message is plainly toxic or emotionally loaded, do not let it sit too low.
            signal_peak = max(severity_score, local_toxicity, hf_toxicity, emotion_risk)
            if signal_peak >= 0.60:
                final_score = max(final_score, 0.32 + (0.42 * signal_peak))
            elif signal_peak >= 0.42:
                final_score = max(final_score, 0.20 + (0.30 * signal_peak))

        if supportive_pair and final_score < 0.35:
            final_score = max(0.0, final_score - 0.04)

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
            "pattern_label": self._safe_pattern_label(pattern_label),
            "pattern_confidence": float(pattern_confidence),
            "pattern_risk": float(pattern_risk),
            "dialog_act_label": str(dialog_act_label),
            "dialog_act_confidence": float(dialog_act_conf),
            "dialog_act_risk": float(dialog_act_risk),
            "forecast_pattern_risk": 0.0,
            "final_score": final_score,
        }

        return final_score

    def log_message(self, role, text, score):
        with open(self.history_file, "r", encoding="utf-8") as f:
            history = json.load(f)

        history.append(
            {
                "role": role,
                "text": text,
                "score": score,
                "pattern_label": self.last_analysis.get("pattern_label", "neutral casual chat"),
                "pattern_confidence": float(self.last_analysis.get("pattern_confidence", 0.0)),
                "pattern_risk": float(self.last_analysis.get("pattern_risk", 0.0)),
                "dialog_act_label": self.last_analysis.get("dialog_act_label", "neutral reply"),
                "dialog_act_confidence": float(self.last_analysis.get("dialog_act_confidence", 0.0)),
                "dialog_act_risk": float(self.last_analysis.get("dialog_act_risk", 0.0)),
            }
        )

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history[-30:], f, ensure_ascii=False)

    def get_fragility(self, sleep, stress, screen_time):
        if sleep < 7:
            sleep_mult = 1.0 + ((7 - sleep) ** 2) * 0.010
        elif 7 <= sleep <= 9:
            sleep_mult = 1.0
        else:
            sleep_mult = 1.0 + (sleep - 9) * 0.025

        screen_impact = max(0, screen_time - 4.0) * 0.015

        df = pd.DataFrame(
            [[sleep, stress, screen_time]],
            columns=["sleep_hours", "stress_level", "screen_time_hours"],
        )

        base_fragility = float(self.resilience.predict(df)[0])
        total_fragility = (base_fragility * sleep_mult) + screen_impact
        return np.clip(total_fragility, 0, 0.98)