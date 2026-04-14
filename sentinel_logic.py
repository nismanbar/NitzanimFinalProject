import json, os, pandas as pd, numpy as np, pickle
from tensorflow.keras.models import load_model
from transformers import pipeline


class SentinelLogic:
    def __init__(self, history_file="chat_history.json"):
        self.history_file, self.state_file = history_file, "model_state.json"
        self.nlp = load_model('sentinel_nlp.keras')
        with open('tokenizer.pkl', 'rb') as f: self.tokenizer = pickle.load(f)
        with open('resilience.pkl', 'rb') as f: self.resilience = pickle.load(f)
        with open('mental_model.pkl', 'rb') as f: self.mental_model = pickle.load(f)

        self.toxic_pipe = pipeline("text-classification", model="unitary/toxic-bert")
        self.emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                                     top_k=None)

        self._init_files()
        self.load_state()

    def _init_files(self, force_reset=False):
        if force_reset or not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f: json.dump([], f)
        if force_reset or not os.path.exists(self.state_file):
            # p_sens and u_sens are the "Learned" global sensitivities
            with open(self.state_file, 'w') as f: json.dump({"u_sens": 1.0, "p_sens": 1.0, "treats": 0}, f)

    def load_state(self):
        with open(self.state_file, 'r') as f: self.state = json.load(f)

    def save_state(self, custom_state=None):
        target = custom_state if custom_state else self.state
        with open(self.state_file, 'w') as f: json.dump(target, f)
        self.state = target

    def get_auto_metrics(self):
        with open(self.history_file, 'r') as f: history = json.load(f)
        chat_volume = min(len(history), 50)
        peer_msgs = [m['score'] for m in history if m['role'] == "Peer"]
        peer_support = 1.0 - np.mean(peer_msgs) if peer_msgs else 0.5
        return chat_volume, peer_support

    def get_forecast(self, sleep, screen, current_stress, prev_stress):
        with open(self.history_file, 'r') as f:
            history = json.load(f)
        chat_volume, peer_support = self.get_auto_metrics()

        # 1. PEER PREDICTION
        p_risks = [m['score'] for m in history if m['role'] == "Peer"][-5:]
        # NEW: If no history, default to a neutral 0.5 (50%) adjusted by learned global sensitivity
        if not p_risks:
            pred_p = 0.5 * self.state.get("p_sens", 1.0)
        else:
            p_slope = np.polyfit(np.arange(len(p_risks)), p_risks, 1)[0] if len(p_risks) > 1 else 0
            pred_p = (p_risks[-1] + p_slope) * self.state.get("p_sens", 1.0)

        # 2. USER PREDICTION
        df_rf = pd.DataFrame([{
            "sleep_hours": sleep, "screen_time_hours": screen,
            "chat_volume": chat_volume, "peer_support_score": peer_support,
            "internal_stress": current_stress, "stress_trend": current_stress - prev_stress
        }])

        mental_risk_prob = self.mental_model.predict_proba(df_rf)[0][1]

        # Small deterministic adjustment so screen time never reduces risk
        screen_boost = max(0, screen - 4.0) * 0.015
        sleep_buffer = max(0, sleep - 8.0) * 0.01  # very mild relief for good sleep

        adjusted_mental_risk = mental_risk_prob + screen_boost - sleep_buffer
        adjusted_mental_risk = float(np.clip(adjusted_mental_risk, 0, 1))

        u_risks = [m['score'] for m in history if m['role'] == "User"][-5:]
        sentiment_momentum = np.mean(u_risks) if u_risks else 0.5

        pred_u = (0.6 * adjusted_mental_risk) + (0.4 * sentiment_momentum)
        pred_u = pred_u * self.state.get("u_sens", 1.0)

        return {
            "user": float(np.clip(pred_u, 0, 1)), "peer": float(np.clip(pred_p, 0, 1)),
            "status": "High Alert" if pred_u > 0.75 or pred_p > 0.75 else "Stable"
        }

    def update_learning(self, role, actual, predicted):
        error = actual - predicted
        key = "u_sens" if role == "User" else "p_sens"
        # Global learning stays preserved across chat clears
        self.state[key] = np.clip(self.state.get(key, 1.0) + (error * 0.05), 0.7, 1.8)
        self.save_state()

    def get_contextual_score(self, text):
        t_low = text.lower().strip()
        tox = self.toxic_pipe(t_low)[0]
        tox_score = tox['score'] if tox['label'] != 'neutral' else 0.0
        emo = {e['label']: e['score'] for e in self.emotion_pipe(t_low)[0]}
        # Capture self-harm/high-risk emotions specifically
        risk_score = max(tox_score, emo.get('sadness', 0), emo.get('fear', 0), emo.get('anger', 0))
        return risk_score

    def log_message(self, role, text, score):
        with open(self.history_file, 'r') as f: history = json.load(f)
        history.append({"role": role, "text": text, "score": score})
        with open(self.history_file, 'w') as f: json.dump(history[-30:], f)

    def get_fragility(self, sleep, stress, screen_time):
        # Sleep curve: still U-shaped, but gentler than before
        if sleep < 7:
            # below 7 hours: rising fragility, but not too extreme
            sleep_mult = 1.0 + ((7 - sleep) ** 2) * 0.010
        elif 7 <= sleep <= 9:
            # optimal zone
            sleep_mult = 1.0
        else:
            # above 9 hours: only a mild increase
            sleep_mult = 1.0 + (sleep - 9) * 0.025

        # Screen time should increase fragility a bit more clearly
        # Baseline starts after 4 hours
        screen_impact = max(0, screen_time - 4.0) * 0.03

        # Base calculation from model
        df = pd.DataFrame([[sleep, stress]], columns=["sleep_hours", "stress_level"])
        base_fragility = float(self.resilience.predict(df)[0])

        # Combine
        total_fragility = (base_fragility * sleep_mult) + screen_impact

        return np.clip(total_fragility, 0, 0.98)