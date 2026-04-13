import json, os, pandas as pd, numpy as np, pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline


class SentinelLogic:
    def __init__(self, history_file="chat_history.json"):
        self.history_file = history_file
        self.nlp = load_model('sentinel_nlp.keras')
        with open('tokenizer.pkl', 'rb') as f: self.tokenizer = pickle.load(f)
        with open('resilience.pkl', 'rb') as f: self.resilience = pickle.load(f)

        # Public Models (No Hardcoding)
        self.toxic_pipe = pipeline("text-classification", model="unitary/toxic-bert")
        self.emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                                     top_k=None)

        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f: json.dump([], f)

    def get_contextual_score(self, text):
        t_low = text.lower().strip()

        # 1. Message-Level Inference
        tox_res = self.toxic_pipe(t_low)[0]
        tox_score = tox_res['score'] if tox_res['label'] != 'neutral' else 0.0

        emo_res = self.emotion_pipe(t_low)[0]
        emo_scores = {e['label']: e['score'] for e in emo_res}
        # Detect Sadness, Fear, or Anger
        vibe_score = max(emo_scores.get('sadness', 0), emo_scores.get('fear', 0), emo_scores.get('anger', 0))

        # 2. Conversation-Level Memory (THE BRAIN)
        with open(self.history_file, 'r') as f:
            history = json.load(f)

        if not history:
            return max(tox_score, vibe_score)

        # Calculate "Conversation Momentum"
        # We look at the average risk of the last 4 messages
        past_risks = [m['score'] for m in history[-4:]]
        momentum = sum(past_risks) / len(past_risks)

        # 3. Decision Engine
        # If the conversation is already "Hot" (High momentum),
        # even a neutral message is treated as a 50% risk minimum.
        if momentum > 0.6:
            # The AI "remembers" the crisis and stays alert
            final_score = max(tox_score, vibe_score, momentum * 0.9)
        else:
            final_score = max(tox_score, vibe_score)

        return final_score

    def log_message(self, role, text, score):
        with open(self.history_file, 'r') as f:
            history = json.load(f)
        history.append({"role": role, "text": text, "score": score})
        with open(self.history_file, 'w') as f:
            json.dump(history[-15:], f)  # Increased memory window

    def get_fragility(self, sleep, stress):
        df = pd.DataFrame([[sleep, stress]], columns=["sleep_hours", "stress_level"])
        return np.clip(float(self.resilience.predict(df)[0]), 0, 1)

    def get_forecast(self):
        with open(self.history_file, 'r') as f:
            history = json.load(f)

        # If the chat is just starting, we return neutral "placeholders"
        if len(history) < 3:
            return 0.0, "Gathering Data...", 0.50  # Added the 0.50 (Confidence) here

        # 1. Calculate the Trend (Slope)
        recent_scores = [m['score'] for m in history[-5:]]
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)

        # Linear regression to find the trend
        slope, intercept = np.polyfit(x, y, 1)

        # 2. Predict the Next Score
        prediction = np.clip(slope * (len(recent_scores)) + intercept, 0, 1)

        # 3. Calculate Confidence (how steady the trend is)
        variance = np.var(recent_scores)
        confidence = np.clip(1.0 - variance, 0.5, 0.99)

        # 4. Status Mapping
        if prediction > 0.75:
            status = "High Risk"
        elif slope > 0.1:
            status = "Rising"
        elif slope < -0.1:
            status = "Falling"
        else:
            status = "Stable"

        # THE FIX: This line MUST have 3 values to match your app.py
        return float(prediction), status, float(confidence)