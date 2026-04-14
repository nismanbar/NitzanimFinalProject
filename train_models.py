import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers


def build_resilience():
    print("--- Phase 1: Training Resilience Model (XGBoost) ---")
    np.random.seed(42)
    n = 10000
    sleep = np.random.uniform(0, 16, n)
    stress = np.random.uniform(1, 10, n)

    sleep_multiplier = []
    for s in sleep:
        if s < 7:
            m = 1.0 + ((7 - s) ** 2) * 0.0122
        elif 7 <= s <= 9:
            m = 1.0
        else:
            m = 1.0 + (s - 9) * 0.021
        sleep_multiplier.append(m)

    base_fragility = (stress / 10) ** 1.5
    vulnerability = base_fragility * np.array(sleep_multiplier)

    df = pd.DataFrame({
        'sleep_hours': sleep,
        'stress_level': stress,
        'target': np.clip(vulnerability, 0, 0.95)
    })

    model = XGBRegressor(n_estimators=1000, max_depth=7, learning_rate=0.03)
    model.fit(df[['sleep_hours', 'stress_level']], df['target'])

    with open('resilience.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("✔ resilience.pkl calibrated.")


def build_mental_health_model():
    print("--- Phase 2: Training Autonomous Mental Health Model ---")

    n = 5000
    data = {
        "sleep_hours": np.random.uniform(2, 12, n),
        "screen_time_hours": np.random.uniform(0, 16, n),
        "chat_volume": np.random.randint(0, 50, n),
        "peer_support_score": np.random.uniform(0.0, 1.0, n),
        "internal_stress": np.random.uniform(1, 10, n),
        "stress_trend": np.random.uniform(-2, 2, n)
    }

    # ✅ NEW: Proper U-curve sleep penalty
    sleep_penalty = []
    for s in data["sleep_hours"]:
        if s < 7:
            p = (7 - s) * 1.2      # strong penalty
        elif 7 <= s <= 9:
            p = 0                 # optimal
        else:
            p = (s - 9) * 0.8     # mild penalty
        sleep_penalty.append(p)

    risk_score = (
        np.array(sleep_penalty) +
        (data["screen_time_hours"] * 0.18) +
        (data["internal_stress"] * 0.8) -
        (data["peer_support_score"] * 3.5)
    )

    y = (risk_score > 9).astype(int)

    df = pd.DataFrame(data)
    X = df[[
        "sleep_hours",
        "screen_time_hours",
        "chat_volume",
        "peer_support_score",
        "internal_stress",
        "stress_trend"
    ]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    with open("mental_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✔ mental_model.pkl saved (fixed sleep behavior).")


def build_placeholder_nlp():
    print("--- Phase 3: Generating NLP Placeholder ---")

    if os.path.exists('sentinel_nlp.keras'):
        return

    model = tf.keras.Sequential([
        layers.Input(shape=(100,)),
        layers.Embedding(10000, 16),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.save('sentinel_nlp.keras')

    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump({"word_index": {"placeholder": 1}}, f)

    print("✔ NLP placeholder created.")


if __name__ == "__main__":
    build_resilience()
    build_mental_health_model()
    build_placeholder_nlp()