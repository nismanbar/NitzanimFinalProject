
import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score

from xgboost import XGBRegressor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def save_json(filename: str, data: dict) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_dataset_report():
    print("\n--- Dataset Report ---")

    mental = load_csv("mental_health.csv")
    toxic = load_csv("toxic_comments.csv")
    cyber = load_csv("cyberbullying.csv")

    mental["date"] = pd.to_datetime(mental["date"])
    mental = mental.sort_values(["student_id", "date"]).copy()

    mental["next_stress"] = mental.groupby("student_id")["stress_level"].shift(-1)
    mental["stress_increase"] = (mental["next_stress"] > mental["stress_level"]).astype(int)
    mental["stress_trend"] = mental.groupby("student_id")["stress_level"].diff()
    mental["mood_drop"] = -mental.groupby("student_id")["mood"].diff()

    mental["fragility_proxy"] = (
        (mental["stress_level"] / 10.0) ** 1.5
        * np.where(
            mental["sleep_hours"] < 7,
            1 + ((7 - mental["sleep_hours"]) ** 2) * 0.010,
            np.where(
                mental["sleep_hours"] <= 9,
                1,
                1 + (mental["sleep_hours"] - 9) * 0.025,
            ),
        )
        + np.maximum(0, mental["screen_time_hours"] - 4.0) * 0.03
    )

    mental_cols = [
        "mood",
        "stress_level",
        "sleep_hours",
        "screen_time_hours",
        "social_interaction_rating",
        "support_feeling",
        "stress_increase",
        "stress_trend",
        "mood_drop",
        "fragility_proxy",
    ]

    mental_corr = mental[mental_cols].corr(numeric_only=True).round(3)
    print("\nMental health correlations:")
    print(mental_corr)

    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    toxic["toxicity_severity"] = (
        toxic["toxic"] * 1
        + toxic["severe_toxic"] * 3
        + toxic["obscene"] * 1
        + toxic["threat"] * 4
        + toxic["insult"] * 2
        + toxic["identity_hate"] * 3
    )

    toxic_corr = toxic[label_cols + ["toxicity_severity"]].corr(numeric_only=True).round(3)
    print("\nToxic comments correlations:")
    print(toxic_corr)

    print("\nCyberbullying class distribution:")
    print(
        cyber["cyberbullying_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts(normalize=True)
        .round(3)
    )

    report = {
        "mental_corr": mental_corr.to_dict(),
        "toxic_corr": toxic_corr.to_dict(),
        "cyber_distribution": (
            cyber["cyberbullying_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts(normalize=True)
            .round(6)
            .to_dict()
        ),
    }

    save_json("dataset_report.json", report)
    print("✔ dataset_report.json saved.")


def build_sentinel_config():
    """
    Runtime configuration shared by train_models.py and the app logic.
    These are calibration defaults, not hard-coded phrases.
    """
    config = {
        "pattern_models": {
            "zero_shot_model_id": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            "dialog_act_model_id": "diwank/dyda-deberta-pair",
        },
        "risk_weights": {
            "base": {
                "toxicity_severity": 0.40,
                "local_toxicity": 0.20,
                "hf_toxicity": 0.12,
                "emotion_risk": 0.08,
                "cyber_boost": 0.05,
            },
            "pattern": {
                "supportive_reassurance": -0.08,
                "neutral_casual_chat": 0.00,
                "casual_exaggeration": 0.02,
                "mild_frustration": 0.06,
                "distress_hopelessness": 0.18,
                "repair_apology": -0.06,
                "harmful_escalation": 0.72,
                "explicit_self_harm_intent": 0.95,
            },
            "dialog_act": {
                "directive": 0.22,
                "commissive": 0.16,
                "inform": 0.00,
                "question": 0.00,
                "dummy": 0.00,
            },
            "forecast": {
                "user_pattern": 0.20,
                "peer_pattern": 0.20,
                "user_momentum": 0.16,
                "peer_pressure": 0.18,
                "supportive_relief": 0.10,
            },
        },
        "thresholds": {
            "user_high": 0.80,
            "peer_high": 0.70,
            "user_check_in": 0.45,
            "peer_concern": 0.45,
        },
    }

    save_json("sentinel_config.json", config)
    print("✔ sentinel_config.json saved.")


def build_resilience():
    print("\n--- Phase 1: Training Resilience Model from REAL mental_health.csv ---")

    mental = load_csv("mental_health.csv")
    mental["date"] = pd.to_datetime(mental["date"])
    mental = mental.sort_values(["student_id", "date"]).copy()

    sleep = mental["sleep_hours"].astype(float).values
    stress = mental["stress_level"].astype(float).values
    screen = mental["screen_time_hours"].astype(float).values

    sleep_multiplier = []
    for s in sleep:
        if s < 7:
            m = 1.0 + ((7 - s) ** 2) * 0.010
        elif 7 <= s <= 9:
            m = 1.0
        else:
            m = 1.0 + (s - 9) * 0.025
        sleep_multiplier.append(m)

    base_fragility = (stress / 10.0) ** 1.5
    screen_impact = np.maximum(0, screen - 4.0) * 0.03
    vulnerability = base_fragility * np.array(sleep_multiplier) + screen_impact

    df = pd.DataFrame({
        "sleep_hours": sleep,
        "stress_level": stress,
        "screen_time_hours": screen,
        "target": np.clip(vulnerability, 0, 0.98),
    })

    X = df[["sleep_hours", "stress_level", "screen_time_hours"]]
    y = df["target"]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X, y)

    with open("resilience.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✔ resilience.pkl calibrated from real mental health data.")


def build_mental_health_model():
    print("\n--- Phase 2: Training Autonomous Mental Health Model ---")

    n = 5000
    data = {
        "sleep_hours": np.random.uniform(2, 12, n),
        "screen_time_hours": np.random.uniform(0, 16, n),
        "chat_volume": np.random.randint(0, 50, n),
        "peer_support_score": np.random.uniform(0.0, 1.0, n),
        "internal_stress": np.random.uniform(1, 10, n),
        "stress_trend": np.random.uniform(-2, 2, n),
    }

    sleep_penalty = []
    for s in data["sleep_hours"]:
        if s < 7:
            p = (7 - s) * 1.2
        elif 7 <= s <= 9:
            p = 0
        else:
            p = (s - 9) * 0.8
        sleep_penalty.append(p)

    risk_score = (
        np.array(sleep_penalty)
        + (data["screen_time_hours"] * 0.18)
        + (data["internal_stress"] * 0.8)
        - (data["peer_support_score"] * 3.5)
    )

    y = (risk_score > 9).astype(int)

    df = pd.DataFrame(data)
    X = df[
        [
            "sleep_hours",
            "screen_time_hours",
            "chat_volume",
            "peer_support_score",
            "internal_stress",
            "stress_trend",
        ]
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Mental model accuracy:", accuracy_score(y_test, y_pred))
    print("Mental model F1:", f1_score(y_test, y_pred))

    with open("mental_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✔ mental_model.pkl saved.")


def build_toxicity_model():
    print("\n--- Phase 3: Training Toxicity Models from toxic_comments.csv ---")

    toxic = load_csv("toxic_comments.csv").copy()
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    toxic["comment_text"] = toxic["comment_text"].fillna("").astype(str)

    toxic["tox_any"] = (toxic[label_cols].sum(axis=1) > 0).astype(int)
    toxic["toxicity_severity"] = (
        toxic["toxic"] * 1
        + toxic["severe_toxic"] * 3
        + toxic["obscene"] * 1
        + toxic["threat"] * 4
        + toxic["insult"] * 2
        + toxic["identity_hate"] * 3
    )

    X_train, X_test, y_train, y_test = train_test_split(
        toxic["comment_text"],
        toxic["tox_any"],
        test_size=0.2,
        random_state=42,
        stratify=toxic["tox_any"],
    )

    toxicity_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            min_df=3,
            max_features=40000,
            ngram_range=(1, 2),
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=1500,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        ))
    ])
    toxicity_model.fit(X_train, y_train)

    y_pred = toxicity_model.predict(X_test)
    print("Toxicity binary accuracy:", accuracy_score(y_test, y_pred))
    print("Toxicity binary F1:", f1_score(y_test, y_pred))

    with open("toxicity_model.pkl", "wb") as f:
        pickle.dump(toxicity_model, f)

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        toxic["comment_text"],
        toxic["toxicity_severity"],
        test_size=0.2,
        random_state=42,
    )

    severity_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            min_df=3,
            max_features=40000,
            ngram_range=(1, 2),
        )),
        ("reg", Ridge(alpha=1.0)),
    ])
    severity_model.fit(X_train_s, y_train_s)

    sev_pred = severity_model.predict(X_test_s)
    print("Toxicity severity MAE:", mean_absolute_error(y_test_s, sev_pred))

    with open("toxicity_severity_model.pkl", "wb") as f:
        pickle.dump(severity_model, f)

    print("✔ toxicity_model.pkl and toxicity_severity_model.pkl saved.")


def build_cyberbullying_model():
    print("\n--- Phase 4: Training Cyberbullying Type Model from cyberbullying.csv ---")

    cyber = load_csv("cyberbullying.csv").copy()
    cyber["tweet_text"] = cyber["tweet_text"].fillna("").astype(str)
    cyber["cyberbullying_type"] = (
        cyber["cyberbullying_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        cyber["tweet_text"],
        cyber["cyberbullying_type"],
        test_size=0.2,
        random_state=42,
        stratify=cyber["cyberbullying_type"],
    )

    cyber_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            min_df=2,
            max_features=30000,
            ngram_range=(1, 2),
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=1500,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        ))
    ])
    cyber_model.fit(X_train, y_train)

    y_pred = cyber_model.predict(X_test)
    print("Cyberbullying type accuracy:", accuracy_score(y_test, y_pred))
    print("Cyberbullying type F1:", f1_score(y_test, y_pred, average="macro"))

    with open("cyberbully_model.pkl", "wb") as f:
        pickle.dump(cyber_model, f)

    print("✔ cyberbully_model.pkl saved.")


def build_placeholder_nlp():
    print("\n--- Phase 5: Generating NLP Placeholder ---")

    if os.path.exists("sentinel_nlp.keras"):
        print("✔ sentinel_nlp.keras already exists.")
        return

    model = tf.keras.Sequential([
        layers.Input(shape=(100,)),
        layers.Embedding(10000, 16),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.save("sentinel_nlp.keras")

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump({"word_index": {"placeholder": 1}}, f)

    print("✔ sentinel_nlp.keras & tokenizer.pkl created.")


if __name__ == "__main__":
    build_dataset_report()
    build_sentinel_config()
    build_resilience()
    build_mental_health_model()
    build_toxicity_model()
    build_cyberbullying_model()
    build_placeholder_nlp()
