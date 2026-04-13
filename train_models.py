import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor


def build_resilience():
    print("Training High-Sensitivity Contextual Model...")
    np.random.seed(42)
    n = 5000
    sleep = np.random.uniform(0, 16, n)
    stress = np.random.uniform(1, 10, n)

    # Asymmetric Sleep Curve (J-Curve)
    sleep_penalty = []
    for s in sleep:
        if s < 7:
            penalty = ((7 - s) ** 1.4) * 0.1  # Aggressive low-sleep penalty
        elif 7 <= s <= 8.5:
            penalty = 0
        else:
            penalty = (s - 8.5) * 0.05
        sleep_penalty.append(penalty)

    # FRAGILITY LOGIC: Stress now has a parabolic impact on Fragility
    # (Stress of 8+ causes fragility to spike exponentially)
    stress_impact = (stress / 10) ** 2
    vulnerability = (0.6 * stress_impact) + (0.4 * np.array(sleep_penalty))

    df = pd.DataFrame({
        'sleep_hours': sleep,
        'stress_level': stress,
        'target': np.clip(vulnerability, 0, 1)
    })

    model = XGBRegressor(n_estimators=1000, max_depth=7, learning_rate=0.01)
    model.fit(df[['sleep_hours', 'stress_level']], df['target'])

    with open('resilience.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Resilience Model Calibrated for High Sensitivity.")


if __name__ == "__main__":
    build_resilience()