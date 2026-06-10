"""
VayuDrishti — PM2.5 Forecasting + Health Advisory
===================================================
Uses:  LSTM  +  Gradient Boosting  +  Ensemble
Input: air_quality_nepal_india.csv
Output:
  - Predictions for 1h, 6h, 24h, 72h ahead
  - Confidence score per horizon
  - Health advisory for 6 groups

Run:
    pip install numpy pandas scikit-learn
    python vayudrishti_predict.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════

def load_data(csv_path: str, city: str = "Kathmandu") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["Location"] == city].copy()
    df["Datetime"] = pd.to_datetime(df[["Year","Month","Day","Hour"]])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # ── Feature engineering ──────────────────────────────────
    df["hour_sin"]  = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["Hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["is_weekend"]= df["Datetime"].dt.dayofweek >= 5

    # Lag features (past readings)
    for lag in [1, 3, 6, 12, 24]:
        df[f"pm25_lag{lag}"] = df["PM2.5"].shift(lag)

    # Rolling statistics
    df["pm25_roll6_mean"]  = df["PM2.5"].rolling(6).mean()
    df["pm25_roll6_std"]   = df["PM2.5"].rolling(6).std()
    df["pm25_roll24_mean"] = df["PM2.5"].rolling(24).mean()

    df = df.dropna().reset_index(drop=True)
    print(f"  Loaded {len(df)} hourly rows for {city}")
    return df


# ══════════════════════════════════════════════════════════════
# STEP 2 — LSTM (pure NumPy, no TensorFlow needed)
# ══════════════════════════════════════════════════════════════

class LSTMCell:
    """Single LSTM cell — forward pass."""
    def __init__(self, in_size: int, hidden: int):
        s = 0.08
        self.Wf = np.random.randn(hidden, in_size + hidden) * s; self.bf = np.ones(hidden)
        self.Wi = np.random.randn(hidden, in_size + hidden) * s; self.bi = np.zeros(hidden)
        self.Wo = np.random.randn(hidden, in_size + hidden) * s; self.bo = np.zeros(hidden)
        self.Wg = np.random.randn(hidden, in_size + hidden) * s; self.bg = np.zeros(hidden)

    def forward(self, x, h, c):
        z  = np.concatenate([x, h])
        si = lambda v: 1 / (1 + np.exp(-np.clip(v, -500, 500)))
        f  = si(self.Wf @ z + self.bf)
        i  = si(self.Wi @ z + self.bi)
        o  = si(self.Wo @ z + self.bo)
        g  = np.tanh(self.Wg @ z + self.bg)
        c  = f * c + i * g
        h  = o * np.tanh(c)
        return h, c


class LSTM:
    """Sequence-to-vector LSTM for PM2.5 forecasting."""
    SEQ = 24   # hours of history

    def __init__(self, hidden=48):
        self.hidden   = hidden
        self.cell     = LSTMCell(1, hidden)
        self.Wout     = np.random.randn(1, hidden) * 0.05
        self.bout     = np.zeros(1)
        self.scaler   = MinMaxScaler()
        self._h_last  = np.zeros(hidden)

    def _run(self, seq):
        h = np.zeros(self.hidden)
        c = np.zeros(self.hidden)
        for val in seq:
            h, c = self.cell.forward(np.array([val]), h, c)
        self._h_last = h.copy()
        return float((self.Wout @ h + self.bout).ravel()[0])

    def fit(self, series: np.ndarray, epochs=60, lr=0.004):
        scaled = self.scaler.fit_transform(series.reshape(-1,1)).ravel()
        X, y   = [], []
        for i in range(len(scaled) - self.SEQ - 1):
            X.append(scaled[i : i + self.SEQ])
            y.append(scaled[i + self.SEQ])
        X, y = np.array(X), np.array(y)

        for ep in range(epochs):
            idx = np.random.permutation(len(X))[:300]
            for i in idx:
                pred  = self._run(X[i])
                err   = pred - y[i]
                # Output layer gradient
                self.Wout -= lr * err * self._h_last
                self.bout  -= lr * err
        print("  LSTM training complete")

    def predict_one(self, recent_series: np.ndarray) -> float:
        seq    = self.scaler.transform(recent_series[-self.SEQ:].reshape(-1,1)).ravel()
        scaled = self._run(seq)
        return float(self.scaler.inverse_transform([[scaled]])[0][0])


# ══════════════════════════════════════════════════════════════
# STEP 3 — GRADIENT BOOSTING (sklearn — no xgboost needed)
# ══════════════════════════════════════════════════════════════

FEATURES = [
    "pm25_lag1","pm25_lag3","pm25_lag6","pm25_lag12","pm25_lag24",
    "pm25_roll6_mean","pm25_roll6_std","pm25_roll24_mean",
    "Temperature","Humidity","hour_sin","hour_cos",
    "month_sin","month_cos","is_weekend"
]

def train_gb(df: pd.DataFrame, horizon: int) -> GradientBoostingRegressor:
    """Train one Gradient Boosting model per forecast horizon."""
    df = df.copy()
    df["target"] = df["PM2.5"].shift(-horizon)
    df = df.dropna()
    X  = df[FEATURES].values
    y  = df["target"].values
    gb = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42
    )
    gb.fit(X, y)
    return gb


# ══════════════════════════════════════════════════════════════
# STEP 4 — ENSEMBLE BLENDER
# ══════════════════════════════════════════════════════════════

def ensemble_weights(horizon: int):
    """
    LSTM is better short-term (sequence memory).
    GB is better long-term (feature-driven conditions).
    """
    lstm_w = max(0.25, 0.70 - horizon * 0.006)
    gb_w   = 1.0 - lstm_w
    return lstm_w, gb_w

def ensemble_predict(lstm_pred, gb_pred, horizon):
    w_l, w_g = ensemble_weights(horizon)
    return w_l * lstm_pred + w_g * gb_pred


# ══════════════════════════════════════════════════════════════
# STEP 5 — AQI + HEALTH ADVISORY
# ══════════════════════════════════════════════════════════════

_PM25_BP = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500),
]

def pm25_to_aqi(pm25: float) -> float:
    for cl, ch, il, ih in _PM25_BP:
        if cl <= pm25 <= ch:
            return round(((ih - il) / (ch - cl)) * (pm25 - cl) + il, 1)
    return 500.0

def aqi_category(aqi: float) -> str:
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

ADVISORIES = {
    "Good": {
        "General":    "Air is clean. Enjoy outdoor activities freely.",
        "Children":   "Safe for all outdoor play.",
        "Elderly":    "No restrictions. Enjoy fresh air.",
        "Respiratory":"Air acceptable. No special precautions needed.",
        "Workers":    "No PPE required for outdoor work.",
        "Schools":    "Outdoor PE and sports can proceed normally.",
    },
    "Moderate": {
        "General":    "Air acceptable. Unusually sensitive people should limit long outdoor activity.",
        "Children":   "Generally safe. Asthmatic children reduce heavy outdoor play.",
        "Elderly":    "Mostly fine outside. Sensitive individuals take short breaks.",
        "Respiratory":"Avoid prolonged strenuous outdoor activity.",
        "Workers":    "Monitor comfort during outdoor labour. Take breaks.",
        "Schools":    "Outdoor activities fine. Watch sensitive students.",
    },
    "Unhealthy for Sensitive Groups": {
        "General":    "Sensitive people may experience effects. Reduce prolonged outdoor exertion.",
        "Children":   "Move vigorous play indoors. Limit outdoor time to 30 min.",
        "Elderly":    "Limit time outdoors. Reduce exertion.",
        "Respiratory":"Avoid outdoor exercise. Keep inhaler handy.",
        "Workers":    "Wear N95 mask outdoors. Limit shifts to 2 hours.",
        "Schools":    "Move PE indoors. Shorten recess to 15 min.",
    },
    "Unhealthy": {
        "General":    "⚠️ Everyone may be affected. Limit outdoor time.",
        "Children":   "Keep children indoors. Cancel outdoor events.",
        "Elderly":    "Stay indoors. Close windows.",
        "Respiratory":"No outdoor activity. Call doctor if breathing worsens.",
        "Workers":    "Mandatory N95 mask. Max 1 hour outdoor per shift.",
        "Schools":    "Cancel all outdoor activities. Alert parents.",
    },
    "Very Unhealthy": {
        "General":    "🚨 Serious health risk. Avoid all outdoor activity.",
        "Children":   "Full indoor lockdown.",
        "Elderly":    "Seal windows. Use air purifier if possible.",
        "Respiratory":"Emergency risk. Contact doctor immediately.",
        "Workers":    "Suspend all outdoor work.",
        "Schools":    "School closure strongly recommended.",
    },
    "Hazardous": {
        "General":    "🆘 EMERGENCY. Everyone at risk. Do not go outside.",
        "Children":   "Full lockdown. Seal all windows and doors.",
        "Elderly":    "Seek shelter. Call emergency services if unwell.",
        "Respiratory":"Seek immediate medical attention.",
        "Workers":    "Halt ALL outdoor work immediately.",
        "Schools":    "MANDATORY school closure. Activate emergency protocols.",
    },
}

def get_advisory(aqi: float) -> dict:
    cat = aqi_category(aqi)
    return {"category": cat, "aqi": aqi, "advice": ADVISORIES.get(cat, ADVISORIES["Moderate"])}

def confidence(horizon: int) -> float:
    return round(max(0.45, 0.95 - horizon * 0.006), 2)


# ══════════════════════════════════════════════════════════════
# STEP 6 — EVALUATE ON TEST SET
# ══════════════════════════════════════════════════════════════

def evaluate(model_name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    print(f"    {model_name:20s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")
    return mae, rmse, r2


# ══════════════════════════════════════════════════════════════
# MAIN — RUN EVERYTHING
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    HORIZONS  = [1, 6, 24, 72]   # hours ahead to predict
    CSV_PATH  = "air_quality_nepal_india.csv"
    CITY      = "Kathmandu"

    print("\n" + "="*60)
    print("  VayuDrishti — PM2.5 Forecasting + Health Advisory")
    print("="*60)

    # ── Load data ─────────────────────────────────────────────
    print("\n📂 Loading data...")
    df   = load_data(CSV_PATH, city=CITY)
    n    = len(df)
    SPLIT= int(n * 0.80)          # 80% train, 20% test
    train= df.iloc[:SPLIT]
    test = df.iloc[SPLIT:]
    print(f"  Train: {len(train)} rows  |  Test: {len(test)} rows")

    # ── Train LSTM ────────────────────────────────────────────
    print("\n🧠 Training LSTM...")
    lstm = LSTM(hidden=48)
    lstm.fit(train["PM2.5"].values, epochs=60, lr=0.004)

    # ── Train Gradient Boosting per horizon ───────────────────
    print("\n🌳 Training Gradient Boosting models...")
    gb_models = {}
    for h in HORIZONS:
        gb_models[h] = train_gb(train, horizon=h)
        print(f"  GB trained for {h:2d}h horizon")

    # ── Evaluate on test set ──────────────────────────────────
    print("\n📊 Evaluation on test set:")
    print("-"*60)

    results = {}
    for h in HORIZONS:
        print(f"\n  Horizon: {h}h ahead")
        cut    = LSTM.SEQ + h
        subset = test.iloc[cut:]

        # GB predictions
        gb_preds = gb_models[h].predict(subset[FEATURES].values)

        # LSTM predictions (use 24h rolling window)
        lstm_preds = []
        pm_series  = df["PM2.5"].values
        for idx in subset.index:
            seq = pm_series[max(0, idx - LSTM.SEQ): idx]
            if len(seq) < LSTM.SEQ:
                seq = np.pad(seq, (LSTM.SEQ - len(seq), 0))
            lstm_preds.append(lstm.predict_one(seq))
        lstm_preds = np.array(lstm_preds)

        # True values
        y_true     = subset["PM2.5"].shift(-h).dropna().values
        min_len    = min(len(y_true), len(gb_preds), len(lstm_preds))
        y_true     = y_true[:min_len]
        gb_preds   = gb_preds[:min_len]
        lstm_preds = lstm_preds[:min_len]

        # Ensemble
        w_l, w_g   = ensemble_weights(h)
        ens_preds  = w_l * lstm_preds + w_g * gb_preds

        mae_l, rmse_l, r2_l = evaluate("LSTM",            y_true, lstm_preds)
        mae_g, rmse_g, r2_g = evaluate("GradientBoosting",y_true, gb_preds)
        mae_e, rmse_e, r2_e = evaluate("Ensemble (best)", y_true, ens_preds)

        results[h] = {
            "ensemble_pred_mean": float(np.mean(ens_preds)),
            "mae": mae_e, "rmse": rmse_e, "r2": r2_e
        }

    # ── Generate live forecast + advisory ─────────────────────
    print("\n" + "="*60)
    print("  FORECAST + HEALTH ADVISORY")
    print("  City:", CITY, " | Based on last 24 hours of test data")
    print("="*60)

    # Use last 24 hours as "current" window
    last_row    = test.iloc[-1]
    last_pm25   = test["PM2.5"].values[-LSTM.SEQ:]
    last_feats  = test[FEATURES].iloc[[-1]].values

    for h in HORIZONS:
        lstm_p = lstm.predict_one(last_pm25)
        gb_p   = float(gb_models[h].predict(last_feats)[0])
        final  = max(0, ensemble_predict(lstm_p, gb_p, h))
        aqi    = pm25_to_aqi(final)
        conf   = confidence(h)
        adv    = get_advisory(aqi)

        print(f"\n  ⏱  {h:2d}h ahead  |  PM2.5 = {final:.1f} µg/m³  |  AQI = {aqi:.0f}  |  Confidence = {int(conf*100)}%")
        print(f"  📌 Category: {adv['category']}")
        print(f"  👨 General:     {adv['advice']['General']}")
        print(f"  🧒 Children:    {adv['advice']['Children']}")
        print(f"  👷 Workers:     {adv['advice']['Workers']}")
        print(f"  🏫 Schools:     {adv['advice']['Schools']}")

    print("\n" + "="*60)
    print("  Done. Stay safe. 🌬️")
    print("="*60 + "\n")
