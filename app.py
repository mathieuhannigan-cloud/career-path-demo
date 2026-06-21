import streamlit as st
import numpy as np
import pandas as pd
import pickle

# =====================================================
# CLASS DEFINITION (باید دقیقاً مثل مدل pickle شده باشه)
# =====================================================
class SugenoNeuroFuzzy:
    def __init__(self, n_features, n_rules, n_classes, centers):
        self.centers = centers.copy()
        self.sigmas = np.full((n_rules, n_features), 0.25)
        self.consequents = np.zeros((n_rules, n_classes, n_features + 1))
        self.n_features = n_features
        self.n_rules = n_rules
        self.n_classes = n_classes

    def forward(self, X):
        batch = X.shape[0]
        diff = X[:, None, :] - self.centers[None, :, :]
        mu = np.exp(-0.5 * (diff / self.sigmas[None, :, :]) ** 2)
        firing = np.prod(mu, axis=2)
        firing_norm = firing / (firing.sum(axis=1, keepdims=True) + 1e-8)
        X_aug = np.hstack([X, np.ones((batch, 1))])
        rule_outputs = np.einsum('bf,rcf->brc', X_aug, self.consequents)
        logits = np.sum(firing_norm[:, :, None] * rule_outputs, axis=1)
        return logits, firing_norm, mu, firing_norm   # ۴ مقدار برمی‌گرداند

    def predict_proba(self, X):
        logits, _, _, _ = self.forward(X)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / (exp.sum(axis=1, keepdims=True) + 1e-8)

# =====================================================
# LOAD
# =====================================================
model = pickle.load(open("neuro_fuzzy_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ... (بقیه کد UI مثل قبل بمونه)

# =====================================================
# PREDICTION با دیباگ کامل
# =====================================================
if st.button("Predict Career Path"):
    X_raw = np.array(values).reshape(1, -1)
    
    st.write("**ورودی خام:**", X_raw[0])
    
    X_scaled = scaler.transform(X_raw)
    st.write("**ورودی بعد از Scale:**", np.round(X_scaled[0], 4))
    
    logits, firing_norm, mu, _ = model.forward(X_scaled)
    
    st.write("**Firing strengths (قوت قوانین):**", np.round(firing_norm[0], 6))
    st.write("**Max Firing:**", firing_norm[0].max())
    st.write("**Logits:**", np.round(logits[0], 4))
    
    probs = model.predict_proba(X_scaled)
    
    pred_idx = np.argmax(probs)
    career = label_encoder.inverse_transform([pred_idx])[0]

    st.success(f"**پیشنهاد:** {career}")
    
    df = pd.DataFrame({
        "Career Path": label_encoder.classes_,
        "Probability (%)": np.round(probs[0] * 100, 2)
    })
    st.bar_chart(df.set_index("Career Path"))
    st.dataframe(df)