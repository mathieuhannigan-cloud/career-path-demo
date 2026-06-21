import streamlit as st
import numpy as np
import pandas as pd
import pickle

# =====================================================
# CLASS DEFINITION (REQUIRED FOR PICKLE)
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

        mu = np.exp(
            -0.5 * (diff / self.sigmas[None, :, :]) ** 2
        )

        firing = np.prod(mu, axis=2)

        firing_norm = firing / (
            firing.sum(axis=1, keepdims=True) + 1e-8
        )

        X_aug = np.hstack([
            X,
            np.ones((batch, 1))
        ])

        rule_outputs = np.einsum(
            'bf,rcf->brc',
            X_aug,
            self.consequents
        )

        logits = np.sum(
            firing_norm[:, :, None] * rule_outputs,
            axis=1
        )

        return logits, firing_norm, mu, firing_norm

    def predict_proba(self, X):
        logits, _, _, _ = self.forward(X)

        exp = np.exp(
            logits - np.max(logits, axis=1, keepdims=True)
        )

        return exp / (
            exp.sum(axis=1, keepdims=True) + 1e-8
        )

# =====================================================
# LOAD FILES
# =====================================================

model = pickle.load(open("neuro_fuzzy_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# =====================================================
# FEATURES
# =====================================================

features = [
    "Reading Comprehension",
    "Mathematics",
    "Science",
    "Critical Thinking",
    "Active Learning",
    "Monitoring",
    "Coordination",
    "Persuasion",
    "Negotiation",
    "Complex Problem Solving",
    "Operations Analysis",
    "Operations Monitoring",
    "Troubleshooting",
    "Judgment and Decision Making",
    "Systems Analysis",
    "Time Management",
    "Management of Financial Resources",
    "Management of Material Resources",
    "Management of Personnel Resources"
]

# =====================================================
# UI
# =====================================================

st.set_page_config(
    page_title="Career Path Recommendation",
    layout="wide"
)

st.title("🎯 Intelligent Career Path Recommendation System")

st.write(
    "Enter your competency scores and receive a recommended career path."
)

values = []

col1, col2 = st.columns(2)

for i, feature in enumerate(features):

    if i < 10:
        value = col1.slider(
            feature,
            min_value=1,
            max_value=5,
            value=3
        )
    else:
        value = col2.slider(
            feature,
            min_value=1,
            max_value=5,
            value=3
        )

    values.append(value)

# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict Career Path"):
    X_raw = np.array(values).reshape(1, -1)
    
    st.write("**ورودی خام (اسلایدرها):**", X_raw[0])
    
    X_scaled = scaler.transform(X_raw)
    st.write("**ورودی بعد از Scale:**", np.round(X_scaled[0], 4))
    
    # Forward pass
    logits, firing_norm, _, _ = model.forward(X_scaled)
    
    st.write("**Logits (قبل از softmax):**", np.round(logits[0], 4))
    st.write("**Firing strengths (قوت قوانین):**", np.round(firing_norm[0], 4))
    
    probs = model.predict_proba(X_scaled)
    
    pred_idx = np.argmax(probs)
    career = label_encoder.inverse_transform([pred_idx])[0]

    st.success(f"**مسیر شغلی پیشنهادی:** {career}")
    
    df = pd.DataFrame({
        "Career Path": label_encoder.classes_,
        "Probability (%)": np.round(probs[0] * 100, 2)
    })
    
    st.subheader("احتمال هر مسیر")
    st.bar_chart(df.set_index("Career Path"))
    st.dataframe(df)