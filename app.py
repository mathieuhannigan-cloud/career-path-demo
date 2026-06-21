import streamlit as st
import numpy as np
import pandas as pd
import pickle

# =====================================================
# CLASS DEFINITION
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
        return logits, firing_norm

    def predict_proba(self, X):
        logits, _ = self.forward(X)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / (exp.sum(axis=1, keepdims=True) + 1e-8)

# =====================================================
# LOAD MODEL
# =====================================================
model = pickle.load(open("neuro_fuzzy_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# =====================================================
# فیچرها - نام فارسی زیبا + ترتیب اصلی مدل
# =====================================================
feature_mapping = [
    ("قدرت تحلیل و درک محیط", "Reading Comprehension"),
    ("گوش دادن فعال", "Active Listening"),
    ("مهارت نوشتاری و مکاتبه", "Writing"),
    ("هوش کلامی", "Speaking"),
    ("هوش تحلیلی و محاسباتی", "Mathematics"),
    ("دانش حرفه‌ای و شغلی", "Science"),
    ("تفکر انتقادی", "Critical Thinking"),
    ("یادگیری فعال", "Active Learning"),
    ("راهبردهای یادگیری", "Learning Strategies"),
    ("قدرت نظارت و کنترل", "Monitoring"),
    ("آگاهی اجتماعی", "Social Perceptiveness"),
    ("هماهنگی بین بخشی", "Coordination"),
    ("رهبری تحول‌گرا", "Persuasion"),
    ("مهارت مذاکره و فن بیان", "Negotiation"),
    ("مهارت حل مسائل پیچیده", "Complex Problem Solving"),
    ("تحلیل عملیات", "Operations Analysis"),
    ("مهارت‌های رایانه‌ای و برنامه‌نویسی", "Programming"),
    ("پایش فرایندهای اجرایی", "Operations Monitoring"),
    ("مهارت مسئله‌یابی", "Troubleshooting"),
    ("مهارت تصمیم‌گیری و قضاوت", "Judgment and Decision Making"),
    ("تفکر و تحلیل سیستمی", "Systems Analysis"),
    ("مدیریت زمان", "Time Management"),
    ("مدیریت منابع مالی", "Management of Financial Resources"),
    ("مدیریت تجهیزات و منابع مادی", "Management of Material Resources"),
    ("مدیریت منابع انسانی", "Management of Personnel Resources")
]

# فقط نام‌های فارسی برای نمایش
features_persian = [name for name, _ in feature_mapping]

# =====================================================
# UI
# =====================================================
st.set_page_config(page_title="توصیه مسیر شغلی", layout="wide", page_icon="🎯")

st.title("🎯 سیستم هوشمند توصیه مسیر شغلی")
st.markdown("### مبتنی بر مدل Neuro-Fuzzy")

st.write("**مهارت‌های خود را از ۱ تا ۵ امتیازدهی کنید:**")

values = []
col1, col2 = st.columns(2)

for i, feature in enumerate(features_persian):
    if i < 13:   # تنظیم ستون‌ها
        value = col1.slider(feature, min_value=1, max_value=5, value=3, step=1)
    else:
        value = col2.slider(feature, min_value=1, max_value=5, value=3, step=1)
    values.append(value)

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔮 پیش‌بینی مسیر شغلی", type="primary", use_container_width=True):
    X_raw = np.array(values).reshape(1, -1)
    
    X_scaled = scaler.transform(X_raw)
    X_input = np.clip(X_scaled / 5.0, 0, 1)
    
    probs = model.predict_proba(X_input)
    
    pred_idx = np.argmax(probs)
    career = label_encoder.inverse_transform([pred_idx])[0]
    
    st.success(f"**مسیر شغلی پیشنهادی شما: {career}**")
    
    st.subheader("احتمال هر مسیر شغلی")
    
    df = pd.DataFrame({
        "مسیر شغلی": label_encoder.classes_,
        "احتمال (%)": np.round(probs[0] * 100, 2)
    })
    
    st.bar_chart(df.set_index("مسیر شغلی"))
    st.dataframe(df.style.format({"احتمال (%)": "{:.2f}"}))

st.caption("توسعه یافته با مدل Neuro-Fuzzy | دمو آموزشی")
