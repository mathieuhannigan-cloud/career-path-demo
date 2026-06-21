if st.button("🔮 Predict Career Path"):
    X_raw = np.array(values).reshape(1, -1)
    
    st.write("**ورودی خام (۱-۵):**", X_raw[0])
    
    # === دیباگ Scaling ===
    X_scaled = scaler.transform(X_raw)
    st.write("**بعد از Scale (باید حدود ۰-۱ باشه):**", np.round(X_scaled[0], 4))
    
    # چک کنیم min و max scaler چقدر بوده
    st.write("**Min داده‌های آموزشی:**", scaler.data_min_.round(2))
    st.write("**Max داده‌های آموزشی:**", scaler.data_max_.round(2))
    
    logits, firing_norm, _, _ = model.forward(X_scaled)
    
    st.write("**Firing Strengths:**", np.round(firing_norm[0], 6))
    st.write("**Max Firing:**", firing_norm[0].max().round(6))
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
