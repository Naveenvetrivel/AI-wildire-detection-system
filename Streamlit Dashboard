
import os
import numpy as np
import streamlit as st
from datetime import datetime
from streamlit_option_menu import option_menu
from predict import predict_image
from database import insert_prediction, get_statistics, get_all_predictions
import plotly.express as px
from analytics import load_data
from report import generate_report  

# PAGE CONFIG
st.set_page_config(
    page_title="AI Wildfire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD CSS

def load_css():
    css_path = os.path.join("css", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# SIDEBAR

with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("<h2 style='text-align:center;color:#00E5FF;'>🌲 AI MONITOR</h2>", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=[
            "Dashboard", "Predict", "Analytics", "History",
            "Reports", "Alert Center", "Settings", "About"
        ],
        icons=[
            "speedometer2", "camera-fill", "bar-chart-fill", "clock-history",
            "file-earmark-bar-graph-fill", "bell-fill", "gear-fill", "info-circle-fill"
        ],
        default_index=0
    )

# HERO

st.markdown("""
<div class="hero">
<h1>🔥 AI WILDFIRE DETECTION SYSTEM</h1>
<h4>Real-Time Forest Fire Monitoring using Deep Learning</h4>
<p>MobileNetV2 • TensorFlow • Streamlit</p>
</div>
""", unsafe_allow_html=True)

# DATE + STATUS

c1, c2, c3 = st.columns(3)
with c1: st.info(f"📅 {datetime.now().strftime('%d-%m-%Y')}")
with c2: st.info(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
with c3: st.success("🟢 System Online")

st.markdown("---")

# KPI CARDS

stats = get_statistics()
k1, k2, k3, k4 = st.columns(4)

with k1: st.markdown(f"<div class='card'><h3>🔥 Wildfires</h3><h1>{stats['wildfire']}</h1></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='card'><h3>⚠ Smoke</h3><h1>{stats['smoke']}</h1></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='card'><h3>✅ Safe</h3><h1>{stats['safe']}</h1></div>", unsafe_allow_html=True)
with k4: st.markdown(f"<div class='card'><h3>📈 Average Confidence</h3><h1>{stats['accuracy']}%</h1></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# MAIN COLUMNS

left, right = st.columns([2,1])

# LEFT PANEL (Prediction)

with left:
    st.markdown("## 📤 Upload Forest Image")
    st.write("Upload a satellite or forest image to detect **Wildfire**, **Smoke**, or **No Fire**.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("📂 Upload an image to begin prediction.")
    else:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(file_path, caption="Uploaded Image", use_container_width=True)
        analyze = st.button("🔥 Analyze Image", use_container_width=True)

        if analyze:
            with st.spinner("🧠 AI is analyzing the image..."):
                label, confidence, probs = predict_image(file_path)
                insert_prediction(uploaded_file.name, label, confidence)

            st.markdown("---")
            st.subheader("Prediction Result")

            if label == "Wildfire":
                st.error("🔥 WILDFIRE DETECTED")
                risk = "🔴 HIGH"
            elif label == "Smoke":
                st.warning("⚠ SMOKE DETECTED")
                risk = "🟠 MEDIUM"
            else:
                st.success("✅ NO FIRE DETECTED")
                risk = "🟢 LOW"

            a, b, c = st.columns(3)
            with a: st.metric("Prediction", label)
            with b: st.metric("Confidence", f"{confidence:.2f}%")
            with c: st.metric("Risk Level", risk)

            st.markdown("---")
            st.subheader("Class Probabilities")
            st.write(f"✅ No Fire : {probs[0]*100:.2f}%"); st.progress(float(probs[0]))
            st.write(f"⚠ Smoke : {probs[1]*100:.2f}%"); st.progress(float(probs[1]))
            st.write(f"🔥 Wildfire : {probs[2]*100:.2f}%"); st.progress(float(probs[2]))

            st.markdown("---")
            st.subheader("AI Recommendation")

            if label == "Wildfire":
                st.error("Immediate action recommended.\n\n• Notify Forest Department\n• Dispatch Fire Response Team\n• Monitor nearby areas continuously.")
            elif label == "Smoke":
                st.warning("Smoke detected.\n\n• Continue monitoring\n• Verify whether smoke is increasing\n• Prepare response team.")
            else:
                st.success("No wildfire detected.\n\nForest condition appears normal.")

            
            # STEP 5: PDF REPORT GENERATION + DOWNLOAD
           
            pdf_file = generate_report(uploaded_file.name, label, confidence, risk)

            st.markdown("---")
            with open(pdf_file, "rb") as pdf:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf,
                    file_name=os.path.basename(pdf_file),
                    mime="application/pdf"
                )

           
            # DEBUG INFO
        
            with st.expander("🔍 Debug Information"):
                st.write("Prediction Array", probs)
                st.write("Argmax", np.argmax(probs))
                st.write("Predicted Label", label)

# RIGHT PANEL (Model Info + Stats)

with right:
    st.markdown("## 🤖 AI Model")
    st.info("""
### MobileNetV2
**Framework:** TensorFlow
**Input Size:** 224 × 224
**Classes:** ✅ No Fire | ⚠ Smoke | 🔥 Wildfire
**Status:** 🟢 Ready
""")

    st.markdown("---")
    st.subheader("📊 System Statistics")
    st.metric("Total Predictions", stats["total"])
    st.metric("Wildfires", stats["wildfire"])
    st.metric("Smoke", stats["smoke"])
    st.metric("Average Confidence", f"{stats['accuracy']}%")
    st.metric("Version", "1.0")

    st.markdown("---")
    st.subheader("🔥 Prediction Distribution")
    st.progress(0.20); st.write("🟢 Low Risk")
    st.progress(0.55); st.write("🟠 Medium Risk")
    st.progress(0.85); st.write("🔴 High Risk")

    st.markdown("---")
    st.subheader("📜 Recent Predictions")
    history = get_all_predictions()
    if history:
        st.dataframe(history, use_container_width=True)
    else:
        st.info("No prediction history found.")

# ANALYTICS + VISUALS

st.markdown("---")
st.subheader("📊 Analytics")

df = load_data()
if not df.empty:
    fig = px.pie(df, names="prediction", hole=0.55, title="Prediction Distribution")
    st.plotly_chart(fig, use_container_width=True)

    count_df = df["prediction"].value_counts().reset_index()
    count_df.columns = ["Prediction", "Count"]
    fig2 = px.bar(count_df, x="Prediction", y="Count", text="Count", title="Prediction Counts")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(df, x="confidence", nbins=15, title="Confidence Distribution")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No analytics data available.")

# RECENT PREDICTIONS

st.markdown("---")
st.subheader("📜 Recent Predictions")

history = get_all_predictions()
if history:
    st.dataframe(history, use_container_width=True)
else:
    st.info("No prediction history found.")

# FOOTER

st.markdown("---")
st.markdown("""
<div class='footer'>
<h3>🌲 AI Wildfire Detection System</h3>
<p>Real-Time Forest Fire Detection using Deep Learning</p>
<p>Model : MobileNetV2 | TensorFlow | Streamlit</p>
<p>Developed By NaveenKumar Vetrivel</p> 
</div>
""", unsafe_allow_html=True)
