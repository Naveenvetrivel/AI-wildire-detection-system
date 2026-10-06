# 🔥 AI-Powered Wildfire Detection and Early Warning System Using Satellite Imagery

## 📌 Project Overview

This project presents an AI-powered wildfire detection system that uses satellite imagery to identify potential wildfire conditions.

The system uses a **MobileNetV2 deep learning model** to classify uploaded satellite images into:

- 🌫️ Smoke
- 🔥 Wildfire

The prediction confidence is then used in a **rule-based risk assessment system** to determine the wildfire risk level.

A **Streamlit dashboard** provides an interactive interface where users can upload an image, view the prediction and confidence score, assess the risk level, and generate reports.

---

## 🎯 Objectives

The main objectives of this project are:

- Detect wildfire-related patterns from satellite imagery.
- Classify images as **Smoke or Wildfire**.
- Use transfer learning with **MobileNetV2**.
- Display prediction confidence for the detected class.
- Calculate a risk score based on model confidence.
- Categorize wildfire risk into different levels.
- Provide an easy-to-use Streamlit dashboard.
- Generate downloadable prediction reports.

---

## 🧠 Methodology

The system follows the workflow:

```text
Satellite Image
      ↓
Image Upload
      ↓
Image Preprocessing
      ↓
MobileNetV2 Model
      ↓
Smoke / Wildfire Prediction
      ↓
Confidence Score
      ↓
Risk Assessment
      ↓
Recommendation
      ↓
Streamlit Dashboard
      ↓
PDF / CSV Report

## DashBoard
<img width="959" height="537" alt="newdashboard2" src="https://github.com/user-attachments/assets/ed28c845-77fe-473f-91eb-236263a3b56e" />
<img width="959" height="536" alt="newdashboard3" src="https://github.com/user-attachments/assets/ae1d9a9b-ca83-49db-a904-3d060501a31e" />
<img width="953" height="538" alt="newdashboard4" src="https://github.com/user-attachments/assets/2ad2aba2-7c6e-42f4-8808-17513b48f28e" />
<img width="959" height="533" alt="newdashboard5" src="https://github.com/user-attachments/assets/fe8399cc-702a-4847-93ef-adf0673dc24f" />
<img width="959" height="542" alt="newdashboard6" src="https://github.com/user-attachments/assets/6a20d0a5-d457-4d19-80fd-76a373e64d57" />
<img width="958" height="569" alt="newdashboard1" src="https://github.com/user-attachments/assets/0263293f-33ed-4e4f-9418-1193560590c7" />

