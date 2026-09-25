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
