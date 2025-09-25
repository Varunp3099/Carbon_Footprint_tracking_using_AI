# Carbon_Footprint_tracking_using_AI

# 🌱 Carbon Emission Tracker

A **Machine Learning + AI-powered application** to estimate and reduce your **carbon footprint** based on lifestyle choices such as diet, transport, energy usage, and recycling habits.

This project combines:

* **Data Preprocessing & Modeling** (📓 `carb.ipynb`)
* **Interactive Web Application** (⚡ `app.py` with **Streamlit** + **Google Gemini AI**)

## 🚀 Features

* 📊 **Carbon Emission Prediction** using a **Random Forest Regressor**
* 🧹 **Data preprocessing pipeline** (handling categorical, ordinal, and multi-label features)
* 📦 **Outlier detection & feature engineering**
* 🖼 **Visualization** with Matplotlib, Seaborn, and Plotly
* 🌍 **Streamlit web app** for user interaction
* 🤖 **AI-Powered Recommendations** using **Gemini AI**
* 📉 **Emission breakdown by categories** (transport, home energy, food, waste, activities)

## 🛠 Tech Stack

* **Python**
* **Pandas, NumPy, Scikit-learn**
* **Matplotlib, Seaborn, Plotly**
* **Streamlit** (for UI)
* **Joblib** (model persistence)
* **Google Gemini AI** (personalized recommendations)


## 📂 Project Structure

```
├── carb.ipynb                 # Data preprocessing, EDA, ML model training
├── app_updated.py                     # Streamlit app for user interaction
├── Carbon Emission.csv        # Dataset used
├── carbon_emission_model.joblib  # Saved trained RandomForest model
└── README.md                  # Project documentation
```


## 📊 Workflow

1. **Data Preprocessing**

   * Handle missing values
   * Encode categorical/ordinal/multi-label features
   * Outlier detection (IQR method)

2. **Model Training** (`carb.ipynb`)

   * Train a **Random Forest Regressor**
   * Evaluate using **MSE & R² Score**
   * Save the trained model with `joblib`

3. **Web App** (`app.py`)

   * Collect user inputs (diet, transport, energy usage, etc.)
   * Encode inputs consistently with training data
   * Predict **annual carbon footprint (kg CO₂e)**
   * Visualize emission breakdown with Plotly
   * Provide **AI recommendations** (via Gemini)


