# 🏠 House Value Prediction Web App

This project is a web application built using **FastAPI** that predicts the **median house value** in California based on user-provided housing data. The prediction is powered by a **pre-trained machine learning model (RandomForestRegressor)** trained on the **California Housing Dataset**.
<br><br>
Users input housing characteristics such as location, number of rooms, income, population, and ocean proximity, and the model outputs the estimated house value.

---

## 🚀 Features

- 📊 **Interactive web form** for user input
- ⚡ **FastAPI backend** for fast and async performance
- 🧠 **Machine Learning model** (Random Forest) trained with engineered features
- 🛠️ **Custom preprocessing pipeline** using Scikit-learn transformers
- 🎨 Clean UI with Jinja2 templating and CSS styling
- 🔁 Form retains input data after prediction for better UX

---
---

## 📁 Project Structure

```
├── app.py                  # FastAPI backend
├── prediction.py           # Preprocessing and ML model
├── templates/
│   └── index.html          # HTML form
├── static/
│   └── style.css           # Optional CSS styling
├── models/
│   └── final_model.pkl     # Trained model
```

---

## 🌐 How It Works (After Deployment)

This FastAPI app is delpoyed at Render :

1. **User Access :**  
   The user will visit the domain at [`https://household-prediction.onrender.com`](https://household-prediction.onrender.com) where the homepage displays a form for entering housing features.

2. **Form Submission :**  
   When the form is submitted:
   - A POST request is sent to the FastAPI server.
   - The data is collected and passed to the `predict_price()` function.

3. **Preprocessing :**
   - The raw input is converted into a structured DataFrame.
   - Numerical features are scaled and custom features like `bedrooms_per_room` are added.
   - Categorical features like `ocean_proximity` are one-hot encoded.

4. **Prediction :**
   - The transformed data is fed to the pre-trained `RandomForestRegressor` model.
   - The model returns a predicted median house value.

5. **Response :**
   - The prediction result is rendered back on the same form page.
   - All previously entered form values remain visible for convenience.

---

## 📝 How to Run Locally

1. Firstly install `uv` if you have not installed
2. Clone the repo:
   ```bash
   git clone https://github.com/abhijitpadhi1/Household-Prediction.git
   cd Household-Prediction
   ```
3. Install all dependencies with one virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate         # For Linux systems
   .venv\Scripts\activate            # For Windows systems
   uv pip install .
   ```
4. Run the `app.py` on `localhost:8000`
   ```bash
   uv run app.py
   ```
---
---

## 🧰 Tech Stack

- **Python**
- **FastAPI**
- **HTML/CSS (Jinja2 templates)**
- **Scikit-learn**
- **pandas, numpy, joblib**

---

## 🧠 What I Learned

Working on this project helped me learn and apply several key concepts:

- **FastAPI Framework**: Creating modern, high-performance APIs
- **Frontend Integration**: Linking forms with ML model predictions
- **ML Deployment**: Real-time prediction using a saved model
- **Feature Engineering**: Adding domain-specific attributes via custom transformer
- **Debugging**: Handling mismatched input/output shape and transforming data properly

---
---
