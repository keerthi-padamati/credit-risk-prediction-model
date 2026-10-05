# 💳 Credit Card Approval Prediction System

An end-to-end Machine Learning web application and REST API that predicts whether a credit card application is likely to be approved based on applicant demographic and financial information.

---

## 🌐 Live Production & Deployment Links

* **Live Web Application**: [https://credit-risk-prediction-model-mht3.onrender.com](https://credit-risk-prediction-model-mht3.onrender.com)
* **Application Form**: [https://credit-risk-prediction-model-mht3.onrender.com/form](https://credit-risk-prediction-model-mht3.onrender.com/form)
* **API Health Check**: [https://credit-risk-prediction-model-mht3.onrender.com/health](https://credit-risk-prediction-model-mht3.onrender.com/health)
* **REST Prediction API**: `POST https://credit-risk-prediction-model-mht3.onrender.com/api/predict`
* **GitHub Repository**: [https://github.com/keerthi-padamati/credit-risk-prediction-model.git](https://github.com/keerthi-padamati/credit-risk-prediction-model.git)

---

## 🛠️ Render Production Settings

* **Environment / Runtime**: `Python 3`
* **Python Version**: `3.11.8` (configured via `.python-version`)
* **Build Command**: `python -m pip install --upgrade pip setuptools wheel && pip install -r requirements.txt && python train.py`
* **Start Command**: `gunicorn app:app`

---

## ✨ Features

* Automated data preprocessing and feature engineering
* Multiple ML models trained and compared
* Best model selected based on F1-score
* Supports Logistic Regression, Decision Tree, Random Forest, and XGBoost
* Handles imbalanced datasets using class weighting
* Real-time predictions using a Flask web application
* Cross-Origin Resource Sharing (CORS) enabled for localhost and production
* RESTful JSON API endpoint for external frontend and client integrations
* Simple and responsive user interface

---

## 🛠️ Technologies Used

* Python 3.11+
* Flask & Flask-CORS
* Gunicorn
* Scikit-learn
* XGBoost
* Pandas
* NumPy
* Matplotlib & Seaborn
* HTML5, CSS3, JavaScript

---

## 📂 Project Structure

```text
credit-card-approval-prediction/
│
├── data/
│   └── application_record.csv
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── assets/
│       └── eda/
│
├── templates/
│   ├── home.html
│   ├── index.html
│   └── result.html
│
├── app.py
├── train.py
├── test_app.py
├── requirements.txt
├── Procfile
├── render.yaml
├── .python-version
├── .gitignore
└── README.md
```

---

## 🚀 Local Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/keerthi-padamati/credit-risk-prediction-model.git
cd credit-risk-prediction-model
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train.py
```

After training, the model files (`model.pkl` and `encoders.pkl`) will be generated automatically.

### 5. Start the Application

```bash
python app.py
```

### 6. Open the Application

Open your browser and visit:
```text
http://127.0.0.1:5000
```

---

## 📡 API Usage & cURL Example

```bash
curl -X POST https://credit-risk-prediction-model-mht3.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "Y",
    "FLAG_OWN_REALTY": "Y",
    "CNT_CHILDREN": "1",
    "AMT_INCOME_TOTAL": "250000",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "DAYS_BIRTH": "-14000",
    "DAYS_EMPLOYED": "-3000",
    "CNT_FAM_MEMBERS": "3",
    "open_month": "-24",
    "end_month": "0"
  }'
```

### Response:
```json
{
  "success": true,
  "approved": true,
  "status": "Approved",
  "probability": 86,
  "risk_probability": 14,
  "prediction_code": 0,
  "model_output": "System processing complete."
}
```
