# 💳 Credit Card Approval Prediction System

An end-to-end Machine Learning web application that predicts whether a credit card application is likely to be approved based on applicant demographic and financial information.

## ✨ Features

* Automated data preprocessing and feature engineering
* Multiple ML models trained and compared
* Best model selected based on F1-score
* Supports Logistic Regression, Decision Tree, Random Forest, and XGBoost
* Handles imbalanced datasets using class weighting
* Real-time predictions using a Flask web application
* Simple and responsive user interface

## 🛠️ Technologies Used

* Python
* Flask
* Scikit-learn
* XGBoost
* Pandas
* NumPy
* Matplotlib
* Seaborn
* HTML
* CSS
* JavaScript

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
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Installation and Setup

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
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train.py
```

After training, the model files will be generated automatically.

### 5. Start the Application

```bash
python app.py
```

### 6. Open the Application

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## 📊 How It Works

1. The user enters applicant information.
2. The application processes the input data.
3. The trained machine learning model analyzes the data.
4. The system predicts the approval result.
5. The result is displayed instantly in the web application.

