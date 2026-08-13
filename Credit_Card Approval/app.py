import os
import pickle
import traceback
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

MODEL_PATH = 'model.pkl'
ENCODERS_PATH = 'encoders.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
except Exception as e:
    print(f"Warning: Could not load models. Ensure they exist in the root directory. Error: {e}")
    model, encoders = None, None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model or not encoders:
            raise ValueError("Model or encoders are not loaded. Please run train.py first.")

        raw_data = dict(request.form)
        data = {}

        for key, val in raw_data.items():
            if val.strip() == "":
                data[key] = 0
                continue
            try:
                data[key] = float(val)
            except ValueError:
                data[key] = val

        if 'DAYS_BIRTH' in data:
            data['DAYS_BIRTH'] = abs(data['DAYS_BIRTH'])
        if 'DAYS_EMPLOYED' in data:
            data['DAYS_EMPLOYED'] = abs(data['DAYS_EMPLOYED'])

        cnt_children = data.get('CNT_CHILDREN', 0)
        cnt_fam_members = data.get('CNT_FAM_MEMBERS', 1)
        cnt_fam_members = cnt_fam_members if cnt_fam_members > 0 else 1
        data['family_dependency'] = cnt_children / cnt_fam_members

        end_month = data.get('end_month', 0)
        open_month = data.get('open_month', 0)
        data['window'] = end_month - open_month

        if 'FLAG_PHONE' not in data:
            data['FLAG_PHONE'] = 0

        if isinstance(encoders, dict):
            le_mapping = {
                'CODE_GENDER': 'gender_le',
                'FLAG_OWN_CAR': 'car_le',
                'FLAG_OWN_REALTY': 'realty_le'
            }
            for col, le_key in le_mapping.items():
                if col in data and isinstance(data[col], str):
                    le = encoders.get(le_key)
                    if le:
                        if data[col] in le.classes_:
                            data[col] = int(le.transform([data[col]])[0])
                        else:
                            data[col] = 0
            
            dict_mapping = {
                'NAME_HOUSING_TYPE': 'housing_map',
                'NAME_INCOME_TYPE': 'income_map',
                'NAME_EDUCATION_TYPE': 'education_map',
                'NAME_FAMILY_STATUS': 'family_map'
            }
            for col, map_key in dict_mapping.items():
                if col in data and isinstance(data[col], str):
                    mapping_dict = encoders.get(map_key, {})
                    data[col] = mapping_dict.get(data[col], 0)

        df = pd.DataFrame([data])
        feature_cols = encoders.get('feature_cols', [])
        
        if feature_cols:
            df = df.reindex(columns=feature_cols, fill_value=0)
        elif hasattr(model, 'feature_names_in_'):
            df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        df = df.astype(float)

        prediction = int(model.predict(df)[0])
        
        probability = 0.0
        if hasattr(model, 'predict_proba'):
            proba_array = model.predict_proba(df)[0]
            probability = proba_array[1] if len(proba_array) > 1 else proba_array[0]

        approved = True if prediction == 0 else False
        prob_percent = int(round(probability * 100))
        
        if approved:
            display_prob = 100 - prob_percent
        else:
            display_prob = prob_percent

        return render_template(
            'result.html', 
            approved=approved, 
            probability=display_prob, 
            prediction_text="System processing complete.",
            form_data=raw_data
        )

    except Exception as e:
        print("\n" + "="*50)
        print("🚨 CRASH DETECTED IN PREDICTION LOGIC 🚨")
        traceback.print_exc()
        print("="*50 + "\n")
        
        error_msg = f"Prediction error: {str(e)}"
        return render_template('result.html', error=True, prediction_text=error_msg)

if __name__ == '__main__':
    app.run(debug=True, port=5000)