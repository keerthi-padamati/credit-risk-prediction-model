import os
import pickle
import traceback
import pandas as pd
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Configure CORS for localhost and production environments
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,X-Requested-With,Accept,Origin'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'encoders.pkl')

if not os.path.exists(MODEL_PATH) and os.path.exists('model.pkl'):
    MODEL_PATH = 'model.pkl'
if not os.path.exists(ENCODERS_PATH) and os.path.exists('encoders.pkl'):
    ENCODERS_PATH = 'encoders.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
except Exception as e:
    print(f"Warning: Could not load models. Ensure they exist in the directory. Error: {e}")
    model, encoders = None, None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "encoders_loaded": encoders is not None,
        "production_url": "https://credit-risk-prediction-model-mht3.onrender.com",
        "cors_enabled": True
    }), 200

def _process_prediction(raw_data):
    if not model or not encoders:
        raise ValueError("Model or encoders are not loaded. Please run train.py first.")

    data = {}
    for key, val in raw_data.items():
        if val is None or (isinstance(val, str) and val.strip() == ""):
            data[key] = 0
            continue
        try:
            data[key] = float(val)
        except (ValueError, TypeError):
            data[key] = val

    if 'DAYS_BIRTH' in data:
        data['DAYS_BIRTH'] = abs(data['DAYS_BIRTH'])
    if 'DAYS_EMPLOYED' in data:
        data['DAYS_EMPLOYED'] = abs(data['DAYS_EMPLOYED'])

    cnt_children = data.get('CNT_CHILDREN', 0)
    cnt_fam_members = data.get('CNT_FAM_MEMBERS', 1)
    try:
        cnt_fam_members = float(cnt_fam_members) if float(cnt_fam_members) > 0 else 1.0
        cnt_children = float(cnt_children)
    except (ValueError, TypeError):
        cnt_fam_members, cnt_children = 1.0, 0.0
    data['family_dependency'] = cnt_children / cnt_fam_members

    end_month = data.get('end_month', 0)
    open_month = data.get('open_month', 0)
    try:
        data['window'] = float(end_month) - float(open_month)
    except (ValueError, TypeError):
        data['window'] = 0.0

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
        probability = float(proba_array[1] if len(proba_array) > 1 else proba_array[0])

    approved = True if prediction == 0 else False
    prob_percent = int(round(probability * 100))
    
    if approved:
        display_prob = 100 - prob_percent
    else:
        display_prob = prob_percent

    return {
        "approved": approved,
        "status": "Approved" if approved else "Rejected",
        "probability": display_prob,
        "risk_probability": prob_percent,
        "prediction_code": prediction,
        "model_output": "System processing complete."
    }

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def api_predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        raw_data = request.get_json(silent=True) or dict(request.form)
        if not raw_data:
            return jsonify({"error": True, "message": "No data provided"}), 400
        result = _process_prediction(raw_data)
        return jsonify({"success": True, **result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": True, "message": str(e)}), 500

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204

    is_json = request.is_json or request.headers.get('Accept') == 'application/json'
    try:
        if request.is_json:
            raw_data = request.get_json(silent=True) or {}
        else:
            raw_data = dict(request.form)

        result = _process_prediction(raw_data)

        if is_json:
            return jsonify({"success": True, **result}), 200

        return render_template(
            'result.html', 
            approved=result["approved"], 
            probability=result["probability"], 
            prediction_text=result["model_output"],
            form_data=raw_data
        )

    except Exception as e:
        print("\n" + "="*50)
        print("🚨 CRASH DETECTED IN PREDICTION LOGIC 🚨")
        traceback.print_exc()
        print("="*50 + "\n")
        
        error_msg = f"Prediction error: {str(e)}"
        if is_json:
            return jsonify({"success": False, "error": True, "message": error_msg}), 500
        return render_template('result.html', error=True, prediction_text=error_msg)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)