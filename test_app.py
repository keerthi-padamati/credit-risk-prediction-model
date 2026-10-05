import json
from app import app

def test_routes():
    client = app.test_client()
    
    # 1. Health check & CORS
    res_health = client.get('/health', headers={'Origin': 'http://localhost:3000'})
    assert res_health.status_code == 200
    data = res_health.get_json()
    assert data['status'] == 'healthy'
    assert data['cors_enabled'] is True
    assert 'Access-Control-Allow-Origin' in res_health.headers
    print("  [PASS] Health check and CORS passed.")

    # 2. Home route
    res_home = client.get('/')
    assert res_home.status_code == 200
    print("  [PASS] Home page (/) passed.")

    # 3. Form route
    res_form = client.get('/form')
    assert res_form.status_code == 200
    print("  [PASS] Form page (/form) passed.")

    # 4. JSON API Predict
    sample_payload = {
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
    }
    res_api = client.post(
        '/api/predict',
        data=json.dumps(sample_payload),
        content_type='application/json',
        headers={'Origin': 'https://credit-risk-prediction-model-mht3.onrender.com'}
    )
    assert res_api.status_code == 200
    api_data = res_api.get_json()
    assert api_data['success'] is True
    assert 'status' in api_data
    print(f"  [PASS] REST API predict passed: {api_data['status']} ({api_data['probability']}%)")

    # 5. Form HTML Predict
    res_predict = client.post('/predict', data=sample_payload)
    assert res_predict.status_code == 200
    assert b"System processing complete" in res_predict.data or b"Approval" in res_predict.data
    print("  [PASS] HTML Form predict passed.")

if __name__ == '__main__':
    print("Running verification tests...")
    test_routes()
    print("\n[SUCCESS] All verification tests passed with 0 errors!")
