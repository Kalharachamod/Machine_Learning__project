from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

# List of features in order
FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Ensure all features are present
    try:
        input_data = [float(data[feature]) for feature in FEATURES]
    except Exception as e:
        return jsonify({'error': f'Missing or invalid input: {e}'}), 400
    # Reshape for prediction
    input_array = np.array(input_data).reshape(1, -1)
    pred = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0][1]
    return jsonify({'prediction': int(pred), 'probability': float(prob)})

if __name__ == '__main__':
    app.run(debug=True) 