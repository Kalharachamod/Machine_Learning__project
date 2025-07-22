from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('crop_recommendation_model.joblib')

# Helper function for input validation
def validate_inputs(form):
    fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    values = []
    for field in fields:
        value = form.get(field)
        if value is None or value.strip() == '':
            return None, f"Field '{field}' is required."
        try:
            val = float(value)
            if field in ['N', 'P', 'K'] and val < 0:
                return None, f"{field} must be non-negative."
            if field == 'ph' and not (0 <= val <= 14):
                return None, "pH must be between 0 and 14."
            values.append(val)
        except ValueError:
            return None, f"Invalid value for {field}. Must be a number."
    return values, None

@app.route('/', methods=['GET', 'POST'])
def index():
    crop = None
    error = None
    if request.method == 'POST':
        features, error = validate_inputs(request.form)
        if not error:
            try:
                features = np.array([features])
                crop = model.predict(features)[0]
            except Exception as e:
                error = f"Prediction error: {e}"
    return render_template('index.html', crop=crop, error=error)

if __name__ == '__main__':
    app.run(debug=True) 