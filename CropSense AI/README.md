# Crop Recommendation System

This project recommends the best crop to grow based on soil and weather parameters using a machine learning model.

## Features
- Input: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, Rainfall
- Output: Recommended crop
- ML Model: Random Forest (scikit-learn)
- Web App: Flask

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the model:**
   ```bash
   python model.py
   ```
   This will create `crop_recommendation_model.joblib`.
3. **Run the web app:**
   ```bash
   python app.py
   ```
4. **Open your browser:**
   Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Files
- `model.py`: Trains and saves the ML model
- `app.py`: Flask web app for crop recommendation
- `Crop_recommendation.csv`: Dataset
- `templates/index.html`: Web UI
- `requirements.txt`: Dependencies

## Dataset
- Source: Kaggle Crop Recommendation Dataset 