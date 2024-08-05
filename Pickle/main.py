from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load your dataset and preprocess
data = pd.read_csv('merged_data_inner.csv')
data = data.dropna()

# Load the Random Forest model and feature names
with open('rf_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    rf_model = model_data['model']
    features = model_data['features']

# Calculate pre-industrial temperature
pre_industrial_temp = data[data['Year'] < 1900]['AverageTemperature'].mean()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        co2_rate = float(request.form['co2_rate'])
        ch4_rate = float(request.form['ch4_rate'])
        no2_rate = float(request.form['no2_rate'])

        # Predict future temperatures
        future_years = np.arange(2014, 2101)
        future_features = pd.DataFrame({
            'Year': future_years,
            'Annual nitrous oxide emissions in CO₂ equivalents': np.zeros(future_years.shape),
            'Annual methane emissions in CO₂ equivalents': np.zeros(future_years.shape),
            'Annual CO₂ emissions': np.zeros(future_years.shape)
        })

        base_no2 = data['Annual nitrous oxide emissions in CO₂ equivalents'].mean()
        base_ch4 = data['Annual methane emissions in CO₂ equivalents'].mean()
        base_co2 = data['Annual CO₂ emissions'].mean()

        for i in range(future_years.shape[0]):
            year = 2014 + i
            future_features.at[i, 'Annual nitrous oxide emissions in CO₂ equivalents'] = base_no2 * (1 + no2_rate) ** (year - 2014)
            future_features.at[i, 'Annual methane emissions in CO₂ equivalents'] = base_ch4 * (1 + ch4_rate) ** (year - 2014)
            future_features.at[i, 'Annual CO₂ emissions'] = base_co2 * (1 + co2_rate) ** (year - 2014)

        rf_future_temperatures = rf_model.predict(future_features[features])
        rf_breach_year = future_years[np.argmax(rf_future_temperatures >= pre_industrial_temp + 2)]

        return jsonify({
            'breach_year': int(rf_breach_year),
            'future_temperatures': rf_future_temperatures.tolist()
        })
    except Exception as e:
        return str(e), 400

@app.route('/tableau')
def tableau():
    return render_template('tableau.html')

if __name__ == '__main__':
    app.run(debug=True)

