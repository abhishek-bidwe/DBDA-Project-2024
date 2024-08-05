import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
data = pd.read_csv('merged_data_inner.csv')

# Preprocessing: Handle missing values if any
data = data.dropna()

# Feature selection
features = ['Year', 'Annual nitrous oxide emissions in CO₂ equivalents',
            'Annual methane emissions in CO₂ equivalents', 'Annual CO₂ emissions']
X = data[features]
y = data['AverageTemperature']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model and the features
with open('rf_model.pkl', 'wb') as f:
    pickle.dump({'model': rf_model, 'features': features}, f)

print("Random Forest model trained and saved successfully.")

