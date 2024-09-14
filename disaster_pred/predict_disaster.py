import json
import pandas as pd
import joblib

# Load current weather data
def load_current_weather(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Predict disaster likelihood
def predict_disaster(current_weather):
    model = joblib.load('disaster_model.pkl')
    current_df = pd.DataFrame([current_weather])
    X_current = current_df[['temperature', 'humidity']]
    prediction = model.predict(X_current)
    return prediction[0]

def main():
    current_weather = load_current_weather('current_weather_data.json')
    if current_weather is None:
        print("No current weather data available.")
        return

    disaster_prediction = predict_disaster(current_weather)
    
    if disaster_prediction == 1:
        print("There is a high chance of a disaster (e.g., flood) occurring.")
    else:
        print("The likelihood of a disaster is low.")

if __name__ == '__main__':
    main()
