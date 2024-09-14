import pandas as pd
import numpy as np


def generate_synthetic_data(num_records):
    timestamps = pd.date_range(start='2024-08-15', periods=num_records, freq='H')
    temperature = np.random.uniform(low=16.0, high=30.0, size=num_records)
    humidity = np.random.uniform(low=23.0, high=80.0, size=num_records)
    disaster = np.random.choice([0, 1], size=num_records)  # 0 = no disaster, 1 = disaster
    data = {
        'timestamp': timestamps,
        'temperature': temperature,
        'humidity': humidity,
        'disaster': disaster
    }
    df = pd.DataFrame(data)
    return df

def save_data():
    df = generate_synthetic_data(100)
    df.to_csv('historical_weather_data.csv', index=False)
    print("CSV file created successfully.")

if __name__ == '__main__':
    save_data()
