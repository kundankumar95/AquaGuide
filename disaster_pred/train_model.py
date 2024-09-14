import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load data
def load_data(filename):
    return pd.read_csv(filename)

# Prepare features and target
def prepare_data(df):
    X = df[['temperature', 'humidity']]
    y = df['disaster']
    return X, y

# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    df = load_data('historical_weather_data.csv')
    X, y = prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, 'disaster_model.pkl')
    print("Model trained and saved successfully.")

if __name__ == '__main__':
    main()
