import requests
import json

def fetch_weather(city):
    API_KEY = '225575b735fc3995f7da7d2c63307ec3'
    URL = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'

    response = requests.get(URL)
    data = response.json()

    if data.get('cod') != 200:
        print(f"Error fetching data for {city}: {data.get('message')}")
        return None

    weather_data = {
        'temperature': data['main']['temp'],
        'humidity': data['main']['humidity']
    }
    with open('current_weather_data.json', 'w') as f:
        json.dump(weather_data, f, indent=4)
    print(f"Current weather data for {city} saved to current_weather_data.json")

if __name__ == '__main__':
    city = "kolkata"
    fetch_weather(city)

