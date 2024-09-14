import subprocess
from fetch_current_weather import fetch_weather


def run_script(script_name):
    subprocess.run(["python", script_name])

def main():
    run_script('backend/disaster_pred/generate_csv_file.py')

    run_script('backend/disaster_pred/train_model.py')
    
    run_script('backend/disaster_pred/fetch_current_weather.py')

    run_script('backend/disaster_pred/predict_disaster.py')

if __name__ == '__main__':
    main()
