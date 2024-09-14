import requests

def main():
    query = input("Enter your query: ")
    
    # Prepare the parameters to be sent to the API
    params = {'query': query}
    
    try:
        # Send GET request to the API server
        response = requests.get('http://127.0.0.1:5000/classify', params=params)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result.get('response', 'No response found')}")
            print(f"Probability: {result.get('probability', 'No probability found')}")
        else:
            error = response.json().get('error', 'Unknown error')
            print(f"Error: {error}")
    except requests.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()
