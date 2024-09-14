from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.chat.chat import Chat


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/classify', methods=['GET'])
def classify():
    query = request.args.get('query')
    
    if not query:
        return jsonify({'error': 'No query parameter provided'}), 400
    
    try:
        response, prob = Chat(query)
        return jsonify({
            'response': response,
            'probability': prob
        })
    except Exception as e:
        app.logger.error(f'Error in Chat function: {e}')
        return jsonify({'error': 'Internal server error'}), 500



if __name__ == '__main__':
    app.run(port=5000, debug=True)

