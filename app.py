# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:8000 app:app

from flask import Flask, render_template
from flask_socketio import SocketIO
import pickle
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bhav-setu-secret-key-!@#'
# Note: You may need to configure CORS if the client and server are on different origins in the future.
# from flask_cors import CORS
# CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Load Model ---
# No MediaPipe or OpenCV needed on the backend anymore.
try:
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

labels_dict = {
    0: 'A', 1: 'Setu', 2: 'Namaste', 3: 'D', 4: 'We', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'We', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'Namaste', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Bhav', 28: 'Thank You', 29: 'Cool', 30: 'Building', 31: 'Are',
    32: 'Bhav'
}

@app.route('/')
def index():
    """Render the main web page."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('process_landmarks')
def handle_landmarks(landmark_data):
    """
    Receives landmark data from the client, runs the model, and sends a prediction back.
    This is much faster as it doesn't process video frames.
    """
    if model is None or not landmark_data:
        return

    data_aux = []
    landmarks = landmark_data['landmarks']

    # Check if landmarks list is not empty
    if not landmarks:
        return

    x_ = [point['x'] for point in landmarks]
    y_ = [point['y'] for point in landmarks]

    # Normalize landmarks relative to the first landmark's position
    # This matches the data preparation used during training
    for i in range(len(landmarks)):
        data_aux.append(landmarks[i]['x'] - min(x_))
        data_aux.append(landmarks[i]['y'] - min(y_))

    try:
        prediction = model.predict([np.asarray(data_aux)])
        prediction_proba = model.predict_proba([np.asarray(data_aux)])
        confidence = float(max(prediction_proba[0]))
        predicted_character = labels_dict.get(int(prediction[0]), 'Unknown')

        # Send only the prediction result back
        socketio.emit('prediction_result', {
            'text': predicted_character,
            'confidence': f"{confidence*100:.2f}"
        })
    except Exception as e:
        print(f"Prediction error: {e}")

if __name__ == '__main__':
    print("Starting Flask server in debug mode...")
    socketio.run(app, debug=True, host='0.0.0.0', port=8000)
