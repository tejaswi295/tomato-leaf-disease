from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from utils import ImageInference
import os
import base64
from PIL import Image
import io
import uuid
import numpy as np
import json

app = Flask(__name__)
# Enable CORS so the React frontend running on port 5173 can query the API
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max limit

print("Loading classification models...")
try:
    classifier = ImageInference(classifier_path='./classifier_checkpoints/best_classifier.pt')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None:
        return jsonify({'error': 'Model not loaded on the server.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided in the request payload'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        # Save temp file
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        
        # If it comes from React webcam, it might be a blob without proper format, force it
        img = Image.open(file.stream).convert('RGB')
        img.save(temp_path)
        
        # Classify the image dynamically
        result = classifier.classify_image(temp_path)
        
        # Generate Grad-CAM output visualization
        gradcam_img = classifier.generate_gradcam(temp_path)
        
        # Clean up temp
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Convert the Grad-CAM numpy array to base64 so React can render it via src=data:image/jpeg;base64...
        gradcam_pil = Image.fromarray(np.uint8(gradcam_img))
        buffered = io.BytesIO()
        gradcam_pil.save(buffered, format="JPEG", quality=90)
        gradcam_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return jsonify({
            'disease': result['class'],
            'confidence': float(result['confidence']),
            'probabilities': {k: float(v) for k, v in result['all_probabilities'].items()},
            'gradcam': gradcam_b64
        })
        
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Retrieve validation and test metrics from the model evaluation JSON."""
    metrics_path = './classifier_checkpoints/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Metrics tracking JSON not found.'}), 404

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """
    General utility endpoint for the frontend to fetch static visualizations
    like training_curves.png or confusion_matrix.png.
    """
    # Check checkpoints dir
    if os.path.exists(os.path.join('./classifier_checkpoints', filename)):
        return send_from_directory('./classifier_checkpoints', filename)
    # Check standard outputs dir
    elif os.path.exists(os.path.join('./output', filename)):
        return send_from_directory('./output', filename)
    
    return jsonify({'error': 'Requested asset file not found'}), 404

if __name__ == '__main__':
    # Running on standard 5000
    app.run(debug=True, port=5000)
