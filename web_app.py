import matplotlib
matplotlib.use('Agg')

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
import traceback

app = Flask(__name__)

# ✅ Keep CORS simple and reliable
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

print("Loading classification models...")
try:
    classifier = ImageInference(classifier_path='./classifier_checkpoints/best_classifier.pt')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None


@app.route('/')
def health_check():
    return jsonify({
        'status': 'online',
        'service': 'Tomato Leaf Disease Detection API',
        'model_loaded': classifier is not None
    })


# ------------------ PREDICT (UNCHANGED) ------------------

@app.route('/predict/', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None:
        return jsonify({'detail': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'detail': 'No file provided'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'detail': 'No file selected'}), 400
        
    try:
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        
        img = Image.open(file.stream).convert('RGB')
        img.save(temp_path)
        
        result = classifier.classify_image(temp_path)
        
        os.remove(temp_path)

        return jsonify({
            'predicted_class': result['class'],
            'confidence_score': float(result['confidence']),
            'class_probabilities': {
                k: float(v) for k, v in result['all_probabilities'].items()
            }
        })
        
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'detail': str(e)}), 500


# ------------------ GRADCAM (SAFE VERSION) ------------------

@app.route('/gradcam/', methods=['POST'])
@app.route('/gradcam', methods=['POST'])
def gradcam():
    if classifier is None:
        return jsonify({'detail': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'detail': 'No file provided'}), 400
        
    file = request.files['file']
    
    try:
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        
        img = Image.open(file.stream).convert('RGB')
        img.save(temp_path)

        # 🔥 TRY GradCAM
        try:
            gradcam_img = classifier.generate_gradcam(temp_path)
        except Exception as e:
            print("⚠️ GradCAM failed:", e)

            # ✅ FALLBACK: return original image instead of crashing
            gradcam_img = np.array(Image.open(temp_path).resize((224, 224)))

        os.remove(temp_path)
            
        gradcam_pil = Image.fromarray(np.uint8(gradcam_img))
        buffered = io.BytesIO()
        gradcam_pil.save(buffered, format="JPEG")

        gradcam_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return jsonify({"gradcam_base64": gradcam_b64})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'detail': str(e)}), 500


# ------------------ METRICS ------------------

@app.route('/metrics/', methods=['GET'])
@app.route('/metrics', methods=['GET'])
def get_metrics():
    metrics_path = './classifier_checkpoints/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'detail': 'Metrics not found'}), 404


# ------------------ GENERATED IMAGES ------------------

@app.route('/generated-images/', methods=['GET'])
@app.route('/generated-images', methods=['GET'])
def generated_images():
    output_dir = './output'
    if not os.path.exists(output_dir):
        return jsonify({"urls": []})

    images = [f for f in os.listdir(output_dir) if f.startswith('generated_epoch_')]
    images.sort()

    image_urls = [f"/assets/{img}" for img in images]
    return jsonify({"urls": image_urls})


# ------------------ STATIC FILES ------------------

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    if os.path.exists(os.path.join('./classifier_checkpoints', filename)):
        return send_from_directory('./classifier_checkpoints', filename)
    elif os.path.exists(os.path.join('./output', filename)):
        return send_from_directory('./output', filename)
    
    return jsonify({'detail': 'File not found'}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5000)
