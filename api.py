from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from utils import ImageInference
import os
import uuid
import base64
from PIL import Image
import io
import numpy as np
import json

app = FastAPI(title="Tomato Leaf Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = None

@app.on_event("startup")
def load_model():
    global classifier
    print("Loading classification models...")
    try:
        classifier = ImageInference(classifier_path='./classifier_checkpoints/best_classifier.pt')
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded on the server.")
        
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
        
    try:
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        contents = await file.read()
        
        # Force convert blob to image correctly
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img.save(temp_path)
            
        result = classifier.classify_image(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            'predicted_class': result['class'],
            'confidence_score': float(result['confidence']),
            'class_probabilities': {k: float(v) for k, v in result['all_probabilities'].items()}
        }
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gradcam/")
async def gradcam(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded on the server.")
        
    try:
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        contents = await file.read()
        
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img.save(temp_path)
            
        gradcam_img = classifier.generate_gradcam(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Return base64 for simplicity in frontend display
        gradcam_pil = Image.fromarray(np.uint8(gradcam_img))
        buffered = io.BytesIO()
        gradcam_pil.save(buffered, format="JPEG", quality=90)
        gradcam_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"gradcam_base64": gradcam_b64}
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/")
async def metrics():
    metrics_path = './classifier_checkpoints/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Metrics not found")

@app.get("/generated-images/")
async def generated_images():
    output_dir = './output'
    if not os.path.exists(output_dir):
        return {"urls": []}
        
    images = [f for f in os.listdir(output_dir) if f.startswith('generated_epoch_') and f.endswith('.png')]
    # Sort logically
    images.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    image_urls = [f"/assets/{img}" for img in images]
    return {"urls": image_urls}

@app.get("/assets/{filename}")
async def fetch_asset(filename: str):
    if os.path.exists(os.path.join('./classifier_checkpoints', filename)):
        return FileResponse(os.path.join('./classifier_checkpoints', filename))
    elif os.path.exists(os.path.join('./output', filename)):
        return FileResponse(os.path.join('./output', filename))
    raise HTTPException(status_code=404, detail="Asset not found")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
