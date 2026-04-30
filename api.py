from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from utils import ImageInference
import os
import uuid
import base64
from PIL import Image
import io
import numpy as np
import json

app = FastAPI(title="Tomato Leaf Disease API")

# ==================== CORS Configuration for Vercel ====================
# Allow requests from Vercel frontend and localhost for development
ALLOWED_ORIGINS = [
    "https://tomato-leaf-disease-alpha.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

classifier = None

@app.on_event("startup")
def load_model():
    global classifier
    print("Loading classification models...")
    try:
        classifier = ImageInference(classifier_path='./classifier_checkpoints/best_classifier.pt')
        print("✓ Models loaded successfully.")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        classifier = None

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "tomato-leaf-disease-api",
            "model_loaded": classifier is not None
        }
    )

@app.options("/predict")
async def options_predict():
    """Handle CORS preflight requests for /predict"""
    return JSONResponse(status_code=200, content={})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict disease class and confidence from uploaded image.
    
    Request: POST /predict with multipart/form-data image file
    Response: JSON with predicted_class, confidence_score, class_probabilities
    """
    if classifier is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "Model not loaded on the server. Service unavailable."
            }
        )
        
    if not file:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "No file provided"
            }
        )
        
    temp_path = None
    try:
        # Generate unique temp filename
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        contents = await file.read()
        
        # Validate file is an image
        if not contents:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "File is empty"
                }
            )
        
        # Force convert blob to proper RGB image
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img.save(temp_path)
            
        # Run classification
        result = classifier.classify_image(temp_path)
        
        # Return proper JSON response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "predicted_class": result['class'],
                "confidence_score": float(result['confidence']),
                "class_probabilities": {
                    k: float(v) for k, v in result['all_probabilities'].items()
                }
            }
        )
        
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Prediction failed: {str(e)}"
            }
        )
    finally:
        # Always clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.options("/gradcam")
async def options_gradcam():
    """Handle CORS preflight requests for /gradcam"""
    return JSONResponse(status_code=200, content={})

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    """
    Generate GradCAM visualization for model explainability.
    
    Request: POST /gradcam with multipart/form-data image file
    Response: JSON with base64-encoded heatmap image
    """
    if classifier is None:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "Model not loaded on the server. Service unavailable."
            }
        )
        
    temp_path = None
    try:
        # Generate unique temp filename
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        contents = await file.read()
        
        # Validate file is an image
        if not contents:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "File is empty"
                }
            )
        
        # Force convert blob to proper RGB image
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img.save(temp_path)
            
        # Generate GradCAM heatmap
        gradcam_img = classifier.generate_gradcam(temp_path)
        
        # Convert numpy array to base64
        gradcam_pil = Image.fromarray(np.uint8(gradcam_img))
        buffered = io.BytesIO()
        gradcam_pil.save(buffered, format="JPEG", quality=90)
        gradcam_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Return proper JSON response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "gradcam_base64": gradcam_b64,
                "format": "jpeg"
            }
        )
        
    except Exception as e:
        print(f"Error in /gradcam: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"GradCAM generation failed: {str(e)}"
            }
        )
    finally:
        # Always clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.get("/metrics")
async def metrics():
    """Get model performance metrics"""
    metrics_path = './classifier_checkpoints/metrics.json'
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "metrics": metrics_data
                }
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Failed to read metrics: {str(e)}"
                }
            )
    
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Metrics not found"
        }
    )

@app.get("/generated-images")
async def generated_images():
    """Get list of GAN-generated images"""
    output_dir = './output'
    try:
        if not os.path.exists(output_dir):
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "urls": []
                }
            )
            
        images = [f for f in os.listdir(output_dir) if f.startswith('generated_epoch_') and f.endswith('.png')]
        # Sort logically by epoch number
        images.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        
        image_urls = [f"/assets/{img}" for img in images]
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "urls": image_urls
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Failed to retrieve images: {str(e)}"
            }
        )

@app.get("/assets/{filename}")
async def fetch_asset(filename: str):
    """Fetch asset files (images, etc)"""
    classifier_path = os.path.join('./classifier_checkpoints', filename)
    output_path = os.path.join('./output', filename)
    
    if os.path.exists(classifier_path):
        return FileResponse(classifier_path)
    elif os.path.exists(output_path):
        return FileResponse(output_path)
    
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Asset not found"
        }
    )

if __name__ == '__main__':
    import uvicorn
    # Use 0.0.0.0 for Render deployment, localhost for development
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
