# FILE: main.py (THE UNIFIED SERVER)
import os
import json
import io
import traceback
import numpy as np
import tensorflow as tf
import faiss
from PIL import Image
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager
import uvicorn

# ==========================================================
# CONFIGURATION
# ==========================================================
VECTOR_DIM = 1280  # MobileNetV2 Output
INDEX_FILE = "faiss.index"
LABELS_FILE = "labels.json"
MATCH_THRESHOLD = 0.70 

# ==========================================================
# GLOBAL RESOURCES
# ==========================================================
model = None

faiss_index = None
LABELS = []

def load_resources():
    global model, faiss_index, LABELS
    print("SYSTEM: Loading MobileNetV2 AI Model...")
    
    # Load Model
    model = tf.keras.applications.MobileNetV2(
        include_top=False, pooling='avg', input_shape=(224, 224, 3)
    )
    model.predict(np.zeros((1, 224, 224, 3), dtype="float32"))
    print("SYSTEM: Model Ready!")

    # Load Database
    if os.path.exists(INDEX_FILE) and os.path.exists(LABELS_FILE):
        try:
            faiss_index = faiss.read_index(INDEX_FILE)
            with open(LABELS_FILE, "r") as f:
                LABELS = json.load(f)
            print(f"SYSTEM: Database loaded with {len(LABELS)} products.")
        except Exception as e:
            print(f"SYSTEM: DB Error ({e}). Creating fresh DB.")
            faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
            LABELS = []
    else:
        print("SYSTEM: No Database found. Starting fresh.")
        faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
        LABELS = []

def extract_embedding(pil_img):
    img = pil_img.resize((224, 224))
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    embedding = model.predict(arr)[0]
    return (embedding / np.linalg.norm(embedding)).astype("float32")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield

app = FastAPI(title="Vision Unified Server", lifespan=lifespan)

# ==========================================================
# 1. UI PAGE (Upload / Train)
# ==========================================================
@app.get("/", response_class=HTMLResponse)
def home():
    """Returns the HTML page to upload images."""
    # UPDATED: The form action is now "/add-images/" to match your Python function
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vision Server</title>
        <style>
            body { font-family: sans-serif; background-color: #222; color: white; padding: 50px; text-align: center; }
            .container { background-color: #333; padding: 40px; border-radius: 10px; display: inline-block; }
            input { padding: 10px; margin: 10px; }
            button { background-color: green; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📷 Vision AI Server</h1>
            <h3>Train New Product</h3>
            <form action="/add-images/" method="post" enctype="multipart/form-data">
                <input type="text" name="product_id" placeholder="Product ID (e.g., 810)" required><br><br>
                <input type="file" name="files" multiple required><br><br>
                <button type="submit">Upload & Store</button>
            </form>
        </div>
    </body>
    </html>
    """

# ==========================================================
# 2. TRAIN API (Store Images)
# ==========================================================

@app.post("/add-images/", response_class=HTMLResponse)
async def add_images(product_id: str = Form(...), files: List[UploadFile] = File(...)):
    """Receives images, converts to vectors, and saves to DB."""
    global LABELS, faiss_index
    count = 0
    print(f"TRAIN: Processing images for Product ID {product_id}...")

    for file in files:
        try:
            content = await file.read()
            pil_img = Image.open(io.BytesIO(content))
            vec = extract_embedding(pil_img)
            
            faiss_index.add(np.array([vec], dtype="float32"))
            LABELS.append(product_id)
            count += 1
        except Exception as e:
            print(f"Skipped file due to error: {e}")

    # Save to disk immediately
    if count > 0:
        faiss.write_index(faiss_index, INDEX_FILE)
        with open(LABELS_FILE, "w") as f:
            json.dump(LABELS, f)
    
    # Return JSON so the browser sees the result
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": f"Saved {count} images for Product ID: {product_id}",
            "product_id": product_id,
            "uploaded_images": count
        }
    )

# ==========================================================
# 3. COMPARE API (Check Images)
# ==========================================================
@app.post("/compare/")
async def compare_image(file: UploadFile = File(...), expected_id: Optional[str] = Form(None)):
    """Receives an image (and optional ID) -> Returns match result."""
    
    # Check if DB is empty
    if not LABELS or faiss_index.ntotal == 0:
        return JSONResponse(status_code=400, content={"error": "Database is empty. Train first."})

    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content))
        vec = extract_embedding(pil_img)

        # Search for closest match
        distances, indices = faiss_index.search(np.array([vec], dtype="float32"), 1)
        best_idx = int(indices[0][0])
        best_dist = float(distances[0][0])
        
        if best_idx == -1: return {"match": False, "message": "Index Error"}

        found_pid = LABELS[best_idx]
        
        # --- VERIFICATION LOGIC (User provided an ID) ---
        if expected_id:
            # 1. Identity Check
            if found_pid != expected_id:
                msg = f"WRONG ITEM: Expected {expected_id}, found {found_pid}"
                print(f"COMPARE: {msg}")
                return {
                    "match": False,
                    "product_id": found_pid,
                    "confidence_score": round(best_dist, 4),
                    "message": msg
                }
            
            # 2. Quality Check
            if best_dist > MATCH_THRESHOLD:
                msg = f"LOW CONFIDENCE: Looks like {found_pid} but unsure ({best_dist:.2f})"
                print(f"COMPARE: {msg}")
                return {
                    "match": False,
                    "product_id": found_pid,
                    "confidence_score": round(best_dist, 4),
                    "message": msg
                }
            
            # 3. Success
            print(f"COMPARE: ✅ VERIFIED {found_pid}")
            return {
                "match": True,
                "product_id": found_pid,
                "confidence_score": round(best_dist, 4),
                "message": "✅ VERIFIED"
            }

        # --- IDENTIFICATION LOGIC (No ID provided) ---
        is_match = best_dist < MATCH_THRESHOLD
        print(f"COMPARE: Best guess {found_pid} ({best_dist:.4f}) -> Match: {is_match}")
        return {
            "match": is_match,
            "product_id": found_pid,
            "confidence_score": round(best_dist, 4)
        }

    except Exception as e:
        print("!!! SERVER ERROR !!!")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5054, reload=False)