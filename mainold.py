import os
import json
import io
import numpy as np
import tensorflow as tf
import faiss
from PIL import Image
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
import uvicorn

# CONFIG
VECTOR_DIM = 1280
INDEX_FILE = "faiss.index"
LABELS_FILE = "labels.json"

# GLOBALS
model = None
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
LABELS = []

def load_resources():
    global model, faiss_index, LABELS
    print("TRAINER: Loading TensorFlow...")
    model = tf.keras.applications.MobileNetV2(
        include_top=False, pooling='avg', input_shape=(224, 224, 3)
    )
    # Warmup
    model.predict(np.zeros((1, 224, 224, 3), dtype="float32"))

    if os.path.exists(INDEX_FILE) and os.path.exists(LABELS_FILE):
        try:
            faiss_index = faiss.read_index(INDEX_FILE)
            with open(LABELS_FILE, "r") as f:
                LABELS = json.load(f)
            print(f"TRAINER: Loaded existing DB with {len(LABELS)} items.")
        except:
            print("TRAINER: DB file error, starting fresh.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield

app = FastAPI(title="Training Service", lifespan=lifespan)

# ==========================================================
# 1. UI PAGE (The Form)
# ==========================================================
@app.get("/", response_class=HTMLResponse)
def get_upload_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Batch Image Uploader</title>
        <style>
            body { font-family: sans-serif; background-color: #1e1e1e; color: white; padding: 50px; text-align: center; }
            .container { background-color: #2d2d2d; padding: 40px; border-radius: 10px; display: inline-block; }
            input { padding: 10px; margin: 10px; border-radius: 5px; border: none; }
            input[type="text"] { width: 200px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; font-size: 16px; border-radius: 5px; }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚀 Bulk Image Uploader</h2>
            <form action="/add-images/" method="post" enctype="multipart/form-data">
                
                <label>Product ID:</label><br>
                <input type="text" name="product_id" placeholder="Enter ID (e.g., 810)" required>
                <br><br>

                <label>Select Images (Ctrl + A works here):</label><br>
                <input type="file" name="files" multiple required>
                <br><br>

                <button type="submit">UPLOAD NOW</button>
            </form>
        </div>
    </body>
    </html>
    """

# ==========================================================
# 2. BACKEND LOGIC
# ==========================================================
def extract_embedding(pil_img):
    img = pil_img.resize((224, 224))
    if img.mode != "RGB": img = img.convert("RGB")
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    embedding = model.predict(arr)[0]
    return (embedding / np.linalg.norm(embedding)).astype("float32")

@app.post("/add-images/", response_class=HTMLResponse)
async def add_images(product_id: str = Form(...), files: List[UploadFile] = File(...)):
    global LABELS, faiss_index
    count = 0
    
    print(f"Receiving {len(files)} images for ID: {product_id}")

    for file in files:
        try:
            content = await file.read()
            img = Image.open(io.BytesIO(content))
            vec = extract_embedding(img)
            
            faiss_index.add(np.array([vec], dtype="float32"))
            LABELS.append(product_id)
            count += 1
        except Exception as e:
            print(f"Skipped file due to error: {e}")

    # Save to disk
    if count > 0:
        faiss.write_index(faiss_index, INDEX_FILE)
        with open(LABELS_FILE, "w") as f:
            json.dump(LABELS, f)
    
    # --- THIS IS THE POPUP LOGIC ---
    # return f"""
    # <html>
    # <body>
    #     <script>
    #         // 1. Show the popup
    #         alert("✅ SUCCESS! \\n\\nSaved {count} images for Product ID: {product_id}");
            
    #         // 2. Go back to the main page to upload more
    #         window.location.href = "/";
    #     </script>
    # </body>
    # </html>
    # """
   
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": f"Saved {count} images for Product ID: {product_id}",
            "product_id": product_id,
            "uploaded_images": count
        }
    )



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5054, reload=False)