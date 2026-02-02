# FILE: main.py (The Training API Server)

import os
import json
import io
import traceback
import numpy as np
import faiss
from PIL import Image
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import configparser
import pyodbc
import onnxruntime as ort

# ==========================================================
# CONFIGURATION & GLOBAL RESOURCES
# ==========================================================
VECTOR_DIM = 2048 
INDEX_FILE = "faiss.index"
LABELS_FILE = "labels.json"
ONNX_MODEL_FILE = "resnet50v2.onnx"

# Global state variables
SQL_SERVER = ""; SQL_DATABASE = ""; SQL_USER = ""; SQL_PASSWORD = ""
ort_session = None; faiss_index = None; LABELS = []
CONFIG_FILE = "" # To be set in load_db_config

def get_base_path():
    """Helper for getting the script's directory."""
    return os.path.dirname(os.path.abspath(__file__))

# --- Database & Model Functions ---

def load_db_config():
    """Loads database configuration from config.ini."""
    global SQL_SERVER, SQL_DATABASE, SQL_USER, SQL_PASSWORD, CONFIG_FILE
    CONFIG_FILE = os.path.join(get_base_path(), "config.ini")
    
    if not os.path.exists(CONFIG_FILE):
        return False, "Configuration file (config.ini) not found."
    
    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        SQL_SERVER = config['DATABASE']['SERVER']
        SQL_DATABASE = config['DATABASE']['DATABASE']
        SQL_USER = config['DATABASE']['USER']
        SQL_PASSWORD = config['DATABASE']['PASSWORD']
        return True, None
    except Exception as e:
        return False, f"Error reading config.ini or missing [DATABASE] section: {e}"

def get_sql_connection():
    """Establishes SQL connection."""
    # Ensure Encrypt=no and TrustServerCertificate=yes are correct for your SQL setup
    try:
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};UID={SQL_USER};PWD={SQL_PASSWORD};Encrypt=no;TrustServerCertificate=yes;"
        return pyodbc.connect(conn_str)
    except: 
        return None

def get_product_name(pid):
    """Fetches product name from SQL based on ID."""
    conn = get_sql_connection()
    if not conn: return None, "SQL Error"
    try:
        cur = conn.cursor()
        # Assumes your SQL table is named 'Products' and the columns are 'ProductName' and 'ProductID'
        cur.execute("SELECT TOP 1 ProductName FROM Products WHERE ProductID=?", pid) 
        row = cur.fetchone()
        conn.close()
        return (row[0], None) if row else (None, "ID Not Found")
    except: 
        conn.close()
        return None, "DB Query Error"

def extract_embedding(pil_img):
    """Preprocesses image and extracts embedding using ONNX model."""
    global ort_session
    if ort_session is None: raise Exception("ONNX model not loaded.")
    
    # 1. Image Preprocessing (ensure 'img' is defined cleanly)
    try:
        w, h = pil_img.size
        crop_factor = 1.0 
        new_w, new_h = w * crop_factor, h * crop_factor
        
        # Crop
        img_cropped = pil_img.crop(((w - new_w)/2, (h - new_h)/2, (w + new_w)/2, (h + new_h)/2))
        
        # Resize/Finalize
        img = img_cropped.resize((224, 224))
        if img.mode != 'RGB': img = img.convert('RGB')
        
    except Exception as e:
        raise Exception(f"Image pre-processing failed: {e}")
        
    # 2. ResNet V2 Preprocessing and Conversion to Tensor (NHWC)
    x = np.array(img).astype(np.float32)
    x = (x - 127.5) / 127.5
    arr = np.expand_dims(x, axis=0) # Shape is (1, 224, 224, 3) - NHWC

    # Transpose is DELETED/OMITTED to keep NHWC format, based on last successful check

    inputs = {ort_session.get_inputs()[0].name: arr}

    # 3. Run ONNX session
    onnx_output = ort_session.run(None, inputs)

    # CRITICAL DEBUG LINE: Check the raw output array shape 
    print(f"DEBUG: Raw ONNX Output Shape: {onnx_output[0].shape}") 

    # Assuming the output is [1, 2048]
    embedding = onnx_output[0][0] 
    
    print(f"DEBUG: Successfully extracted embedding shape: {embedding.shape}")

    return (embedding / np.linalg.norm(embedding)).astype("float32")


def load_resources():
    """Initializes ONNX model and Faiss database."""
    global ort_session, faiss_index, LABELS
    
    # 1. Load Config
    success, msg = load_db_config()
    if not success: print(f"SYSTEM: Config Error: {msg}")

    # 2. Load ONNX Model
    try:
        if not os.path.exists(ONNX_MODEL_FILE):
             print(f"SYSTEM: Missing ONNX Model: {ONNX_MODEL_FILE}")
        ort_session = ort.InferenceSession(ONNX_MODEL_FILE)
        print("SYSTEM: ONNX Model Ready!")
    except Exception as e:
        print(f"SYSTEM: Model Loading Error: {e}")
        
    # 3. Load Faiss Database
    if os.path.exists(INDEX_FILE) and os.path.exists(LABELS_FILE):
        try:
            faiss_index = faiss.read_index(INDEX_FILE)
            with open(LABELS_FILE, "r") as f:
                LABELS = json.load(f)
            print(f"SYSTEM: Database loaded with {len(LABELS)} vectors.")
        except Exception as e:
            print(f"SYSTEM: DB Error ({e}). Creating fresh DB.")
            faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
            LABELS = []
    else:
        print("SYSTEM: No Database found. Starting fresh.")
        faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
        LABELS = []

# --- FastAPI Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield

app = FastAPI(title="Vision Training Server", lifespan=lifespan)

# ==========================================================
# 1. TRAIN API (Store Images)
# ==========================================================

@app.post("/train/add-images/")
async def add_images(
    product_id: str = Form(...),
    product_name: str = Form(...), 
    files: List[UploadFile] = File(...)
):
    """Receives images, verifies product ID against SQL, converts to vectors, and saves to DB."""
    global LABELS, faiss_index
    count = 0
    
    if not ort_session or faiss_index is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Server resources (model/database) not fully initialized."})

    # --- 1. SQL Check and ID Verification ---
    pname_sql, error = get_product_name(product_id) 
    
    if error == "SQL Error":
        return JSONResponse(status_code=500, content={"status": "error", "message": "SQL Connection Failed. Check config.ini."})
    if error == "DB Query Error":
        return JSONResponse(status_code=500, content={"status": "error", "message": "SQL Query Error. Check table/database structure."})
    if pname_sql is None and error == "ID Not Found":
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Product ID {product_id} not found in SQL database."})
    
    print(f"TRAIN: Processing {len(files)} images for Product ID {product_id} (Client Name: {product_name})...")

    # --- 2. Process and Store Images ---
    for file in files:
        try:
            content = await file.read() 
            pil_img = Image.open(io.BytesIO(content)) 
        
            # Extract embedding vector
            vec = extract_embedding(pil_img)
            
            # Add to Faiss Index and Label list
            faiss_index.add(np.array([vec], dtype="float32"))
            LABELS.append(product_id)
            count += 1
        except Exception as e:
            # Enhanced error logging (Critical for debugging)
            print(f"Skipped file '{file.filename}' due to error: {e}")
            traceback.print_exc() # Print full stack trace

    # --- 3. Save to Disk ---
    if count > 0:
        try:
            faiss.write_index(faiss_index, INDEX_FILE)
            with open(LABELS_FILE, "w") as f:
                json.dump(LABELS, f)
            print(f"TRAIN: Successfully saved {count} embeddings.")
        except Exception as e:
            print(f"TRAIN: Error saving DB to disk: {e}")
            return JSONResponse(status_code=500, content={"status": "error", "message": f"Failed to save embeddings to disk: {e}"})

    # --- 4. Success Response ---
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": f"Successfully stored {count} embeddings for {product_name} (ID: {product_id}). Total vectors now: {faiss_index.ntotal}",
            "product_id": product_id,
            "product_name": product_name, 
            "uploaded_images": count
        }
    )

# ==========================================================
# 2. STATUS API (Optional Health Check)
# ==========================================================
@app.get("/status/")
def status_check():
    return {
        "status": "online",
        "db_size": faiss_index.ntotal if faiss_index else 0,
        "model_loaded": ort_session is not None,
        "vector_dim": VECTOR_DIM,
    }


if __name__ == "__main__":
    # Ensure this runs the 'app' instance defined above
    uvicorn.run("main:app", host="0.0.0.0", port=5054, reload=False)