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
MATCH_THRESHOLD = 0.650
TOP_K_NEIGHBORS = 5 

# Global state variables
SQL_SERVER = ""; SQL_DATABASE = ""; SQL_USER = ""; SQL_PASSWORD = ""
ort_session = None; faiss_index = None; LABELS = []
CONFIG_FILE = "" 

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
    """Establishes SQL connection (with detailed logging on failure)."""
    try:
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};UID={SQL_USER};PWD={SQL_PASSWORD};Encrypt=no;TrustServerCertificate=yes;"     
        return pyodbc.connect(conn_str)
    except Exception as e: 
        print(f"\n--- CRITICAL SQL CONNECTION ERROR --- \nError Details: {e}\n-------------------------------------\n")
        return None

def get_product_name(pid):
    """Fetches product name from SQL based on ID."""
    conn = get_sql_connection()
    if not conn: return None, "SQL Error"
    try:
        cur = conn.cursor()
        # Using 'Products' table name for consistency with your Training API logic
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
    
    # 1. Image Preprocessing
    w, h = pil_img.size
    crop_factor = 1.0 
    new_w, new_h = w * crop_factor, h * crop_factor
    img_cropped = pil_img.crop(((w - new_w)/2, (h - new_h)/2, (w + new_w)/2, (h + new_h)/2))
    img = img_cropped.resize((224, 224))
    if img.mode != 'RGB': img = img.convert('RGB')
        
    # 2. ResNet V2 Preprocessing (Normalization)
    x = np.array(img).astype(np.float32)
    x = (x - 127.5) / 127.5
    arr = np.expand_dims(x, axis=0) # Shape: (1, 224, 224, 3) - NHWC
    
    inputs = {ort_session.get_inputs()[0].name: arr}

    # 3. Run ONNX session
    embedding = ort_session.run(None, inputs)[0][0] 
    
    # L2 Normalization
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

app = FastAPI(title="Unified Vision Server (Train/Compare/Identify)", lifespan=lifespan)

# ==========================================================
# 1. TRAINING ENDPOINT (formerly main.py)
# ==========================================================

@app.post("/train/add-images/")
async def add_images(
    product_id: str = Form(...),
    product_name: str = Form(...), 
    files: List[UploadFile] = File(...)
):
    """Receives images, verifies product ID, converts to vectors, and saves to DB."""
    global LABELS, faiss_index
    count = 0
    
    if not ort_session or faiss_index is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Server resources (model/database) not fully initialized."})

    # --- 1. SQL Check and ID Verification ---
    pname_sql, error = get_product_name(product_id) 
    
    if error == "SQL Error" or error == "DB Query Error":
        return JSONResponse(status_code=500, content={"status": "error", "message": f"SQL Error: {error}. Check config/DB structure."})
    if pname_sql is None and error == "ID Not Found":
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Product ID {product_id} not found in SQL database."})
    
    print(f"TRAIN: Processing {len(files)} images for ID {product_id}...")

    # --- 2. Process and Store Images ---
    for file in files:
        try:
            content = await file.read() 
            pil_img = Image.open(io.BytesIO(content)) 
            vec = extract_embedding(pil_img)
            
            # Add to Faiss Index and Label list
            faiss_index.add(np.array([vec], dtype="float32"))
            LABELS.append(product_id)
            count += 1
        except Exception as e:
            print(f"Skipped file '{file.filename}' due to error: {e}")
            traceback.print_exc()

    # --- 3. Save to Disk ---
    if count > 0:
        try:
            faiss.write_index(faiss_index, INDEX_FILE)
            with open(LABELS_FILE, "w") as f:
                json.dump(LABELS, f)
            print(f"TRAIN: Successfully saved {count} embeddings. Total: {faiss_index.ntotal}")
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": f"Failed to save embeddings to disk: {e}"})

    # --- 4. Success Response ---
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": f"Successfully stored {count} embeddings for {product_name} (ID: {product_id}).",
            "total_vectors": faiss_index.ntotal
        }
    )

# ==========================================================
# 2. COMPARISON ENDPOINT (formerly compare_api.py / validation)
# ==========================================================

@app.post("/compare/")
async def compare_image(file: UploadFile = File(...), expected_id: str = Form(...)):
    """Receives an image and expected ID -> Returns match result with product name."""
    
    if not LABELS or faiss_index.ntotal == 0 or ort_session is None:
        return JSONResponse(status_code=503, content={"match": False, "message": "Server/Database not ready."})

    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content))
        vec = extract_embedding(pil_img)

        # --- Initial Checks and Name Fetch ---
        pname, error = get_product_name(expected_id)
        if error == "SQL Error" or error == "DB Query Error":
            return JSONResponse(status_code=500, content={"match": False, "message": "SQL connection/query error."})
        if pname is None and error == "ID Not Found":
            return JSONResponse(status_code=400, content={"match": False, "message": f"Expected ID {expected_id} not found in SQL."})
        if expected_id not in LABELS:
            return JSONResponse(status_code=400, content={"match": False, "message": f"ID {expected_id} has no trained images."})

        # --- ROBUST MATCHING LOGIC (Top K Search) ---
        k = TOP_K_NEIGHBORS
        D, I = faiss_index.search(np.array([vec], dtype="float32"), k)
        
        best_dist = float(D[0][0])
        match_found = False
        
        for i in range(k):
            if D[0][i] < MATCH_THRESHOLD and LABELS[I[0][i]] == expected_id:
                match_found = True
                break 

        # --- Final Response ---
        if match_found:
            print(f"COMPARE: ✅ VERIFIED {expected_id} (Dist: {best_dist:.4f})")
            return {
                "match": True,
                "product_id": expected_id,
                "product_name": pname,
                "distance": round(best_dist, 4),
                "message": "✅ VERIFIED"
            }
        else:
            best_match_id = LABELS[I[0][0]]
            print(f"COMPARE: ❌ FAILED {expected_id} (Closest: {best_match_id}, Dist: {best_dist:.4f})")
            
            return {
                "match": False,
                "product_id": expected_id,
                "product_name": pname,
                "distance": round(best_dist, 4),
                "message": f"❌ FAILED. Closest match found was ID {best_match_id}."
            }

    except Exception as e:
        print("!!! COMPARE SERVER ERROR !!!")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"match": False, "message": f"Internal server error: {e}"})


# ==========================================================
# 3. IDENTIFY ENDPOINT (The third API / blind search)
# ==========================================================
@app.post("/identify/")
async def identify_image(file: UploadFile = File(...)):
    """Receives an image and returns the closest matching product name from the vector database."""

    if not LABELS or faiss_index.ntotal == 0 or ort_session is None:
        return JSONResponse(status_code=503, content={"match": False, "message": "Server/Database not ready or empty."})

    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content))
        vec = extract_embedding(pil_img)

        # --- Blind Search Logic (K=1) ---
        k = 1
        D, I = faiss_index.search(np.array([vec], dtype="float32"), k)

        best_dist = float(D[0][0])
        closest_index = I[0][0]
        closest_id = LABELS[closest_index]

        # Check if the closest match is within the acceptable distance threshold
        if best_dist < MATCH_THRESHOLD:
            # Match found, now fetch the product name from SQL
            pname, error = get_product_name(closest_id)
            
            if error == "SQL Error" or error == "DB Query Error":
                 pname = "SQL Name Fetch Failed"

            print(f"IDENTIFY: ✅ MATCH FOUND (ID: {closest_id}, Dist: {best_dist:.4f})")
            return {
                "match": True,
                "identified_id": closest_id,
                "product_name": pname,
                "distance": round(best_dist, 4),
                "message": f"Successfully identified product: {pname}"
            }
        else:
            print(f"IDENTIFY: ❌ NO MATCH (Closest Dist: {best_dist:.4f})")
            return {
                "match": False,
                "identified_id": None,
                "product_name": None,
                "distance": round(best_dist, 4),
                "message": "❌ FAILED. Closest vector was outside the matching threshold."
            }

    except Exception as e:
        print("!!! IDENTIFY SERVER ERROR !!!")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"match": False, "message": f"Internal server error: {e}"})

# ==========================================================
# 4. STATUS & REFRESH ENDPOINTS (Health and Data Sync)
# ==========================================================

@app.get("/refresh/")
def refresh_db():
    """Forces the server to immediately reload the Faiss index and labels from disk."""
    load_resources() 
    return {
        "status": "success", 
        "message": f"Database reloaded from disk. New vector count: {faiss_index.ntotal if faiss_index else 0}"
    }

@app.get("/status/")
def status_check():
    """Combined health check for the server."""
    return {
        "status": "online",
        "db_size": faiss_index.ntotal if faiss_index else 0,
        "model_loaded": ort_session is not None,
        "vector_dim": VECTOR_DIM,
        "match_threshold": MATCH_THRESHOLD,
        "top_k_check": TOP_K_NEIGHBORS
    }


if __name__ == "__main__":
    # Runs the unified server on a single port (e.g., 5054 or 5055/5056)
    uvicorn.run("main:app", host="0.0.0.0", port=5054, reload=False)