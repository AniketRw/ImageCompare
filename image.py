import os
import json
import io
import traceback
import numpy as np
import faiss
from PIL import Image
from typing import Optional
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
MATCH_THRESHOLD = 0.40 
TOP_K_NEIGHBORS = 5 

SQL_SERVER = ""; SQL_DATABASE = ""; SQL_USER = ""; SQL_PASSWORD = ""
ort_session = None; faiss_index = None; LABELS = []

def get_base_path():
    return os.path.dirname(os.path.abspath(__file__))

# --- Database & Model Functions ---

def load_db_config():
    """Loads database configuration from config.ini."""
    global SQL_SERVER, SQL_DATABASE, SQL_USER, SQL_PASSWORD
    CONFIG_FILE = os.path.join(get_base_path(), "config.ini")
    
    if not os.path.exists(CONFIG_FILE):
        print("SYSTEM: Configuration file (config.ini) not found.")
        return False
    
    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)
        SQL_SERVER = config['DATABASE']['SERVER']
        SQL_DATABASE = config['DATABASE']['DATABASE']
        SQL_USER = config['DATABASE']['USER']
        SQL_PASSWORD = config['DATABASE']['PASSWORD']
        return True
    except Exception as e:
        print(f"SYSTEM: Error reading config.ini: {e}")
        return False

def get_sql_connection():
    """Establishes SQL connection (with error logging)."""
    try:
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SQL_SERVER};DATABASE={SQL_DATABASE};UID={SQL_USER};PWD={SQL_PASSWORD};Encrypt=no;TrustServerCertificate=yes;"     
        return pyodbc.connect(conn_str)
    except Exception as e: 
        # CRITICAL: Print the detailed error for debugging
        print("\n--- CRITICAL SQL CONNECTION ERROR ---")
        print(f"Error Details: {e}") 
        print(f"Connection String used: {conn_str}")
        print("-------------------------------------\n")
        return None


def get_product_name(pid):
    """Fetches product name from SQL based on ID."""
    conn = get_sql_connection()
    if not conn: return None, "SQL Error"
    try:
        cur = conn.cursor()
        # NOTE: Updated SQL query based on your last provided code
        cur.execute("SELECT TOP 1 Name FROM rplus_productmaster WHERE ProductID=?", pid)
        row = cur.fetchone()
        conn.close()
        return (row[0], None) if row else (None, "ID Not Found")
    except: conn.close(); return None, "DB Query Error"

def extract_embedding(pil_img):
    """Preprocesses image and extracts embedding using ONNX model."""
    global ort_session
    if ort_session is None: raise Exception("ONNX model not loaded.")
    
    # Resizing and Preprocessing
    w, h = pil_img.size
    crop_factor = 1.0 
    new_w, new_h = w * crop_factor, h * crop_factor
    img_cropped = pil_img.crop(((w - new_w)/2, (h - new_h)/2, (w + new_w)/2, (h + new_h)/2))
    img = img_cropped.resize((224, 224))
    if img.mode != 'RGB': img = img.convert('RGB')
    
    # ResNet V2 Preprocessing
    x = np.array(img).astype(np.float32)
    x = (x - 127.5) / 127.5
    arr = np.expand_dims(x, axis=0)
    
    inputs = {ort_session.get_inputs()[0].name: arr}
    embedding = ort_session.run(None, inputs)[0][0]
    return (embedding / np.linalg.norm(embedding)).astype("float32")

def load_resources():
    """Initializes ONNX model and Faiss database."""
    global ort_session, faiss_index, LABELS
    
    # 1. Load Config
    load_db_config()

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
            print(f"SYSTEM: Database loaded with {len(LABELS)} products.")
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

app = FastAPI(title="Vision Compare Server", lifespan=lifespan)

# ==========================================================
# 1. COMPARE API (The core validation endpoint)
# ==========================================================
@app.post("/compare/")
async def compare_image(file: UploadFile = File(...), expected_id: str = Form(...)):
    """Receives an image and expected ID -> Returns match result with product name."""
    
    # Check server status
    if not LABELS or faiss_index.ntotal == 0 or ort_session is None:
        return JSONResponse(status_code=503, content={"match": False, "message": "Server/Database not ready."})

    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content))
        vec = extract_embedding(pil_img)

        # --- Initial Checks and Name Fetch ---
        pname, error = get_product_name(expected_id)
        if error == "SQL Error" or error == "DB Query Error":
            return JSONResponse(status_code=500, content={"match": False, "message": "SQL connection error."})
        if pname is None and error == "ID Not Found":
            return JSONResponse(status_code=400, content={"match": False, "message": f"Expected ID {expected_id} not found in SQL."})
        if expected_id not in LABELS:
            return JSONResponse(status_code=400, content={"match": False, "message": f"ID {expected_id} has no trained images."})

        # --- ROBUST MATCHING LOGIC (Top K Search) ---
        
        # Search the top K nearest neighbors
        k = TOP_K_NEIGHBORS
        D, I = faiss_index.search(np.array([vec], dtype="float32"), k)
        
        best_dist = float(D[0][0])
        match_found = False
        
        # Check if ANY of the top 'k' results match the expected ID AND pass the distance threshold
        for i in range(k):
            # Ensure index is valid and distance is below threshold
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
            best_match_id = LABELS[I[0][0]] # The ID of the closest item found
            print(f"COMPARE: ❌ FAILED {expected_id} (Closest: {best_match_id}, Dist: {best_dist:.4f})")
            
            return {
                "match": False,
                "product_id": expected_id,
                "product_name": pname,
                "distance": round(best_dist, 4),
                "message": f"❌ FAILED. Closest match found was ID {best_match_id}."
            }

    except Exception as e:
        print("!!! SERVER ERROR !!!")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"match": False, "message": f"Internal server error: {e}"})

# ==========================================================
# 2. IDENTIFY API (Blind search for the closest match)
# ==========================================================
@app.post("/identify/")
async def identify_image(file: UploadFile = File(...)):
    """Receives an image and returns the closest matching product name from the vector database."""

    # Check server status
    if not LABELS or faiss_index.ntotal == 0 or ort_session is None:
        return JSONResponse(status_code=503, content={"match": False, "message": "Server/Database not ready or empty."})

    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content))
        vec = extract_embedding(pil_img)

        # --- Blind Search Logic ---

        # Search the single nearest neighbor (K=1)
        k = 1
        D, I = faiss_index.search(np.array([vec], dtype="float32"), k)

        best_dist = float(D[0][0])
        closest_index = I[0][0]
        closest_id = LABELS[closest_index]

        # Check if the closest match is within the acceptable distance threshold
        if best_dist < MATCH_THRESHOLD:
            # Match found, now fetch the product name from SQL
            pname, error = get_product_name(closest_id)
            
            # Note: We must handle SQL errors here too, as the identification is done, but name fetch failed
            if error == "SQL Error" or error == "DB Query Error":
                 pname = "SQL Name Fetch Failed"

            print(f"IDENTIFY: ✅ MATCH FOUND (ID: {closest_id}, Dist: {best_dist:.4f})")
            return {
                "match": True,
                "identified_id": closest_id,
                "product_name": pname,
                "message": f"Successfully identified product: {pname}"
            }
        else:
            print(f"IDENTIFY: ❌ NO MATCH (Closest Dist: {best_dist:.4f})")
            return {
                "match": False,
                "identified_id": None,
                "product_name": None,
                "message": "❌ FAILED. Closest vector was outside the matching threshold."
            }

    except Exception as e:
        print("!!! IDENTIFY SERVER ERROR !!!")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"match": False, "message": f"Internal server error: {e}"})

@app.get("/refresh/")
def refresh_db():
    # This function would call the Faiss and LABELS loading logic
    # You would call this API endpoint immediately after training is complete.
    load_resources() 
    return {"status": "success", "message": "Database reloaded from disk."}


# ==========================================================
# 3. STATUS API (Optional Health Check)
# ==========================================================
@app.get("/status/")
def status_check():
    return {
        "status": "online",
        "db_size": faiss_index.ntotal if faiss_index else 0,
        "model_loaded": ort_session is not None,
        "vector_dim": VECTOR_DIM,
        "match_threshold": MATCH_THRESHOLD,
        "top_k_check": TOP_K_NEIGHBORS
    }


if __name__ == "__main__":
    # Runs the comparison server on port 5055
    uvicorn.run("image:app", host="0.0.0.0", port=5056, reload=False)