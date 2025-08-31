import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import shutil
import uuid
import insightface
import cv2
import numpy as np

# -----------------------------
# Setup paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
LIBRARY_DIR = os.path.join(BASE_DIR, "library_headshots")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_photos")

os.makedirs(LIBRARY_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# Initialize app and model
# -----------------------------
app = FastAPI(title="Face Finder")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

from fastapi.staticfiles import StaticFiles

# Serve uploaded photos so frontend can access them
app.mount("/uploaded_photos", StaticFiles(directory=UPLOAD_DIR), name="uploaded_photos")


# Load InsightFace model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1)  # CPU

# -----------------------------
# Utility functions
# -----------------------------
def save_upload(file: UploadFile, dest_folder: str) -> str:
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(dest_folder, filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return path

def extract_face_embedding(img_path: str):
    img = cv2.imread(img_path)
    faces = model.get(img)
    if not faces:
        return None
    return faces[0].embedding  # single face assumed per image

# -----------------------------
# Build library embeddings
# -----------------------------
library_embeddings = {}

def build_library_embeddings():
    library_embeddings.clear()
    for file in os.listdir(LIBRARY_DIR):
        path = os.path.join(LIBRARY_DIR, file)
        img = cv2.imread(path)
        faces = model.get(img)
        if faces:
            library_embeddings[file] = faces[0].embedding

# Call this once on startup
build_library_embeddings()

# -----------------------------
# Serve frontend
# -----------------------------
@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# -----------------------------
# API: Index library headshots
# -----------------------------
@app.post("/api/index")
async def index_files(files: List[UploadFile] = File(...)):
    photos_saved = 0
    embeddings_indexed = 0

    for file in files:
        path = save_upload(file, LIBRARY_DIR)
        photos_saved += 1
        emb = extract_face_embedding(path)
        if emb is not None:
            embeddings_indexed += 1
            library_embeddings[path] = emb.tolist()

    return JSONResponse({
        "photos_saved": photos_saved,
        "faces_indexed": embeddings_indexed,
        "index_size": len(library_embeddings)
    })

# -----------------------------
# API: Search uploaded image
# -----------------------------
@app.post("/api/search")
async def search_file(
    file: UploadFile = File(...),
    threshold: float = Form(..., gt=0, lt=1)
):
    uploaded_path = save_upload(file, UPLOAD_DIR)
    uploaded_emb = extract_face_embedding(uploaded_path)
    if uploaded_emb is None:
        return JSONResponse({"message": "No faces detected."})

    # Compare to library
    matches = []
    for lib_file, lib_emb in library_embeddings.items():
        dist = np.linalg.norm(np.array(uploaded_emb) - np.array(lib_emb))
        if dist < threshold:
            matches.append({"file": os.path.basename(lib_file), "score": float(dist)})

    return JSONResponse({
        "matches": matches,
        "count": len(matches)
    })

# -----------------------------
# API: Classify photos
# -----------------------------
@app.post("/api/classify_photos")
async def classify_photos(files: List[UploadFile] = File(...), threshold: float = 0.6):
    matches = []
    no_matches = []

    for file in files:
        # Save file with UUID
        file_path = save_upload(file, UPLOAD_DIR)
        img = cv2.imread(file_path)
        faces = model.get(img)
        contains_known = False

        for face in faces:
            embedding = face.embedding
            for lib_file, lib_emb in library_embeddings.items():
                similarity = np.dot(embedding, lib_emb) / (np.linalg.norm(embedding) * np.linalg.norm(lib_emb))
                if similarity >= threshold:
                    contains_known = True
                    break
            if contains_known:
                break

        # Use saved filename (UUID) instead of original
        saved_filename = os.path.basename(file_path)
        if contains_known:
            matches.append(saved_filename)
        else:
            no_matches.append(saved_filename)

    return {"matches": matches, "no_matches": no_matches}