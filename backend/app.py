import os
import shutil
import uuid
from typing import List

import cv2
import numpy as np
import insightface
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
LIBRARY_DIR = os.path.join(BASE_DIR, "library_headshots")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_photos")

os.makedirs(LIBRARY_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# App and model
# -----------------------------
app = FastAPI(title="Face Finder")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
# Serve uploaded photos so the frontend can render result thumbnails.
app.mount("/uploaded_photos", StaticFiles(directory=UPLOAD_DIR), name="uploaded_photos")

# InsightFace runs on CPU (ctx_id=-1).
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1)

# Maps library filename -> face embedding (np.ndarray).
library_embeddings = {}


# -----------------------------
# Helpers
# -----------------------------
def save_upload(file: UploadFile, dest_folder: str) -> str:
    """Save an upload under a random filename and return its path."""
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(dest_folder, filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


def extract_face_embedding(img_path: str):
    """Return the embedding of the first detected face, or None."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = model.get(img)
    if not faces:
        return None
    return faces[0].embedding  # single face assumed per image


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_library_embeddings():
    """Index every image already present in the library directory."""
    library_embeddings.clear()
    for filename in os.listdir(LIBRARY_DIR):
        emb = extract_face_embedding(os.path.join(LIBRARY_DIR, filename))
        if emb is not None:
            library_embeddings[filename] = emb


build_library_embeddings()


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/api/index")
async def index_files(files: List[UploadFile] = File(...)):
    """Add headshots to the searchable library."""
    photos_saved = 0
    faces_indexed = 0

    for file in files:
        path = save_upload(file, LIBRARY_DIR)
        photos_saved += 1
        emb = extract_face_embedding(path)
        if emb is not None:
            faces_indexed += 1
            library_embeddings[os.path.basename(path)] = emb

    return JSONResponse({
        "photos_saved": photos_saved,
        "faces_indexed": faces_indexed,
        "index_size": len(library_embeddings),
    })


@app.post("/api/classify_photos")
async def classify_photos(
    files: List[UploadFile] = File(...),
    threshold: float = Form(0.38),
):
    """Split uploaded photos into those that contain a known face and those that don't."""
    matches = []
    no_matches = []

    for file in files:
        file_path = save_upload(file, UPLOAD_DIR)
        img = cv2.imread(file_path)
        faces = model.get(img) if img is not None else []

        contains_known = any(
            cosine_similarity(face.embedding, lib_emb) >= threshold
            for face in faces
            for lib_emb in library_embeddings.values()
        )

        saved_filename = os.path.basename(file_path)
        (matches if contains_known else no_matches).append(saved_filename)

    return {"matches": matches, "no_matches": no_matches}
