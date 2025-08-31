# FaceDetect

Face Detect is a web application that allows you to **index a library of headshots** and **classify uploaded photos** based on facial similarity using [InsightFace](https://github.com/deepinsight/insightface). It supports **threshold adjustments** for classification updates.

---

## Features

- Drag-and-drop interface for **library headshots** and **photos to classify**.
- Automatically extracts face embeddings using InsightFace.
- Compares uploaded photos against indexed library images.
- Supports **updates** via threshold slider.
- Displays **matches** and **non-matches** with thumbnails.

---

## Tech Stack

- **Backend:** FastAPI, OpenCV, InsightFace, NumPy
- **Frontend:** HTML, CSS, Vanilla JavaScript
- **Storage:** Local file system (for library and uploaded photos)

---