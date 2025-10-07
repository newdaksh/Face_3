# Face Search GUI

A simple Tkinter GUI to query a face index built with `index_faces.py` and `query_faces.py`.

Features:

- Select a query image
- Configure index directory, top_k and threshold
- View matched image thumbnails
- Click thumbnails to open an enlarged preview

Requirements

- Python 3.8+
- Packages from `requirements.txt` (DeepFace, mtcnn, faiss, pillow, tqdm, numpy)

Run

1. Ensure you already built the index (folder `face_index` with `face_index.index`, `embeddings.npy`, `meta.pkl`).
2. From the project folder run:

```
python gui.py
```

Usage notes

- By default the GUI assumes dataset images are located in a sibling folder `dataset_images` next to the index directory. If your dataset files are stored elsewhere, update the `Index dir` field accordingly.
- If thumbnails don't appear, check the 'file' entries in `face_index/meta.pkl` to see how file paths were stored (relative vs absolute) and adjust accordingly.

Troubleshooting

- If DeepFace model download occurs on first run it may take some time. Ensure you have internet access the first time.
- On Windows, FAISS may not be available prebuilt. If faiss import fails, consider using a CPU-only faiss wheel or run the query script in an environment where faiss is installed.
