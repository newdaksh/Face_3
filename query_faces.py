import os
import argparse
import pickle
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from PIL import Image
import faiss
def _preprocess_for_model(face_np, target_size=(160,160)):
    im = Image.fromarray(face_np).resize(target_size)
    arr = np.asarray(im).astype('float32')
    arr = (arr - 127.5) / 128.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def _get_embedding_from_model(model, preproc):
    try:
        if hasattr(model, 'predict'):
            out = model.predict(preproc)
            return np.asarray(out[0], dtype=np.float32)
        if hasattr(model, 'model') and hasattr(model.model, 'predict'):
            out = model.model.predict(preproc)
            return np.asarray(out[0], dtype=np.float32)
        if callable(model):
            out = model(preproc)
            return np.asarray(out[0], dtype=np.float32)
        for name in ('get_embedding', 'get_feature', 'extract', 'embeddings', 'forward'):
            fn = getattr(model, name, None)
            if callable(fn):
                out = fn(preproc)
                if isinstance(out, (list, tuple, np.ndarray)):
                    return np.asarray(out[0], dtype=np.float32)
                return np.asarray(out, dtype=np.float32)
    except Exception:
        raise
    raise RuntimeError('Unable to obtain embedding from model.')

# Utils (same as index file)
def crop_face_from_bbox(pil_img, box, margin=0.25):
    x, y, w, h = box
    img_w, img_h = pil_img.size
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(img_w, x + w + mx)
    y2 = min(img_h, y + h + my)
    return pil_img.crop((x1, y1, x2, y2)), (x1, y1, x2 - x1, y2 - y1)

def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def load_index(index_dir):
    index_path = os.path.join(index_dir, "face_index.index")
    embeddings_path = os.path.join(index_dir, "embeddings.npy")
    meta_path = os.path.join(index_dir, "meta.pkl")
    index = faiss.read_index(index_path)
    embs = np.load(embeddings_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, embs, meta

def embed_faces_in_image(img_path, model_name='Facenet', model=None, detector=None):
    pil_img = Image.open(img_path).convert("RGB")
    detector = detector or MTCNN()
    results = detector.detect_faces(np.array(pil_img))
    if not results:
        return []
    out_embs = []
    for i, res in enumerate(results):
        try:
            face_img, bbox = crop_face_from_bbox(pil_img, res['box'])
            # Compute embedding via DeepFace.represent (or fallback)
            try:
                rep = DeepFace.represent(face_img, model_name=model_name, model=model, enforce_detection=False, detector_backend='skip')
            except Exception:
                face_np = np.asarray(face_img)
                try:
                    preproc = _preprocess_for_model(face_np, target_size=(160,160))
                except Exception:
                    preproc = _preprocess_for_model(face_np, target_size=(160,160))
                emb_vec = _get_embedding_from_model(model, preproc)
                rep = emb_vec.tolist()

            if isinstance(rep, dict) and 'embedding' in rep:
                vec = np.asarray(rep['embedding'], dtype=np.float32)
            else:
                if isinstance(rep, list) and len(rep)>0 and isinstance(rep[0], (list, tuple, np.ndarray)):
                    vec = np.asarray(rep[0], dtype=np.float32)
                else:
                    vec = np.asarray(rep, dtype=np.float32)

            vec = l2_normalize(vec)
            out_embs.append((vec, bbox))
        except Exception as e:
            print("face embed failed", e)
            continue
    return out_embs

def query(index_dir, query_image, top_k=10, threshold=0.4, model_name='Facenet'):
    print("Loading index...")
    index, embs, meta = load_index(index_dir)
    print("Loading DeepFace model...")
    model = DeepFace.build_model(model_name)
    detector = MTCNN()

    q_embs = embed_faces_in_image(query_image, model_name=model_name, model=model, detector=detector)
    if not q_embs:
        print("No faces found in query image.")
        return []

    # if multiple faces in query, decide policy: take first, or average
    # We'll average embeddings of all faces in query (works when query contains multiple photos of same person or multiple angles)
    q_vectors = np.array([v for v, bbox in q_embs], dtype=np.float32)
    q_vec = np.mean(q_vectors, axis=0)
    q_vec = l2_normalize(q_vec).astype('float32')
    q_vec = q_vec.reshape(1, -1)

    # Search
    D, I = index.search(q_vec, top_k)  # inner products (cosine)
    scores = D[0]  # similarity scores (cosine since vectors are normalized)
    indices = I[0]
    results = []
    for sim, idx in zip(scores, indices):
        if idx < 0: continue
        if sim < threshold:
            continue
        info = meta[idx].copy()
        info['score'] = float(sim)
        results.append(info)
    # dedupe files (we may get multiple faces from same file). We'll return top result per file with max score
    by_file = {}
    for r in results:
        f = r['file']
        if f not in by_file or r['score'] > by_file[f]['score']:
            by_file[f] = r
    results_unique = sorted(by_file.values(), key=lambda x: -x['score'])
    return results_unique

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.45, help="cosine similarity threshold (0..1)")
    parser.add_argument("--model", default="Facenet")
    args = parser.parse_args()

    res = query(args.index_dir, args.query, args.top_k, args.threshold, args.model)
    if not res:
        print("No matches found above threshold (0.40).")
    else:
        print("Matches (file, score):")
        for r in res:
            print(r['file'], round(r['score'], 4), r['bbox'])
