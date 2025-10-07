import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from deepface import DeepFace
import faiss
def _preprocess_for_model(face_np, target_size=(160,160)):
    """Minimal local preprocessing used when deepface.commons.functions is not available.
    Resizes to target_size and scales pixels similar to Facenet preprocessing.
    """
    # face_np is a HxWxC uint8 RGB image
    im = Image.fromarray(face_np).resize(target_size)
    arr = np.asarray(im).astype('float32')
    # Facenet-like scaling
    arr = (arr - 127.5) / 128.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def _get_embedding_from_model(model, preproc):
    """Try several ways to obtain embedding from model object."""
    # prefer predict
    try:
        if hasattr(model, 'predict'):
            out = model.predict(preproc)
            return np.asarray(out[0], dtype=np.float32)
        # Keras-like wrapper
        if hasattr(model, 'model') and hasattr(model.model, 'predict'):
            out = model.model.predict(preproc)
            return np.asarray(out[0], dtype=np.float32)
        # callable
        if callable(model):
            out = model(preproc)
            return np.asarray(out[0], dtype=np.float32)
        # common wrapper method names
        for name in ('get_embedding', 'get_feature', 'extract', 'embeddings', 'forward'):
            fn = getattr(model, name, None)
            if callable(fn):
                out = fn(preproc)
                # if returns array for single sample
                if isinstance(out, (list, tuple, np.ndarray)):
                    return np.asarray(out[0], dtype=np.float32)
                return np.asarray(out, dtype=np.float32)
    except Exception as e:
        raise
    raise RuntimeError('Unable to obtain embedding from model.')

# ---------- Utils ----------
def crop_face_from_bbox(pil_img, box, margin=0.25):
    # box: [x, y, width, height] from mtcnn
    x, y, w, h = box
    img_w, img_h = pil_img.size
    # add margin
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

# ---------- Main ----------
def build_index(dataset_dir, out_dir, model_name='Facenet', detector_backend='mtcnn', img_exts=(".jpg",".jpeg",".png"), verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    detector = MTCNN()
    # Build / load deepface model
    print("Loading DeepFace model:", model_name)
    model = DeepFace.build_model(model_name)
    print("Model loaded.")

    embeddings = []
    meta = []  # each entry maps to an embedding
    for fname in tqdm(os.listdir(dataset_dir)):
        if not fname.lower().endswith(img_exts):
            continue
        path = os.path.join(dataset_dir, fname)
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as e:
            if verbose: print("Failed to open", path, e)
            continue

        # Detect faces with MTCNN (gives list)
        results = detector.detect_faces(np.array(pil_img))
        if not results:
            if verbose: print("No face:", fname)
            continue

        # For each face found, crop, compute embedding
        for i, res in enumerate(results):
            bbox = res['box']  # [x, y, width, height]
            try:
                face_img, face_bbox = crop_face_from_bbox(pil_img, bbox)
                # DeepFace expects an image path or numpy array; we'll use aligned face as numpy array
                face_np = np.asarray(face_img)
                
                try:
                    rep = DeepFace.represent(face_img, model_name=model_name, model=model, enforce_detection=False, detector_backend='skip')
                except Exception:
                    
                    try:
                        from deepface.commons import functions
                        preproc = functions.preprocess_face(img=face_np, target_size=(160,160), enforce_detection=False, detector_backend='skip')
                    except Exception:
                        preproc = _preprocess_for_model(face_np, target_size=(160,160))
                    emb_vec = _get_embedding_from_model(model, preproc)
                    # ensure list-like for downstream handling
                    rep = emb_vec.tolist()

                # rep may be list/array or dict depending on deepface versions
                if isinstance(rep, dict) and 'embedding' in rep:
                    vec = np.asarray(rep['embedding'], dtype=np.float32)
                else:
                    
                    if isinstance(rep, list) and len(rep)>0 and isinstance(rep[0], (list, tuple, np.ndarray)):
                        vec = np.asarray(rep[0], dtype=np.float32)
                    else:
                        vec = np.asarray(rep, dtype=np.float32)

                # normalize
                vec = l2_normalize(vec)
                embeddings.append(vec.astype('float32'))
                meta.append({'file': fname, 'face_id': i, 'bbox': face_bbox})
            except Exception as e:
                if verbose: print("face processing failed", fname, e)
                continue

    if len(embeddings) == 0:
        raise RuntimeError("No embeddings extracted. Check dataset and detector.")

    embeddings_np = np.stack(embeddings)  # (N, D)
    # Build Faiss Index using Inner Product over L2-normalized vectors => cosine similarity
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_np)

    # Save index and metadata
    index_path = os.path.join(out_dir, "face_index.index")
    embeddings_path = os.path.join(out_dir, "embeddings.npy")
    meta_path = os.path.join(out_dir, "meta.pkl")
    faiss.write_index(index, index_path)
    np.save(embeddings_path, embeddings_np)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"Built index with {index.ntotal} faces. Saved to {out_dir}")
    return index_path, embeddings_path, meta_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to folder with images")
    parser.add_argument("--out_dir", default="./face_index", help="Where to save index and meta")
    parser.add_argument("--model", default="Facenet", help="DeepFace model name (Facenet, ArcFace, VGG-Face, etc.)")
    args = parser.parse_args()
    build_index(args.dataset, args.out_dir, model_name=args.model)
