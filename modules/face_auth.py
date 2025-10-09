import os
import pickle
import numpy as np
from deepface import DeepFace

# ---------------- CONFIG ----------------
# Directory where folders of authorized people are stored (e.g., /authorized_faces/person_a/face1.jpg)
AUTHORIZED_DIR = r"static/authorized_faces" 
CACHE_FILE = os.path.join(AUTHORIZED_DIR, "embeddings_cache.pkl")
MODEL_NAME = "Facenet512"
DETECTOR = "retinaface"
THRESHOLD = 0.65  # Cosine similarity threshold
# ----------------------------------------

def _create_and_cache_embeddings(db_path):
    """
    Scans the authorized faces directory, creates embeddings, and saves them to a pickle file.
    """
    authorized_db = {}
    for person in os.listdir(db_path):
        person_path = os.path.join(db_path, person)
        if os.path.isdir(person_path):
            embeddings = []
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                try:
                    # Generate embedding using DeepFace.represent
                    emb = DeepFace.represent(
                        img_path=img_path,
                        model_name=MODEL_NAME,
                        detector_backend=DETECTOR,
                        enforce_detection=False
                    )[0]["embedding"]
                    embeddings.append(emb)
                except Exception as e:
                    print(f"[AUTH WARN] Could not process {img_file} for {person}: {e}")
            
            if embeddings:
                authorized_db[person] = np.array(embeddings)
                print(f"[AUTH INFO] Cached {len(embeddings)} face(s) for {person}.")

    # Save the dictionary of embeddings to the cache file
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(authorized_db, f)
    
    return authorized_db

def load_or_create_authorized_db():
    """
    Loads the authorized face embeddings from cache if it exists, otherwise creates it.
    """
    if os.path.exists(CACHE_FILE):
        print("[AUTH INFO] Loading authorized faces from cache...")
        with open(CACHE_FILE, "rb") as f:
            authorized_db = pickle.load(f)
        print(f"[AUTH INFO] Loaded {len(authorized_db)} authorized user(s) from cache ✅")
    else:
        print("[AUTH INFO] No cache found. Building authorized faces database...")
        authorized_db = _create_and_cache_embeddings(AUTHORIZED_DIR)
        print(f"[AUTH INFO] Built and cached database for {len(authorized_db)} user(s) ✅")
        
    return authorized_db

def match_face(frame_face_crop, authorized_db):
    """
    Finds the best match for a given face crop from the authorized database.
    This is the core, non-threaded matching logic.
    """
    if frame_face_crop is None or frame_face_crop.size == 0:
        return None, 0.0

    try:
        # 1. Get embedding for the face detected in the video frame
        frame_emb = DeepFace.represent(
            img_path=frame_face_crop,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False
        )[0]["embedding"]
        
        best_match_name = None
        best_similarity = 0.0

        # 2. Compare with all cached embeddings using cosine similarity
        for name, emb_array in authorized_db.items():
            for ref_emb in emb_array:
                # Cosine similarity calculation
                sim = np.dot(frame_emb, ref_emb) / (np.linalg.norm(frame_emb) * np.linalg.norm(ref_emb))
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_name = name

        # 3. Return the best match if it passes the threshold
        if best_similarity >= THRESHOLD:
            return best_match_name, best_similarity
        else:
            return "Unauthorized", best_similarity # Return a specific label for clarity
            
    except Exception:
        # This can happen if DeepFace fails to find a face in the crop
        return None, 0.0

# --- Load the database ONCE when the application starts ---
authorized_db = load_or_create_authorized_db()