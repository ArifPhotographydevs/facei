# worker_job.py
import os
import tempfile
import logging
import numpy as np
from helpers import s3, BUCKET, create_presigned_url, send_link_email
import time
import threading
import sys
from contextlib import ExitStack
from botocore.exceptions import ClientError

# face_recognition import heavy libs
import face_recognition

logger = logging.getLogger("worker_job")
logger.setLevel(logging.INFO)

# Simple in-process cache for embeddings per shoot to avoid repeated S3 downloads.
# Key: shoot_id, Value: dict with 'keys' list and 'embeddings' numpy array and 'loaded_at' timestamp
_SHOOT_CACHE = {}
_SHOOT_CACHE_LOCK = threading.Lock()
_CACHE_TTL_SECONDS = int(os.environ.get("SHOOT_EMB_CACHE_TTL", 60 * 10))  # default 10 minutes

def _load_embeddings_for_shoot(shoot_id):
    now = time.time()
    with _SHOOT_CACHE_LOCK:
        cached = _SHOOT_CACHE.get(shoot_id)
        if cached and (now - cached.get('loaded_at', 0)) < _CACHE_TTL_SECONDS:
            return cached['keys'], cached['embeddings']
        else:
            _SHOOT_CACHE.pop(shoot_id, None)
    key = f"projects/gallery/{shoot_id}/embeddings.npz"
    try:
        with ExitStack() as stack:
            tmp = stack.enter_context(tempfile.NamedTemporaryFile(suffix=".npz"))
            s3.download_file(Filename=tmp.name, Bucket=BUCKET, Key=key)
            data = np.load(tmp.name, allow_pickle=True)
            keys = data['keys'].tolist()
            embeddings = data['embeddings'].astype(np.float32)
            with _SHOOT_CACHE_LOCK:
                _SHOOT_CACHE[shoot_id] = {'keys': keys, 'embeddings': embeddings, 'loaded_at': now}
            cache_size = sum(sys.getsizeof(v['embeddings']) + sys.getsizeof(v['keys']) for v in _SHOOT_CACHE.values())
            logger.info("Cache size: %d bytes", cache_size)
            logger.info("Loaded embeddings for shoot %s (N=%d)", shoot_id, len(keys))
            return keys, embeddings
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.error("Embeddings file not found for shoot %s: %s", shoot_id, key)
            return [], np.array([])
        raise
    except Exception as e:
        logger.exception("Failed to load embeddings for shoot %s: %s", shoot_id, e)
        return [], np.array([])

def match_shoot_job(selfie_embedding_list, shoot_id, recipient_email, name, phone, job_id, top_k=None):
    """
    Worker job that receives:
      - selfie_embedding_list: small list of floats (no image)
      - shoot_id: which shoot to search
    Returns a dict with matches (key, distance, presigned_url) and counts.
    """
    try:
        top_k = min(int(top_k or os.environ.get("DEFAULT_TOP_K", 20)), 100)  # Cap at 100
        # convert selfie embedding into numpy
        selfie_emb = np.asarray(selfie_embedding_list, dtype=np.float32)
        if selfie_emb.size == 0:
            return {"success": False, "message": "Empty selfie embedding", "job_id": job_id}

        # load shoot embeddings (cached)
        keys, embeddings = _load_embeddings_for_shoot(shoot_id)
        if embeddings.size == 0:
            return {"success": False, "message": "No embeddings for shoot", "job_id": job_id}

        # compute vectorized Euclidean distances
        # (embeddings shape: N x D)
        diff = embeddings - selfie_emb[None, :]
        dists = np.sqrt(np.sum(diff * diff, axis=1))

        # select top_k smallest distances
        idx_sorted = np.argsort(dists)[:top_k]
        results = []
        for idx in idx_sorted:
            dist = float(dists[idx])
            match_key = keys[idx]
            presigned = create_presigned_url(BUCKET, match_key, expiration=3600)
            results.append({"key": match_key, "distance": dist, "presigned_url": presigned})

        # Optionally send an email with link to results (commented out by default)
        # zip_url = None
        # if results:
        #     zip_url = create_presigned_url(BUCKET, some_zip_key, expiration=3600)
        #     send_link_email(zip_url, recipient_email, name, phone)

        return {"success": True, "matched_count": len(results), "matches": results, "job_id": job_id}
    except Exception as e:
        logger.exception("match_shoot_job failed: %s", e)
        return {"success": False, "message": str(e), "job_id": job_id}