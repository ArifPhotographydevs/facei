#!/usr/bin/env python3
"""
Robust ingest_shoot script - improved error handling, resource controls, and logging.

Replaces the original ingest script with:
- safer image size checks and resizing
- bounded process pool sized to CPU (with cap) to avoid native heap corruption
- deterministic closing of file-like objects
- defensive handling of S3 download/upload failures
- explicit logging of memory usage and progress
- consistent returns from process_image to avoid breaking the aggregator loop
"""

# --- IMPORTANT: limit native threads BEFORE importing numpy/PIL/face_recognition ---
import os

# Reduce native parallelism to avoid native-thread / heap corruption issues in C extensions.
# Must be set before importing numpy, Pillow, face_recognition, or other native libs.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import io
import tempfile
import logging
import numpy as np
from PIL import Image, ImageFile
import face_recognition
from helpers import s3, BUCKET
from botocore.exceptions import ClientError, EndpointConnectionError
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import multiprocessing
import time
from typing import Tuple, Optional, List

# PIL can raise "OSError: image file is truncated" on some files; enabling LOAD_TRUNCATED_IMAGES helps.
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-shoot")

# Configuration - tune these if necessary
MAX_PIXELS = 4000 * 4000            # reject images with more than 16,000,000 pixels (approx 16MP)
THUMBNAIL_SIZE = (1024, 1024)       # downscale images to this box for face recognition
JPEG_QUALITY = 85
MAX_WORKERS_CAP = 1                 # conservative default to avoid memory/native issues; raise if you have RAM
S3_DOWNLOAD_RETRIES = 2
S3_UPLOAD_RETRIES = 2
S3_TIMEOUT_SECONDS = 30             # if your boto3 config uses connect/read timeouts, set them there

def list_shoots() -> List[str]:
    """
    List shoot IDs under projects/gallery/ in the S3 bucket.
    Returns sorted list of shoot ids (strings).
    """
    try:
        prefix = "projects/gallery/"
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter='/')
        shoots = []
        for page in page_iterator:
            for cp in page.get("CommonPrefixes", []):
                p = cp.get("Prefix", "")
                if p.startswith(prefix):
                    shoot_id = p[len(prefix):].rstrip('/')
                    if shoot_id:
                        shoots.append(shoot_id)
        return sorted(shoots)
    except ClientError as e:
        logger.exception(f"Failed to list shoots: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error listing shoots: {e}")
        return []

def list_shoot_keys(shoot_id: str) -> List[str]:
    """
    Return a sorted list of image keys (jpg/jpeg/png) for the shoot.
    """
    try:
        prefix = f"projects/gallery/{shoot_id}/"
        paginator = s3.get_paginator('list_objects_v2')
        keys = []
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith((".jpg", ".jpeg", ".png")):
                    keys.append(key)
        logger.info(f"Found {len(keys)} images for shoot {shoot_id} at prefix {prefix}")
        return sorted(keys)
    except ClientError as e:
        logger.exception(f"Failed to list keys for shoot {shoot_id}: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error listing keys for shoot {shoot_id}: {e}")
        return []

def download_to_bytes(key: str) -> Optional[io.BytesIO]:
    """
    Download S3 object into a BytesIO and return it. Retries a few times for transient errors.
    Caller must close the returned BytesIO when finished.
    """
    attempt = 0
    while attempt <= S3_DOWNLOAD_RETRIES:
        attempt += 1
        try:
            bio = io.BytesIO()
            s3.download_fileobj(Bucket=BUCKET, Key=key, Fileobj=bio)
            bio.seek(0)
            return bio
        except EndpointConnectionError as e:
            logger.warning(f"S3 endpoint connection error while downloading {key} (attempt {attempt}): {e}")
        except ClientError as e:
            logger.exception(f"S3 client error downloading {key} (attempt {attempt}): {e}")
            # If it's a 404 / NoSuchKey, no point in retrying
            if getattr(e, 'response', {}).get('Error', {}).get('Code') in ('NoSuchKey', '404'):
                break
        except Exception as e:
            logger.exception(f"Unexpected error downloading {key} (attempt {attempt}): {e}")
        time.sleep(1 * attempt)
    return None

def _safe_load_pil(bio: io.BytesIO) -> Optional[Image.Image]:
    """
    Load an image via PIL from BytesIO. Returns a PIL.Image or None.
    This function clones the bytes into a new BytesIO because face_recognition and PIL sometimes
    move the stream pointer.
    """
    try:
        bio.seek(0)
        # create a copy to avoid pointer issues
        tmp = io.BytesIO(bio.read())
        tmp.seek(0)
        img = Image.open(tmp)
        return img
    except Exception as e:
        logger.warning(f"Failed to open image via PIL: {e}")
        return None

def compute_embedding_from_image(pil_img: Image.Image) -> Optional[np.ndarray]:
    """
    Given a PIL.Image (RGB), compute the first face embedding (128-d float32) or return None.
    """
    try:
        # convert PIL image to numpy array expected by face_recognition
        arr = np.asarray(pil_img.convert('RGB'))
        # First detect face locations (quicker) then compute encodings
        locations = face_recognition.face_locations(arr, model='hog')  # 'hog' is faster; 'cnn' is more accurate if GPU available
        if not locations:
            return None
        encs = face_recognition.face_encodings(arr, known_face_locations=locations)
        if not encs:
            return None
        return np.asarray(encs[0], dtype=np.float32)
    except Exception as e:
        logger.exception(f"Face encoding error: {e}")
        return None

def process_image(key: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Download image from S3 key, validate/resize, compute face embedding.
    Always returns a tuple (key or None, embedding or None).
    This function is safe to call in a separate process (ProcessPoolExecutor).
    """
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / 1024 / 1024
    logger.debug(f"[{key}] Starting process_image; memory_before={mem_before:.2f} MB")
    bio = None
    pil_img = None
    try:
        bio = download_to_bytes(key)
        if bio is None:
            logger.warning(f"[{key}] Download failed or object missing")
            return None, None

        # Load via PIL to inspect size without fully converting to numpy
        pil_img = _safe_load_pil(bio)
        if pil_img is None:
            logger.info(f"[{key}] PIL failed to open image, skipping")
            return None, None

        width, height = pil_img.size
        pixels = width * height
        # Reject extremely large images early
        if pixels > MAX_PIXELS:
            logger.warning(f"[{key}] Image too large ({width}x{height} = {pixels} px); rejecting")
            return None, None

        # Convert to RGB if necessary
        if pil_img.mode not in ('RGB', 'L'):
            pil_img = pil_img.convert('RGB')

        # Resize to thumbnail for faster face detection; use LANCZOS for quality
        pil_img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)

        # compute embedding
        emb = compute_embedding_from_image(pil_img)
        if emb is None:
            logger.info(f"[{key}] No face found or encoding failed after resize")
            return None, None

        mem_after = proc.memory_info().rss / 1024 / 1024
        logger.debug(f"[{key}] Completed embedding; memory_after={mem_after:.2f} MB")
        return key, emb

    except Exception as e:
        logger.exception(f"[{key}] Unexpected error processing image: {e}")
        return None, None
    finally:
        # Ensure resources are closed and references dropped
        try:
            if pil_img:
                pil_img.close()
        except Exception:
            pass
        try:
            if bio:
                bio.close()
        except Exception:
            pass

def ingest_shoot(shoot_id: str, upload_embeddings_key: Optional[str] = None) -> dict:
    """
    Ingest a single shoot: list image keys, compute embeddings in parallel (processes),
    and upload embeddings.npz back to S3.
    Returns a result dict similar to your original format.
    """
    process = psutil.Process(os.getpid())
    logger.info(f"Starting ingest for {shoot_id}, initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    keys = list_shoot_keys(shoot_id)
    logger.info(f"Processing {len(keys)} images for shoot {shoot_id}")
    if not keys:
        logger.warning(f"No images found for shoot {shoot_id}")
        return {"success": True, "message": "No images found", "count": 0}

    # Determine worker count conservatively to avoid memory explosion
    cpu_count = multiprocessing.cpu_count() or 1
    # For process-based workers, be conservative: each process loads native libs and uses memory.
    max_workers = min(MAX_WORKERS_CAP, max(1, cpu_count // 2))
    logger.info(f"Using ProcessPoolExecutor with max_workers={max_workers}")

    embeddings: List[np.ndarray] = []
    kept_keys: List[str] = []

    # Use ProcessPoolExecutor to isolate native extensions per-process (safer for heap integrity)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(process_image, key): key for key in keys}
        for future in as_completed(future_to_key):
            src_key = future_to_key[future]
            try:
                # Give processes more time than threads; if a process hangs, it will be reaped later.
                key, emb = future.result(timeout=600)
            except Exception as e:
                logger.exception(f"Worker failed processing {src_key}: {e}")
                continue
            if emb is not None and key:
                embeddings.append(emb)
                kept_keys.append(key)
            else:
                logger.info(f"Skipping {src_key} due to no face or processing error")

    if not embeddings:
        logger.warning(f"No embeddings computed for shoot {shoot_id}")
        return {"success": True, "message": "No embeddings found", "count": 0}

    try:
        embs = np.stack(embeddings, axis=0)
        keys_arr = np.array(kept_keys, dtype=object)
    except Exception as e:
        logger.exception(f"Failed to stack embeddings for shoot {shoot_id}: {e}")
        return {"success": False, "message": "Failed to assemble embeddings", "count": 0}

    # Write compressed npz to a temporary file, ensure close before upload
    tmp_file = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp_file = tmp.name
        tmp.close()  # close so numpy can open it on Windows-like platforms
        np.savez_compressed(tmp_file, keys=keys_arr, embeddings=embs)
        upload_key = upload_embeddings_key or f"projects/gallery/{shoot_id}/embeddings.npz"

        # Upload with retry loop
        attempt = 0
        while attempt <= S3_UPLOAD_RETRIES:
            attempt += 1
            try:
                s3.upload_file(Filename=tmp_file, Bucket=BUCKET, Key=upload_key)
                logger.info(f"Uploaded embeddings to {upload_key}")
                return {"success": True, "upload_key": upload_key, "count": len(kept_keys)}
            except ClientError as e:
                logger.exception(f"S3 upload_client_error for {upload_key} (attempt {attempt}): {e}")
            except Exception as e:
                logger.exception(f"Unexpected S3 upload error for {upload_key} (attempt {attempt}): {e}")
            time.sleep(1 * attempt)

        return {"success": False, "message": f"S3 upload failed after {S3_UPLOAD_RETRIES} retries", "count": 0}
    finally:
        # cleanup temporary file
        try:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass
        mem_final = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logger.info(f"Finished ingest for {shoot_id}. Final memory usage: {mem_final:.2f} MB")

def ingest_all_shoots() -> dict:
    shoots = list_shoots()
    if not shoots:
        logger.warning("No shoots found in S3 bucket")
        return {"success": True, "message": "No shoots found", "processed": 0}
    
    results = []
    for shoot_id in shoots:
        logger.info(f"Processing shoot {shoot_id}")
        try:
            result = ingest_shoot(shoot_id)
        except Exception as e:
            logger.exception(f"Top-level error ingesting shoot {shoot_id}: {e}")
            result = {"success": False, "message": str(e), "count": 0}
        results.append(result)
    
    success_count = sum(1 for r in results if r.get("success"))
    logger.info(f"Processed {len(shoots)} shoots, {success_count} successful")
    return {
        "success": True,
        "message": f"Processed {len(shoots)} shoots",
        "results": results,
        "processed": len(shoots)
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = ingest_shoot(sys.argv[1])
        print(res)
        sys.exit(0)
    else:
        res = ingest_all_shoots()
        print(res)
        sys.exit(0)
