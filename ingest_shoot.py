#!/usr/bin/env python3
"""
ingest_shoot.py

Robust ingest + built-in scheduler/watchdog suitable for running inside Docker.

Features:
- Limits native thread usage BEFORE importing numpy/PIL/face_recognition
- Uses multiprocessing spawn start method to avoid fork-related corruption
- Per-worker s3 client initialization
- Worker isolation via ProcessPoolExecutor; synchronous fallback if workers crash
- Chunked processing to reduce memory pressure
- Writes embeddings.npz back to S3
- Built-in scheduler/watchdog mode with exponential backoff and graceful shutdown
- By default (no CLI args) runs the scheduler loop so container doesn't exit immediately
- Configurable via environment variables:
    INGEST_MAX_WORKERS (default 2)
    INGEST_IMAGE_MAX_DIM (default 1600)
    INGEST_INTERVAL_SECONDS (default 300)
    INGEST_BACKOFF_BASE (default 5)
    S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_ENDPOINT
    INGEST_DAEMON=true to run scheduler by env
"""

# ---------------------------
# IMPORTANT: limit native thread env vars BEFORE importing native libs
# ---------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# standard libs
import sys
import time
import uuid
import signal
import logging
import tempfile
import traceback
import argparse
from typing import Optional, Tuple, List
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

# set spawn start method early (helps avoid fork/native lib problems)
try:
    multiprocessing.set_start_method('spawn', force=True)
except Exception:
    pass

# now safe to import heavy native libs
import io
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

import face_recognition
import psutil
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError
from botocore.config import Config

# -------------- Logging --------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("ingest-multiface")

# -------------- Configuration (via env or defaults) --------------
BUCKET = os.environ.get("S3_BUCKET", "arif12")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", os.environ.get("ACCESS_KEY", ""))
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", os.environ.get("SECRET_KEY", ""))
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", f"https://s3.{AWS_REGION}.wasabisys.com")
BOTO_MAX_POOL = int(os.environ.get("BOTO_MAX_POOL", "50"))

MAX_WORKERS = max(1, int(os.environ.get("INGEST_MAX_WORKERS", "2")))
IMAGE_MAX_DIM = int(os.environ.get("INGEST_IMAGE_MAX_DIM", "1600"))
FUTURE_TIMEOUT = int(os.environ.get("INGEST_FUTURE_TIMEOUT", "300"))
CHUNK_MULTIPLIER = int(os.environ.get("INGEST_CHUNK_MULTIPLIER", "4"))
RETRIES_PER_IMAGE = int(os.environ.get("INGEST_RETRIES_PER_IMAGE", "1"))
S3_PAGE_SIZE = int(os.environ.get("INGEST_S3_PAGE_SIZE", "1000"))

# Scheduler config
INGEST_INTERVAL_SECONDS = int(os.environ.get("INGEST_INTERVAL_SECONDS", "300"))  # poll every 5 minutes by default
INGEST_BACKOFF_BASE = int(os.environ.get("INGEST_BACKOFF_BASE", "5"))  # base seconds for exponential backoff
INGEST_DAEMON = os.environ.get("INGEST_DAEMON", "false").lower() in ("1", "true", "yes")

# S3 client factory
def create_s3_client():
    cfg = Config(max_pool_connections=BOTO_MAX_POOL)
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY or None,
        aws_secret_access_key=AWS_SECRET_KEY or None,
        region_name=AWS_REGION,
        endpoint_url=S3_ENDPOINT,
        config=cfg
    )

# global s3 client for main process
s3 = create_s3_client()

# per-worker s3 in initializer
_worker_s3 = None
def _worker_init():
    global _worker_s3
    # re-ensure env thread limits in worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    _worker_s3 = create_s3_client()

# ----------------- Utility functions -----------------
def list_shoots(prefix: str = "projects/gallery/") -> List[str]:
    try:
        paginator = s3.get_paginator('list_objects_v2')
        shoots = set()
        page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter='/', PaginationConfig={'PageSize': S3_PAGE_SIZE})
        for page in page_iterator:
            for cp in page.get("CommonPrefixes", []):
                p = cp.get("Prefix", "")
                if p.startswith(prefix):
                    sid = p[len(prefix):].rstrip('/')
                    if sid:
                        shoots.add(sid)
        if not shoots:
            # fallback
            page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=prefix, PaginationConfig={'PageSize': S3_PAGE_SIZE})
            for page in page_iterator:
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if key and key.startswith(prefix):
                        remaining = key[len(prefix):]
                        parts = remaining.split('/')
                        if parts and parts[0]:
                            shoots.add(parts[0])
        return sorted(shoots)
    except Exception as e:
        logger.exception("list_shoots failed: %s", e)
        return []

def list_shoot_keys(shoot_id: str) -> List[str]:
    prefix = f"projects/gallery/{shoot_id}/"
    keys = []
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, PaginationConfig={'PageSize': S3_PAGE_SIZE}):
            for obj in page.get("Contents", []):
                k = obj.get("Key")
                if k and not k.endswith('/') and k.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")):
                    keys.append(k)
    except Exception as e:
        logger.exception("list_shoot_keys failed for %s: %s", shoot_id, e)
    logger.info("Found %d images for shoot %s", len(keys), shoot_id)
    return sorted(keys)

def download_to_bytesio(client, key: str, attempts: int = 2) -> Optional[io.BytesIO]:
    attempt = 0
    while attempt < attempts:
        attempt += 1
        try:
            bio = io.BytesIO()
            client.download_fileobj(Bucket=BUCKET, Key=key, Fileobj=bio)
            bio.seek(0)
            return bio
        except EndpointConnectionError as e:
            logger.warning("S3 endpoint error for %s attempt %d: %s", key, attempt, e)
        except ClientError as e:
            logger.warning("S3 client error for %s attempt %d: %s", key, attempt, e)
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                break
        except Exception as e:
            logger.exception("Unexpected download error for %s attempt %d: %s", key, attempt, e)
        time.sleep(1 * attempt)
    return None

def safe_load_pil_from_bytes(bio: io.BytesIO) -> Optional[Image.Image]:
    try:
        bio.seek(0)
        dup = io.BytesIO(bio.read())
        dup.seek(0)
        img = Image.open(dup)
        return img
    except Exception as e:
        logger.debug("safe_load_pil_from_bytes failed: %s", e)
        return None

def resize_image_in_memory(img: Image.Image, max_dim: int = IMAGE_MAX_DIM) -> Image.Image:
    try:
        w, h = img.size
        if max(w, h) <= max_dim:
            return img
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return img
    except Exception:
        return img

# worker function (picklable)
def worker_process_image(key: str) -> Tuple[Optional[str], Optional[np.ndarray], Optional[str]]:
    global _worker_s3
    try:
        if _worker_s3 is None:
            _worker_s3 = create_s3_client()
        bio = download_to_bytesio(_worker_s3, key, attempts=2)
        if not bio:
            return None, None, f"download_failed:{key}"
        img = safe_load_pil_from_bytes(bio)
        if img is None:
            bio.close()
            return None, None, f"pil_open_failed:{key}"
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img = resize_image_in_memory(img, IMAGE_MAX_DIM)
        arr = np.asarray(img)
        encs = face_recognition.face_encodings(arr)
        try:
            img.close()
        except Exception:
            pass
        bio.close()
        if not encs:
            return None, None, f"no_face_found:{key}"
        return key, np.asarray(encs[0], dtype=np.float32), None
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("worker_process_image exception for %s: %s\n%s", key, e, tb)
        return None, None, f"worker_exception:{str(e)}"

# synchronous fallback (main process)
def sync_process_image(key: str) -> Tuple[Optional[str], Optional[np.ndarray], Optional[str]]:
    try:
        bio = download_to_bytesio(s3, key, attempts=2)
        if not bio:
            return None, None, f"download_failed:{key}"
        img = safe_load_pil_from_bytes(bio)
        if img is None:
            bio.close()
            return None, None, f"pil_open_failed:{key}"
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img = resize_image_in_memory(img, IMAGE_MAX_DIM)
        arr = np.asarray(img)
        encs = face_recognition.face_encodings(arr)
        try:
            img.close()
        except Exception:
            pass
        bio.close()
        if not encs:
            return None, None, f"no_face_found:{key}"
        return key, np.asarray(encs[0], dtype=np.float32), None
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("sync_process_image exception for %s: %s\n%s", key, e, tb)
        return None, None, f"sync_exception:{str(e)}"

# ingest a single shoot
def ingest_shoot(shoot_id: str, upload_key: Optional[str] = None) -> dict:
    proc = psutil.Process(os.getpid())
    logger.info("Starting ingest for %s; mem=%.2f MB", shoot_id, proc.memory_info().rss / 1024 / 1024)
    keys = list_shoot_keys(shoot_id)
    if not keys:
        logger.warning("No images found for shoot %s", shoot_id)
        return {"success": True, "message": "No images found", "count": 0}

    embeddings: List[np.ndarray] = []
    kept_keys: List[str] = []
    chunk_size = max(1, MAX_WORKERS * CHUNK_MULTIPLIER)
    chunks = [keys[i:i+chunk_size] for i in range(0, len(keys), chunk_size)]

    for ci, chunk in enumerate(chunks, start=1):
        logger.info("Processing chunk %d/%d size=%d", ci, len(chunks), len(chunk))
        try:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_worker_init) as exe:
                futures = {exe.submit(worker_process_image, k): k for k in chunk}
                for fut in as_completed(futures):
                    src_key = futures[fut]
                    try:
                        key, emb, err = fut.result(timeout=FUTURE_TIMEOUT)
                    except TimeoutError:
                        logger.warning("Future timeout for %s; trying sync fallback", src_key)
                        key, emb, err = sync_process_image(src_key)
                    except Exception as e:
                        logger.exception("Future exception for %s: %s", src_key, e)
                        key, emb, err = sync_process_image(src_key)
                    if emb is not None and key:
                        embeddings.append(emb)
                        kept_keys.append(key)
                        logger.info("Embedding found for %s", src_key)
                    else:
                        logger.info("No embedding for %s (%s)", src_key, err)
        except Exception as e:
            logger.exception("ProcessPool failed for chunk starting %s: %s; falling back to sync for chunk", chunk[0] if chunk else "N/A", e)
            for k in chunk:
                key, emb, err = sync_process_image(k)
                if emb is not None and key:
                    embeddings.append(emb)
                    kept_keys.append(key)
                    logger.info("Sync embedding found for %s", k)
                else:
                    logger.info("Sync no embedding for %s (%s)", k, err)

        mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logger.info("After chunk %d mem=%.2f MB embeddings=%d", ci, mem, len(embeddings))

    if not embeddings:
        logger.warning("No embeddings for shoot %s", shoot_id)
        return {"success": True, "message": "No embeddings found", "count": 0}

    try:
        embs = np.stack(embeddings, axis=0).astype(np.float32)
        keys_arr = np.array(kept_keys, dtype=object)
    except Exception as e:
        logger.exception("Failed to stack embeddings: %s", e)
        return {"success": False, "message": "Failed to assemble embeddings", "count": 0}

    tmp_file = None
    try:
        tmpf = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp_file = tmpf.name
        tmpf.close()
        np.savez_compressed(tmp_file, keys=keys_arr, embeddings=embs)
        upload_key = upload_key or f"projects/gallery/{shoot_id}/embeddings.npz"
        s3.upload_file(Filename=tmp_file, Bucket=BUCKET, Key=upload_key)
        logger.info("Uploaded embeddings for %s -> %s (count=%d)", shoot_id, upload_key, len(kept_keys))
        return {"success": True, "upload_key": upload_key, "count": len(kept_keys)}
    except Exception as e:
        logger.exception("Failed uploading embeddings for %s: %s", shoot_id, e)
        return {"success": False, "message": str(e), "count": 0}
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass
        mem_final = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        logger.info("Finished ingest for %s mem=%.2f MB", shoot_id, mem_final)

# ingest all shoots
def ingest_all_shoots() -> dict:
    shoots = list_shoots()
    if not shoots:
        logger.warning("No shoots found")
        return {"success": True, "message": "No shoots found", "processed": 0}
    results = []
    for sid in shoots:
        logger.info("Ingesting shoot %s", sid)
        try:
            r = ingest_shoot(sid)
        except Exception as e:
            logger.exception("Top-level ingest error for %s: %s", sid, e)
            r = {"success": False, "message": str(e), "count": 0}
        results.append(r)
    success_count = sum(1 for r in results if r.get("success"))
    logger.info("Processed %d shoots, %d successful", len(shoots), success_count)
    return {"success": True, "message": f"Processed {len(shoots)} shoots", "results": results, "processed": len(shoots)}

# ---------------- Scheduler / watchdog ----------------
_shutdown = False
def _signal_handler(signum, frame):
    global _shutdown
    logger.info("Received signal %s, shutting down gracefully...", signum)
    _shutdown = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def run_scheduler_loop(interval_seconds: int = INGEST_INTERVAL_SECONDS, backoff_base: int = INGEST_BACKOFF_BASE):
    """
    Run continuous ingest_all_shoots() every `interval_seconds`.
    On failure, apply exponential backoff using backoff_base.
    This loop never exits unless SIGINT/SIGTERM received.
    """
    logger.info("Starting scheduler loop: interval=%ds backoff_base=%ds", interval_seconds, backoff_base)
    consecutive_failures = 0
    while not _shutdown:
        start = time.time()
        try:
            res = ingest_all_shoots()
            # success -> reset failure counter
            consecutive_failures = 0
            logger.info("Scheduler run completed: %s", res.get("message", "done"))
        except Exception as e:
            consecutive_failures += 1
            logger.exception("Scheduler ingestion failed (attempt #%d): %s", consecutive_failures, e)
            # backoff
            wait = backoff_base * (2 ** (consecutive_failures - 1))
            wait = min(wait, 3600)  # cap to 1 hour
            logger.info("Backing off for %ds before retrying...", wait)
            # sleep with shutdown checks
            slept = 0
            while slept < wait and not _shutdown:
                time.sleep(1)
                slept += 1
        # normal interval wait
        if _shutdown:
            break
        elapsed = time.time() - start
        to_wait = max(0, interval_seconds - int(elapsed))
        logger.info("Next scheduled run in %ds", to_wait)
        slept = 0
        while slept < to_wait and not _shutdown:
            time.sleep(1)
            slept += 1
    logger.info("Scheduler loop exiting gracefully")

# --------------- CLI Entrypoint ---------------
def main():
    parser = argparse.ArgumentParser(description="Ingest shoots and optionally run as scheduler/watchdog")
    parser.add_argument("--shoot-id", help="Process single shoot id")
    parser.add_argument("--ingest-all", action="store_true", help="Process all shoots once")
    parser.add_argument("--daemon", action="store_true", help="Run continuous scheduler loop (same as INGEST_DAEMON env)")
    args = parser.parse_args()

    # Priority: explicit flags -> env var -> default behavior (daemon)
    if args.shoot_id:
        out = ingest_shoot(args.shoot_id)
        print(out)
        return 0
    if args.ingest_all:
        out = ingest_all_shoots()
        print(out)
        return 0
    if args.daemon or INGEST_DAEMON:
        run_scheduler_loop()
        return 0

    # Default behavior when no args passed: run scheduler loop (so container keeps running)
    logger.info("No CLI arguments passed. Starting scheduler loop by default (to keep container running).")
    run_scheduler_loop()
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as e:
        logger.exception("Unhandled exception in main: %s", e)
        exit_code = 1
    sys.exit(exit_code)
