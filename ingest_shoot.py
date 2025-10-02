import os
import io
import tempfile
import logging
import numpy as np
from PIL import Image
import face_recognition
from helpers import s3, BUCKET
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-shoot")

def list_shoots():
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

def list_shoot_keys(shoot_id):
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

def download_to_bytes(key):
    try:
        bio = io.BytesIO()
        s3.download_fileobj(Bucket=BUCKET, Key=key, Fileobj=bio)
        bio.seek(0)
        return bio
    except ClientError as e:
        logger.exception(f"Failed to download {key}: {e}")
        return None

def compute_embedding_from_bytes(bio):
    try:
        img = face_recognition.load_image_file(bio)
    except Exception as e:
        logger.warning(f"Failed to load image from bytes for {bio}: {e}")
        return None
    encs = face_recognition.face_encodings(img)
    if not encs:
        logger.info(f"No face found in image from {bio}")
        return None
    return np.asarray(encs[0], dtype=np.float32)

def process_image(key):
    bio = download_to_bytes(key)
    if bio is None:
        return None, None
    try:
        with Image.open(bio) as pil_img:
            # Increase size limit to 1600x1600 (2,560,000 pixels)
            if pil_img.size[0] * pil_img.size[1] > 1600 * 1600:
                logger.warning(f"Image {key} too large, skipping: {pil_img.size}")
                return None, None
            if pil_img.mode not in ('RGB', 'L'):
                pil_img = pil_img.convert('RGB')
            pil_img.thumbnail((800, 800))
            tmp_b = io.BytesIO()
            pil_img.save(tmp_b, format="JPEG", quality=85)
            tmp_b.seek(0)
            bio = tmp_b
            pil_img.close()
    except (IOError, ValueError, Exception) as e:
        logger.warning(f"Failed to process image {key}: {e}")
        if bio:
            bio.close()
        return None, None
    try:
        img = face_recognition.load_image_file(bio)
        encs = face_recognition.face_encodings(img)
        if not encs:
            logger.info(f"No face found in image {key}")
            return None, None
        return key, np.asarray(encs[0], dtype=np.float32)
    except Exception as e:
        logger.error(f"Face encoding failed for {key}: {e}")
        return None, None
    finally:
        if bio:
            bio.close()

def ingest_shoot(shoot_id, upload_embeddings_key=None):
    process = psutil.Process(os.getpid())
    logger.info(f"Starting ingest for {shoot_id}, initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    keys = list_shoot_keys(shoot_id)
    logger.info(f"Processing {len(keys)} images for shoot {shoot_id}")
    if not keys:
        logger.warning(f"No images found for shoot {shoot_id}")
        return {"success": True, "message": "No images found", "count": 0}

    embeddings = []
    kept_keys = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_key = {executor.submit(process_image, key): key for key in keys}
        for future in as_completed(future_to_key):
            key, emb = future.result()
            if emb is not None:
                embeddings.append(emb)
                kept_keys.append(key)
            else:
                logger.info(f"Skipping {key} due to no face or processing error")

    if not embeddings:
        logger.warning(f"No embeddings computed for shoot {shoot_id}")
        return {"success": True, "message": "No embeddings found", "count": 0}

    embs = np.stack(embeddings, axis=0)
    keys_arr = np.array(kept_keys, dtype=object)

    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        try:
            np.savez_compressed(tmp.name, keys=keys_arr, embeddings=embs)
            upload_key = upload_embeddings_key or f"projects/gallery/{shoot_id}/embeddings.npz"
            s3.upload_file(Filename=tmp.name, Bucket=BUCKET, Key=upload_key)
            logger.info(f"Uploaded embeddings to {upload_key}")
            return {"success": True, "upload_key": upload_key, "count": len(kept_keys)}
        except ClientError as e:
            logger.exception(f"Failed to upload embeddings for shoot {shoot_id}: {e}")
            return {"success": False, "message": f"S3 upload error: {e}", "count": 0}
        except Exception as e:
            logger.exception(f"Failed to process embeddings for shoot {shoot_id}: {e}")
            return {"success": False, "message": str(e), "count": 0}

def ingest_all_shoots():
    shoots = list_shoots()
    if not shoots:
        logger.warning("No shoots found in S3 bucket")
        return {"success": True, "message": "No shoots found", "processed": 0}
    
    results = []
    for shoot_id in shoots:
        logger.info(f"Processing shoot {shoot_id}")
        result = ingest_shoot(shoot_id)
        results.append(result)
    
    success_count = sum(1 for r in results if r["success"])
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