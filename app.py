#!/usr/bin/env python3
"""
Improved full app.py for Face Recognition matcher.

Changes and fixes included:
- Robust .npz loading (many possible field names)
- No synthetic placeholder keys returned (if keys array missing -> fallback to direct scan)
- Uses all detected faces per image and keeps best distance per image
- Picks best (minimum) distance for images that appear multiple times
- Corrected variable name typos from previous code
- MAX_WORKERS configurable via environment variable; conservative default
- Better resource cleanup and defensive error handling
- Presigned URLs generated for copied objects in client folder (downloadable)
- Email sending is best-effort and will not break response on failure
- CLI supports runserver, ingest, and optional --daemon flag for possible scheduler loop

NOTE: This file contains sensitive example credentials copied from your earlier message.
In production you should set credentials via environment variables or IAM role and never keep
secrets inside source code.
"""

import os
import sys
import io
import argparse
import tempfile
import uuid
import logging
import base64
from zipfile import ZipFile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from PIL import Image, UnidentifiedImageError
import face_recognition
import numpy as np

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ===== Logging =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("face-matcher")

# ===== Configuration (edit these for your environment or use env vars) =====
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "SW5I2XCNJAI7GTB7MRIW")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "eKNEI3erAhnSiBdcK0OltkTHIe2jJYJVhPu1eazJ")
REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
BUCKET = os.environ.get("S3_BUCKET", "arif12")
ENDPOINT_URL = os.environ.get("S3_ENDPOINT", f"https://s3.{REGION}.wasabisys.com")
BOTO_CONFIG = Config(max_pool_connections=int(os.environ.get("BOTO_MAX_POOL", "50")))

# Email config (for production supply via environment variables)
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "githubarifphotography@gmail.com")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "utuz rvgk kmsv sntz")

# Matching hyperparams
IMAGE_MAX_SIZE = (1600, 1600)  # reasonable default; images will be resized for processing
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", min(4, (os.cpu_count() or 2))))
MAX_MATCHES = int(os.environ.get("MAX_MATCHES", "200"))
FACE_COMPARE_TOLERANCE = float(os.environ.get("FACE_COMPARE_TOLERANCE", "0.65"))  # L2 distance threshold

logger.info(f"Configuration: REGION={REGION}, BUCKET={BUCKET}, MAX_WORKERS={MAX_WORKERS}, TOL={FACE_COMPARE_TOLERANCE}")

# ===== S3 client (main process) =====
s3 = boto3.client(
    "s3",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION,
    endpoint_url=ENDPOINT_URL,
    config=BOTO_CONFIG
)

# ===== Flask app =====
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ===== Helper utilities =====
def safe_resize_image(path: str, max_size: Tuple[int, int] = IMAGE_MAX_SIZE) -> None:
    """Resize in-place preserving aspect ratio, convert to RGB if necessary."""
    try:
        with Image.open(path) as img:
            if hasattr(Image, "Resampling"):
                resample = Image.Resampling.LANCZOS
            else:
                resample = Image.LANCZOS
            img.thumbnail(max_size, resample)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            img.save(path)
    except Exception as e:
        logger.warning(f"safe_resize_image failed for {path}: {e}")

def _create_s3_client_for_worker():
    """Create an s3 client for use inside worker processes (picklable)."""
    return boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
        endpoint_url=ENDPOINT_URL,
        config=Config(max_pool_connections=5)
    )

def create_presigned_url(*, Bucket=None, Key=None, bucket=None, key=None, expiration=3600, **kwargs):
    """
    Create a presigned URL for an S3 object.
    Accepts both (Bucket, Key) and (bucket, key) forms.
    """
    bucket_name = Bucket or bucket
    object_key = Key or key

    if not bucket_name or not object_key:
        logger.error("create_presigned_url missing bucket or key")
        return None

    try:
        return s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration
        )
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL for {object_key}: {e}")
        return None

def send_link_email(download_url: str, recipient_email: str, name: str, phone: str, shoot_id: str) -> None:
    """Send a simple plaintext email with link. Best-effort; exceptions are logged and re-raised."""
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = f"Face Recognition ZIP Download Link for {name} - {shoot_id}"
        body = f"""Hi {name},

Thank you for using the Face Matching service for shoot {shoot_id}.
Phone: {phone}

Here is your temporary download link for all matched images (valid for 1 hour):
{download_url}

If you have issues downloading, please contact support.

Best,
Face Matcher Bot
"""
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.starttls()
            if SENDER_EMAIL and SENDER_PASSWORD:
                try:
                    server.login(SENDER_EMAIL, SENDER_PASSWORD)
                except Exception as e:
                    logger.warning(f"SMTP login failed (will still attempt send): {e}")
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        logger.info(f"Sent download link to {recipient_email}")
    except Exception as e:
        logger.exception(f"Failed to send email to {recipient_email}: {e}")
        # Do not raise to avoid breaking normal flow; caller can choose to ignore/email best-effort.
        raise

# ===== S3 listing helpers =====
def list_shoot_ids(prefix: str = "projects/gallery/") -> List[str]:
    """Return sorted list of shoot ids found under the given prefix."""
    paginator = s3.get_paginator('list_objects_v2')
    shoots_set = set()
    try:
        page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter='/')
        for page in page_iterator:
            for cp in page.get("CommonPrefixes", []):
                key = cp.get("Prefix")
                if key and key.startswith(prefix):
                    parts = key[len(prefix):].split('/')
                    if parts and parts[0]:
                        shoots_set.add(parts[0])
    except Exception as e:
        logger.warning(f"Delimiter-based listing failed: {e}")

    if not shoots_set:
        try:
            page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=prefix)
            for page in page_iterator:
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if not key:
                        continue
                    if key.startswith(prefix):
                        remaining = key[len(prefix):]
                        parts = remaining.split('/')
                        if parts and parts[0]:
                            shoots_set.add(parts[0])
        except Exception as e:
            logger.exception(f"Full listing fallback failed: {e}")

    return sorted(shoots_set)

# ===== Ingest / embeddings creation (workers) =====
def compute_face_encodings_for_local_image(path: str) -> List[np.ndarray]:
    """Return list of encodings (list of numpy arrays) for faces in given image path."""
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as e:
        logger.debug(f"Image verify failed for {path}: {e}")
        return []
    try:
        safe_resize_image(path)
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        return encs or []
    except Exception as e:
        logger.exception(f"Failed to encode image {path}: {e}")
        return []

def _worker_compute_encoding(s3_key: str) -> List[Tuple[str, np.ndarray]]:
    """
    Worker executed in separate process: download image to temp file,
    compute face encodings, and return list of (s3_key, encoding).
    """
    local_client = _create_s3_client_for_worker()
    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(s3_key)[1] or ".jpg")
    os.close(fd)
    try:
        local_client.download_file(Bucket=BUCKET, Key=s3_key, Filename=tmp_path)
        encs = compute_face_encodings_for_local_image(tmp_path)
        results = []
        for enc in encs:
            results.append((s3_key, np.array(enc, dtype=np.float32)))
        return results
    except Exception as e:
        logger.warning(f"Worker failed for {s3_key}: {e}")
        return []
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

def ingest_shoot_embeddings(shoot_id: str, save_overwrite: bool = True) -> bool:
    """
    Compute face encodings for all images in a shoot and upload embeddings.npz to S3.
    The npz contains keys (string array) and encodings (N x D float32).
    """
    prefix = f"projects/gallery/{shoot_id}/"
    logger.info(f"Starting ingestion for shoot: {shoot_id} (prefix={prefix})")

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=prefix)
    image_keys: List[str] = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            k = obj.get("Key")
            if k and k.lower().endswith((".jpg", ".jpeg", ".png")) and not k.endswith('/'):
                image_keys.append(k)

    logger.info(f"Found {len(image_keys)} images for shoot {shoot_id}")

    encodings: List[np.ndarray] = []
    keys: List[str] = []

    if image_keys:
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(image_keys)))) as exe:
            futures = {exe.submit(_worker_compute_encoding, k): k for k in image_keys}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if res:
                        for key, enc in res:
                            keys.append(key)
                            encodings.append(enc)
                except Exception as e:
                    logger.exception(f"Exception in worker future: {e}")

    if not keys:
        logger.warning(f"No face encodings found for shoot {shoot_id}; not writing embeddings file.")
        return False

    try:
        encodings_np = np.stack(encodings).astype(np.float32)  # shape (N, D)
        keys_arr = np.array(keys, dtype=np.str_)
    except Exception as e:
        logger.exception(f"Failed to assemble numpy arrays for {shoot_id}: {e}")
        return False

    fd, tmp_npz = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_npz, keys=keys_arr, encodings=encodings_np)
        target_key = f"{prefix}embeddings.npz"
        s3.upload_file(Filename=tmp_npz, Bucket=BUCKET, Key=target_key)
        logger.info(f"Uploaded embeddings for shoot {shoot_id} to s3://{BUCKET}/{target_key} ({len(keys)} encodings)")
        return True
    except Exception as e:
        logger.exception(f"Failed to save/upload embeddings for {shoot_id}: {e}")
        return False
    finally:
        try:
            if os.path.exists(tmp_npz):
                os.unlink(tmp_npz)
        except Exception:
            pass

# ===== Worker used for direct scanning (top-level so picklable) =====
def worker_compare(args: Tuple[str, np.ndarray]) -> Optional[dict]:
    """
    Compare known_encoding to faces in a given image key.
    Returns {"key": key, "distance": best_distance} if matched else None.
    """
    key, known_enc = args
    local_client = _create_s3_client_for_worker()
    fd, tmp_file = tempfile.mkstemp(suffix=os.path.splitext(key)[1] or ".jpg")
    os.close(fd)
    try:
        local_client.download_file(Bucket=BUCKET, Key=key, Filename=tmp_file)
        try:
            with Image.open(tmp_file) as im:
                im.verify()
        except Exception:
            return None
        safe_resize_image(tmp_file)
        image = face_recognition.load_image_file(tmp_file)
        encs = face_recognition.face_encodings(image)
        if not encs:
            return None
        best_d = None
        for enc in encs:
            d = float(np.linalg.norm(np.array(enc, dtype=np.float32) - known_enc))
            if best_d is None or d < best_d:
                best_d = d
        if best_d is not None and best_d <= FACE_COMPARE_TOLERANCE:
            return {"key": key, "distance": float(best_d)}
        return None
    except Exception as e:
        logger.debug(f"Direct worker failed for {key}: {e}")
        return None
    finally:
        try:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
        except Exception:
            pass

# ===== Flask routes =====
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/shoots', methods=['GET'])
def get_shoots():
    try:
        shoot_ids = list_shoot_ids(prefix="projects/gallery/")
        shoots = []
        for sid in shoot_ids:
            embeddings_key = f"projects/gallery/{sid}/embeddings.npz"
            has_embeddings = False
            try:
                s3.head_object(Bucket=BUCKET, Key=embeddings_key)
                has_embeddings = True
            except ClientError as e:
                code = e.response.get('Error', {}).get('Code', '')
                if code not in ('404', 'NoSuchKey'):
                    logger.warning(f"Error checking embeddings for {sid}: {e}")
            shoots.append({"shoot_id": sid, "has_embeddings": has_embeddings})
        return jsonify({"success": True, "shoots": shoots}), 200, {'Access-Control-Allow-Origin': '*'}
    except Exception as e:
        logger.exception(f"Failed to list shoots: {e}")
        return jsonify({"success": False, "message": str(e)}), 500, {'Access-Control-Allow-Origin': '*'}

@app.route('/match', methods=['POST', 'OPTIONS'])
def match_face():
    if request.method == 'OPTIONS':
        return jsonify({}), 200, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }

    try:
        data = request.get_json(force=True)
        for field in ["selfie", "email", "name", "phone"]:
            if not data.get(field):
                return jsonify({"success": False, "message": f"Missing {field}"}), 400, {'Access-Control-Allow-Origin': '*'}

        name = data["name"]
        recipient_email = data["email"]
        phone = data["phone"]
        shoot_id = data.get("shoot_id")

        # Auto-select shoot_id if not provided
        if not shoot_id:
            shoot_ids = list_shoot_ids(prefix="projects/gallery/")
            if not shoot_ids:
                return jsonify({"success": False, "message": "No shoot folders found in S3"}), 400, {'Access-Control-Allow-Origin': '*'}
            shoot_id = shoot_ids[0]
            logger.info(f"Auto-selected shoot_id: {shoot_id}")

        # Save selfie to temp, validate and compute encoding
        selfie_bytes = base64.b64decode(data["selfie"])
        selfie_fd, selfie_path = tempfile.mkstemp(suffix=".jpg")
        os.close(selfie_fd)
        with open(selfie_path, "wb") as f:
            f.write(selfie_bytes)

        try:
            with Image.open(selfie_path) as test_img:
                test_img.verify()
        except (UnidentifiedImageError, Exception) as e:
            logger.warning(f"Uploaded selfie is not a valid image: {e}")
            try:
                os.unlink(selfie_path)
            except Exception:
                pass
            return jsonify({"success": False, "message": "Uploaded selfie is not a valid image."}), 400, {'Access-Control-Allow-Origin': '*'}

        # Resize selfie (if large) and compute known encoding(s)
        safe_resize_image(selfie_path, max_size=IMAGE_MAX_SIZE)
        known_image = face_recognition.load_image_file(selfie_path)
        known_encodings = face_recognition.face_encodings(known_image)
        try:
            os.unlink(selfie_path)
        except Exception:
            pass

        if not known_encodings:
            return jsonify({"success": False, "message": "No face found in selfie"}), 400, {'Access-Control-Allow-Origin': '*'}

        # Use the first detected face in selfie as the probe encoding (common practice)
        known_encoding = np.array(known_encodings[0], dtype=np.float32)

        # Try to use precomputed embeddings
        embeddings_key = f"projects/gallery/{shoot_id}/embeddings.npz"
        matched = []
        npz_tmp_path = None
        used_embeddings_file = False
        try:
            fd, tmp_npz = tempfile.mkstemp(suffix=".npz")
            os.close(fd)
            npz_tmp_path = tmp_npz
            s3.download_file(Bucket=BUCKET, Key=embeddings_key, Filename=tmp_npz)
            npz = np.load(tmp_npz, allow_pickle=True)
            logger.info(f"Loaded embeddings.npz fields: {npz.files}")

            # heuristics for encodings & keys fields
            encodings_arr = None
            keys_arr = None

            possible_enc_keys = ['encodings', 'embeddings', 'data', 'arr_0', 'array', 'encoding', 'encs']
            possible_key_keys = ['keys', 'filenames', 'names', 'paths', 'keys_arr', 'arr_1', 'arr_0']

            for k in possible_enc_keys:
                if k in npz.files:
                    encodings_arr = npz[k]
                    logger.info(f"Using encodings from npz['{k}']")
                    break

            if encodings_arr is None:
                for f in npz.files:
                    try:
                        arr = np.array(npz[f])
                        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                            encodings_arr = arr
                            logger.info(f"Inferred encodings from npz['{f}']")
                            break
                    except Exception:
                        continue

            for k in possible_key_keys:
                if k in npz.files:
                    keys_arr = npz[k]
                    logger.info(f"Using keys from npz['{k}']")
                    break

            keys_list = None
            if keys_arr is not None:
                try:
                    keys_list = [(k.decode('utf-8') if isinstance(k, (bytes, bytearray)) else str(k)) for k in np.array(keys_arr).tolist()]
                except Exception:
                    keys_list = [str(k) for k in np.array(keys_arr).tolist()]

            if encodings_arr is None:
                raise ValueError(f"No numeric encodings array found in embeddings file; available fields: {npz.files}")

            encodings = np.array(encodings_arr, dtype=np.float32)
            if encodings.ndim != 2:
                raise ValueError(f"encodings array has unexpected shape {encodings.shape}")

            # Only accept keys_list if length matches encodings; otherwise fallback to direct scan
            if keys_list is None or len(keys_list) != encodings.shape[0]:
                raise ValueError("embeddings file missing a valid keys array or length mismatch; falling back to direct scan.")

            used_embeddings_file = True
            logger.info(f"Using embeddings file: {encodings.shape[0]} encodings; computing distances")

            # compute distances and filter by tolerance
            diffs = encodings - known_encoding
            dists = np.linalg.norm(diffs, axis=1)
            tol = FACE_COMPARE_TOLERANCE
            idxs = np.where(dists <= tol)[0]
            logger.info(f"Found {len(idxs)} indices within tolerance in embeddings file")

            # group by key - keep best distance per key (in case multiple faces from same image)
            best_by_key = {}
            for idx in idxs:
                key = keys_list[idx]
                dist = float(dists[idx])
                if not key or key.startswith("idx_"):
                    # avoid synthetic placeholders (defensive)
                    logger.debug(f"Skipping synthetic or invalid key {key}")
                    continue
                if key not in best_by_key or dist < best_by_key[key]:
                    best_by_key[key] = dist

            # create matched list sorted by distance ascending
            for k, d in sorted(best_by_key.items(), key=lambda kv: kv[1])[:MAX_MATCHES]:
                matched.append({"key": k, "distance": float(d)})

        except ClientError as e:
            logger.warning(f"Embeddings file not found or unreadable for shoot {shoot_id}: {e}; falling back to full scan.")
            matched = []
            used_embeddings_file = False
        except Exception as e:
            logger.warning(f"Embeddings file unusable for shoot {shoot_id}: {e}; falling back to full scan.")
            matched = []
            used_embeddings_file = False
        finally:
            if npz_tmp_path and os.path.exists(npz_tmp_path):
                try:
                    os.unlink(npz_tmp_path)
                except Exception:
                    pass

        # If embeddings didn't yield matches, fallback to scanning all images (slower).
        if not matched:
            logger.info("Falling back to scanning images directly (this can be slow).")
            paginator = s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=f"projects/gallery/{shoot_id}/")
            image_keys: List[str] = []
            for page in page_iterator:
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if key and key.lower().endswith((".jpg", ".jpeg", ".png")) and not key.endswith('/'):
                        image_keys.append(key)

            logger.info(f"Found {len(image_keys)} images to scan for direct matching.")

            if image_keys:
                with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(image_keys)))) as exe:
                    futures = {exe.submit(worker_compare, (k, known_encoding)): k for k in image_keys}
                    for fut in as_completed(futures):
                        try:
                            res = fut.result()
                            if res:
                                matched.append(res)
                                logger.info(f"Matched (direct scan) key: {res['key']} dist={res['distance']}")
                                if len(matched) >= MAX_MATCHES:
                                    logger.info("Reached MAX_MATCHES limit during direct scan.")
                                    break
                        except Exception as e:
                            logger.exception(f"Exception in direct-scan future: {e}")

            # dedupe and keep best per key
            if matched:
                best_by_key = {}
                for m in matched:
                    k = m['key']
                    d = float(m['distance'])
                    if k not in best_by_key or d < best_by_key[k]:
                        best_by_key[k] = d
                matched = [{"key": k, "distance": float(best_by_key[k])} for k in sorted(best_by_key, key=lambda x: best_by_key[x])][:MAX_MATCHES]

        logger.info(f"Total matched images after filtering: {len(matched)}")

        zip_url = None
        final_matches = []

        if matched:
            # Create zip with matched images (download each match)
            zip_fd, zip_path = tempfile.mkstemp(suffix=".zip")
            os.close(zip_fd)
            try:
                with ZipFile(zip_path, "w") as zipf:
                    for m in matched:
                        key = m["key"]
                        if not key or key.startswith("idx_"):
                            logger.debug(f"Skipping non-downloadable key {key}")
                            continue
                        fd, local_tmp = tempfile.mkstemp(suffix=os.path.splitext(key)[1] or ".jpg")
                        os.close(fd)
                        try:
                            s3.download_file(Bucket=BUCKET, Key=key, Filename=local_tmp)
                            # store in zip under the original basename (keep readable name)
                            zipf.write(local_tmp, arcname=os.path.basename(key))
                            os.unlink(local_tmp)
                        except Exception as e:
                            logger.warning(f"Failed to download matched key {key}: {e}")
                temp_zip_key = f"temp_zips/{uuid.uuid4()}.zip"
                s3.upload_file(Filename=zip_path, Bucket=BUCKET, Key=temp_zip_key)
                zip_url = create_presigned_url(Bucket=BUCKET, Key=temp_zip_key)
                try:
                    os.unlink(zip_path)
                except Exception:
                    pass

                # send email with download url (best-effort)
                try:
                    send_link_email(zip_url, recipient_email, name, phone, shoot_id)
                except Exception:
                    logger.warning("Failed to send email; continuing to return response.")

            except Exception as e:
                logger.exception(f"Error creating/uploading zip: {e}")
                if os.path.exists(zip_path):
                    try:
                        os.unlink(zip_path)
                    except Exception:
                        pass

            # copy matched objects to client folder and generate per-file presigned urls
            user_id = str(uuid.uuid4())
            user_folder = f"clients/{user_id}/"
            for m in matched:
                key = m["key"]
                if not key or key.startswith("idx_"):
                    logger.debug(f"Skipping placeholder or invalid key {key} for copy step")
                    continue
                try:
                    # Confirm object exists before copying
                    try:
                        s3.head_object(Bucket=BUCKET, Key=key)
                    except ClientError as e:
                        logger.warning(f"Matched key not available in S3 at copy-time: {key}: {e}")
                        continue

                    src = {"Bucket": BUCKET, "Key": key}
                    dest_key = f"{user_folder}{os.path.basename(key)}"
                    s3.copy_object(Bucket=BUCKET, CopySource=src, Key=dest_key)
                    presigned = create_presigned_url(Bucket=BUCKET, Key=dest_key)
                    final_matches.append({"key": key, "presigned_url": presigned, "distance": m.get("distance")})
                except Exception:
                    logger.exception(f"Failed to copy matched object {key} to {user_folder}")

            share_url = f"https://{BUCKET}.s3.{REGION}.wasabisys.com/{user_folder}" if final_matches else ""
            response = {
                "success": True,
                "message": f"Matched {len(final_matches)} images. Download link sent to {recipient_email}." if final_matches else "No matches found.",
                "matched_count": len(final_matches),
                "zip_download_url": zip_url,
                "shared_url": share_url,
                "matches": final_matches
            }
        else:
            response = {"success": True, "message": "No matches found.", "matched_count": 0, "zip_download_url": None, "shared_url": "", "matches": []}

        logger.info("Matching operation completed.")
        return jsonify(response), 200, {'Access-Control-Allow-Origin': '*'}

    except Exception as e:
        logger.exception(f"Matching failed: {e}")
        return jsonify({"success": False, "message": str(e)}), 500, {'Access-Control-Allow-Origin': '*'}

# ===== CLI / entrypoint =====
def main():
    parser = argparse.ArgumentParser(description="Face matcher server and ingestion utility.")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")

    run_parser = subparsers.add_parser("runserver", help="Run Flask development server")
    run_parser.add_argument("--host", default="0.0.0.0")
    run_parser.add_argument("--port", default=5000, type=int)
    run_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest embeddings for shoots")
    ingest_group = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument("--shoot-id", help="Single shoot id to ingest (projects/gallery/<shoot_id>/)")
    ingest_group.add_argument("--ingest-all", action="store_true", help="Ingest all shoots found under projects/gallery/")
    ingest_parser.add_argument("--daemon", action="store_true", help="Run continuous scheduler loop (legacy)")

    args = parser.parse_args()

    if args.cmd == "runserver":
        logger.info("Starting Flask server...")
        app.run(host=args.host, port=args.port, debug=args.debug)
    elif args.cmd == "ingest":
        if args.ingest_all:
            shoots = list_shoot_ids(prefix="projects/gallery/")
            logger.info(f"Will ingest {len(shoots)} shoots.")
            for sid in shoots:
                logger.info(f"Ingesting {sid} ...")
                success = ingest_shoot_embeddings(sid)
                logger.info(f"Ingest {sid} success: {success}")
                # If daemon/scheduler desired, caller can use --daemon and external scheduler (keeps script simple)
        else:
            sid = args.shoot_id
            logger.info(f"Ingesting single shoot: {sid}")
            ok = ingest_shoot_embeddings(sid)
            logger.info(f"Ingest finished: {ok}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
