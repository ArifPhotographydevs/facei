import boto3
import os
from PIL import Image, ImageOps
import logging
from botocore.exceptions import ClientError
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
import zipfile
import io
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("helpers")

BUCKET = "arif12"
PROJECT_NAME = "FaceMatcher"

session = boto3.Session(
    aws_access_key_id="SW5I2XCNJAI7GTB7MRIW",
    aws_secret_access_key="eKNEI3erAhnSiBdcK0OltkTHIe2jJYJVhPu1eazJ",
    region_name="ap-northeast-1"
)
s3 = session.client('s3', endpoint_url="https://s3.ap-northeast-1.wasabisys.com")

def safe_resize_image(file_path):
    try:
        with Image.open(file_path) as img:
            img = ImageOps.equalize(img)  # Histogram equalization for contrast
            img.thumbnail((800, 800))
            img.save(file_path, format="JPEG")
    except Exception as e:
        raise Exception(f"Failed to resize image: {e}")

def create_presigned_url(bucket, key, expiration=3600):
    try:
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL for {key}: {e}")
        return None

def create_zip_for_matches(match_keys, shoot_id, job_id):
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, key in enumerate(match_keys):
                try:
                    response = s3.get_object(Bucket=BUCKET, Key=key)
                    image_data = response['Body'].read()
                    filename = f"match_{i+1}.jpg"
                    zip_file.writestr(filename, image_data)
                except ClientError as e:
                    logger.warning(f"Failed to add {key} to zip: {e}")
                    continue
        
        zip_buffer.seek(0)
        
        zip_key = f"projects/gallery/{shoot_id}/matches/{job_id}.zip"
        s3.put_object(Bucket=BUCKET, Key=zip_key, Body=zip_buffer)
        zip_url = create_presigned_url(BUCKET, zip_key, expiration=86400)  # 24 hours
        return zip_url
    except Exception as e:
        logger.error(f"Failed to create zip for matches: {e}")
        return None

def send_link_email(match_keys, recipient_email, name, phone, shoot_id, job_id):
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_user = "githubarifphotography@gmail.com"  # Replace with your email
        smtp_password = "utuz rvgk kmsv sntz"  # Replace with your password or App Password
        zip_url = create_zip_for_matches(match_keys, shoot_id, job_id)
        
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = recipient_email
        msg['Subject'] = f"🎉 Face Match Results for {name} - {shoot_id}"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #4a90e2;">Hello {name}!</h2>
                <p>Your face matching results from the <strong>{shoot_id}</strong> shoot are ready. We've found matches for your selfie!</p>
                
                <div style="text-align: center; margin: 20px 0;">
                    <img src="cid:first_match" alt="First Match Preview" style="max-width: 300px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    <p style="margin-top: 10px; font-size: 14px; color: #666;">Preview of your first match</p>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h3 style="color: #4a90e2;">📥 Download All Matches</h3>
                    <p>Click the button below to download a ZIP file containing all {len(match_keys)} matched images:</p>
                    <a href="{zip_url}" style="background-color: #4a90e2; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0;">Download ZIP ({len(match_keys)} Images)</a>
                    <p style="font-size: 12px; color: #666; margin-top: 10px;">Link expires in 24 hours</p>
                </div>
                
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h4 style="color: #28a745;">Contact Information</h4>
                    <p><strong>Name:</strong> {name}</p>
                    <p><strong>Email:</strong> {recipient_email}</p>
                    <p><strong>Phone:</strong> {phone}</p>
                </div>
                
                <p>If you have any questions, feel free to reply to this email. Enjoy your matched photos!</p>
                
                <hr style="margin: 30px 0;">
                <p style="font-size: 12px; color: #666; text-align: center;">
                    Sent by <strong>FaceMatcher</strong> • {PROJECT_NAME}
                </p>
            </div>
        </body>
        </html>
        """

        if match_keys:
            first_key = match_keys[0]
            try:
                response = s3.get_object(Bucket=BUCKET, Key=first_key)
                image_data = response['Body'].read()
                image = MIMEImage(image_data)
                image.add_header('Content-ID', '<first_match>')
                msg.attach(image)
            except Exception as e:
                logger.warning(f"Failed to attach preview image: {e}")

        msg.attach(MIMEText(html_body, 'html'))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Email sent successfully to {recipient_email} with ZIP download link")
    except Exception as e:
        logger.error(f"Failed to send email to {recipient_email}: {e}")
        raise