import boto3
from botocore.config import Config
from helpers import ACCESS_KEY, SECRET_KEY, REGION, ENDPOINT_URL, BOTO_CONFIG

s3 = boto3.client(
    "s3",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION,
    endpoint_url=ENDPOINT_URL,
    config=BOTO_CONFIG
)

BUCKET = "arif12"

def list_all_prefixes(bucket, delimiter='/', max_pages=10):
    prefixes = set()
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Delimiter=delimiter)
    page_count = 0
    for page in page_iterator:
        page_count += 1
        if page_count > max_pages:
            break
        for cp in page.get("CommonPrefixes", []):
            prefixes.add(cp.get("Prefix", "").rstrip('/'))
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if '/' in key and key.rstrip('/') not in prefixes:
                prefixes.add(key.rsplit('/', 1)[0])
    return sorted(list(prefixes))

print("All top-level prefixes in bucket 'arif12':")
prefixes = list_all_prefixes(BUCKET)
for p in prefixes:
    print(f"- {p}")

# Specifically check for projects/gallery
print("\nChecking under 'projects/gallery/':")
paginator = s3.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket=BUCKET, Prefix='projects/gallery/', Delimiter='/')
shoots = []
for page in page_iterator:
    for cp in page.get("CommonPrefixes", []):
        p = cp.get("Prefix", "")
        if p.startswith('projects/gallery/'):
            shoot_id = p[len('projects/gallery/'):].rstrip('/')
            if shoot_id:
                shoots.append(shoot_id)
print("Folders under projects/gallery/:")
for s in sorted(shoots):
    print(f"- {s}")
