# TODO for Task: Update Project Name from Wasabi Gallery and Fix Timeouts

- [x] Update helpers.py: Add PROJECT_NAME="Arif CRM Gallery" after BUCKET definition, and update BOTO_CONFIG to include connect_timeout=30, read_timeout=30.

- [x] Update app.py: In list_shoots(), change prefix from "shoots/" to "projects/gallery/"; increase job_timeout from 60*5 to 60*10 in match_face_enqueue().

- [x] Update ping_redis.py: Change socket_connect_timeout from 5 to 10.

- [x] Test changes: Execute `python ping_redis.py` to verify Redis connection; start the app with `python app.py` and check /shoots endpoint for new prefix; monitor for timeout issues.
