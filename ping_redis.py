import os, redis
host = os.environ.get("REDIS_HOST", "localhost")
port = int(os.environ.get("REDIS_PORT", 6379))
r = redis.Redis(host=host, port=port, socket_connect_timeout=10)
print("Connecting to", f"{host}:{port}")
print("Ping:", r.ping())
