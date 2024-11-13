from config import SERVICE_HOST, SERVICE_PORT, WORKERS
loglevel = "debug"
errorlog = "-"
accesslog = "-"
bind = str(SERVICE_HOST) + ":" + str(SERVICE_PORT)
workers = WORKERS
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 3 * 60
keepalive = 24 * 60 * 60
capture_output = True