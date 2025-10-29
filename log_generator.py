import os
import redis
from time import sleep
from dotenv import load_dotenv
from helpers import generate_server_logs, connect_redis_cloud

load_dotenv()

# redis_server = redis.Redis(host=os.getenv('redis_host'), port=6379, db=0)
redis_server = connect_redis_cloud()


def stream_server_logs():
    data = generate_server_logs()
    stream_key = "server_logs"
    redis_server.xadd(stream_key, data)
    return {"Message":"Event streamed successfully", "Event_id":stream_key}


if __name__ == '__main__':
    while True:
        print("Streaming..")
        stream_server_logs()
        print("Streaming completed, waiting 30 seconds to restart")
        sleep(30)  # Seconds


