import os
import redis
import random
import uuid
from datetime import datetime
from time import sleep
from dotenv import load_dotenv

load_dotenv()

user_agents = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
               'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:54.0) Gecko/20100101 Firefox/54.0',
               'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko',
               'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 OPR/45.0.2552.898',
               'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0',
               'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36']

ip = ".".join(str(random.randint(0, 255)) for _ in range(4))
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
method = random.choice(["GET","POST","PUT"])
url = '/' + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10)))
protocol = 'HTTP/1.1'
status_code = random.choice([200, 201, 204, 301, 302, 400, 401, 403, 404, 500])
size = random.randint(100, 10000)
refurl = random.choice(['https://www.example.com/','https://www.google.com/','https://www.ibm.com/',
                        'https://www.msn.com/','https://www.geovac.com','https://www.nsukka.com',
                        'https://www.anambra.com','https://www.enugu.com','https://www.odidama.com'])
user_agent = random.choice(user_agents)

timestamp_a = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
timestamp_b = datetime.now().strftime("%Y/%m/%d %H:%M:%S,%f")
timestamp_c = datetime.now().strftime("%d/%M/%Y %H:%M:%S,%f")
timestamp_d = datetime.now().strftime("%Y-%m-%d")

time_formats = [timestamp_a, timestamp_b, timestamp_c]

time_len = round(random.uniform(0, 2), 6)
small_float = round(random.uniform(0, 2), 2)
big_float = round(random.uniform(10, 5000), 2)

randint_ = random.randint(100, 5000)
srandint_ = random.randint(1, 80)
log_sample = f"{ip} - [{timestamp}] \"{method} {url} {protocol}\" {status_code} {size} \"{refurl}\" \"{user_agent}\"\n"

source = ["AnalyticsEngine","BillingSystem","LegacyCRM","ModernCRM","ModernHR","ThirdPartyAPI"]
target_label = ["HTTP Status","Critical Error","Security Alert","Error","System Notification",
                "Resource Usage","User Action","Workflow Error","Deprecation Warning"]




messages = {
            "Critical Error": [
                f"System component failure occurred: component ID Component{srandint_}",
                "Email delivery failure resulted in service issue",
                "Detection of multiple failed disks in RAID setup",
                "Kernel malfunction caused boot process to terminate",
                f"Essential system part malfunction: part ID Component{srandint_}",
                "Email service affected by failed transmission",
                "Irreparable failure detected in fundamental application",
                "Delivery issue with email service caused outage",
                "Fatal issue encountered in central system module",
                "Global settings have been compromised",
                "Systemic configuration inconsistencies detected",
                f"Key system element crashed: element ID Component{srandint_}",
                "Unfixable problem discovered in essential application module"
            ],
            "HTTP Status": [
                f"nova.metadata.wsgi.server [-] {ip},{ip} {method} /latest/meta-data/ HTTP/1.1 len: {randint_} time: {time_len}",
                f"nova.compute.claims [req-{str(uuid.uuid4())} - {str(uuid.uuid4())} --] [instance: {str(uuid.uuid4())} - ] Total vcpu: {srandint_} VCPU, used: {small_float}",
                f"nova.metadata.wsgi.server [-] 10.11.21.138,10.11.10.1 '{random.choice(method)} /openstack/2013-10-17 HTTP/1.1' RCODE  {status_code} len: {randint_} time: {small_float}"
            ],
            "Security Alert": [
                "Unauthorized access to data was attempted",
                f"Multiple bad login attempts detected on user {randint_} account",
                f"Alert: brute force login attempt from {ip} detected",
                f"Suspicious login activity detected from {ip}",
                f"Denied access attempt on restricted account Account2682",
                f"Abnormal system behavior on server {srandint_}, potential security breach",
                f"User {randint_} tried to bypass API security measures",
                f"API intrusion detection system flagged user {randint_}",
                f"Detection of admin privilege misuse by user {randint_}",
                f"Server {srandint_} is under potential security threat, review necessary",
                f"Abnormal behavior found on server 10, possible security threat",
                f"Security breach suspected from IP address {ip}",
                f"Elevation of admin privileges detected for user {randint_}",
                f"Multiple login failures were detected for user {randint_}",
                f"API security system detected suspicious activity from user {randint_}",
                " Suspicious data export activity was identified",
                f"Potential security risk from 192.168.60.100 identified",
                f"User {randint_} had multiple login attempts rejected",
                f"Detected a data transfer attempt with insufficient credentials",
                f"User {randint_} did not have permission to access API"
            ],
            "Error": [
                "Shard 6 replication task ended in failure",
                "Email server encountered a sending fault",
                f"Data replication task for shard {srandint_} did not complete",
                f"Server {srandint_} restarted without warning during data migration",
                f"Server {srandint_} crashed unexpectedly while syncing data",
                f"Data replication task failed for shard {srandint_}",
                f"Replication of data to shard {srandint_} failed",
                f"Unexpected server {srandint_} stoppage occurred during data conversion",
                f"Server {srandint_} restarted without warning during data migration",
                f"Data replication task for shard {srandint_} did not complete",
                f"Mail service encountered a delivery glitch",
                "Email delivery system encountered an error",
                "Module X failed to process input due to formatting error"
                f"Server {srandint_} suffered an abrupt restart during data import",
                f"Unforeseen server {srandint_} reboot occurred during data export",
                f"Shard {srandint_} data transfer failed",
                f"Server {srandint_} experienced an unplanned restart during data replication",
                "Invalid SSL certificate resulted in a failed service health check."
                "Input data format in module X was invalid or corrupted"
            ],
            "System Notification": [
                f"Backup ended at {timestamp_a}.",
                f"Backup started at {timestamp_a}",
                f"File data_1503.csv uploaded successfully by user User{randint_}",
                f"System reboot initiated by user User{randint_}",
                f"System updated to version {small_float}"
            ],
            "Resource Usage": [
                f"nova.compute.claims [req-{str(uuid.uuid4())} {str(uuid.uuid4())} {str(uuid.uuid4())}- - -] [instance: {str(uuid.uuid4())}] Total disk: {srandint_} GB, used: {small_float} GB",
                f"nova.compute.claims [req-{str(uuid.uuid4())} {str(uuid.uuid4())} {str(uuid.uuid4())}- - -] [instance: {str(uuid.uuid4())}] Attempting claim: memory {randint_} MB, disk {srandint_} GB, vcpus {srandint_} CPU",
                f"nova.compute.claims [req-{str(uuid.uuid4())} {str(uuid.uuid4())} {str(uuid.uuid4())}- - -] [instance: {str(uuid.uuid4())}] memory limit: {big_float} MB, free: {big_float} MB",
                f"nova.compute.claims [req-{str(uuid.uuid4())} {str(uuid.uuid4())} {str(uuid.uuid4())}- - -] [instance: {str(uuid.uuid4())}] disk limit not specified, defaulting to unlimited",
                f"nova.compute.claims [req-{str(uuid.uuid4())} {str(uuid.uuid4())} {str(uuid.uuid4())}- - -] [instance: {str(uuid.uuid4())}] Total memory: {randint_} MB, used: {big_float} MB",
            ],
            "User Action": [
                f"User User{randint_} logged out.",
                f"User User{randint_} logged in.",
                f"Account with ID {randint_} created by User{randint_}."

            ],
            "Workflow Error": [
                f"Lead conversion failed for prospect ID {randint_} due to missing contact information.",
                f"Customer follow-up process for lead ID {randint_} failed due to missing next action",
                f"Escalation rule execution failed for ticket ID {randint_} - undefined escalation level.",
                f"Task assignment for TeamID {randint_} could not complete due to invalid priority level."
            ],
            "Deprecation Warning": [
                f"API endpoint 'getCustomerDetails' is deprecated and will be removed in version {small_float} Use 'fetchCustomerInfo' instead.",
                f"The 'ExportToCSV' feature is outdated. Please migrate to 'ExportToXLSX' by the end of Q3.",
                f"Support for legacy authentication methods will be discontinued after {timestamp_d}"
            ]
        }


def connect_redis_cloud():
    redis_url = os.getenv('cl_redis_url')
    try:
        r = redis.from_url(redis_url)
        return r
    except redis.exceptions.ConnectionError as e:
        print(f"Could not connect to Redis Cloud: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# redis_server = redis.Redis(host=os.getenv('redis_host'), port=6379, db=0)
redis_server = connect_redis_cloud()

def generate_server_logs():
    target = random.choice([target for target in messages])
    target_log_msg = random.choice(messages[target])

    server_log_data = {
        "timestamp": random.choice(time_formats),
        "source": random.choice(source),
        "log_message": target_log_msg,
        "target_label": target
    }

    return server_log_data


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


