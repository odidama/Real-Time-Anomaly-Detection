from sqlalchemy import create_engine
import requests
import random
import uuid
import os
from groq import Groq
import pandas as pd
from datetime import datetime
import pickle
import redis
from redis.client import Redis, ConnectionPool
import streamlit as st
import re
from sentence_transformers import  SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

engine = None

groq = Groq()

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


def connect_to_db():

    global engine
    if engine is None:

        # DB_USER = os.getenv("db_user")
        DB_USER = st.secrets["db_user"]
        # DB_PASSWORD = os.getenv("db_password")
        DB_PASSWORD = st.secrets["db_password"]
        # DB_HOST = os.getenv("db_host")
        DB_HOST = st.secrets["db_host"]
        # DB_PORT = os.getenv("db_port")
        DB_PORT = st.secrets["db_port"]
        # DB_NAME = os.getenv("db_name")
        DB_NAME = st.secrets["db_name"]

        pgdb_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

        engine = create_engine(pgdb_url)
    return engine
    # return None

conn = connect_to_db()

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

def connect_redis_cloud():
    redis_url = st.secrets['cl_redis_url']
    try:
        r = redis.from_url(redis_url)
        return r
        # r = redis.Redis(
        #     host= st.secrets['cl_redis_host'],
        #     port= 16478,
        #     decode_responses=True,
        #     username=st.secrets['cl_redis_user'],
        #     password=st.secrets['cl_redis_password']
        # )
    except redis.exceptions.ConnectionError as e:
        st.error(f"Could not connect to Redis Cloud: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        # success = r.set('foo', 'bar')
        # True

        # result = r.get('foo')
        # print(result)
        # >>> bar



redis_server = connect_redis_cloud()

def consume_from_redis_q(stream):
    """
    args / stream keys can either be firewall_logs or server_logs
    :param stream:
    :return:
    """
    # redis_server = redis.Redis(host=os.getenv('redis_host'), port=6379, db=0)
    # redis_server = redis.Redis(host=st.secrets['redis_host'], port=6379, db=0)
    stream_key = stream
    last_id = '0-0'

    redis_messages = redis_server.xread({stream_key: '0-0'}, count=1000)
    processed_streams = []
    message_id = []

    for stream_id, fields in redis_messages[0][1]:
        entry_data = {"message_id": stream_id}
        message_id.append(stream_id)
        for key, value in fields.items():
            entry_data[key] = value
        processed_streams.append(entry_data)


    for msg_id in message_id:
        redis_server.xdel(stream_key, msg_id)

    return processed_streams



def convert_logs_to_dataframe(logs):

    df = pd.DataFrame(logs)
    # print(read_into_df)
    return df


def perform_clustering(logs, eps, min_sample, metric):
    # convert the extracted logs into a dataframe using pandas
    read_into_df = convert_logs_to_dataframe(logs)

    # generate embeddings for log messages
    embeddings = model.encode(read_into_df['log_message'].tolist())
    # print(pd.DataFrame(embeddings[:2]))

    # perform clustering with DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_sample, metric=metric)
    clusters = dbscan.fit_predict(embeddings)

    # create a new column to hold cluster info
    read_into_df['clusters'] = clusters

    return read_into_df

def classify_pd_with_regex(log_message):

    regex_patterns = {
        r"Account with ID \d+ created by User\d+": "User Action",
        r"Escalation rule execution failed for ticket ID \d+ - undefined *": "Workflow Error",
        r"nova.compute.claims* memory*" : "Resource Usage",
        r"nova.compute.claims* Total vcpu*" : "Resource Usage",
        r"nova.compute.claims* Attempting*" : "Resource Usage",
        r"nova.compute.claims* disk*" : "Resource Usage",
        r"User User\d+ logged (out|in)*" : "User Action",
        r"Account with ID * created*" : "User Action",
        r"nova.metadata.wsgi.server*" : "HTTP Status",
        r"Lead conversion failed for prospect*" : "Workflow Error",
        r"Customer follow-up process for lead ID*" : "Workflow Error",
        r"Backup (started|ended) at*" : "System Notification",
        r"Email service affected by failed*" : "Critical Error",
        r"Email delivery system*" : "Error",
        r"The * feature is outdated. Please migrate to*" : "Deprecation Warning",
        r"API endpoint * is deprecated and will be removed*" : "Deprecation Warning",
        r"Support for legacy * methods will be discontinued*" : "Deprecation Warning",
        r"Task assignment for * could not complete*" : "Workflow Error",
        r"System updated to version*" : "System Notification",
        r"File * uploaded successfully by user*" : "System Notification",
        r"Detection of multiple failed disks*" : "Critical Error",
        r"Global settings have been compromised*" : "Critical Error",
        r"Irreparable failure detected in fundamental" : "Critical Error",
        r"Kernel malfunction caused boot process*" : "Critical Error",
        r"Systemic configuration inconsistencies*" : "Critical Error"
    }
    for regex_pattern, label in regex_patterns.items():
        if re.search(regex_pattern, log_message, re.IGNORECASE):
            return label

    return None



def classifier_model_lin_reg(df_no_regex):

    non_regex_embeddings = model.encode(df_no_regex['log_message'].tolist())

    X = non_regex_embeddings
    y = df_no_regex['target_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)

    print(report)

    filename = 'models/log_classifier.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(clf, file)

    return None


# load sentence transformer model to compute log_msg embeddings
transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# load the classification model
with open('models/log_classifier.pkl', 'rb') as file:
    classifier_model = pickle.load(file)


def classify_with_bert(log_message):

    # compute the embeddings for the log message
    msg_embeddings = transformer_model.encode(log_message)

    probabilities = classifier_model.predict_proba([msg_embeddings])[0]

    if max(probabilities) < 0.5:
        return "Unclassified"

    # perform classification using the loaded - classifier - model
    predicted_class = classifier_model.predict([msg_embeddings])[0]

    return  predicted_class



def classify_log_msg_with_llm(log_message):

    prompt = f'''Classify the log message into one of the following categories:
    (1) Workflow Error, (2) Deprecation Warning. If you cannot figure out the category, return "Unclassified".
    Only return the category name. No preamble.
    Lod message: {log_message}'''

    chat_completions = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ])
    return chat_completions.choices[0].message.content




def run_pd_sql(stmnt):
    output = pd.read_sql(stmnt, con=conn)

    return output


def get_news_article():
    topics = ['ai', 'cyber security']
    news_url = f"https://newsapi.org/v2/everything?q={random.choice(topics)}&apiKey={st.secrets['NEWS_API_KEY']}&pageSize=1"
    result = requests.get(news_url)
    news_result = result.json()
    # print(type(news_result))
    for i in news_result["articles"]:
        news_source = i["source"]["name"]
        news_author = i["author"]
        news_title = i["title"]
        news_description = i["description"]
        news_url = i["url"]
        # print(i)
    return news_source, news_author, news_title, news_description, news_url
