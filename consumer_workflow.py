from time import sleep
from datetime import datetime
from helpers import consume_from_redis_q, convert_logs_to_dataframe, classify_pd_with_regex, classify_with_bert, connect_to_pg_db, perform_clustering
import pandas as pd
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

# logging.basicConfig(level=logging.INFO, filename=f'logs/consumer_workflow_{datetime.now().strftime('%Y%m%d-%H%M%S.%f')[:-3]}.txt',
#                     format='%(process)d--%(asctime)s--%(levelname)s--%(message)s')

conn = connect_to_pg_db()

# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# load pretrained sentence transformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def consume_redis_stream(stream_name):

    # consume logs from redis streams - specify the stream name [server_logs]
    logs = consume_from_redis_q(stream_name)

    return logs

def convert_logs_to_df(logs):

    read_into_df = convert_logs_to_dataframe(logs)

    return read_into_df


def perform_log_column_clustering(logs_df):

    # logs, eps = 0.5, min_sample = 5, metric = 'cosine'
    cluster = perform_clustering(logs_df, eps=0.5, min_sample=5, metric='cosine')
    print("df Clustering done...")

    return cluster

# --------------------------------------------------------------------------------------------------------------------
# After running the data thru the regex, write the classified to a table - like regex classified
# then write the unclassified to a separate table for further analysis with ai and co
#  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# --------------------------------------------------------------------------------------------------------------------

def regex_classify_logs_dataframe(read_into_df):
    # run the regex classify fxn on the dataframe and write to db
    read_into_df['regex_label'] = read_into_df['log_message'].apply(classify_pd_with_regex)
    print(f"regex_classify done \n {read_into_df}")

    # make a copy of only regex classified records - without null
    df_regex_classified = read_into_df[read_into_df['regex_label'].notnull()].copy()
    print(f"made copy of df")

    # add a timestamp and write the regex classified records output to db
    df_regex_classified['workflow_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f" \n Data to load \n{df_regex_classified}")
    df_regex_classified.to_sql(name='regex_classified', con=conn, if_exists='append', index=False)
    print(f"Data loaded into regex_classified table")

    # create another df with unclassified messages to be used for the linear regression fxn
    df_no_regex = read_into_df[read_into_df['regex_label'].isnull()].copy()
    print(f"df without regex: \n {df_no_regex}")

    return df_no_regex



# create embeddings for the non_regex classified table. This is for the log regression
# classifier_model_lin_reg(df_no_regex)

def bert_classify_unclassified_logs(df_no_regex):
    # classify with BERT and create a new bert_label column to hold the values
    df_no_regex['bert_label'] = df_no_regex['log_message'].apply(classify_with_bert)

    # remove the regex label column as it is no longer required at this stage
    df_no_regex.drop('regex_label', axis=1, inplace=True)

    # make a copy of the same table to hold the unclassified rows, which will be passed to the llm model
    df_no_regex_no_bert = df_no_regex[df_no_regex['bert_label'].isnull()].copy()

    # add a timestamp and write the df to a pg table
    df_no_regex['workflow_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_no_regex.to_sql(name='bert_classified', con=conn, if_exists='append', index=False)
    conn.close()

    return df_no_regex_no_bert


def consumer_workflow():

    logs = consume_redis_stream("server_logs")

    logs_df = convert_logs_to_dataframe(logs)

    cluster = perform_log_column_clustering(logs_df)

    df_no_regex = regex_classify_logs_dataframe(cluster)

    df_no_regex_no_bert = bert_classify_unclassified_logs(df_no_regex)

    print(df_no_regex_no_bert.head())


if __name__ == "__main__":
    while True:
        print("Processing Redis Stream logs...")
        consumer_workflow()
        print("Process completed, waiting 5 Minutes to restart")
        sleep(300000)  # Seconds

