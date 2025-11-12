import streamlit as st
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
import helpers

st.markdown("""
<style>
.block-container {
    padding-top: 0rem; /* Adjust this value as needed, 0rem will remove all padding */
    padding-bottom: 0rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

# SQL Statements
# date_today = '2025-10-25'
date_today = datetime.now().strftime("%Y-%m-%d")

st_autorefresh(interval=3600000, key="data_refresh")

# news_source, news_author, news_title, news_description, news_url = helpers.get_news_article()

sql_all_critical_events_cnt_today = (f"with union_select as (select * from regex_classified rc union all select * from bert_classified bc) "
                                     f"SELECT count(1) as count from union_select us where "
                                     f"regex_label in ('Workflow Error', 'Security Alert', 'Critical Error') and CAST(workflow_timestamp as DATE) = '{date_today}'")

# sql_all_critical_events_cnt_today = (f"select count(1) from regex_classified rc "
#                                  f"where CAST(rc.workflow_timestamp as DATE) = '{date_today}' "
#                                  f"and regex_label in ('Workflow Error', 'Security Alert', 'Critical Error')")

sql_all_events_today = f"select count(1) from regex_classified rc where CAST(rc.workflow_timestamp as DATE) = '{date_today}'"

sql_count_grp_per_event_type = (f"with union_select as (select * from regex_classified rc union all "
             f"select * from bert_classified bc) select regex_label, count(1) as number_of_events "
             f"from union_select us group by regex_label, workflow_timestamp "
             f"having CAST(workflow_timestamp as DATE) = '{date_today}' ")


sql_security_alert = (f"select count(1) from regex_classified rc where CAST(rc.workflow_timestamp as DATE) = '{date_today}'"
                      f"and rc.regex_label = 'Security Alert'")

sql_suspicious_user_actions = (f" with union_select as (select * from regex_classified rc union all "
                               f"select * from bert_classified bc) select count(1) as UserActions "
                               f"from union_select us where regex_label = 'User Action' "
                               f"and CAST(workflow_timestamp as DATE) = '{date_today}'")

sql_source_system = (f"with union_select as (select * from regex_classified rc union all "
                     f"select * from bert_classified bc) SELECT distinct(source), count(1) as count "
                     f"from union_select us group by source, workflow_timestamp "
                     f"having CAST(workflow_timestamp as DATE) = '{date_today}'")

sql_src_msg_trgt = (f"with union_select as (select * from regex_classified rc union all "
                    f"select * from bert_classified bc) SELECT us.source  Source_System, "
                    f"target_label Action, log_message Log_Message,"
                    f"workflow_timestamp as Event_Timestamp from "
                    f"union_select us where CAST(workflow_timestamp as DATE) = '{date_today}' "
                    f"order by CAST(workflow_timestamp as DATE) desc limit 9")

sql_latest_news = (f"select news_author, news_title, news_url  from news "
                   f"order by random() limit 1")

sql_security_events_cnt_hist = (f"with union_select as (select * from regex_classified rc  union all "
                                f"select * from bert_classified bc) SELECT CAST(workflow_timestamp as DATE),"
                                f" count(1) from union_select us group by CAST(workflow_timestamp as DATE)), "
                                f"us.target_label having us.target_label = 'Security Alert' "
                                f"and CAST(us.workflow_timestamp as DATE) is not null")

critical_events_count_today = helpers.run_pd_sql(sql_all_critical_events_cnt_today).iloc[0, 0].item()
all_events_today = helpers.run_pd_sql(sql_all_events_today).iloc[0, 0].item()
sql_count_grp_per_event_type = helpers.run_pd_sql(sql_count_grp_per_event_type)
sql_suspicious_user_actions = helpers.run_pd_sql(sql_suspicious_user_actions).iloc[0, 0].item()
source_systems = helpers.run_pd_sql(sql_source_system)
src_msg_trgt = helpers.run_pd_sql(sql_src_msg_trgt)
latest_news = helpers.run_pd_sql(sql_latest_news)


st.set_page_config(page_title="AI/ML-Powered SIEM platform", layout="wide", initial_sidebar_state="expanded")

st.sidebar.header(f"Real-Time Log Analytics & Anomaly Detection Pipeline")
st.sidebar.markdown(f":green[**Tech Stack:** Python, Pandas, Redis Streams, Scikit-learn, BERT, Logistic Regression, Regex]")
st.sidebar.markdown(f":small[A near-real-time log analytics and anomaly detection platform to help organizations "
                    f"proactively identify issues in their distributed systems.Using Python, Redis Streams, "
                    f"the solution processed and analyzed continuous log data with high throughput and automation.]")
st.sidebar.markdown(f":small[A Regex-based classifier reduced data noise, while DBSCAN clustering and BERT + "
                    f"Logistic Regression models detected anomalies and unknown patterns — improving precision by 30%.]")
st.sidebar.markdown(":small[The modular, scalable design supports automated retraining, trend visualization, "
                    "and alerting, enabling faster incident response and reducing downtime risks.]")
st.sidebar.markdown(f":small[This project demonstrated expertise in Python data engineering, AI model integration, "
                    f"and reliable workflow automation — skills directly applicable to operational analytics, "
                    f"fraud detection, and intelligent monitoring systems.]")
st.sidebar.markdown(f":small[Created by] nnaemeka.okeke@gmail.com")

# [0.28,0.4,0.3]
col1, col2, col3 = st.columns(3)
col1.metric(label=f":green[{datetime.now().strftime('%a. %b %d, %Y')}] - **All Events Count :** ", value=f"{all_events_today}", border=True)

if critical_events_count_today  == 0 or sql_suspicious_user_actions == 0:
    col2.metric(label=f"**Critical | Suspicious Events:** ", value=f"{0} | {0}", border=True)
else:
    col2.metric(label=f"**Critical | Suspicious Events:** ", value=f"{critical_events_count_today} | {sql_suspicious_user_actions}", border=True)

with col3:

    # st.metric(label=f"News** ", value=f"{latest_news['news_title'].item()}", border=True)
    with st.container(border=True, height=210):
        st.markdown("CyberSec & A.I Events:")
        st.markdown(f":grey[:small[{latest_news['news_title'].item()}]]")
        st.markdown(f":grey[:small[by {latest_news['news_author'].item()}]]")
        st.markdown(f":small[{latest_news['news_url'].item()}]")
st.write("")
col4, col5 = st.columns([0.8, 1.5])
with col4:
    with st.container(height=300, border=False, vertical_alignment='bottom', horizontal_alignment='left'):
        fig4 = go.Figure(data=[go.Pie(labels=sql_count_grp_per_event_type['regex_label'],
                                      values=sql_count_grp_per_event_type['number_of_events'], hole=0.6)])
        fig4.update_layout(width=380, height=380, showlegend=False)
        fig4.update_traces(textinfo='label')
        fig4.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.04,  # Adjust y to move it slightly above the plot if needed
            xanchor="center", x=1.06))
        st.plotly_chart(fig4, width=True)

with col5:
    with st.container(height=280, border=False, vertical_alignment='center'):
        # custom_colors = ['#2e5cb8', '#003399', '#29293d', '#002b80', '#00134d', '#264d73']
        colors = ['lightslategray', 'crimson', 'lightslategray', 'lightslategray', 'lightslategray', '#29293d']
        # bar_trace = go.Bar(x=source_systems['source'], y=source_systems['count'])
        # fig5 = go.Figure(data=[bar_trace])
        fig5 = go.Figure(go.Bar(x=source_systems['source'], y=source_systems['count']))
        fig5.update_layout(height=320, showlegend=False, title_text='Source Systems')
        fig5.update_traces(width=0.4)
        st.plotly_chart(fig5, use_container_width=True)


with st.container(height=300, border=False, vertical_alignment='top', horizontal_alignment='right'):
    st.markdown("**Latest Server Log Events**")
    st.dataframe(src_msg_trgt, height=250, width='stretch', hide_index=True, use_container_width=True)

st.markdown('''
<style>
/*center metric label*/
[data-testid="stMetricLabel"] p {
    font-size: 1.5rem;
}

div[data-testid="stMetric"] {
        #border: 1px solid grey; 
        border-left: 0.5rem solid #9AD8E1 !important;
        border-right: 0rem !important;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
        border-radius: 5px; /* Optional: Adds rounded corners */
        padding: 10px; /* Optional: Adds padding inside the border */
        padding-top: 1px;
        background-color: #001a33;
    }

[data-testid="stMetricValue"] > div:nth-child(1) {
    justify-content: right;
    font-size: 100px; /* Adjust this value as needed */;
}

</style>
''', unsafe_allow_html=True)
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 400px !important;  # Set the desired width in pixels
    }
    </style>
    """,
    unsafe_allow_html=True,
)