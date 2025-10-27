import streamlit as st
from datetime import datetime
from streamlit_extras.stylable_container import stylable_container
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
from helpers import run_pd_sql, get_news_article

# SQL Statements
date_today = '2025-10-25'
st_autorefresh(interval=3600000, key="data_refresh")

news_source, news_author, news_title, news_description, news_url = get_news_article()

sql_all_critical_events_cnt_today = ("select count(1) from regex_classified rc "
                                 "where CAST(rc.workflow_timestamp as DATE) = '2025-10-25' "
                                 "and regex_label in ('Workflow Error', 'Security Alert', 'Critical Error')")

sql_all_events_today = "select count(1) from regex_classified rc where CAST(rc.workflow_timestamp as DATE) = '2025-10-25'"

sql_count_grp_per_event_type = (f"with union_select as (select * from regex_classified rc union all "
             f"select * from bert_classified bc) select regex_label, count(1) as number_of_events "
             f"from union_select us group by regex_label, TO_DATE(us.workflow_timestamp, 'YYYY-MM-DD') "
             f"having TO_DATE(us.workflow_timestamp, 'YYYY-MM-DD') = '{date_today}' ")


sql_security_alert = ("select count(1) from regex_classified rc where CAST(rc.workflow_timestamp as DATE) = '2025-10-25'"
                      "and rc.regex_label = 'Security Alert'")

sql_suspicious_user_actions = (f" with union_select as (select * from regex_classified rc union all "
                               f"select * from bert_classified bc) select count(1) as UserActions "
                               f"from union_select us where regex_label = 'User Action' "
                               f"and TO_DATE(us.workflow_timestamp, 'YYYY-MM-DD') = '{date_today}'")

sql_source_system = (f"with union_select as (select * from regex_classified rc union all "
                     f"select * from bert_classified bc) SELECT distinct(us.source), count(1) "
                     f"from union_select us group by distinct(us.source), TO_DATE(us.workflow_timestamp, 'YYYY-MM-DD') "
                     f"having TO_DATE(us.workflow_timestamp, 'YYYY-MM-DD') = '{date_today}'")

sql_src_msg_trgt = (f"with union_select as (select * from regex_classified rc union all "
                    f"select * from bert_classified bc) SELECT us.source  Source_System, "
                    f"target_label Action, log_message Log_Message,"
                    f"us.workflow_timestamp as Event_Timestamp from "
                    f"union_select us where TO_DATE(us.workflow_timestamp, 'YYYY-MM-DD') = '{date_today}' "
                    f"order by TO_DATE(us.workflow_timestamp, 'YYYY-MM-DD') desc limit 9")

critical_events_count_today = run_pd_sql(sql_all_critical_events_cnt_today).iloc[0, 0].item()
all_events_today = run_pd_sql(sql_all_events_today).iloc[0, 0].item()
sql_count_grp_per_event_type = run_pd_sql(sql_count_grp_per_event_type)
sql_suspicious_user_actions = run_pd_sql(sql_suspicious_user_actions).iloc[0, 0].item()
source_systems = run_pd_sql(sql_source_system)
src_msg_trgt = run_pd_sql(sql_src_msg_trgt)



st.set_page_config(page_title="AI/ML-Powered SIEM platform", layout="wide", initial_sidebar_state="expanded")

st.sidebar.header('Real-Time Log Analytics & Anomaly Detection Pipeline')
st.sidebar.markdown("Tech Stack: Python, Pandas, Redis Streams, Scikit-learn, BERT, Logistic Regression, Regex")
st.sidebar.markdown("A near-real-time log analytics and anomaly detection platform to help organizations "
                    "proactively identify issues in their distributed systems.Using Python, Redis Streams, "
                    "the solution processed and analyzed continuous log data with high throughput and automation.")
st.sidebar.markdown("A Regex-based classifier reduced data noise, while DBSCAN clustering and BERT + Logistic "
                    "Regression models detected anomalies and unknown patterns — improving precision by 30%.")
st.sidebar.markdown("The modular, scalable design supports automated retraining, trend visualization, and alerting, "
                    "enabling faster incident response and reducing downtime risks.")
st.sidebar.markdown("This project demonstrated expertise in Python data engineering, AI model integration, "
                    "and reliable workflow automation — skills directly applicable to operational analytics, "
                    "fraud detection, and intelligent monitoring systems.")
st.sidebar.markdown("Created by nnaemeka.okeke@gmail.com")

# [0.28,0.4,0.3]
col1, col2, col3 = st.columns(3)
col1.metric(label=f":green[{datetime.now().strftime('%a. %b %d, %Y')}] - **All Events Count :** ", value=f"{all_events_today}", border=True)
col2.metric(label=f"**Critical Events:** ", value=f"{critical_events_count_today} ", border=True)
col3.metric(label=f"**Suspicious User Activity:** ", value=f"{sql_suspicious_user_actions}", border=True)
st.write("")
col4, col5 = st.columns([0.8, 1.5])
with col4:
    with st.container(height=270, border=False, vertical_alignment='center', horizontal_alignment='left'):
        fig4 = go.Figure(data=[go.Pie(labels=sql_count_grp_per_event_type['regex_label'], values=sql_count_grp_per_event_type['number_of_events'], hole=0.6)])
        fig4.update_layout(width=380, height=380, showlegend=False)
        fig4.update_traces(textinfo='label')
        fig4.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.04,  # Adjust y to move it slightly above the plot if needed
            xanchor="center", x=1.06))
        st.plotly_chart(fig4, use_container_width=True)

with col5:
    with st.container(height=270, border=False, vertical_alignment='top', horizontal_alignment='right'):
        st.markdown("**Latest Server Log Events**")
        st.dataframe(src_msg_trgt, height=250, width='stretch', hide_index=True, use_container_width=True)

st.write("")

col6_news, col7 = st.columns([0.9, 1.3])
with col6_news:
    with st.container(height=280, border=False, vertical_alignment='bottom'):

        with stylable_container(
                key="my_container_with_left_border",
                css_styles="""
                    {
                        border-left: 0.1rem solid #33cc33 !important;
                        border-right: 0.1rem solid #33cc33 !important;
                        #border-right: 0rem !important;
                        #box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;
                        height: 280px;
                        font-size: 0.8rem;
                        border-radius: 5px; /* Optional: Adds rounded corners */
                        padding: 10px; /* Optional: Adds padding inside the border */
                        padding-top: 1px;
                        #background-color: #000000;
                    }
                    """
        ):
            st.write(f"""
                :grey[{news_title}]
                <br>:grey[by {news_author}]
                <br>{news_description}
                <br>{news_url}
                """, unsafe_allow_html=True)

with col7:
    with st.container(height=280, border=False, vertical_alignment='bottom'):
        custom_colors = ['#2e5cb8', '#24478f', '#193366', '#0f1f3d', '#050a14']
        fig5 = px.bar(source_systems, x=source_systems['source'], y=source_systems['count'], color=source_systems['source'], color_discrete_sequence=custom_colors)
        fig5.update_layout(height=300, showlegend=False, title_text='')
        fig5.update_traces(width=0.4)
        st.plotly_chart(fig5, use_container_width=False)

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
