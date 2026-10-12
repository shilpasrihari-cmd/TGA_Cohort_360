import plotly.express as px
import streamlit as st
from data import filter_data, counts, community_metrics, geographic_counts
from src.mock_dataset import load_mock
from night_globe import render_globe
from learning_ui import render_learning, render_ai_insights
from src.topic_classifier import TOPICS

st.set_page_config(page_title='TGA Cohort 360', page_icon='🌐', layout='wide')
st.markdown('''<style>
.block-container {padding-top:2rem; max-width:1500px;}
[data-testid="stMetric"] {background:var(--secondary-background-color); border-radius:12px; padding:16px; height:100%;}
[data-testid="stMetricLabel"] {white-space:normal;}
</style>''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    people,messages,_ = load_mock()
    return people,messages

people, questions = load_data()
def reset_filters():
    for key in ['regions', 'countries', 'topics', 'ai_topics']:
        st.session_state[key] = []

st.title('TGA Cohort 360')
st.markdown('### Global Community & Learning Intelligence')
st.info('SYNTHETIC DEMO DATA · Entirely fictional messages and participants. No original TGA transcript is read or represented. AI results use local inference only.')

with st.sidebar:
    st.header('Community filters')
    regions = st.multiselect('Regions', sorted(people.region.unique()), key='regions')
    available = people[people.region.isin(regions)] if regions else people
    valid_countries = sorted(available.country.unique())
    if 'countries' in st.session_state:
        st.session_state.countries = [c for c in st.session_state.countries if c in valid_countries]
    countries = st.multiselect('Countries', valid_countries, key='countries')
    st.caption('Country and region filters apply to all tabs. Empty selection means all.')
    st.header('Learning filters')
    topics = st.multiselect('Topics', sorted(questions.topic.unique()), key='topics')
    st.multiselect('AI topic classifications', TOPICS, key='ai_topics')
    st.caption('Topic filters affect Learning Intelligence and AI Insights, not the geographic dataset.')
    st.button('Reset filters', width='stretch', on_click=reset_filters)

filtered_people, _ = filter_data(people, questions, regions, countries, topics)
st.caption(f'Synthetic demo · Geography: {", ".join(regions + countries) or "all locations"} · Topics: {", ".join(topics) or "all topics"}')

def metrics(labels, values):
    columns = st.columns(4)
    for col, label, value in zip(columns, labels, values):
        col.metric(label, f'{value:,}' if isinstance(value, int) else value,
                   icon=':material/group:' if label == 'Total Participants' else None)
    st.caption('All KPI values are synthetic and reflect the active filters.')

def bar(frame, column, title, limit=10):
    st.subheader(title)
    result = counts(frame, column).head(limit)
    if result.empty:
        st.info('No synthetic records match these filters. Clear a filter to explore more.')
        return
    fig = px.bar(result.sort_values('Count'), x='Count', y=column, orientation='h', text='Count', color_discrete_sequence=['#36b7bc'])
    fig.update_layout(xaxis_title='Synthetic count', yaxis_title=None, margin=dict(l=0,r=10,t=10,b=0), height=360)
    st.plotly_chart(fig, width='stretch')

community, learning, insights = st.tabs(['🌎 Global Community', '📊 Learning Intelligence', '🧠 AI Insights & Recommendations'])
with community:
    metrics(['Total Participants', 'Countries', 'Cities', 'Top Country'], community_metrics(filtered_people))
    st.subheader('Community across the world')
    st.caption('Synthetic participants by country · Hover for counts · Drag to rotate · Scroll to zoom.')
    if filtered_people.empty:
        st.info('No synthetic participants match these locations. Adjust the community filters.')
    else:
        locations = geographic_counts(filtered_people)
        render_globe(locations)
    left, right = st.columns(2)
    with left:
        bar(filtered_people, 'country', 'Top 10 countries')
    with right:
        cities = filtered_people.assign(location=filtered_people.city + ', ' + filtered_people.country)
        bar(cities, 'location', 'Top 10 cities')

with learning:
    learning_context = render_learning(people, questions, filtered_people, topics, metrics, bar)

with insights:
    render_ai_insights(learning_context)
st.divider()
st.caption('TGA Cohort 360 · Layout prototype · Synthetic data only · No live integrations')
