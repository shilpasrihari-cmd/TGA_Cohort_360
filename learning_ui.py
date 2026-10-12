"""Learning Intelligence presentation, with computation isolated in src/."""
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from src.embedding_model import LocalEmbedder, MODEL_DIR, ModelUnavailableError
from src.topic_classifier import TopicClassifier
from src.synthetic_chat import prepare_synthetic_messages
from src.pipeline import analyze_messages, filter_intelligence
from src.learning_analytics import summarize_learning, privacy_safe_export
from src.mock_dataset import load_mock
from src.model_evaluation import evaluate

@st.cache_data(max_entries=1)
def model_evaluation(messages, truth, revision, _embedder, _classifier):
    result = intelligence_results(messages, revision, _embedder, _classifier)
    return evaluate(result, truth, _embedder, _classifier)

@st.cache_resource(max_entries=1)
def local_resources(model_revision):
    embedder = LocalEmbedder()
    return embedder, TopicClassifier(embedder)

@st.cache_data(max_entries=2)
def intelligence_results(messages, model_revision, _embedder, _classifier):
    return analyze_messages(messages, _embedder, _classifier)

def render_learning(participants, fixture_questions, filtered_participants, source_topics, metrics, bar):
    st.caption('SYNTHETIC · Questions are detected with Python rules; topic and theme similarities use a real, local all-MiniLM-L6-v2 model.')
    try:
        manifest_path = MODEL_DIR / 'download_manifest.json'
        revision = json.loads(manifest_path.read_text(encoding='utf-8'))['revision'] if manifest_path.exists() else 'local'
        embedder, classifier = local_resources(revision)
        messages = prepare_synthetic_messages(participants, fixture_questions)
        result = intelligence_results(messages, revision, embedder, classifier)
    except ModelUnavailableError as error:
        st.warning(str(error))
        st.info('Learning AI results are unavailable. No placeholder AI classifications or themes are displayed. The other dashboard tabs remain available.')
        return
    chosen_topics = st.session_state.get('ai_topics', [])
    st.caption('Sidebar Topics filters fictional chat channels; AI topic classifications filters model topics. Both Learning Intelligence and AI Insights use the same selection.')
    selected = filter_intelligence(result.questions, filtered_participants.participant_id, source_topics, chosen_topics)
    analytics = summarize_learning(selected)
    values = [value if value is not None else 'Not assessed' for value in analytics['metrics']]
    metrics(['Total Questions', 'Learning Topics', 'Repeated Question Themes', 'Questions Without Identifiable Chat Answers'], values)
    st.caption('Total Questions counts detected messages, including clarification and support requests. Repeated themes have at least two questions. Answer availability comes from simulated fixture flags; this engine does not identify chat answers.')
    left, right = st.columns(2)
    with left:
        bar(selected[selected.category == 'Learning'], 'topic', 'Most frequently discussed AI topics')
    with right:
        st.subheader('Top recurring question themes')
        recurring = analytics['recurring'].head(10).copy()
        if recurring.empty:
            st.info('No recurring synthetic themes match the selected filters.')
        else:
            recurring['label'] = recurring.theme + ' · ' + recurring.cluster_id
            fig = px.bar(recurring.sort_values('question_count'), x='question_count', y='label',
                         orientation='h', text='question_count', color_discrete_sequence=['#7c83e8'],
                         hover_data=['topic', 'participant_count', 'needs_review'])
            fig.update_layout(height=360, xaxis_title='Synthetic questions', yaxis_title=None, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, width='stretch')
    st.subheader('Learning vs. support questions')
    if selected.empty:
        st.info('No synthetic questions match these filters. Select another topic or clear a location filter.')
    else:
        distribution = analytics['distribution']
        fig = px.pie(distribution, names='category', values='question_count', hole=.65, color='category',
                     color_discrete_map={'Learning':'#36b7bc','Support':'#7c83e8'})
        fig.update_layout(height=320, margin=dict(t=0,b=0))
        st.plotly_chart(fig, width='stretch')
    st.caption('Synthetic question counts measure discussion frequency, not learning proficiency. Support means a course-access topic or a rule-detected troubleshooting request.')
    with st.expander('How the local intelligence works'):
        st.write('Python rules detect questions, clarification requests, support requests, reactions, and ambiguous messages. Ambiguous nonquestions remain outside the statistics and need human review.')
        st.write('The local embedding model compares each detected message with eight topic descriptions. Scores below 0.30 become Other / Unclear; low scores or close alternatives are flagged. Scores are similarities, not calibrated probabilities.')
        st.write('Within each topic, complete-link clustering requires cosine similarity of at least 0.70 for every pair. Groups below 0.80 minimum similarity, borderline singletons, and uncertain classifications are flagged. Clusters are formed once before filters; displayed counts are recalculated after filters.')
        st.write('Theme names use model similarity to curated descriptions. pandas calculates totals and distinct pseudonymous participants. Follow-up suggestions and export suppression use deterministic rules. No generative AI or external AI API is used.')
        cohort_detections = result.detections[result.detections.participant_id.isin(filtered_participants.participant_id)]
        if source_topics:
            cohort_detections = cohort_detections[cohort_detections.source_topic.isin(source_topics)]
        st.caption(f'Synthetic detection audit for geography/channel selection: {len(cohort_detections)} messages; {int(cohort_detections.is_question.sum())} detected requests; {int(cohort_detections.detection_review.sum())} ambiguous messages. AI topic filters do not change this detection audit.')

    return selected, analytics

def render_ai_insights(context):
    if context is None:
        st.info('AI insights are unavailable until the local model is ready. No placeholder results are displayed.')
        return
    selected, analytics = context
    st.subheader('AI-Discovered Question Themes')
    st.caption(f'SYNTHETIC · {analytics["review_count"]} questions in this selection need review. Names are selected from a curated theme vocabulary using embedding similarity; they are not generated summaries.')
    if analytics['themes'].empty:
        st.info('No AI-discovered themes for this selection.')
    else:
        st.dataframe(analytics['themes'].rename(columns={'cluster_id':'Cluster ID', 'theme':'Theme', 'topic':'Learning topic',
                     'question_count':'Questions', 'participant_count':'Distinct pseudonymous participants', 'needs_review':'Needs review'}),
                     hide_index=True, width='stretch')
    st.subheader('Suggested Follow-up Topics')
    st.caption('Proposed learning opportunities, not confirmed learning gaps. Eligibility uses the existing rules: repeated themes, at least two distinct participants, no review flag, and a non-course-support topic. Human validation is still required; no completed review is implied.')
    followups = analytics['followups']
    if followups.empty:
        st.info('No sufficiently clear recurring learning themes qualify for a follow-up in this selection.')
    else:
        st.dataframe(followups[['suggested_session','topic','question_count','participant_count']].rename(
                     columns={'suggested_session':'Suggested session title','topic':'Learning topic','question_count':'Questions','participant_count':'Distinct participants'}),
                     hide_index=True, width='stretch')
    st.download_button('Download synthetic aggregate themes', privacy_safe_export(selected),
                       file_name='synthetic_learning_themes.csv', mime='text/csv')
    st.caption('Export contains controlled aggregate labels and counts only. Groups with fewer than three distinct participants are suppressed; no messages or participant identifiers are exported.')
    st.subheader('AI Model Evaluation')
    _, messages, truth = load_mock()
    manifest = MODEL_DIR / 'download_manifest.json'
    revision = json.loads(manifest.read_text(encoding='utf-8'))['revision'] if manifest.exists() else 'local'
    embedder, classifier = local_resources(revision)
    scores = model_evaluation(messages.rename(columns={'message':'text'}).drop(columns=['topic']), truth, revision, embedder, classifier)
    st.caption(f'SYNTHETIC DEMO DATA · {scores["model"]} · Full fixed fixture: {scores["evaluated_messages"]} messages, {scores["evaluated_questions"]} authored requests. Evaluation is independent of sidebar filters.')
    st.write('Ground truth is authored evaluation data, not model output. Question detection uses Python rules; topic and cluster predictions use local model embeddings. Labels are never passed into inference.')
    st.dataframe(pd.DataFrame([
        {'Evaluation':'Question detection (rules)', **scores['detection']},
        {'Evaluation':'Topic classification (embeddings)', 'accuracy':scores['topic_accuracy'],'macro-F1':scores['topic_macro_f1']},
        {'Evaluation':f'Cluster membership ({scores["clustering"]["pairs"]} independent pairs)', **{k:v for k,v in scores['clustering'].items() if k!='pairs'}},
    ]).round(4), hide_index=True, width='stretch')
    st.write(f'{scores["review_count"]} model-pipeline question predictions require human review in the full fixture.')
    st.caption('Topic macro-F1 averages the eight authored topic classes; Other / Unclear predictions count as missed labels. Pair positives share question intent; negatives differ in intent, including within the same topic. Pair scores evaluate final cluster membership, not only pairwise cosine similarity.')
    st.warning('Small synthetic benchmark: repeated template wording overlaps curated references, making inventory scores optimistic. The 12 separately authored pairs provide a limited independent check, not evidence of real-chat performance. Review flags do not mean review is complete.')
    with st.expander('Topic confusion matrix and fictional classification errors'):
        st.caption('Rows: authored ground truth. Columns: local model prediction.')
        st.dataframe(scores['confusion'], width='stretch')
        st.dataframe(scores['errors'].rename(columns={'topic_truth':'Authored topic','topic_prediction':'Model topic','text':'Fictional question'}),hide_index=True,width='stretch')
        st.caption('Credential-security questions can span Development Tools and AI Safety. Low similarity may trigger Other / Unclear instead of a confident classification. Duplicate templates can repeat the same error; error counts are not independent examples.')
        st.dataframe(scores['pair_results'],hide_index=True,width='stretch')
