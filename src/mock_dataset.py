"""Authored fictional data only: no transcript, names, or AI service input."""
import random
from pathlib import Path
import pandas as pd
from data import LOCATIONS
from .topic_classifier import THEME_DESCRIPTIONS,TOPIC_DESCRIPTIONS

SEED=1360
DIRECTORY=Path(__file__).resolve().parents[1]/'data/synthetic'
FILES=['mock_messages.csv','mock_participants.csv','mock_locations.csv','mock_ground_truth.csv']
LABEL='SYNTHETIC DEMO DATA'
ALTERNATIVES={
 'Machine learning basics':'How does machine learning find patterns in training examples?',
 'Training and inference':'Please explain the difference between model training and inference.',
 'Tool calling':'Can you explain how an AI agent chooses a tool to call?',
 'Multi-agent orchestration':'What is a good way to orchestrate several AI agents together?',
 'Prompt structure':'What prompt structure works for clear instructions and few-shot examples?',
 'Structured outputs':'How can I make language model responses follow a JSON schema?',
 'Retrieval quality':'What can improve the accuracy of retrieved documents in a RAG system?',
 'Document chunking':'How should I choose document chunk sizes for a retrieval pipeline?',
 'Prompt injection defenses':'How do I defend an AI application from malicious prompt injection?',
 'Privacy and data protection':'What safeguards protect sensitive information in an AI workflow?',
 'Python environment setup':'What steps create a Python virtual environment with the right packages?',
 'API credential setup':'How should I store API credentials in a local development environment?',
 'Production monitoring':'What should I monitor after deploying my AI application?',
 'Application deployment':'What deployment pattern should I choose for my AI application?',
 'Recording access':'Where is the recording from our fictional learning session?',
 'Session links':'How do I find the link to join the next class meeting?',
}
EXPLANATIONS={
 'AI Foundations':'Training adjusts model parameters using examples; inference applies them to new inputs.',
 'Agentic AI':'An agent can combine planning, tool calls and feedback to complete a bounded task.',
 'Prompt Engineering':'Clear instructions, examples and an output schema help make prompts easier to evaluate.',
 'Models & Architecture':'Retrieval supplies relevant context; transformer attention weights relationships between tokens.',
 'AI Safety & Security':'Input validation, access controls and privacy review are useful layers in AI security.',
 'Development Tools':'A virtual environment isolates package dependencies for one Python project.',
 'Project Implementation':'A deployment checklist can include tests, monitoring and rollback planning.',
 'Course Support':'The fictional course portal has separate areas for recordings and meeting links.',
}

def generate_mock(seed=SEED):
    rng=random.Random(seed)
    locations=pd.DataFrame([dict(location_id=f'TGA-L{i+1:03}',country=c,iso3=iso,region=r,city=city,latitude=lat,longitude=lon,data_status=LABEL)
                           for i,(c,iso,r,city,lat,lon) in enumerate(LOCATIONS)])
    people=[]
    for i in range(60):
        place=i if i<len(locations) else rng.choices(range(len(locations)),weights=[15,10,8,7,5,5,3,3,4,2,2,2,2,2])[0]
        people.append(dict(participant_id=f'TGA-P{i+1:03}',participant_label=f'Fictional participant {i+1:03}',location_id=locations.iloc[place].location_id,data_status=LABEL))
    entries=[]
    for topic in TOPIC_DESCRIPTIONS:
        themes=list(THEME_DESCRIPTIONS[topic])[:2]
        for i in range(20):
            theme=themes[i%2]
            text=THEME_DESCRIPTIONS[topic][theme] if i//2%2==0 else ALTERNATIVES[theme]
            intent='clarification' if text.startswith(('Please explain','Can you explain')) else 'question'
            entries.append((topic,theme,text,intent,True))
        for i in range(5):
            entries.append((topic,'',EXPLANATIONS[topic]+['',' A small example makes this concrete.',' We can test this with a toy project.',' This is a starting point.',' The next exercise explores this idea.'][i],'explanation',False))
        for i in range(2):
            entries.append((topic,'',f'I plan to explore {topic.lower()} in a small fictional practice project.'+(' I will compare notes afterward.' if i else ''),'discussion',False))
    for i in range(8):
        entries.append(('Development Tools','Python environment setup',['My Python package is not working in the virtual environment; can someone help?','I cannot install a Python package in my virtual environment; please help.'][i%2],'support_request',True))
        entries.append(('Course Support','Recording access',['I cannot access the fictional course recording; can someone help?','The course recording link is not working; please help me access it.'][i%2],'support_request',True))
    for i in range(18):
        entries.append((list(TOPIC_DESCRIPTIONS)[i%8],'',['Thanks!','Great!','Awesome!','Got it.','Nice!'][i%5],'reaction',False))
    rng.shuffle(entries)
    messages=[];truth=[];elapsed=0
    for i,(topic,theme,text,intent,question) in enumerate(entries):
        elapsed+=rng.randint(8,24)
        mid=f'TGA-SYN-M{i+1:04}'
        messages.append(dict(message_id=mid,participant_id=f'TGA-P{i%60+1:03}',session_id='TGA-SYN-WEEK1',
            session_relative_seconds=elapsed,timestamp=f'T+{elapsed//3600:02}:{elapsed//60%60:02}:{elapsed%60:02}',message=text,
            message_type='fictional_chat',source_topic=topic,has_identifiable_answer=rng.random()>.25 if question else None,data_status=LABEL))
        truth.append(dict(message_id=mid,question_intent=intent,is_question=question,topic=topic,recurring_question_theme=theme,data_status=LABEL))
    return pd.DataFrame(messages),pd.DataFrame(people),locations,pd.DataFrame(truth)

def validate_mock(messages,people,locations,truth):
    if len(messages)!=250 or len(people)!=60:
        raise ValueError('Expected 250 fictional messages and 60 fictional participants.')
    for frame,key in [(messages,'message_id'),(people,'participant_id'),(locations,'location_id'),(truth,'message_id')]:
        if frame[key].isna().any() or frame[key].duplicated().any() or not frame.data_status.eq(LABEL).all():
            raise ValueError('Invalid synthetic IDs or provenance labels.')
    if set(messages.message_id)!=set(truth.message_id) or set(messages.participant_id)!=set(people.participant_id):
        raise ValueError('Inconsistent mock message/participant references.')
    if not set(people.location_id)<=set(locations.location_id):
        raise ValueError('Invalid location references.')
    if not people.participant_id.str.fullmatch(r'TGA-P\d{3}').all() or not messages.message_id.str.fullmatch(r'TGA-SYN-M\d{4}').all():
        raise ValueError('Use fictional ID namespaces only.')
    if not people.participant_label.str.fullmatch(r'Fictional participant \d{3}').all():
        raise ValueError('Human names are forbidden in participant labels.')
    reference=pd.DataFrame(LOCATIONS,columns=['country','iso3','region','city','latitude','longitude'])
    if len(locations.merge(reference,on=list(reference.columns)))!=len(locations) or not locations.latitude.between(-90,90).all() or not locations.longitude.between(-180,180).all():
        raise ValueError('Geography must match the established location catalog.')
    if not messages.session_relative_seconds.is_monotonic_increasing or messages.session_relative_seconds.duplicated().any():
        raise ValueError('Relative times must increase uniquely.')
    expected=messages.session_relative_seconds.map(lambda s:f'T+{s//3600:02}:{s//60%60:02}:{s%60:02}')
    if not messages.timestamp.eq(expected).all():
        raise ValueError('Invalid fictional session-relative timestamps.')
    if not truth.question_intent.isin(['question','clarification','support_request','explanation','reaction','discussion']).all() or not truth.topic.isin(TOPIC_DESCRIPTIONS).all():
        raise ValueError('Invalid authored intent/topic labels.')
    if not truth.is_question.eq(truth.question_intent.isin(['question','clarification','support_request'])).all():
        raise ValueError('Inconsistent ground-truth question labels.')
    merged=messages.merge(truth,on='message_id',validate='one_to_one')
    if not merged.source_topic.eq(merged.topic).all():
        raise ValueError('Channel and authored topic labels disagree.')
    for row in truth.itertuples():
        if (row.is_question and row.recurring_question_theme not in THEME_DESCRIPTIONS[row.topic]) or (not row.is_question and row.recurring_question_theme):
            raise ValueError('Invalid authored recurring theme.')
    if messages.message.str.contains(r'@|https?://|\bsk-(?:proj|ant)-',regex=True).any():
        raise ValueError('Emails, URLs and credential-like content are forbidden.')

def write_mock():
    from .storage_security import canonical_path
    destination=canonical_path(DIRECTORY)
    tables=generate_mock();validate_mock(*tables)
    destination.mkdir(parents=True,exist_ok=True)
    for name,frame in zip(FILES,tables):
        frame.to_csv(canonical_path(destination/name),index=False)
    return tables

def load_mock():
    from .storage_security import canonical_path
    tables=[pd.read_csv(canonical_path(DIRECTORY/name),keep_default_na=False) for name in FILES]
    messages,people,locations,truth=tables
    messages['has_identifiable_answer']=messages.has_identifiable_answer.map({True:True,False:False,'True':True,'False':False,'':None})
    validate_mock(*tables)
    people=people.merge(locations.drop(columns='data_status'),on='location_id',validate='many_to_one')
    messages['topic']=messages.source_topic
    messages.attrs.update(synthetic=True,data_mode='synthetic')
    return people,messages,truth
