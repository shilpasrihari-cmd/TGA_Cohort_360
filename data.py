"""Deterministic fictional fixtures and processing; no chat data or network calls."""
import random
from pathlib import Path
import pandas as pd

def country_coordinates():
    """Local country centers from Google's DSPL countries.csv (see README).

    Maintain this lookup when adding fixture countries. Labels can be positioned
    independently, but the marker remains at the published country center.
    """
    coordinates = pd.read_csv(Path(__file__).with_name('country_coordinates.csv'))
    if coordinates.iso3.duplicated().any() or coordinates.country.duplicated().any():
        raise ValueError('Country coordinate lookup must have unique countries and ISO3 codes.')
    if not coordinates.iso3.str.fullmatch(r'[A-Z]{3}').all():
        raise ValueError('Invalid ISO3 country coordinate key.')
    if not (coordinates.latitude.between(-90, 90).all() and coordinates.longitude.between(-180, 180).all()):
        raise ValueError('Country coordinates must be finite and within geographic bounds.')
    return coordinates

def geographic_counts(people):
    locations = people.groupby(['country', 'iso3']).size().reset_index(name='Participants')
    result = locations.merge(country_coordinates(), on=['country', 'iso3'], how='left', validate='one_to_one')
    if result[['latitude', 'longitude', 'label_position']].isna().any().any():
        raise ValueError('Add missing countries to the validated country-coordinate lookup.')
    return result

LOCATIONS = [
    ('India', 'IND', 'Asia', 'Bengaluru', 12.97, 77.59),
    ('India', 'IND', 'Asia', 'Mumbai', 19.08, 72.88),
    ('Canada', 'CAN', 'North America', 'Toronto', 43.65, -79.38),
    ('United States', 'USA', 'North America', 'New York', 40.71, -74.01),
    ('United States', 'USA', 'North America', 'San Francisco', 37.77, -122.42),
    ('United Kingdom', 'GBR', 'Europe', 'London', 51.51, -0.13),
    ('Germany', 'DEU', 'Europe', 'Berlin', 52.52, 13.41),
    ('Australia', 'AUS', 'Oceania', 'Sydney', -33.87, 151.21),
    ('Singapore', 'SGP', 'Asia', 'Singapore', 1.35, 103.82),
    ('Brazil', 'BRA', 'South America', 'São Paulo', -23.55, -46.63),
    ('Nigeria', 'NGA', 'Africa', 'Lagos', 6.52, 3.38),
    ('South Africa', 'ZAF', 'Africa', 'Cape Town', -33.92, 18.42),
    ('France', 'FRA', 'Europe', 'Paris', 48.86, 2.35),
    ('Japan', 'JPN', 'Asia', 'Tokyo', 35.68, 139.69),
]
TOPICS = {
    'AI agents': ['Tool calling', 'Agent orchestration', 'Agent evaluation'],
    'RAG': ['Retrieval quality', 'Chunking strategies', 'Vector databases'],
    'Prompt engineering': ['Prompt structure', 'Output consistency'],
    'Model evaluation': ['Evaluation datasets', 'Quality metrics'],
    'Deployment': ['Production monitoring', 'Deployment patterns'],
    'Course access': ['Recording access', 'Session links'],
    'Environment setup': ['Package installation', 'API key setup'],
}

def synthetic_data():
    rng = random.Random(360)
    rows = []
    for i in range(480):
        location = rng.choices(LOCATIONS, weights=[20, 14, 12, 10, 8, 9, 5, 6, 7, 4, 4, 3, 3, 3])[0]
        rows.append((f'P{i+1:04}', *location))
    participants = pd.DataFrame(rows, columns=['participant_id', 'country', 'iso3', 'region', 'city', 'latitude', 'longitude'])
    questions = []
    for i in range(760):
        topic = rng.choices(list(TOPICS), weights=[23, 21, 15, 12, 10, 10, 9])[0]
        questions.append({
            'question_id': f'Q{i+1:04}', 'participant_id': rng.choice(participants.participant_id.tolist()),
            'topic': topic, 'theme': rng.choice(TOPICS[topic]),
            'category': 'Support' if topic in ['Course access', 'Environment setup'] else 'Learning',
            'has_identifiable_answer': rng.random() > .22,
        })
    return participants, pd.DataFrame(questions)

def filter_data(participants, questions, regions, countries, topics):
    people = participants.copy()
    if regions:
        people = people[people.region.isin(regions)]
    if countries:
        people = people[people.country.isin(countries)]
    selected = questions[questions.participant_id.isin(people.participant_id)]
    if topics:
        selected = selected[selected.topic.isin(topics)]
    return people, selected.copy()

def counts(frame, column, label='Count'):
    return frame.groupby(column).size().reset_index(name=label).sort_values([label, column], ascending=[False, True])

def community_metrics(people):
    ranked = counts(people, 'country')
    return [len(people), people.country.nunique(), people[['country', 'city']].drop_duplicates().shape[0], ranked.iloc[0].country if len(ranked) else '—']

def learning_metrics(questions):
    themes = counts(questions, 'theme')
    return [len(questions), questions[questions.category == 'Learning'].topic.nunique(), int((themes.Count > 1).sum()), int((~questions.has_identifiable_answer).sum())]

def meetup_hubs(people, minimum=10):
    hubs = people.groupby(['city', 'country']).size().reset_index(name='Participants')
    return hubs[hubs.Participants >= minimum].sort_values('Participants', ascending=False).head(8)
