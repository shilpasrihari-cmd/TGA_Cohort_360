"""Fictional fixture validation; never opens a restricted source."""
import pandas as pd
import pytest
from data import geographic_counts, community_metrics, filter_data
from night_globe import marker_records
from src.mock_dataset import generate_mock, validate_mock, load_mock, LABEL, DIRECTORY, FILES
from src.question_detector import detect_messages

def test_reproducible_fictional_dataset():
    first = generate_mock()
    validate_mock(*first)
    for left, right in zip(first, generate_mock()):
        pd.testing.assert_frame_equal(left, right)
    messages, people, locations, truth = first
    assert len(messages) == 250 and len(people) == 60
    assert truth.is_question.sum() == 176
    assert truth.topic.nunique() == 8
    assert truth.question_intent.nunique() == 6
    assert truth.loc[truth.is_question, 'recurring_question_theme'].nunique() == 16
    assert messages.message.nunique() < len(messages)
    assert all(frame.data_status.eq(LABEL).all() for frame in first)
    assert not {'display_name','email','username'} & set(messages.columns)
    assert messages.timestamp.str.startswith('T+').all()

def test_authored_intents_match_detector():
    messages, _, _, truth = generate_mock()
    detected = detect_messages(messages.rename(columns={'message':'text'}))
    assert detected.is_question.tolist() == truth.is_question.tolist()

def test_chat_adapter_never_uses_legacy_theme_generation(monkeypatch):
    import src.synthetic_chat as adapter
    people, messages, _ = load_mock()
    def forbidden(*args):
        raise AssertionError('Chat rows must not enter legacy generation')
    monkeypatch.setattr(adapter, 'synthetic_messages', forbidden)
    prepared = adapter.prepare_synthetic_messages(people, messages)
    assert prepared.text.tolist() == messages.message.tolist()
    assert len(prepared) == 250 and 'topic' not in prepared
    assert prepared.attrs['synthetic'] is True
    pd.testing.assert_frame_equal(adapter.prepare_synthetic_messages(people,prepared), prepared)
    unsafe = messages.copy(); unsafe.attrs.clear()
    with pytest.raises(ValueError): adapter.prepare_synthetic_messages(people,unsafe)

@pytest.mark.parametrize('regions,countries', [([], []), (['Europe'], []), ([], ['India']), (['North America'], ['Canada']), (['Europe'], ['India'])])
def test_mock_geography_reconciles(regions,countries):
    people, messages, _ = load_mock()
    selected, _ = filter_data(people,messages,regions,countries,[])
    counts = geographic_counts(selected)
    assert int(counts.Participants.sum()) == community_metrics(selected)[0] == len(selected)
    assert sum(marker['count'] for marker in marker_records(counts)) == len(selected)

def test_loader_only_opens_four_synthetic_files(monkeypatch):
    original = pd.read_csv
    opened = []
    def recording(path, **kwargs):
        opened.append(path)
        return original(path, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', recording)
    _, messages, _ = load_mock()
    assert opened == [(DIRECTORY / name).resolve() for name in FILES]
    assert messages.attrs['synthetic'] is True
    assert 'question_intent' not in messages and 'recurring_question_theme' not in messages

@pytest.mark.parametrize('change', ['duplicate','participant','coordinates','label'])
def test_invalid_mock_rejected(change):
    messages,people,locations,truth = generate_mock()
    if change == 'duplicate': messages.loc[1,'message_id'] = messages.loc[0,'message_id']
    if change == 'participant': messages.loc[0,'participant_id'] = 'unknown'
    if change == 'coordinates': locations.loc[0,'latitude'] = 91
    if change == 'label': truth.loc[0,'is_question'] = not truth.loc[0,'is_question']
    with pytest.raises(ValueError): validate_mock(messages,people,locations,truth)
