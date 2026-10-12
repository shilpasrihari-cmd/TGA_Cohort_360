"""Run with python -m unittest test_dashboard.py."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from data import synthetic_data, filter_data, community_metrics, learning_metrics, geographic_counts, country_coordinates
from charts import community_map
from night_globe import marker_records, globe_html
from src.synthetic_chat import synthetic_messages
from src.question_detector import detect_messages
from src.mock_dataset import load_mock

class DashboardTests(unittest.TestCase):
    def test_empty_selection(self):
        people, questions = synthetic_data()
        p, q = filter_data(people, questions, ['Europe'], ['India'], [])
        self.assertEqual(community_metrics(p), [0, 0, 0, '—'])
        self.assertEqual(learning_metrics(q), [0, 0, 0, 0])

    def test_ui_and_filters(self):
        app = AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 3)
        self.assertEqual(app.metric[0].value, '60')
        self.assertEqual(len(app.radio), 0)
        self.assertEqual(app.metric[0].proto.icon, ':material/group:')
        app.multiselect(key='countries').set_value(['India']).run()
        self.assertFalse(app.exception)
        self.assertLess(int(app.metric[0].value), 60)
        app.multiselect(key='topics').set_value(['Course Support']).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[5].value, '0')
        app.multiselect(key='regions').set_value(['Europe']).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.multiselect(key='countries').value, [])
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, '60')
        self.assertEqual(app.multiselect(key='topics').value, [])

    def test_map_counts_and_filters(self):
        people, questions = synthetic_data()
        for regions, countries in [([], []), ([], ['India']), (['Europe'], []), (['North America'], ['Canada']), (['Europe'], ['India'])]:
            with self.subTest(regions=regions, countries=countries):
                selected, _ = filter_data(people, questions, regions, countries, [])
                locations = geographic_counts(selected)
                expected = selected.groupby('iso3').size().to_dict()
                self.assertEqual(dict(zip(locations.iso3, locations.Participants)), expected)
                self.assertEqual(int(locations.Participants.sum()), community_metrics(selected)[0])
                records = marker_records(locations)
                self.assertEqual(sum(row['count'] for row in records), community_metrics(selected)[0])
                self.assertEqual({row['country']:row['count'] for row in records}, dict(zip(locations.country, locations.Participants)))
                fig = community_map(locations)
                self.assertEqual(fig.data[0].type, 'choropleth')
                self.assertEqual(fig.layout.geo.projection.type, 'orthographic')
                self.assertEqual(fig.layout.paper_bgcolor, '#ffffff')
                self.assertEqual(fig.layout.geo.bgcolor, '#ffffff')
                self.assertTrue(fig.layout.geo.showcountries)
                self.assertEqual(fig.layout.coloraxis.colorscale[-1][1], 'rgb(8,48,107)')
                marker = fig.data[1]
                self.assertEqual(marker.type, 'scattergeo')
                self.assertEqual(marker.textfont.size, 10)
                self.assertEqual(marker.marker.size, 5)
                self.assertEqual(list(marker.text), locations.Participants.astype(str).tolist())
                self.assertEqual(list(marker.lat), locations.latitude.tolist())
                self.assertEqual(list(marker.lon), locations.longitude.tolist())
                self.assertEqual(dict(marker.customdata), dict(zip(locations.country, locations.Participants)))

    def test_coordinate_validation(self):
        from unittest.mock import patch
        coordinates = country_coordinates()
        people, _ = synthetic_data()
        self.assertEqual(set(coordinates.iso3), set(people.iso3))
        for invalid in [coordinates.assign(latitude=91), coordinates.assign(longitude=float('nan')), coordinates.assign(iso3='bad')]:
            with patch('data.pd.read_csv', return_value=invalid):
                with self.assertRaises(ValueError):
                    country_coordinates()
        with patch('data.pd.read_csv', return_value=coordinates.iloc[:1]):
            with self.assertRaises(ValueError):
                geographic_counts(people)

    def test_globe_payload_security(self):
        people, _ = synthetic_data()
        locations = geographic_counts(people)
        locations.loc[0, 'country'] = '</script><img src=x onerror=alert(1)>'
        html = globe_html(locations)
        import json
        payload = html.split('<script type="application/json" id="marker-data">')[1].split('</script>')[0]
        self.assertNotIn('<', payload)
        records = json.loads(payload)
        self.assertEqual(records[0]['country'], locations.iloc[0].country)
        self.assertEqual(set(records[0]), {'country', 'count', 'lat', 'lng', 'position'})
        self.assertIn("connect-src 'none'", html)

    def test_geographic_ui_totals_and_other_tabs(self):
        people, questions, _ = load_mock()
        app = AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=30)
        for regions, countries in [(['Europe'], []), (['Europe'], ['Germany']), ([], ['India'])]:
            app.multiselect(key='regions').set_value(regions).run()
            app.multiselect(key='countries').set_value(countries).run()
            self.assertFalse(app.exception)
            selected, selected_questions = filter_data(people, questions, regions, countries, [])
            self.assertEqual(int(app.metric[0].value.replace(',', '')), len(selected))
            detections = detect_messages(questions.rename(columns={"message":"text"}))
            expected_questions = detections[detections.is_question & detections.participant_id.isin(selected.participant_id)]
            self.assertEqual(int(app.metric[4].value.replace(',', '')), len(expected_questions))
            self.assertTrue(any(s.value == 'AI-Discovered Question Themes' for s in app.subheader))
            self.assertEqual(len(app.tabs), 3)
            self.assertTrue(any(s.value == 'Suggested Follow-up Topics' for s in app.tabs[2].subheader))
            self.assertFalse(any(s.value == 'Common support issues' for s in app.subheader))

    def test_ai_topic_filter_and_empty_state(self):
        app = AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=30)
        app.multiselect(key='ai_topics').set_value(['Course Support']).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[4].value, '28')
        self.assertEqual(app.metric[5].value, '0')
        app.multiselect(key='ai_topics').set_value(['AI Safety & Security']).run()
        app.multiselect(key='topics').set_value(['Course Support']).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[4].value, '0')
        self.assertTrue(any('No AI-discovered themes' in alert.value for alert in app.info))
        app.button[0].click().run()
        self.assertEqual(app.multiselect(key='ai_topics').value, [])
        self.assertEqual(app.metric[4].value, '176')

    def test_model_unavailable_shows_no_fake_ai_results(self):
        from unittest.mock import patch
        import learning_ui
        from src.embedding_model import ModelUnavailableError
        with patch.object(learning_ui, 'local_resources', side_effect=ModelUnavailableError('Local model missing')):
            app = AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any('Local model missing' in alert.value for alert in app.warning))
        self.assertFalse(any(s.value == 'AI-Discovered Question Themes' for s in app.subheader))
        self.assertEqual(app.metric[0].value, '60')
        self.assertTrue(any('AI insights are unavailable' in alert.value for alert in app.tabs[2].info))

    def test_sections_relocated_without_duplicates(self):
        app = AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual([tab.label for tab in app.tabs], ['🌎 Global Community', '📊 Learning Intelligence', '🧠 AI Insights & Recommendations'])
        for title in ['AI-Discovered Question Themes', 'Suggested Follow-up Topics']:
            self.assertEqual(sum(s.value == title for s in app.subheader), 1)
            self.assertFalse(any(s.value == title for s in app.tabs[1].subheader))
            self.assertTrue(any(s.value == title for s in app.tabs[2].subheader))
        self.assertEqual(len(app.tabs[1].get('plotly_chart')), 3)
        table = app.tabs[2].dataframe[0].value
        self.assertEqual(list(table.columns), ['Cluster ID','Theme','Learning topic','Questions','Distinct pseudonymous participants','Needs review'])
        self.assertTrue(table.Questions.is_monotonic_decreasing)
        self.assertEqual(int(table.Questions.sum()), 176)
        flagged = int(table.loc[table['Needs review'], 'Questions'].sum())
        self.assertTrue(any(f'{flagged} questions' in caption.value for caption in app.tabs[2].caption))
        app.multiselect(key='ai_topics').set_value(['Course Support']).run()
        self.assertFalse(app.exception)
        table = app.tabs[2].dataframe[0].value
        self.assertEqual(set(table['Learning topic']), {'Course Support'})
        self.assertEqual(int(table.Questions.sum()), 28)
        self.assertEqual(app.metric[0].value, '60')

if __name__ == '__main__':
    unittest.main()
