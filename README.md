# TGA Cohort 360

The active dashboard uses **SYNTHETIC DEMO DATA**: 250 entirely fictional messages from 60 fictional participants, across eight learning topics and 14 city assignments. No original transcript is read, represented, or used as a generation source. Existing authored topic/theme vocabulary and fictional templates supply the content; country/city names are public geography, not participant information.

Regenerate the four checked-in mock CSVs reproducibly with seed `1360`:

```powershell
..\venv\Scripts\python.exe scripts/generate_mock_dataset.py
```

`data/synthetic/mock_messages.csv` contains fictional message IDs, participant references, session-relative times, text, fictional topic channels, and simulated answer flags. `mock_participants.csv` contains TGA-P001–TGA-P060, fictional labels, and location references. `mock_locations.csv` supplies fictional country/city assignments. `mock_ground_truth.csv` supplies authored intent, topic, and recurring-theme labels for evaluation, not AI predictions. All four explicitly declare their synthetic status. The loader opens only these fixed synthetic paths; it has no transcript input option. This provenance boundary avoids inspecting private data even for comparison.

The detector uses deterministic rules. Topic predictions and semantic clusters use the existing offline MiniLM model; ground-truth classifications are not passed into inference. Simulated answer flags do not establish actual answers. Reactions and explanations remain in the mock inventory but are excluded from question totals. Local model files must already be installed. The globe, camera, red markers, charts, and existing filter behavior remain unchanged.

For the separate preapproval real-data CLI milestone, see [Local restricted processing](LOCAL_PROCESSING.md). It is tested on generated data only; the dashboard still uses synthetic fixtures.

The preflight security findings are addressed by the [Week 1 fixed snapshot policy and storage safeguards](SECURITY_RELEASE_POLICY.md). Real processing and public release remain separate approval steps; neither has been executed.

Streamlit layout prototype with three tabs, one unified dashboard, geographic and topic filters, and deterministic fictional data. No private chat data, credentials, or external AI services are used.

From this directory:

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements.txt
..\venv\Scripts\python.exe -m streamlit run app.py
```

Open the Local URL printed by Streamlit. The globe uses locally bundled assets; no participant data is sent to an AI service.

`data.py` contains fixtures, aggregation, and filter logic. `app.py` renders the UI. All figures and suggestions are synthetic. Geographic filters apply across tabs; topic filters apply to questions and related actions. Blank selections include all records. Meetup hubs require 10 fictional participants. Repeated themes have at least two questions. Answer availability is a randomly generated demo flag.

The provided three-tab specification is the layout reference; no separate wireframe file was supplied.

The Total Participants card includes a native people icon. Global Community uses a nighttime Globe.GL/Three.js Earth inside the supported Streamlit `st.iframe` API. Counts come from filtered participant rows, never from the coordinate file. Red markers and white 10px count labels use differing positions to reduce collisions in Europe. Markers behind the Earth are occluded; rotate to see them. The original Plotly globe helper remains in `charts.py` for reference. Top-country/city charts still use Plotly.

`country_coordinates.csv` is the maintained local lookup for the 12 fixture countries. Country center coordinates were checked against [Google's DSPL countries.csv](https://developers.google.com/public-data/docs/canonical/countries_csv), last updated June 26, 2024, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). ISO alpha-2 codes were mapped to the dataset's ISO alpha-3 keys; label positions are local display metadata. These are representative country centers, not calculated polygon centroids. Add and verify lookup entries when adding countries. Runtime validation rejects duplicate keys, invalid coordinate ranges, and missing coverage. The app does not fetch this lookup or call external APIs.

Run the existing unittest-compatible tests with pytest:

```powershell
..\venv\Scripts\python.exe -m pip install pytest
..\venv\Scripts\python.exe -m pytest -q
```

Tests cover KPI/geographic count reconciliation, choropleth and marker values, coordinates, country/region filters, unified UI, topic filters, reset, empty selections, and the other tabs.

## Night globe prototype

`night_globe.py` embeds locally stored JavaScript and texture in a self-contained iframe. Only country names, counts, coordinates, and label positions enter the HTML. JSON HTML delimiters are escaped; tooltips escape HTML and count labels use DOM text nodes. No participant identifiers or question records are transmitted. See [asset attribution](assets/globe/ATTRIBUTION.md) for NASA texture usage and JavaScript licenses.

The initial camera faces latitude 22°, longitude 40°, showing Africa, Europe, and India. Camera distance fits the sphere to 83% of the smaller container dimension: approximately 564px diameter in the 680px desktop container. Mouse drag rotates the actual WebGL sphere; mouse wheel zooms. Zoom limits reserve room for the atmosphere to prevent clipping, so marker focus is deliberately modest. Click a count label or point to animate toward its validated coordinates and show a country/count tooltip. Background clicks use Globe.GL's Three.js sphere hit-testing to restore the view; pointer movement and wheel guards prevent accidental resets. The local Reset Globe button also restores orientation and zoom without a Streamlit rerun. Auto-rotation, startup animation, point transitions, and control damping are disabled. Device pixel ratio is capped at 1.5. Rendering pauses when the browser document is hidden; ordinary WebGL frame rendering remains active while visible.

Tradeoffs: WebGL is required (otherwise an explanatory message is shown). The iframe is 680px tall and fits its available width; the sphere shrinks on narrow screens. Filters rebuild the iframe and reset the camera. Resizing resets the camera to ensure the globe fits. Labels can still overlap at small widths or extreme zoom. Assets are embedded in the HTML (about 3MB), so reruns resend them; a packaged component with separate cached assets and persistent camera state is a future optimization. The globe's navy background is local to the visualization; the dashboard theme is untouched. No reference image was attached, so appearance follows the written brief.

The staging page `globe_preview.py` was browser-verified before switching `app.py`. Browser checks are separate from pytest:

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
..\venv\Scripts\python.exe -m playwright install chromium
..\venv\Scripts\python.exe verify_globe_browser.py http://127.0.0.1:8511 --dashboard
```

The URL must point to a running dashboard with filters reset. Browser checks verify texture loading, dark background, real mouse rotation, wheel zoom, India and Europe marker/KPI reconciliation, hover label content, and other tabs. `assets/globe/verified-preview.png` records the rendered browser result.

## Local Learning Intelligence

Install the runtime requirements, then perform the explicit one-time model download:

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements.txt
..\venv\Scripts\python.exe scripts\download_model.py
..\venv\Scripts\python.exe -m streamlit run app.py
```

The model setup script downloads public all-MiniLM-L6-v2 files at a pinned commit. No chat content is sent. See [model provenance and license](models/README.md). Inference runs on CPU using only local safe-tensor weights, with Hugging Face offline mode, telemetry disabled, local-files-only loading, and remote model code disabled. If the files are absent, the Learning tab explains that AI results are unavailable; it does not invent results or silently download weights.

Pipeline components:

| File | Behavior | Uses model? |
| --- | --- | --- |
| `src/synthetic_chat.py` | Creates 792 fictional messages from existing fixtures plus explicitly fictional questions/reactions | No |
| `src/question_detector.py` | English rules distinguish questions, clarification, support, reactions, statements and ambiguous messages; explains each decision | No |
| `src/embedding_model.py` | Real 384-dimensional, normalized MiniLM embeddings from local files | Yes |
| `src/topic_classifier.py` | Nearest topic descriptions; selects curated theme names using similarity | Yes, embeddings; no generative summaries |
| `src/question_clusterer.py` | Complete-link grouping within predicted topics using scikit-learn cosine similarity and SciPy hierarchy | Uses model embeddings; grouping is deterministic |
| `src/learning_analytics.py` | pandas topic/category counts, recurring themes, distinct pseudonymous participants, session suggestions and suppressed exports | No |
| `src/pipeline.py` | Synthetic-only validation and orchestration; filters derived questions without changing cluster assignments | No additional model |
| `learning_ui.py` | Cached model/resources and results; existing Learning layout plus aggregate theme/session tables and download | Displays actual pipeline results |

The final tabs are 🌎 Global Community, 📊 Learning Intelligence, and 🧠 AI Insights & Recommendations. Global Community is unchanged. Learning Intelligence retains its four KPIs and three charts. AI Insights contains the relocated AI-Discovered Question Themes and Suggested Follow-up Topics sections, aggregate export, review counts and explanatory captions. The old Community & Action Insights UI is removed; reusable datasets and analytics functions are retained.

Sidebar Topics selects original fictional chat channels; the AI topic-classification filter is also in the sidebar. Both learning tabs reuse one filtered result, including geographic selection by pseudonymous participant IDs. Topic filters never reduce the geographic dataset. Reset filters clears both topic filters. Theme tables default to descending question counts and support native column sorting. Suggestions use existing eligibility rules (recurrence, multiple participants, no review flag, non-course-support topic); they are proposed opportunities, not confirmed gaps or evidence of completed human review.

How decisions work:

- A message containing an explicit question, clarification, or support request counts once. Multiple questions inside one message are not split. Ambiguous nonquestions are excluded and counted in the detection audit for review.
- The model embeds each question and compares it to curated descriptions of the eight concrete topics. A maximum cosine similarity below 0.30 maps to `Other / Unclear`. Scores below 0.38 or a best-vs-second margin below 0.035 are flagged. These scores are not confidence probabilities.
- Complete-link clusters require every pair to have cosine similarity at least 0.70. This avoids joining unrelated endpoints through chains. Minimum cluster similarity below 0.80 and borderline singleton neighbors trigger review. Theme names use a fixed vocabulary and nearest embedding description; raw message excerpts are never used as displayed names. Two separate clusters can have the same name and are distinguished by cluster ID.
- Clusters are computed on the full synthetic cohort, then filtered counts are recomputed. IDs are stable under message-row reordering but may change when the input corpus or settings change.
- Follow-up suggestions require at least two questions, at least two distinct participants, no review flag and a non-course-support topic. This is a deterministic shortlist, not an LLM recommendation or proof of learner demand.
- The unanswered-question KPI uses simulated boolean flags from the original fixtures. Answer matching is outside this milestone. Missing flags show `Not assessed`; no answer status is invented.

Limitations: English rule detection and a general-purpose embedding model can misclassify negation, mixed topics, sarcasm and domain-specific language. Thresholds have only been exercised on synthetic examples, not validated on actual course messages. Long inputs can be truncated by MiniLM; messages over 120 whitespace tokens are conservatively flagged. Complete-link clustering uses quadratic memory, so this prototype accepts at most 5,000 messages per batch. This Windows environment blocks scikit-learn's `_libsvm` DLL loaded transitively by `sklearn.cluster`; SciPy performs the equivalent complete-link hierarchy while scikit-learn still computes cosine similarities.

Privacy: this milestone has no private-data loaders or upload controls. `analyze_messages` requires an explicit synthetic provenance flag; the flag is a guardrail, not proof that arbitrary text is anonymous. Model inference and UI use only generated fixtures. Message-level results stay in memory. Public CSV export is constructed from an allowlist of controlled names and aggregate fields, suppressing groups with fewer than three distinct pseudonymous participants. It exports no message text, IDs, representative snippets, or detection reasons. Suppression is not a formal privacy guarantee and must be reviewed before any future real-data export.

Run and review actual synthetic results:

```powershell
..\venv\Scripts\python.exe scripts\run_intelligence.py
..\venv\Scripts\python.exe -m pytest -q
..\venv\Scripts\python.exe verify_intelligence_browser.py http://127.0.0.1:8513
```

The CLI writes only aggregate artifacts: `reports/synthetic_intelligence.json` and `reports/synthetic_learning_themes.csv`. The tests run real local-model topic/similarity checks while outbound sockets are blocked, plus detection, conservative clustering, count reconciliation, empty inputs, missing model handling, exports, filters, and unchanged dashboard features. The browser check verifies the user-facing CSV download and all three tabs. Install `requirements-dev.txt` and Playwright Chromium for browser tests.
