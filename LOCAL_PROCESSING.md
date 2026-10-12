# Local restricted inventory processing

Security update: [Week 1 fixed release policy](SECURITY_RELEASE_POLICY.md) supersedes the original draft-output and storage descriptions below. New identity evidence is separated into `identity_vault/`, analytics into `analysis/`, and public generation requires a separate approved immutable snapshot. Staging drafts are restricted and must never be served directly.

Status: implemented and tested on generated synthetic inventory only. No real TGA CSV has been located, read, modified, or processed. The dashboard and globe are unchanged and remain connected only to synthetic fixtures. This CLI does not publish or connect its candidate outputs to the dashboard.

## Commands

From `tga-cohort-360`, run a generated synthetic fixture with a new run ID:

```powershell
..\venv\Scripts\python.exe scripts\process_inventory.py --mode synthetic --run-id synthetic-preapproval-check-03
```

Synthetic mode generates a fictional CSV in `data/staging/synthetic/fixtures/`. It rejects `--input`; it cannot be pointed at an arbitrary private transcript. Existing run IDs are never overwritten.

After the user explicitly approves the following command and input, the real-mode command is:

```powershell
..\venv\Scripts\python.exe scripts\process_inventory.py --mode real --input ".\data\private\real\authorized_inventory.csv" --run-id tga-local-001 --minimum-participants 5 --approve-real-processing
```

`authorized_inventory.csv` is a documented destination, not a file that was searched for or read. Substitute the approved path if needed, inside `data/private/real/` or `data/staging/real/`. The approval flag records deliberate command execution; it does not technically authenticate a human's consent. Do not run real mode before approval. This milestone ends before that step.

Timestamps must be timezone-aware ISO-8601 datetimes. If the authorized source has naive timestamps, add `--assume-timezone` with the independently confirmed source timezone, for example `--assume-timezone America/Toronto`. The CLI never guesses from the computer timezone and rejects ambiguous or nonexistent daylight-saving times. Do not assume Toronto merely because it appears in this example.

Duplicate message IDs stop processing. If authorized removal of exact repeated records is appropriate, add `--deduplicate-exact`; conflicting records sharing an ID still fail. Repeated question text under different message IDs is retained as genuine repeated messages. `--model-dir` can select a locally stored embedding model directory; inference never downloads one. The existing local all-MiniLM-L6-v2 directory is the default.

## Schema and validation

Exactly these columns are required: `message_id`, `display_name`, `timestamp`, `message`, `message_type`, `topic`, `is_question`, `speaker_role`. Unknown or duplicate columns are rejected to prevent accidentally forwarding extra sensitive fields.

- Message IDs, display names, timestamps and message text must be nonempty. Missing names are rejected instead of inventing separate identities that could inflate the suppression threshold.
- Message IDs must be unique after any explicitly requested exact-copy removal. Source record numbers preserve the link to copies; they refer to parsed CSV records, not physical lines.
- Timestamps are validated and converted to UTC. Invalid dates and ambiguous locale formats are rejected.
- Preliminary `is_question` accepts boolean-like values (`true/false`, `yes/no`, `1/0`) or blank/unknown. Blank preliminary topic, role and message-type metadata are flagged for review. These labels are preserved only as restricted evidence, never used as validated ground truth or classifier inputs.
- English question detection uses the existing rules. Local embeddings provide advisory review signals for potentially missed requests, not automatic promotion to confirmed questions. The existing embedding topic classifier and semantic clustering produce independent topic and theme suggestions.

## Restricted storage and traceability

Outputs are separated by mode:

```text
data/private/<synthetic-or-real>/identity_vault/pseudonym.key
data/private/<synthetic-or-real>/runs/<run-id>/
    identity_vault/
        validated_inventory.csv
        identity_mapping.csv
        source_trace.csv
        review_queue.csv
        cluster_membership.csv
    analysis/
        detected_messages.csv
        classified_questions.csv
        topic_counts.csv
        recurring_themes.csv
    run_manifest.json
data/staging/<synthetic-or-real>/aggregate_candidates/<run-id>/
    recurring_themes.csv
    topic_counts.csv
```

The restricted inventory retains names, text, preliminary labels and source message IDs. Classified analytics retain restricted pseudonyms and random question references, without source IDs, names or raw text. Source traces stay in the identity vault. Cluster membership supplies a direct source-ID-to-cluster trace. The review queue includes raw validation evidence, scores, ambiguity and preliminary-label disagreement reasons, and `Pending human review` status. Scores are similarities, not probabilities, and absence of an uncertainty flag is not completed human review.

Pseudonyms use HMAC-SHA256 over Unicode-normalized, whitespace-normalized, case-folded display names with a random 32-byte key stored privately. Keys are separate for synthetic and real modes and reused for stable internal deduplication. Never publish the key or pseudonyms. Display names are not reliable account identifiers: shared names can merge people, while aliases can split them. Human identity validation is required before treating these counts as verified people.

Raw source files are not modified. Directory boundaries use resolved paths and reject escape paths, redirected storage roots and cross-mode synthetic inputs. Private and staging directories are excluded from Git. They are not served by this dashboard or passed to a Streamlit data cache.

These directories are a logical access boundary, not encryption. POSIX private run/key permissions are restricted; New Windows outputs receive protected owner-and-LocalSystem DACLs; existing workspace/input ACLs still require operational confirmation. Store the project on an appropriately access-controlled local volume before placing actual restricted data in it. CSV review files contain raw source values; use a text editor or import them as literal text in a spreadsheet to avoid interpreting formula-like chat text.

## Aggregate candidates

Only controlled topic/theme vocabulary, counts, an uncertainty flag and fixed provenance/status labels enter the candidate CSVs. No names, raw text, source IDs, timestamps, pseudonyms, cluster IDs or message-level records are included. Output labels cannot be supplied by preliminary topics or arbitrary message text.

Themes need at least two messages and five distinct pseudonyms. The threshold cannot be reduced below five. Topic counts are calculated only from released themes; full parent totals, suppressed counts and grand totals are omitted so the missing contribution cannot be recovered by simple subtraction. Full count reconciliation remains in the restricted manifest.

Candidates and their unapproved review request remain restricted under staging with `Draft - human review required` status. Public generation uses only the separate, explicitly approved fixed snapshot policy. They are not dashboard inputs and are not automatically published. Suppression and name-based pseudonyms do not provide formal anonymization or differential privacy; repeated releases, new filters and identity reliability require further review before publishing real-data aggregates.

## Offline model behavior

The existing local model loader uses CPU inference, a filesystem path, `local_files_only=True`, safe-tensor weights, no remote model code, Hugging Face offline mode and disabled telemetry. A missing local model fails before reading the CSV, including for empty datasets. The CLI has no model-download call, external AI client or transcript upload.

The synthetic dashboard entry remains synthetic-only. A distinct authorized entry is used exclusively by the restricted CLI. Shared classification/clustering logic is reused; no display or data-schema changes were made to the dashboard. Preliminary answer status is not present in this inventory, so answer identification is not fabricated. The current clustering batch limit remains 5,000 messages.

## Tests and stopping point

```powershell
..\venv\Scripts\python.exe -m pytest -q
```

Tests use generated synthetic records only. They verify input validation, count reconciliation, pseudonymous deduplication, exact-copy handling, timestamp policy, restricted traceability, review queues, safe export allowlists, suppression and parent-total consistency, mode/path boundaries, empty inventories and the pre-read real-mode approval gate. Local inference is tested with outbound sockets blocked. No test runs approved real mode or reads a real CSV.

Implementation stops here. Await explicit approval for the real processing command above; moving data into the dashboard or publishing outputs is a separate future step.
