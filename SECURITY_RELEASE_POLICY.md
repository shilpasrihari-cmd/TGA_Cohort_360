# Week 1 security remediation and release policy

No real participant CSV was read or processed during this remediation. No `--approve-real-processing` command was executed. Tests use invented rows and temporary synthetic storage. The current Streamlit app and globe are unchanged and remain synthetic-only.

## One fixed public snapshot

`policies/week1-release.json` defines `week1-v1`. `src/public_release.py` is the only supported route from restricted results to public snapshot data. Processing writes private/staging drafts and an unapproved review request, never public outputs.

Public generation requires explicit approval of the exact dataset fingerprint and policy hash, a completed privacy review, verification of the participant identity scope, and a private review reference. A flag alone does not authenticate a reviewer or prove that review occurred. Protect the approval file with the reviewer/operator's filesystem access controls.

The publisher consumes an exclusive release slot under `data/private/<mode>/release_control/` before writing `data/public/<mode>/snapshot.json`. The slot is permanent for Week 1, across runs and policy versions. A crash after reservation fails closed; the code never clears it or automatically retries a replacement. Repeated downloads of the same sealed snapshot are permitted; repeated generation, a five-person replacement of a six-person snapshot, and changed-version overlapping releases are rejected.

Approved filters are named, fixed views from the sealed policy: global/all, individual learning topics, and the Agentic AI + Models & Architecture union. No dates, participants, countries, regions, exclusions or arbitrary topic combinations are accepted. This inventory has no geography fields, so **global is the only approved geographic aggregation**. Adding a geographic breakdown requires a separately reviewed policy and data source; none is fabricated. Exports select already released rows and never read restricted message CSVs or recompute cohort counts.

Same-vocabulary clusters are merged before public release. Themes require at least five distinct pseudonyms and no unresolved model review flag. Parent topic tables include only the sum of released question counts; parent distinct-person counts, grand totals and suppressed counts are omitted. Thus public filters cannot request the six-person population and its five-person subset, and parent totals cannot reveal omitted small theme contributions. Public payloads contain controlled topic/theme names, counts, fixed mode/version and the fixed view policy only. They contain no source IDs, pseudonyms, question references, cluster IDs, names, raw text, dataset fingerprints or reviewer identifiers. Private approval metadata is not embedded publicly.

The legacy Streamlit demo exporter now requires explicit synthetic provenance and rejects real or unmarked data. It cannot be used as an alternate real-data release route. The current dashboard has **no real-snapshot selector**. A future approved selection must consume this fixed artifact with its integrity check; it must not reconnect live filters to restricted rows.

## Identity and analytics separation

HMAC identifiers are stable under a protected scope key but are **not irreversible anonymity**. Authorized vault holders can reconstruct identities from mappings. Display-name aliases can inflate apparent participants and shared names can merge them. The public release gate therefore requires explicit identity-scope verification; the code cannot establish five actual people from five display names alone.

New private output structure:

```text
data/private/<mode>/identity_vault/pseudonym.key
data/private/<mode>/runs/<run-id>/identity_vault/
    validated_inventory.csv       # names, raw text and preliminary metadata
    identity_mapping.csv          # pseudonym -> display name
    source_trace.csv              # random question reference -> source ID
    review_queue.csv              # restricted raw review evidence
    cluster_membership.csv        # restricted source-ID validation links
data/private/<mode>/runs/<run-id>/analysis/
    classified_questions.csv      # random question references, pseudonyms, model results
    detected_messages.csv         # references, pseudonyms, detection flags
    topic_counts.csv
    recurring_themes.csv
```

Analytics files omit names, raw messages, source message IDs and timestamps. Per-run random references preserve validation only through the separately restricted source trace. Pseudonyms remain restricted internal analytics fields; neither kind of identifier enters the public snapshot. Vault evidence is deliberately recoverable for validation. Give an analyst only the analysis directory, not the identity vault, key or source CSV. The default local operator can access both; distinct roles need distinct OS accounts and an approved ACL configuration.

Existing outputs from the earlier synthetic prototype are not migrated, deleted or read by this remediation. Legacy private exports must not be treated as public artifacts. Before any future real run, confirm retention/access rules and use the new layout.

## Physical storage and permissions

`src/storage_security.py` walks every existing path component using non-following metadata. It rejects symbolic links, Windows junctions and every reparse-point type, unexpected canonical redirects, cross-mode physical aliases, hard-linked inputs, escaped roots and input/output overlaps. These checks occur before model loading, CSV reading and output creation. Failure to inspect a component stops the operation. Synthetic fixture creation uses the same checks. Directory creation is revalidated; files use exclusive creation rather than overwrite.

New restricted directories/files use owner-only POSIX permissions (0700/0600). On Windows, the code applies protected DACLs containing only the current operator SID and LocalSystem, removing inherited grants. Enforcement failures stop processing before sensitive writes. The native junction and protected-DACL tests use temporary fictional files. The implementation uses Microsoft's [SetNamedSecurityInfoW](https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-setnamedsecurityinfow) API and an explicitly constructed DACL.

Windows operational requirements still need confirmation:

- Use a local filesystem supporting persistent ACLs; confirm permissions on the existing workspace, approved input CSV, backups and parent directories. The tests do not inspect any actual real-data file or its ACL.
- Protect the identity vault/key and release registry under appropriate operator/reviewer accounts. Dashboard users must not receive access to either.
- Do not relocate the release registry, clone the project to create another release authority, delete a consumed release slot, or publish independent exports of the same population. Filesystem administrators can bypass application-level restrictions.
- Back up the registry and keys together with the agreed privacy scope. Encryption at rest, machine security, retention/deletion, and trustworthy reviewer identity are operational controls, not guarantees made by this code.

No software path check eliminates races against a privileged actor who can replace trusted parent directories concurrently. The filesystem and process account must be trusted. No raw source values are included in CLI error messages; controlled validation errors are now reported generically too. Captured success and failure log tests include synthetic sentinel names, IDs and text.

## Timezones

Offset-aware ISO timestamps are converted to UTC. Naive timestamps require an independently confirmed IANA region (`--assume-timezone America/Toronto` only if that really is the source zone), or explicitly confirmed UTC. Abbreviations such as EST, numeric offset strings used as zone names, and invalid zones are rejected. Both spring-forward nonexistent times and fall-back ambiguous times fail validation. No host timezone is inferred.

## Approval workflow (not executed)

After separately approved local processing, an operator can prepare a private release request using `scripts/release_snapshot.py --action prepare`, with the analysis question CSV and a new approval path under the matching private mode namespace. The reviewer verifies identity scope, the exact population, the complete fixed table set, residual risks and policy. Only then may the three approval booleans be set to true with a review reference.

Publishing is a separate `--action publish --approve-public-release` operation using that same private approval file. The CLI rejects publishing without the approval flag before reading restricted input. This remediation did not execute a real processing or real release operation. Public dashboard selection would require a separate explicit change/approval.

## Residual risks

The fixed snapshot closes the implemented overlapping-query/release paths; a five-person threshold alone does not guarantee anonymity. External knowledge, distinctive semantic topics, membership known to observers, aliases, a dishonest reviewer, administrator access, compromised keys/registries, independent releases elsewhere or edits to the application can permit re-identification. Week 1 intentionally provides a single privacy-reviewed snapshot rather than an unrestricted privacy engine or formal differential privacy. Future repeated releases and geographic slicing require a new reviewed mechanism, not deletion of the current registry.

Run the complete suite and the post-remediation audit:

```powershell
..\venv\Scripts\python.exe -m pytest -q
..\venv\Scripts\python.exe -m pytest -q test_preflight_audit.py test_security_remediation.py
```

All processing in these tests uses generated synthetic fixtures. They also confirm the unchanged dashboard still loads synthetic data and that unapproved real processing is rejected before input access.
