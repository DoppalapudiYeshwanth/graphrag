# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use GitHub's private reporting instead:
**Security tab, then "Report a vulnerability"**. (Repository owners: enable it under
Settings, then Code security, then Private vulnerability reporting.)

Include what you found, how to reproduce it, and what an attacker could do. You will get an acknowledgement and a
fix or a clear answer as soon as the maintainer can manage; this is a volunteer-run project with no guaranteed
response time.

## Supported versions

Only the latest commit on `main` is supported.

## How the application defends itself

| Risk | Defence | Where |
| --- | --- | --- |
| One visitor reading another's documents | Random 128-bit workspace id; entity keys namespaced per workspace; every query filtered by scope; isolation tests that fail if the filter is removed | `graph.py`, `retrieval.py`, `tests/test_workspaces_integration.py` |
| Cypher injection | Values are always parameters; identifiers that cannot be parameters are validated against a strict pattern first | `graph.py` |
| Malicious uploads | Extension allow-list, size and length limits, PDF page cap, parse errors reported without details, no execution of file content | `ingest.py` |
| Cross-site scripting | All server text rendered with `textContent` (web page) or Streamlit's escaped Markdown; strict Content-Security-Policy | `static/app.js`, `api.py` |
| Cost or quota exhaustion | Daily budgets stored in the database, per-visitor and per-address rate limits, concurrent-upload cap | `graph.py`, `api.py` |
| Leaked secrets | `.env`, `data/` and local secrets are git-ignored; secrets are read from the environment; `SecretStr` keeps them out of logs | `config.py`, `.gitignore` |
| Vulnerable dependencies | Weekly Dependabot updates and an advisory `pip-audit` job in CI | `.github/` |

## Known limitations

- **No accounts.** Anyone holding a workspace link or cookie can open that workspace. Do not use the public
  deployment for sensitive documents.
- **Uploaded text is sent to a third-party LLM provider (Groq).** This is shown to visitors on the page.
- Per-visitor rate limits are kept in memory, so they are per process. The daily budgets are the hard cap.
- Some transitive dependencies (for example Pillow, used only by Streamlit's image support) have fixes that need
  Python 3.11 or newer. Run on Python 3.12 to get them (see `requirements.lock` and the CI audit output).
