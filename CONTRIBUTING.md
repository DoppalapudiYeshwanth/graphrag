# Contributing

Thanks for helping. This project has two hard rules that shape every change:

1. **It must stay free to run.** No feature may require a paid service or a payment method.
2. **A visitor must never see another visitor's data.** Anything reachable from the web page is scoped to the
   visitor's own workspace plus the shared demo corpus.

## Set up

```bash
git clone <your fork or this repository> && cd graphrag
python3.12 -m venv venv && source venv/bin/activate      # 3.10 also works; 3.12 is recommended
pip install -r requirements-dev.txt                      # exact tested versions: requirements.lock
cp .env.example .env                                     # then fill in GROQ_API_KEY and NEO4J_PASSWORD
docker run -d --name neo4j-graphrag -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<your-password> neo4j:5
```

Never commit `.env`, `data/` or `.streamlit/secrets.toml`. They are git-ignored; keep it that way.

## Run the checks (the same ones CI runs)

```bash
ruff check . && ruff format --check .      # lint and formatting (ruff format . fixes formatting)
python -m pytest                           # unit tests; integration tests are skipped
```

The integration tests **wipe the database they connect to**. Start a separate, throwaway Neo4j and point the
tests at it. They also refuse to run against a database that already holds data they did not create:

```bash
docker run -d --name neo4j-test -p 7688:7687 -e NEO4J_AUTH=neo4j/testpassword neo4j:5
TEST_NEO4J_URI=bolt://localhost:7688 python -m pytest --cov --cov-fail-under=85
```

The LLM and the embedding model are faked in tests, so running them costs no API quota.

## Workflow

1. Branch from `main`: `git switch -c feature/short-description` (or `fix/...`, `docs/...`).
2. Make a focused change. Add or update tests; behaviour without a test is not finished.
3. Run the checks above. CI must be green before review.
4. Commit with a message that says **what changed and why**, for example
   `Retry the primary model after an hour so long-running servers recover from quota resets`.
5. Open a pull request using the template and describe how you tested it.

## Code standards

- Python 3.10 compatible, type hints on public functions, docstrings that explain *why*, not just *what*.
- Validate at the boundary: user input, model output and settings go through Pydantic.
- Keep modules single-purpose (`config`, `data`, `models`, `graph`, `retrieval`, `ingest`, `api`, UI).
- Any Cypher built from a string must validate the interpolated identifier first; values are always parameters.
- Render server text in the browser with `textContent`, never `innerHTML`.

## Changes that need extra care

| Area | What to do |
| --- | --- |
| Retrieval or graph queries | Keep the workspace scope filter. Add an isolation test that breaks if the filter is removed. |
| Anything that calls the LLM | Respect the fallback chain and the daily budgets; do not add a path that bypasses them. |
| Dependencies | Add direct dependencies to `requirements.txt` (runtime) or `requirements-dev.txt`; regenerate `requirements.lock`. |
| New models | Add to `GROQ_FALLBACK_MODELS`; verify with `groq.models.list()` that your key can use it. |
| Hosting | Check the provider's current free-tier rules (cost, ports, sleep) before writing code for it. |

## Reporting problems

Bugs and ideas: open an issue using the templates. Security problems: do **not** open a public issue; see
[SECURITY.md](SECURITY.md).
