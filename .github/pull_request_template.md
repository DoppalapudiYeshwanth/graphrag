## What and why

<!-- What does this change, and why is it needed? Link the issue if there is one. -->

## How it was tested

- [ ] `ruff check . && ruff format --check .`
- [ ] `python -m pytest` (add `TEST_NEO4J_URI=...` against a throwaway Neo4j for the integration tests)
- [ ] I added or updated tests for behaviour I changed

## Safety checklist

- [ ] No secrets, `.env`, `data/` or personal files are included
- [ ] Anything a visitor can reach stays scoped to their own workspace
- [ ] Docs updated (README, `docs/`, `DEPLOYMENT.md`) if behaviour or settings changed
