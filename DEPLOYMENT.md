# Deploying Graph RAG

One container serves both the **web app** (upload documents, ask questions) and the **API**. All state
lives in Neo4j, so the deployment is two pieces: a managed Neo4j (AuraDB) holding the demo corpus and
visitors' private workspaces, and a Cloud Run service running the container.

```
browser --cookie--> Cloud Run (FastAPI: web page + /api/*  +  /ask) --bolt+s--> AuraDB
                              |                                        graph, vectors, daily budgets
                              +--HTTPS--> Groq (extraction + answers)
```

> **Status of this guide.** The app, both front ends, the 190 tests and the local tunnel are verified. The
> hosted routes (Streamlit Cloud, Render, Cloud Run) were NOT executed by the author, because they need your
> accounts. Free-tier rules change, so check each provider's current terms before relying on them.

## Choose a route (read first)

| Route | Card on file? | Can it bill you? | Always on? | Status |
| --- | --- | --- | --- | --- |
| **A. Your machine + free Cloudflare tunnel** | No | No | Only while your Mac runs the script | **Built and tested end to end** |
| **B. Streamlit Community Cloud + Aura Free** | Aura: no (per its FAQ). Streamlit: needs only a GitHub account | No payment method involved | Sleeps after 12 idle hours; Aura pauses after 3 idle days | **Code and probe built.** Run the probe first: whether Streamlit's servers can reach port 7687 is unverified |
| C. Render free web service | Unverified (docs do not say) | Docs say overage charges are possible when a payment method exists | Sleeps after 15 idle minutes | Not built. Free instances have 512 MB RAM; the embedding model alone measured about 434 MB, so it would need a lighter embedder |
| D. Google Cloud Run + AuraDB (sections 0 to 11) | Yes (an open billing account) | Yes: Google has no hard cap, only alerts | Yes, scales to zero | Guide written, not run (your billing accounts are closed) |
| E. Hugging Face Docker Space | Needs a paid plan | n/a | n/a | **Not viable**: Docker Spaces require a paid plan, and free Spaces only reach ports 80, 443 and 8080, so Aura (7687) is blocked |

Rule of thumb for "never bill me": if no service in the chain has your payment method, nothing can bill you.

## Route A: share from your own machine (free, no card, no account)

```bash
brew install cloudflared            # once
scripts/share_public.sh             # prints a public https URL; Ctrl-C closes it
```

What the script does: starts your local Neo4j container, starts the app on `127.0.0.1` only, keeps the
Mac awake while it runs, and opens a Cloudflare quick tunnel. Nothing is deployed anywhere, and the
URL is random and changes every run.

Safety applied automatically while public:

| Control | Value |
| --- | --- |
| Daily caps, shared by all visitors | 60 questions and 100 upload chunks (override with `DAILY_ASK_BUDGET`, `DAILY_CHUNK_BUDGET`) |
| `/docs` and `/openapi.json` | Hidden (`DOCS_ENABLED=false`) |
| Programmatic `/ask` | Needs a random `API_KEY` created fresh each run |
| Database | Never exposed; only the app port goes through the tunnel |
| Visitor addresses | Taken from proxy headers, so per-visitor rate limits work |
| Cookies | `Secure`, `HttpOnly`, `SameSite=Lax` over https |
| Retention | Idle visitor workspaces are deleted on each start |

Tested through a real tunnel: health, headers, hidden docs, 401 without a key, database not reachable,
streamed upload progress, a private answer with citations, and cleanup. The tunnel closes cleanly on Ctrl-C.

Know the limits:
- **The site is up only while the script runs and your Mac is awake and online.**
- Quick tunnels have **no uptime guarantee**. A brand-new URL can take up to a minute to resolve on your
  machine (a stale DNS cache); other networks usually see it sooner.
- **You are hosting other people's uploads on your machine.** Do not leave it running unattended, and do
  not advertise it for sensitive documents. Uploaded text goes to Groq.
- Capacity is the free Groq quota. When the daily caps are reached, visitors get a clear message.

To share the URL for a limited time: run the script, send the link, and press Ctrl-C when you are done.

## Route B: Streamlit Community Cloud + Aura Free

Everything here is free and needs no payment method, **if** Streamlit's servers can reach Aura on port 7687.
That is the one thing nobody has verified, so test it before building anything on it.

### Step 1: probe (10 minutes, decides everything)

1. Create an [AuraDB Free](https://neo4j.com/cloud/aura-free/) instance. Download the credentials file when
   prompted; the password is shown once. Note the `neo4j+s://...` URI.
2. Push this repository to its own GitHub repository (a public repository is simplest). It must be the repository root: Streamlit and GitHub Actions both expect `requirements.txt` and `.github/` at the top level.
3. In Streamlit Community Cloud, create an app from the repository with **main file `deploy/streamlit_probe.py`**
   and choose Python 3.12 under Advanced settings.
4. In Advanced settings, Secrets, paste:

```toml
NEO4J_URI = "neo4j+s://<id>.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "<password from the credentials file>"
```

5. Open the app. Every line is a real check run from Streamlit's servers: DNS, port 443 (control), **port 7687**,
   a real Neo4j connection and query, reachability of Groq, and loading the embedding model with its peak memory.

| Probe result | Meaning |
| --- | --- |
| All green | Route B works. Continue to step 2. |
| Port 443 green, **7687 red** | Streamlit blocks Bolt. Route B is dead; use Route A, or Aura's HTTP query API with a code change. |
| The page says the secrets "point at your own computer" (or `Target host: localhost`) | `NEO4J_URI` is still your local database address. Put the Aura `neo4j+s://<id>.databases.neo4j.io` address in Secrets and reboot the app. Nothing else on the page is meaningful until this is fixed. |
| Embedding model red or killed | Out of memory or install problem; Streamlit's documented range is 690 MB to 2.7 GB (February 2024 figures); a real probe on Streamlit's host measured 668 MB peak with the model loaded. |

### Step 2: load the demo graph into Aura (free: reuses the extraction cache, no LLM tokens)

```bash
source venv/bin/activate
export NEO4J_URI='neo4j+s://<id>.databases.neo4j.io' NEO4J_USERNAME=neo4j NEO4J_PASSWORD='<password>'
python -m scripts.build_graph
```

The demo graph is about 5,200 nodes and 8,600 relationships. Aura's documentation lists a 200,000-node limit while its
free-tier FAQ page lists an older, smaller one; the graph fits under either. Verify the counts in the Aura browser.

### Step 3: deploy the real app

1. In Streamlit Community Cloud, create (or edit) the app with **main file `streamlit_app.py`**, Python 3.12.
2. Secrets: the same three Neo4j lines plus `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_FALLBACK_MODELS`, `ANSWER_MODEL` and
   low public caps. A ready template is `.streamlit/secrets.toml.example`.
3. Never put these in the repository. `.streamlit/secrets.toml` is git-ignored.

### What to expect

- Streamlit apps sleep after 12 idle hours and wake on the next visit; Aura Free pauses after 3 idle days (resume it in
  the Aura console; data is kept).
- Visitors are identified by the `?ws=` value in the page URL, so **anyone holding that link can open that workspace**.
  The page says so. Per-session rate limits are in memory; the shared daily budgets in Neo4j are the hard cap.
- `requirements.txt` carries a CPU-only PyTorch index line. If Streamlit's installer rejects it, the install log will
  say so; fall back to removing that line (the install is then larger).

The rest of this guide (sections 0 to 11) covers Route D, Google Cloud Run.

## 0. What you are about to make public

Read this before you deploy; it is what separates a demo from an incident.

| Topic | How it works | What you must accept |
| --- | --- | --- |
| **Private workspaces** | Each visitor gets a random, unguessable cookie id. Their uploads, entities and chunks carry that id and are only searchable by them (plus the shared demo corpus). | Anyone holding the cookie can access that workspace. There are no accounts. |
| **Privacy** | Uploaded text is sent to Groq for analysis and answering. The page says so. | Do not advertise this for sensitive documents. |
| **Cost and quota** | Groq's free tier is the capacity limit. Hard daily caps are stored in Neo4j: `DAILY_CHUNK_BUDGET` (default 250 text chunks, about 1.2 to 1.6k tokens each) and `DAILY_ASK_BUDGET` (default 150 questions). | When a cap is hit, uploads or questions are refused with a clear message until the next UTC day. |
| **Abuse limits** | Per-visitor and per-address rate limits, upload size and document-length limits, at most 5 documents per visitor, at most 2 uploads processed at once. | Rate limits are in memory, so they are per instance. The daily budgets are the hard cap. |
| **Retention** | Idle workspaces are deleted after `WORKSPACE_TTL_DAYS` (default 7) by a scheduled job (section 8). | Without the job, data stays until the visitor deletes it. |
| **The `/ask` API** | Searches the shared demo corpus only. Protected by `X-API-Key` (set `API_KEY`). The web page does not use it and never holds a secret. | The web endpoints are intentionally open, so keep the budgets low. |

## 1. Local check (do this first)

```bash
docker build -t graphrag-api:local .
docker run --rm -p 8080:8080 --env-file .env \
  -e NEO4J_URI=bolt://host.docker.internal:7687 graphrag-api:local
# open http://localhost:8080 , upload a small .txt, ask a question
curl localhost:8080/health
```

## 2. Create the AuraDB instance

1. In the Aura console, create an AuraDB instance (the free tier should fit this graph: about 5,200
   nodes and 8,600 relationships plus visitors' documents; confirm the current free-tier limits).
2. **Download the credentials file when prompted.** The password is shown once.
3. Note the URI (`neo4j+s://<id>.databases.neo4j.io`) and the database name (normally `neo4j`).

## 3. Load the demo corpus into Aura

This reuses the extraction cache, so it costs **no LLM tokens**; only local embedding runs.

```bash
source venv/bin/activate
export NEO4J_URI='neo4j+s://<id>.databases.neo4j.io'
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD='<password from the credentials file>'
export NEO4J_DATABASE=neo4j
python -m scripts.build_graph
```

Environment variables override `.env`, so your local database is untouched. Verify in the Aura browser:

```cypher
MATCH (c:Chunk) RETURN count(c);            // expect 800
MATCH (e:Entity) RETURN count(e);           // expect about 3,961
SHOW VECTOR INDEXES;                         // chunk_embeddings, ONLINE
```

The service also creates the schema (constraints, workspace indexes, vector index) on start-up, so
visitors' uploads work on a fresh database.

## 4. Store secrets in Secret Manager

```bash
gcloud config set project PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com

openssl rand -hex 32 > /tmp/api_key.txt           # for programmatic clients of /ask
printf '%s' "$GROQ_API_KEY"   | gcloud secrets create groq-api-key  --data-file=-
printf '%s' "$NEO4J_PASSWORD" | gcloud secrets create neo4j-password --data-file=-
gcloud secrets create api-key --data-file=/tmp/api_key.txt
shred -u /tmp/api_key.txt 2>/dev/null || rm -P /tmp/api_key.txt   # copy it to a password manager first

PROJECT_NUMBER=$(gcloud projects describe PROJECT_ID --format='value(projectNumber)')
for s in groq-api-key neo4j-password api-key; do
  gcloud secrets add-iam-policy-binding $s \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor
done
```

## 5. Build the image in the cloud

Build with Cloud Build, not from your Apple Silicon laptop: a local `docker build` produces an `arm64`
image, and Cloud Run runs `amd64`.

```bash
gcloud artifacts repositories create graphrag --repository-format=docker --location=REGION
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/graphrag/api:v1
```

## 6. Deploy

```bash
gcloud run deploy graphrag-api \
  --image REGION-docker.pkg.dev/PROJECT_ID/graphrag/api:v1 \
  --region REGION \
  --memory 2Gi --cpu 1 --cpu-boost \
  --min-instances 0 --max-instances 2 --concurrency 8 --timeout 600 \
  --allow-unauthenticated \
  --set-env-vars NEO4J_URI='neo4j+s://<id>.databases.neo4j.io',NEO4J_USERNAME=neo4j,NEO4J_DATABASE=neo4j,GROQ_MODEL=openai/gpt-oss-20b,GROQ_FALLBACK_MODELS='openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-safeguard-20b',ANSWER_MODEL=qwen/qwen3.8-27b,EXPANSION_MODE=co_mention,DAILY_CHUNK_BUDGET=250,DAILY_ASK_BUDGET=150 \
  --set-secrets GROQ_API_KEY=groq-api-key:latest,NEO4J_PASSWORD=neo4j-password:latest,API_KEY=api-key:latest
```

The commas inside `GROQ_FALLBACK_MODELS` break `--set-env-vars` parsing in some `gcloud` versions. If
the deploy rejects it, put the variables in a YAML file and use `--env-vars-file`.

| Flag | Reason |
| --- | --- |
| `--allow-unauthenticated` | Required so anyone can open the web page. The web routes protect themselves with per-visitor cookies, limits and budgets. |
| `--timeout 600` | An upload is processed while the request stays open (progress streams to the page). A 15,000-character document can take a couple of minutes. |
| `--concurrency 8` | Uploads hold a request open for a long time; the app also caps itself at 2 concurrent uploads. |
| `--max-instances 2` | Caps spend. Remember rate limits are per instance. |
| `--memory 2Gi`, `--cpu-boost` | PyTorch plus the model need headroom; boost speeds cold start. |
| `--min-instances 0` | Scales to zero when idle. Use 1 to avoid cold starts (small ongoing cost). |

## 7. Smoke test the live URL

1. Open the service URL in a browser. You should see the page and the capacity line.
2. Upload a small `.txt`, watch the progress bar finish, then ask about it with "My documents".
3. Open the URL in a private window: that visitor must NOT see the first visitor's document.
4. Command-line checks:

```bash
URL=$(gcloud run services describe graphrag-api --region REGION --format='value(status.url)')
curl -s "$URL/health"
curl -s -o /dev/null -w '%{http_code}\n' "$URL/ask" -H 'Content-Type: application/json' \
  -d '{"question":"no key"}'            # expect 401
```

Also confirm the rate limit really keys on the visitor's address. Behind Cloud Run the client address
comes from `X-Forwarded-For`; the container runs uvicorn with `--proxy-headers`. If every visitor
appears to share one address (you hit 429 quickly from different networks), the headers are not
being honoured; check the service logs.

## 8. Operate it

**Retention job.** Deletes workspaces idle longer than `WORKSPACE_TTL_DAYS`. The image contains the script.

```bash
gcloud run jobs create graphrag-cleanup \
  --image REGION-docker.pkg.dev/PROJECT_ID/graphrag/api:v1 --region REGION \
  --command python --args=-m,scripts.cleanup_workspaces \
  --set-env-vars NEO4J_URI='neo4j+s://<id>.databases.neo4j.io',NEO4J_USERNAME=neo4j,NEO4J_DATABASE=neo4j \
  --set-secrets NEO4J_PASSWORD=neo4j-password:latest,GROQ_API_KEY=groq-api-key:latest
gcloud run jobs execute graphrag-cleanup --region REGION          # try it once
# then schedule it daily in Cloud Scheduler (console: target = Cloud Run job)
```

Other commands:

```bash
gcloud run services logs read graphrag-api --region REGION --limit 50
gcloud run revisions list --service graphrag-api --region REGION
gcloud run services update-traffic graphrag-api --region REGION --to-revisions <REVISION>=100   # roll back
```

Set a billing budget alert in the Cloud console before sharing the URL.

## 9. Know the limits

- **Capacity is the Groq quota.** Roughly 120 to 170 chunks per model per day for uploads (budget default
  250 across the fallback chain) and roughly 50 to 70 questions per day on one answer model. Measure
  with real traffic and tune `DAILY_CHUNK_BUDGET` and `DAILY_ASK_BUDGET`. A paid Groq tier removes this.
- **A typical upload** (a few thousand characters) is 3 to 8 chunks; the 15,000-character maximum is
  about 20.
- **Answering has no fallback chain yet**; extraction does.
- **Aura free instances may pause when idle**; if `/health` returns 503, resume the instance.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Container does not start | Missing secret access, out of memory, or Aura unreachable | Check logs; confirm secretAccessor bindings; keep `--memory 2Gi` |
| `/health` 503 | Aura paused, wrong URI or password | Resume Aura; URI must be `neo4j+s://` |
| Upload says "capacity is used up" | Daily chunk budget reached | Wait for the next UTC day or raise `DAILY_CHUNK_BUDGET` |
| Upload says "quota is used up" | Every Groq model hit its daily limit | Wait, or use a paid tier |
| Page loads but uploads fail | Request timeout too low | Use `--timeout 600` |
| Everyone gets 429 | All visitors seen as one address | See the proxy-headers note in section 7 |
| Visitor "lost" their documents | Cookie cleared or 7 idle days passed | Expected; documents cannot be recovered without the cookie |

## 11. Security checklist

- [ ] Secrets live in Secret Manager, never in the image, the repo or `--set-env-vars`.
- [ ] `API_KEY` is set (protects `/ask`); `.env` and `data/` are ignored by git and Docker.
- [ ] Daily budgets and `--max-instances` are set to values you can afford.
- [ ] A billing budget alert is configured.
- [ ] The retention job is scheduled.
- [ ] The privacy notice on the page matches reality (text goes to Groq).
- [ ] You rotate the Groq key and the API key if either is ever exposed.

## Alternatives

Any host that runs a container with 1 to 2 GB RAM, injects `PORT` and allows long requests works the
same way (Fly.io, Render, Railway). The configuration surface is the environment variables above.
