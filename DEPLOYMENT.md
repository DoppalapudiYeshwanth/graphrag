# Deploying Graph RAG: Cloud Run + Neo4j AuraDB

The API (`graphrag/api.py`) is a stateless container. All state lives in Neo4j, so the deployment is
two pieces: a managed Neo4j (AuraDB) holding the graph, and a Cloud Run service running the container.

```
client --X-API-Key--> Cloud Run (FastAPI + MiniLM) --bolt+s--> AuraDB (graph + vector index)
                              |
                              +--HTTPS--> Groq (answer generation)
```

> **Status of this guide.** The API, its tests and the Docker image are verified locally (see
> "Local check" below). The AuraDB and Cloud Run steps below were NOT executed by the author; they use
> standard `gcloud` and Aura flows. Free-tier limits and console wording change, so check each against
> the current provider docs as you go.

## 0. Before you start

You need: a Google Cloud project with billing enabled, `gcloud` installed and logged in, an AuraDB
account, your Groq key, and the local project data (`data/raw/corpus.csv` and
`data/processed/extractions.jsonl`). Replace the placeholders `PROJECT_ID`, `REGION` (for example
`us-central1`) and the Aura values.

## 1. Local check (do this first)

```bash
docker build -t graphrag-api:local .
docker run --rm -p 8080:8080 --env-file .env \
  -e NEO4J_URI=bolt://host.docker.internal:7687 -e API_KEY=local-test-key graphrag-api:local
curl localhost:8080/health
curl -s localhost:8080/ask -H 'X-API-Key: local-test-key' -H 'Content-Type: application/json' \
  -d '{"question":"How does homomorphic encryption protect biometric data?"}'
```

## 2. Create the AuraDB instance

1. In the Aura console, create a new AuraDB instance (the free tier is enough for this graph:
   about 5,200 nodes and 8,600 relationships; confirm the current free-tier limits).
2. **Download the credentials file when prompted.** The password is shown once.
3. Note the connection URI (`neo4j+s://<id>.databases.neo4j.io`) and the database name shown in the
   console (normally `neo4j`).

## 3. Load the graph into Aura

This reuses the extraction cache, so it costs **no LLM tokens**; only local embedding runs.

```bash
source venv/bin/activate
export NEO4J_URI='neo4j+s://<id>.databases.neo4j.io'
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD='<password from the credentials file>'
export NEO4J_DATABASE=neo4j
python -m scripts.build_graph
```

Environment variables override `.env`, so your local database is untouched. Verify in the Aura
browser:

```cypher
MATCH (c:Chunk) RETURN count(c);            // expect 800
MATCH (e:Entity) RETURN count(e);           // expect about 3,961
SHOW VECTOR INDEXES;                         // chunk_embeddings, ONLINE
```

## 4. Store secrets in Secret Manager

```bash
gcloud config set project PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com

openssl rand -hex 32 > /tmp/api_key.txt           # the key your clients will send
printf '%s' "$GROQ_API_KEY"   | gcloud secrets create groq-api-key  --data-file=-
printf '%s' "$NEO4J_PASSWORD" | gcloud secrets create neo4j-password --data-file=-
gcloud secrets create api-key --data-file=/tmp/api_key.txt
shred -u /tmp/api_key.txt 2>/dev/null || rm -P /tmp/api_key.txt   # keep a copy in your password manager first

# let the Cloud Run runtime service account read them
PROJECT_NUMBER=$(gcloud projects describe PROJECT_ID --format='value(projectNumber)')
for s in groq-api-key neo4j-password api-key; do
  gcloud secrets add-iam-policy-binding $s \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor
done
```

## 5. Build the image in the cloud

Build with Cloud Build, not from your Apple Silicon laptop: a local `docker build` produces an
`arm64` image, and Cloud Run runs `amd64`.

```bash
gcloud artifacts repositories create graphrag --repository-format=docker --location=REGION
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/graphrag/api:v1
```

(The first build is slow because PyTorch and the embedding model are baked into the image.)

## 6. Deploy

```bash
gcloud run deploy graphrag-api \
  --image REGION-docker.pkg.dev/PROJECT_ID/graphrag/api:v1 \
  --region REGION \
  --memory 2Gi --cpu 1 --cpu-boost \
  --min-instances 0 --max-instances 2 --concurrency 4 --timeout 60 \
  --allow-unauthenticated \
  --set-env-vars NEO4J_URI='neo4j+s://<id>.databases.neo4j.io',NEO4J_USERNAME=neo4j,NEO4J_DATABASE=neo4j,GROQ_MODEL=openai/gpt-oss-20b,ANSWER_MODEL=qwen/qwen3.8-27b,EXPANSION_MODE=co_mention \
  --set-secrets GROQ_API_KEY=groq-api-key:latest,NEO4J_PASSWORD=neo4j-password:latest,API_KEY=api-key:latest
```

Why these flags:

| Flag | Reason |
| --- | --- |
| `--memory 2Gi` | PyTorch plus the model need headroom; 512 MB will be killed |
| `--cpu-boost` | Faster cold start while the embedding model loads |
| `--min-instances 0` | Scales to zero when idle so you pay nothing. Set 1 to avoid cold starts (small ongoing cost) |
| `--max-instances 2` | Caps spend and caps how fast Groq tokens can be burned |
| `--concurrency 4` | Each request is CPU and I/O bound; low concurrency keeps latency steady |
| `--allow-unauthenticated` | The app enforces `X-API-Key` itself. For stricter access, drop this flag and use Cloud Run IAM instead |

## 7. Smoke test the live URL

```bash
URL=$(gcloud run services describe graphrag-api --region REGION --format='value(status.url)')
curl -s "$URL/health"
curl -s "$URL/ask" -H "X-API-Key: <your key>" -H 'Content-Type: application/json' \
  -d '{"question":"How does homomorphic encryption protect biometric data?"}'
curl -s -o /dev/null -w '%{http_code}\n' "$URL/ask" -H 'Content-Type: application/json' \
  -d '{"question":"no key"}'            # expect 401
```

Done when: `/health` returns `{"status":"ok"}`, `/ask` returns a cited answer in under about 10 seconds
(first request after idle can take longer), and the keyless request returns 401.

## 8. Know the limits

- **Groq quota is your real capacity.** The free tier allows 200,000 tokens per day per model. One
  answer uses roughly 3,000 to 4,000 tokens (seven passages, graph facts, reasoning), so expect
  about 50 to 70 questions per day. Measure with real traffic. When the quota is gone, `/ask`
  returns 429.
- **Rate limiting is per instance and in memory** (`RATE_LIMIT_PER_MINUTE`, default 20 per client).
  With `--max-instances 2` the real ceiling is up to twice that.
- **Aura free instances may pause when idle**; if `/health` returns 503, check and resume the instance
  in the Aura console.
- **No answer fallback chain yet.** Extraction has a model fallback chain; answering uses a single
  `ANSWER_MODEL`. Adding the same chain to answering is the obvious next hardening step.

## 9. Operating it

```bash
gcloud run services logs read graphrag-api --region REGION --limit 50   # logs
gcloud run services update graphrag-api --region REGION --set-secrets API_KEY=api-key:latest  # after rotating a secret
gcloud run revisions list --service graphrag-api --region REGION         # find a previous revision to roll back to
gcloud run services update-traffic graphrag-api --region REGION --to-revisions <REVISION>=100
```

Set a billing budget alert in the Cloud console before sharing the URL.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Deploy fails, container does not start | Missing secret access, or out of memory | Check logs; confirm the secretAccessor bindings; keep `--memory 2Gi` |
| `/health` 503 | Aura paused, wrong URI or password | Resume Aura; check env vars; URI must be `neo4j+s://` |
| `/ask` 401 | Missing or wrong `X-API-Key` | Send the key stored in `api-key` |
| `/ask` 429 | Per-client limit, or Groq daily quota | Wait, or lower traffic; see section 8 |
| `/ask` 502 | Groq unreachable or erroring | Retry; check the Groq status page |
| Slow first request | Cold start loading the model | Use `--cpu-boost`, or `--min-instances 1` |
| Empty or irrelevant answers | Graph not loaded into Aura | Rerun section 3 and verify counts |

## 11. Security checklist

- [ ] `API_KEY` is set (the service logs a warning at start-up if it is not).
- [ ] Secrets live in Secret Manager, never in the image, the repo or `--set-env-vars`.
- [ ] `.env` and `data/` are git-ignored and docker-ignored (they are).
- [ ] A billing budget alert is configured.
- [ ] `--max-instances` is capped.
- [ ] You rotate the Groq key and the API key if either is ever pasted somewhere public.

## Alternatives

Any host that runs a container with 1 to 2 GB RAM and injects `PORT` works the same way (Fly.io,
Render, Railway). The configuration surface is only the environment variables listed in section 6.
