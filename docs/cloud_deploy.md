# Cloud worker deploy (job + worker pool)

One Cloud Build config builds the container and rolls out both runtimes.

## Deploy from your machine

```bash
cd moyo
gcloud builds submit --config=cloudbuild.yaml --project=senteguard-website
```

The build pushes `us-central1-docker.pkg.dev/senteguard-website/moyo/worker:${BUILD_ID}`,
updates job `moyo-report-worker-no-vpc`, and updates worker pool
`moyo-interactive-worker` (`python pool_worker.py`).

## GCP setup (outside this repo)

### 1. Cloud Build service account permissions

Find the build SA (default legacy):

`226988471174@cloudbuild.gserviceaccount.com`

Or the user-managed Cloud Build SA if you switched.

It needs at least:

- **Artifact Registry**: push to `moyo/worker` (usually `roles/artifactregistry.writer` on the repo or project).
- **Cloud Run**: update the job and worker pool — typically **`roles/run.admin`** on project `senteguard-website`, or equivalent custom permissions including `run.jobs.update` and worker pool update.

If the worker-pools step fails with `PERMISSION_DENIED`, grant `roles/run.admin` to the Cloud Build SA in IAM, wait a minute, and re-run the build.

### 2. Optional: deploy on push to `main`

GitHub Actions CI does **not** run this file. To automate:

1. [Cloud Build Triggers](https://console.cloud.google.com/cloud-build/triggers?project=senteguard-website)
2. **Create trigger** → connect the `moyo` GitHub repo (2nd gen connection if needed).
3. Event: push to branch `main` (or release tag).
4. Configuration: Cloud Build configuration file → `cloudbuild.yaml`, source location `moyo/` (or repo root with path `moyo/cloudbuild.yaml` depending on how the repo is connected).
5. Save. Pushes to `main` then build + deploy job + pool.

### 3. Verify after a build

```bash
# Job image
gcloud run jobs describe moyo-report-worker-no-vpc --region=us-central1 \
  --format='value(template.template.containers[0].image)'

# Pool image
gcloud run worker-pools describe moyo-interactive-worker --region=us-central1 \
  --format='value(template.containers[0].image)'
```

Both should end with the same `BUILD_ID` from the successful Cloud Build.

### 4. What this does *not* deploy

- **moyomapwebpage** (Netlify / Firebase functions) — separate pipeline.
- **Worker pool env/secrets** (`SUBSCRIPTION_ID`, API keys, `moyo-worker@…` SA) — only the image and `pool_worker.py` command are updated; change env in Console or export/replace YAML if needed.

### 5. Override pool or region (rare)

```bash
gcloud builds submit --config=cloudbuild.yaml --project=senteguard-website \
  --substitutions=_WORKER_POOL=other-pool,_REGION=us-central1
```
