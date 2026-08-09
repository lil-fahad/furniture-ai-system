# Cloud training on Google Cloud (Vertex AI)

Train the three furniture-ai-system models — room classifier, floor-plan
segmenter, supplier ranker — on Vertex AI custom jobs with GPUs, with datasets
staged in Google Cloud Storage. Arabic guide: `docs/GCP_TRAINING_AR.md`.

## Prerequisites

- A GCP project with **billing enabled** (e.g. `round-office-505007-q4`).
- `gcloud` CLI authenticated (`gcloud auth login`); Cloud Shell is pre-authenticated.
- GPU quota for `NVIDIA_L4` in your region (default `us-central1`).
- Python 3.11+ with the repo installed locally only if you run the staging /
  submission steps outside Cloud Build.

## One-command bootstrap

From the repository root:

```bash
scripts/gcp_bootstrap.sh --project <your-project> --region us-central1
```

This will (idempotently):

1. check authentication and project access,
2. enable the `aiplatform`, `storage`, `cloudbuild`, `artifactregistry` APIs,
3. create the Artifact Registry repo `furniture-ai`,
4. create the bucket `gs://furniture-ai-training-<project>`,
5. build the training image (`cloud/Dockerfile.training`) with Cloud Build,
6. stage datasets to `gs://<bucket>/datasets/` (`python -m training.data_ingest.stage_all`),
7. submit three Vertex AI custom jobs (`python -m cloud.vertex_jobs --task all`).

Useful flags: `--skip-data`, `--skip-image`, `--skip-jobs` (safe re-runs),
`--bucket B` to override the default bucket name.

Submit a single task manually:

```bash
python -m cloud.vertex_jobs --task room \
  --project <your-project> --region us-central1 --bucket <bucket>
```

Defaults live in `cloud/config.yaml` and can be overridden with CLI flags or
the `GCP_PROJECT` / `GCP_REGION` / `GCS_BUCKET` environment variables.

## Monitoring

```bash
# list jobs
gcloud ai custom-jobs list --region us-central1 --project <your-project>

# stream logs for a job
gcloud ai custom-jobs stream-logs <JOB_ID> --region us-central1 --project <your-project>
```

Or open **Vertex AI → Training** in the Cloud Console.

## Fetching artifacts

Every run writes checkpoints, `metrics.json` and `logs.txt` to
`gs://<bucket>/runs/<RUN_ID>/` (see SPEC §2.2/§2.3):

```bash
gcloud storage cp -r gs://<bucket>/runs .
```

## Cost notes (approximate, us-central1)

- One job: `g2-standard-4` + 1× NVIDIA L4, spot pricing ≈ **$0.4–0.8/hour**.
- Typical full run (3 tasks, default epochs) finishes in well under a few
  hours; expect a few USD total with spot capacity.
- Storage for staged datasets + run artifacts is small (GBs) — cents/month.
- Use `--no-spot` on `cloud.vertex_jobs` only if spot capacity keeps getting
  preempted (cost rises to ≈ $1.0–1.3/hour per job).
