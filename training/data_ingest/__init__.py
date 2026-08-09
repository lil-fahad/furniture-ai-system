"""Data ingestion pipeline for cloud training (WP-B).

Stages real and synthetic datasets locally under ``data/staging/`` and uploads
them to Google Cloud Storage following the layout in SPEC.md section 2.2.

Submodules:
- ``openimages_furniture``: Open Images furniture subset downloader (rooms).
- ``floorplans``: synthetic floor-plan image/mask pair generator (plans).
- ``catalog``: supplier catalog decoder (catalog).
- ``gcs``: lazy ``google-cloud-storage`` upload/download helpers.
- ``stage_all``: CLI orchestrator for staging + upload + manifest.
"""
