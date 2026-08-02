# Consolidation and deduplication

The former repository collection was not copied verbatim. Functionality was reviewed and reimplemented into one coherent package.

## Kept and strengthened

- Typed contracts and FastAPI control plane.
- Secure upload limits, CORS restrictions, and server-side secret handling.
- OpenCV floor-plan extraction as a deterministic baseline.
- Shapely constraint-based furniture placement.
- Safe PyTorch checkpoint loading with `weights_only=True`.
- Room-classifier and floor-plan-segmenter training paths.
- Catalog, consultation booking, Docker, Streamlit, and CI.

## Removed as duplicate or unsuitable

- Multiple FastAPI and Streamlit entrypoints.
- Node backends duplicating catalog, authentication, and booking functions.
- Random furniture placement and random inference state.
- Simulated marketplace responses presented as real integrations.
- Face-generation GAN weights unrelated to furniture design.
- A DQN checkpoint whose architecture and four-dimensional state did not match the room-mask planner.
- Demo credentials, environment files, Git histories, and submodule references.
- The quarantined legacy monolith with contaminated history.

The repository now contains one implementation per capability and one dependency definition.
