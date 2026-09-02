# ML Model and Dataset Adoption Checklist

Use this checklist before adding any external ML model or dataset to FurnitureAI.

- [ ] Repository/model/dataset identifier recorded.
- [ ] License and redistribution terms verified.
- [ ] Provenance and revision recorded.
- [ ] Intended task matches the FurnitureAI contract.
- [ ] Input preprocessing and output schema documented.
- [ ] Known limitations and failure modes documented.
- [ ] Data leakage and train/test contamination considered.
- [ ] Benchmark protocol is reproducible.
- [ ] Baseline comparison is complete.
- [ ] Confidence calibration or abstention behavior is evaluated where applicable.
- [ ] CPU/memory/GPU requirements are known.
- [ ] Model artifact is integrity-pinned before deployment.
- [ ] Security review covers serialization/loading behavior.
- [ ] Rollback to the previous artifact is tested.
- [ ] No production claim is made from synthetic, annotation-only, or insufficient evaluation data.

## Floor-plan priority

For floor-plan intelligence, prioritize real labeled evaluation of walls, rooms, doors, and windows. Room-type semantics must not be inferred from room area alone when a trained classifier/segmenter is unavailable. Physical dimensions must not be claimed unless a reliable scale is present.
