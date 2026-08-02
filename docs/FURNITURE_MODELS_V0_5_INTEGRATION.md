# FurnitureAI recovered-model integration v0.5

## Decision

Keep the recovered professional-model toolkit behind an explicit package boundary. The current monorepo already owns the `furniture_ai` namespace, while the recovered toolkit contains a separate CLI, local service, dataset tooling, archive import logic, OpenAI labeling, and model download/verification code. Blind file replacement would regress the current API and UI.

## Changes prepared

1. Record the six verified model/checkpoint entries and release archive hashes.
2. Preserve the partial-recovery report and missing-range evidence.
3. Add a standalone checkpoint verifier.
4. Document that model weights stay out of Git and are verified after download/extraction.
5. Require explicit OpenAI model selection and billable opt-in in the standalone source package.

## Verified release results

- Automated tests: 45 passed.
- Full archive ZIP test: pass; 314 members.
- Source archive ZIP test: pass; 75 members.
- Wheel ZIP test and installed-resource smoke test: pass.

## Safe merge path

Create adapters in the monorepo for classification, detection, segmentation, and relative-depth tasks. Keep checkpoint paths configurable, call the toolkit's hash verification before loading, and do not import its training-data claims as complete because the first archive segment is unavailable.
