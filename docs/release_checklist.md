# Release Checklist

## Release 0.1.0

The first portfolio release is complete for local/staging review. Its launch status
remains **HOLD** for real-world use.

### Scope and metadata

- [x] Package version and metadata command report `0.1.0`.
- [x] Changelog, MIT license text, and citation metadata are tracked.
- [x] Architecture, policy, methodology, operations, and interview notes are
  linked from the README.
- [x] Portfolio release summary points to the canonical generated reports.
- [x] No notebook or large binary artifact is included.

### Verification

- [x] `pytest -q`
- [x] `ruff check .`
- [x] `make check`
- [x] `make smoke`
- [x] `bash scripts/generate_demo.sh`
- [x] `make release-check`
- [x] Demo outputs are deterministic and reviewable.
- [x] Docker build and health-check scripts are present and shell-valid.

### Evidence and safety

- [x] Baseline and contextual policies are compared on common synthetic inputs.
- [x] OPE reports propensity-aware estimates, matched counts, and ESS.
- [x] Promotion gates consider support, exploration, concentration, improvement,
  and regret.
- [x] Default promotion decision is `HOLD`.
- [x] Policy card distinguishes oracle-assisted benchmarks from deployable
  learners.
- [x] Observability fixtures exercise warning paths.

### Intentionally incomplete

- [ ] Representative production data and domain-specific action definitions
- [ ] Privacy, security, threat-model, and fairness reviews
- [ ] Production authentication, durable logging, and distributed metrics
- [ ] Confidence intervals and cross-fitted OPE
- [ ] Controlled staging experiment with rollback criteria
- [ ] Cloud deployment, on-call ownership, and service-level objectives

## Version-change procedure

1. Confirm the release scope and keep feature work out of the polish commit.
2. Run `make release-check` and `make demo`.
3. Review generated diffs for seeds, counts, decisions, and unexpected artifacts.
4. Update the version in package, citation, changelog, and release summary.
5. Build and smoke-test Docker where an engine is available.
6. Commit with a release-specific message and tag only after review.
