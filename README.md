# Ledger ML

Production-oriented Kubernetes operator for regulated financial-services ML workloads.

Ledger ML provides a Kubernetes-native control plane for reproducible, auditable fraud
and risk-model training, evaluation, approval, deployment, monitoring, and rollback.

## MVP

Ledger ML's first vertical is **fraud scoring**. The MVP will run a complete, reproducible
model lifecycle on a local Kubernetes cluster:

```text
synthetic transactions
        -> training job
        -> model artifact
        -> evaluation
        -> explicit approval
        -> inference deployment
        -> prediction and metrics
```

### MVP outcome

An engineer can submit one `RiskModel` resource and use Kubernetes-native status to
follow the model from training through an approval-gated deployment. The system must
make the model version, training inputs, evaluation result, and deployment state visible.

### MVP target scope (planned)

- Go Kubernetes operator using controller-runtime
- `RiskModel` custom resource with validation and status conditions
- Reproducible training Job using a small synthetic fraud dataset
- Versioned model artifact stored in a local object-storage-compatible service
- Evaluation gate with a documented quality threshold
- Explicit approval before production deployment
- HTTP inference service for fraud-risk predictions
- Prometheus metrics for workload state, predictions, latency, and errors
- Retry-safe reconciliation, terminal failure states, and basic rollback
- Local kind-based development workflow
- Architecture, threat model, runbook, and failure-mode documentation

### Not included in the MVP

- Real customer data or claims of regulatory compliance
- Production cloud deployment
- Distributed multi-GPU training
- Automated model retraining
- Feature store implementation
- Full model registry
- Advanced drift detection
- Multi-tenant billing or chargeback

These are deliberate exclusions. The MVP should teach the lifecycle and establish
production-quality boundaries before adding scale.

## Current implementation status (Milestone 3: governance and evidence foundation)

Implemented in this milestone:

- Go module and controller-runtime manager bootstrap
- Versioned API type: `ledger.ledgerml.io/v1alpha1`, kind `RiskModel`
- `RiskModelSpec` fields for:
  - task
  - training image
  - dataset reference
  - model artifact/output reference
  - immutable lineage identity (`trainingImageDigest`, dataset/output versions, config digest)
  - resource requests/limits
  - serving configuration
  - governance policy (resource bounds + approval semantics)
  - explicit approvals (human-supplied records)
- `RiskModelStatus` fields for:
  - phase
  - conditions
  - observed generation
  - lineage hash
  - approvers counted toward policy
  - governance evidence records
  - model version (reserved for later milestones)
  - reason/message
- Idempotent reconciler that creates a single owned `batch/v1` training Job per
  `RiskModel` and does not recreate it on every reconcile.
- Policy gate before Job creation. Unsafe resource declarations are rejected with clear status/events.
- Status mapping from Job + governance state to phases:
  `Rejected`, `TrainingPending`, `TrainingRunning`, `TrainingSucceeded`,
  `TrainingFailed`, `AwaitingApproval`, `Approved`.
- Structured audit evidence + Kubernetes Events for key decisions:
  accepted/rejected, job created, training observed/failed, approval observed.
- Focused unit tests for API defaulting/validation, Job construction, and status mapping.

### Current training workload contract (teaching boundary)

The operator currently treats the training container image as a **contract boundary**
for a smoke workload, not yet the real fraud model trainer.

The Job injects these environment variables from `RiskModel.spec`:

- `LEDGERML_TASK`
- `LEDGERML_DATASET_KIND`
- `LEDGERML_DATASET_NAME`
- `LEDGERML_DATASET_PATH`
- `LEDGERML_DATASET_VERSION`
- `LEDGERML_OUTPUT_KIND`
- `LEDGERML_OUTPUT_NAME`
- `LEDGERML_OUTPUT_PATH`
- `LEDGERML_OUTPUT_ARTIFACT_VERSION`
- `LEDGERML_TRAINING_IMAGE_DIGEST`
- `LEDGERML_CONFIGURATION_DIGEST`

The controller intentionally does **not** override container `command`/`args` in this
milestone; the training image entrypoint defines execution behavior.

This defines how trainer images should consume inputs/outputs without adding fake
fraud logic yet. Real training implementation, metrics, and artifact integrations are
later milestones.

## Governance model (teaching scope)

### Invariants enforced now

1. **Immutable lineage identity**: task, training image/refs, lineage digest/version fields,
   resource profile, serving intent, and governance policy are immutable after create.
   New lineage => create a new `RiskModel`. This is enforced in the controller by
   comparing persisted `status.lineageHash` to the newly computed hash before any Job action.
2. **Resource policy gate**: requested CPU/memory and optional GPU limits must stay within
   declared policy bounds before the controller creates a training Job.
3. **Approval gate**: successful training does not imply production readiness.
   The controller only marks `Approved` when explicit approval records satisfy policy.
4. **No self-approval by controller**: approvals are read from `spec.approvals`; the
   controller does not write approvals on your behalf.
5. **No production-ready claim yet**: `Ready` remains false with `PromotionNotImplemented`
   until serving/promotion milestones exist.

### Evidence boundaries

- Evidence records and Events capture operator decisions, reasons, and timestamps.
- Evidence intentionally excludes raw transaction payloads, labels, or other PII.
- This is operational governance evidence, **not** regulatory certification and **not**
  cryptographic/non-repudiation proof yet.
- Admission webhook enforcement is not installed yet; immutable lineage is currently
  controller-level enforcement and should be hardened with webhooks in a later milestone.

Not implemented yet (later milestones): real fraud trainer behavior, drift detection,
continuous retraining, real external data integrations, feature store, inference
deployment, object storage/cloud integration, or cryptographic attestation.

## Local development prerequisites

- Go 1.26+
- A Kubernetes cluster for runtime testing (for example kind/minikube)
- `kubectl`

`controller-gen` is executed via `go run` in Makefile targets, so no global install is required.

## Development commands

```bash
make fmt         # format Go code
make generate    # generate deepcopy methods
make manifests   # generate CRD manifests into config/crd/bases
make test        # run unit tests
make build       # compile manager binary
```

## Code structure and operator concepts

- `api/v1alpha1/`: CRD-facing domain model (`spec` = desired state, `status` = observed state).
- `internal/controller/`: reconciliation loop that continuously converges observed cluster state toward desired state.
- `main.go`: manager process wiring scheme, health endpoints, and controller registration.

Teaching focus in this milestone:

- **CRD modeling**: encode operator intent in declarative API fields.
- **Desired vs observed state**: users set `spec`, controller reports progress in `status`.
- **Idempotency**: reconcile is safe to run repeatedly and avoids duplicate Job creation.
- **Conditions**: machine-readable readiness/progress signals for Kubernetes-native observability.
- **Workload contract**: the operator and trainer image communicate through explicit, versionable inputs.
- **Governance gates**: policy and approval checks block unsafe or premature progression.
- **Auditability**: evidence and Events explain lifecycle decisions without leaking sensitive data.

## Teaching path

Each milestone has both a working outcome and a concept to learn:

1. **Model the domain** — understand fraud classification, labels, features, precision,
   recall, false positives, and why accuracy alone is insufficient.
2. **Build the operator skeleton** — learn CRDs, reconciliation, desired versus observed
   state, idempotency, and Kubernetes conditions.
3. **Run training as a workload** — learn containers, Jobs, resource requests, artifacts,
   logs, retries, and reproducibility.
4. **Add evaluation and approval** — learn model promotion, quality gates, lineage, and
   human control over production changes.
5. **Serve predictions** — learn model loading, readiness, request latency, metrics, and
   the difference between batch training and online inference.
6. **Harden the platform** — learn rollback, failure injection, authorization, network
   isolation, PII-safe telemetry, and operational runbooks.

## Initial success criteria

The MVP is complete when:

- A clean checkout can create a kind cluster and deploy Ledger ML using documented steps.
- Submitting a `RiskModel` creates a training workload without manual pod changes.
- A failed training job is observable and does not create a falsely healthy deployment.
- A successful run produces an immutable model version and evaluation result.
- Deployment is blocked until approval is recorded.
- The inference endpoint returns a prediction containing the model version.
- Prometheus exposes request count, latency, errors, and current model state.
- Re-running reconciliation does not duplicate jobs, deployments, or model versions.
- A documented failure test demonstrates rollback to the previous approved model.
