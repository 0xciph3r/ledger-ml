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

## Current implementation status (Milestone 7: governed preparation and training stages)

Implemented in this milestone:

- Go module and controller-runtime manager bootstrap
- Versioned API type: `ledger.ledgerml.io/v1alpha1`, kind `RiskModel`
- `RiskModelSpec` fields for:
  - task
  - training image
  - dataset reference
  - optional dataset preparation image, curated dataset output, and resources
  - optional evaluation image, resource policy, and measurable quality gates
  - model artifact/output reference
  - immutable lineage identity (`trainingImageDigest`, source/prepared dataset versions,
    dataset/output versions, config digest)
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
  - immutable promotion record reference and timestamp
  - reason/message
- Idempotent reconciler that creates a single owned preparation Job and training Job
  per `RiskModel` when preparation is enabled, and does not recreate either on every
  reconcile.
- Training is blocked until preparation succeeds; the training Job consumes the
  curated dataset reference and prepared dataset version rather than the raw source.
- Approval is blocked until the optional evaluation Job succeeds. The evaluator
  receives minimum recall, minimum PR-AUC, and maximum false-negative gates and
  must exit successfully only when the artifact passes them.
- Once approved, an immutable ConfigMap promotion record is created with the exact
  artifact reference and lineage hash. Serving rollout is intentionally separate.
- When `serving.enabled=true`, the operator creates an internal ClusterIP Service
  and shadow Deployment using the approved artifact. No ingress, external route, or
  production traffic switch is created.
- The reference CPU inference service is in `serving/`. It exposes `/healthz`,
  `/readyz`, `/predict`, and Prometheus-compatible `/metrics`, and returns the
  model version and lineage hash with every prediction.
- With `serving.mode=Canary`, the operator creates a Gateway API `HTTPRoute` that
  references an existing stable Service and the promoted candidate Service. Traffic
  is expressed in basis points and defaults to zero candidate traffic.
- Automated canary progression is opt-in with `serving.canaryProgression.enabled=true`.
  The supported progression model is `prometheus_query` and the fail-closed rollback
  behavior is `zero_candidate`. Configure strictly increasing `trafficStepsBPS`, a
  positive `observationWindow`, a Prometheus instant-query `url` and `query`, and
  the maximum healthy `maxErrorRateBPS` (for example, `100` means 1%):

  ```yaml
  serving:
    enabled: true
    mode: Canary
    canaryProgression:
      enabled: true
      progressionModel: prometheus_query
      rollbackBehavior: zero_candidate
      trafficStepsBPS: [100, 1000, 10000]
      observationWindow: 10m
      prometheus:
        url: http://prometheus.monitoring.svc/api/v1/query
        query: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))
      maxErrorRateBPS: 100
  ```

  The query must return one Prometheus scalar/vector value containing an error-rate
  ratio from 0 to 1. A failed query, malformed result, or threshold breach sets
  candidate traffic to zero, records a warning/evidence failure, and pauses
  progression. The controller advances only after the full observation window and
  a healthy query. Status exposes `currentStep`, `currentWeightBPS`,
  `lastEvaluation`, and `progressionState`.
- Setting annotation `ledger.ledgerml.io/rollback-canary: "true"` forces the route
  to 100% stable and 0% candidate traffic without changing model lineage. This is
  an explicit, auditable rollback control and takes precedence over automated
  progression.
- Policy gate before Job creation. Unsafe resource declarations are rejected with clear status/events.
- Status mapping from Job + governance state to phases:
  `Rejected`, `PreparationPending`, `PreparationRunning`, `PreparationSucceeded`,
  `PreparationFailed`, `TrainingPending`, `TrainingRunning`, `TrainingSucceeded`,
  `TrainingFailed`, `EvaluationPending`, `EvaluationRunning`,
  `EvaluationSucceeded`, `EvaluationFailed`, `AwaitingApproval`, `Approved`.
- Structured audit evidence + Kubernetes Events for key decisions:
  accepted/rejected, job created, training observed/failed, approval observed.
- Real `training/` Python project with deterministic synthetic fraud data generation,
  scikit-learn preprocessing + logistic regression training, immutable artifact output,
  and machine-readable evaluation/lineage JSON.
- Sanitized double-entry ledger snapshot adapter with strict accounting validation,
  proxy-label generation, leakage guardrails, and stable transaction-level feature extraction.
- Deterministic dataset preparation command that writes separate feature/label files
  and a content-addressed manifest with source lineage, point-in-time cutoff, quality
  counts, schema versions, and double-entry validation evidence.
- Executable evaluator workload in `training/Dockerfile.evaluator` that validates
  `evaluation-lineage.json` and exits nonzero when a quality gate fails.
- Focused unit tests for API/controller behavior plus trainer determinism, imbalance,
  required features, metrics output, missing environment validation, and snapshot invariants.

### Current training workload contract

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

When preparation is enabled, the preparation Job additionally receives:

- `LEDGERML_PREPARATION_OUTPUT_KIND`
- `LEDGERML_PREPARATION_OUTPUT_NAME`
- `LEDGERML_PREPARATION_OUTPUT_PATH`
- `LEDGERML_PREPARED_DATASET_VERSION`

The preparation Job receives the raw `LEDGERML_DATASET_*` values. The training Job
receives the curated preparation output and `LEDGERML_PREPARED_DATASET_VERSION`.
Preparation is disabled by default to preserve the synthetic MVP path.

When evaluation is enabled, the evaluator Job additionally receives:

- `LEDGERML_EVALUATION_PATH`
- `LEDGERML_EVALUATION_MIN_RECALL`
- `LEDGERML_EVALUATION_MIN_PR_AUC`
- `LEDGERML_EVALUATION_MAX_FALSE_NEGATIVES`

The evaluator reads the trainer's `evaluation-lineage.json`. Missing or malformed
metrics fail closed, and a threshold violation exits nonzero so Kubernetes records
the evaluation Job as failed.

Evaluation thresholds are represented as integer basis points in the Kubernetes API
to avoid floating-point CRD portability problems:

- `minRecallBPS: 8000` means recall must be at least `0.80`.
- `minPRAUCBPS: 3500` means PR-AUC must be at least `0.35`.
- `maxFalseNegatives: 7` caps false negatives in the evaluation report.

Zero disables an individual threshold. A failed evaluator Job is a failed quality
gate, not an approval request.

Dataset selection is now controlled by `LEDGERML_DATASET_KIND`:

- `LocalPath` (existing synthetic path)
- `DoubleEntryLedgerSnapshot` (sanitized snapshot adapter)

The controller intentionally does **not** override container `command`/`args` in this
milestone; the training image entrypoint defines execution behavior.

The trainer now uses this contract to generate synthetic transactions, train a real
fraud classifier, and emit immutable outputs.

## CPU inference service

The serving container loads the immutable `model.joblib` artifact using:

- `LEDGERML_MODEL_PATH`
- `LEDGERML_OUTPUT_ARTIFACT_VERSION`
- `LEDGERML_LINEAGE_HASH`

`POST /predict` accepts a JSON object containing the exact trained feature set:

```json
{
  "features": {
    "amount": 125.0,
    "hour": 13.0,
    "merchant_risk": 0.2,
    "distance_km": 4.0,
    "tx_count_1h": 1.0,
    "tx_count_24h": 3.0,
    "device_risk": 0.1
  }
}
```

The service rejects missing, unknown, boolean, non-numeric, and non-finite feature
values. It does not log raw requests. The initial implementation is intentionally
CPU-oriented; GPU serving will be justified later with measured latency and cost data.

### Local PVC artifact contract

The current runnable artifact adapter uses a Kubernetes `PersistentVolumeClaim`.
Training, evaluation, and serving workloads mount `spec.outputRef.name` at
`/mnt/model-artifacts`. Training writes the versioned model and evaluation evidence
under `spec.outputRef.path`; evaluation reads that evidence from the same mount; and
serving loads `model.joblib` from the promoted artifact version. For `ObjectStore`,
the same paths are immutable S3 keys and workloads use the S3-compatible adapter
instead of a volume mount.

The object-store contract uses `spec.outputRef.name` as the bucket and builds keys as
`<outputRef.path>/<artifactVersion>/<filename>`. Workloads use ambient AWS credentials
(IRSA, workload identity, node credentials, or injected Kubernetes Secret-backed
environment variables). S3-compatible endpoints can be supplied with
`LEDGERML_S3_ENDPOINT_URL` or `AWS_ENDPOINT_URL`; AWS S3 uses the default region and
endpoint behavior.

## Drift detection foundation

The `monitoring/` package establishes a training baseline and compares later feature
windows against it. The first detector covers:

- Missing-column and missing-value-rate changes.
- Numeric feature distribution changes using PSI-style comparisons.
- Model and dataset version identity in every report.

Example:

```bash
PYTHONPATH=monitoring python monitoring/build_baseline.py \
  --features curated/features.csv \
  --model-version fraud-model-v1 \
  --dataset-version curated-dataset-v1 \
  --output baselines/fraud-model-v1.json

PYTHONPATH=monitoring python monitoring/detect_drift.py \
  --baseline baselines/fraud-model-v1.json \
  --current-features windows/current.csv \
  --output reports/drift.json
```

The operator can now schedule the detector as an owned Kubernetes CronJob through
`spec.driftMonitoring`. It passes the immutable model version, baseline/current data
references, and thresholds into each run. The detector report is consumed from a
configured ConfigMap reference and reflected in `RiskModel.status` and governance
evidence. Drift signals investigation or candidate retraining; it does not
automatically promote a new model.

## Outcome monitoring

The `monitoring/ledgerml_monitoring/outcomes.py` module evaluates delayed operational
outcomes against versioned predictions:

```bash
PYTHONPATH=monitoring python monitoring/evaluate_outcomes.py \
  --predictions windows/predictions.csv \
  --outcomes windows/outcomes.csv \
  --model-version fraud-model-v1 \
  --output reports/outcomes.json
```

The join is keyed by `transaction_id`, requires one prediction and at most one
outcome per transaction, and reports coverage, precision, recall, and a confusion
matrix. Unmatched predictions remain visible as delayed-label coverage rather than
being silently treated as negative outcomes. Outcome fields are monitoring labels
only and never become inference features.

An optional `spec.outcomeMonitoring` report reference lets the operator consume the
JSON report from a ConfigMap. It validates the report model version, publishes
coverage and recall in basis points through `RiskModel.status`, and sets an
`OutcomeQuality` condition. Minimum coverage and recall thresholds are observational
quality policies; a breach records evidence but does not automatically retrain,
rollback, or promote a model.

## Double-entry ledger snapshot adapter (sanitized local fixture only)

Ledger ML now supports a local, sanitized snapshot that represents common double-entry
ledger schema shapes without copying real records:

- `ledgers.csv`
- `accounts.csv`
- `transactions.csv`
- `entries.csv`
- `reversals.csv` (optional)

Reference format: `training/double_entry_snapshot_format.md`

Exact required columns:

- `ledgers.csv`: `ledger_id,currency`
- `accounts.csv`: `account_id,ledger_id,account_type,currency`
- `transactions.csv`: `transaction_id,ledger_id,transaction_type,created_at,currency,settlement_state,session_reference,external_reference`
- `entries.csv`: `entry_id,transaction_id,account_id,entry_type,amount_base_units,currency`
- `reversals.csv` (optional): `reversal_id,transaction_id,reversed_at,reason_code`

### Table mapping and integrity model

- `ledgers` + `accounts` define account and currency context.
- `transactions` provides event headers and timestamps.
- `entries` provides debit/credit postings in **integer base units**.
- `reversals` and settlement outcomes provide operational proxy labels.

Double-entry validations enforced before feature extraction:

- at least two entries per transaction
- positive integer base-unit amounts (no decimal money math)
- debit total equals credit total
- single currency per transaction and currency consistency across ledger/account/entry

Invalid snapshots fail fast with explicit errors.

### Dataset preparation boundary

The snapshot adapter is the source-validation boundary. The preparation command
turns a validated snapshot into a curated dataset for a separate training stage:

```bash
PYTHONPATH=training python3 training/prepare_dataset.py \
  --snapshot training/tests/fixtures/double_entry_snapshot \
  --output training/tests/.artifacts/curated \
  --source-version double-entry-snapshot-v1
```

The output contains:

- `features.csv` with transaction identifiers and leakage-safe feature columns.
- `labels.csv` with labels kept separate from features.
- `dataset-manifest.json` with source digest, point-in-time cutoff, schema versions,
  row counts, validation results, label definition, and curated content digest.

This local contract mirrors the intended AWS boundary: RDS/DMS data lands in raw
storage, Glue performs validation and transformation, and the training Job consumes
only the immutable curated dataset plus its manifest.

### Proxy labels and leakage boundary

- Label is `1` for failed/reversed outcomes or explicit reversal markers.
- This is an **operational risk proxy**, not confirmed fraud ground truth.
- Outcome fields (settlement state, reversal markers, post-transaction status) are
  intentionally excluded from model feature columns.

### Feature contract (stable names)

Snapshot training emits stable transaction-level features including:

- total amount in base units
- entry/debit/credit counts
- debit/credit account-type composition
- unique account count
- prior-account velocity counts (1h and 24h windows using only prior timestamps)
- hour/day/weekend features
- safe presence indicators for session/external references
- transaction-type indicators

## ML teaching model (synthetic fraud only)

### Data-generating assumptions and label definition

The synthetic generator creates transaction rows with these features:

- `amount`
- `hour`
- `merchant_risk`
- `distance_km`
- `tx_count_1h`
- `tx_count_24h`
- `device_risk`

Fraud labels are sampled from a logistic risk function with a low base rate and
higher probability for riskier merchants/devices, high transaction velocity, long
distance, and late-night activity. This produces intentional class imbalance and avoids
real customer or financial data.

### Why accuracy is insufficient

Fraud is rare; a model can score high accuracy by predicting "not fraud" almost always.
Ledger ML therefore records precision, recall, F1, PR-AUC, ROC-AUC, and confusion
matrix counts (`tn`, `fp`, `fn`, `tp`) for honest quality signals.

### Artifact and evaluation contract

For each run, the trainer writes:

- `model.joblib` (preprocessing + classifier pipeline payload)
- `evaluation-lineage.json` with:
  - dataset reference + version
  - configuration digest input and computed digest
  - feature names
  - class balance
  - threshold
  - precision/recall/F1/PR-AUC/ROC-AUC
  - confusion matrix
  - model artifact path/version

Outputs are written under:
`<LEDGERML_OUTPUT_NAME>/<LEDGERML_OUTPUT_PATH>/<LEDGERML_OUTPUT_ARTIFACT_VERSION>/`

### Local training command

```bash
python3 -m venv training/.venv
. training/.venv/bin/activate
pip install -r training/requirements.txt

LEDGERML_TASK=fraud-scoring \
LEDGERML_DATASET_KIND=LocalPath \
LEDGERML_DATASET_NAME=synthetic \
LEDGERML_DATASET_PATH=profiles/default \
LEDGERML_DATASET_VERSION=synthetic-fraud-v1 \
LEDGERML_OUTPUT_KIND=LocalPath \
LEDGERML_OUTPUT_NAME=training/local-output \
LEDGERML_OUTPUT_PATH=runs \
LEDGERML_OUTPUT_ARTIFACT_VERSION=fraud-model-v1 \
LEDGERML_TRAINING_IMAGE_DIGEST=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
LEDGERML_CONFIGURATION_DIGEST=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
LEDGERML_CLASSIFICATION_THRESHOLD=0.15 \
PYTHONPATH=. \
python training/train.py
```

### Local fixture training command (double-entry snapshot path)

Use only sanitized fixture data:

```bash
LEDGERML_TASK=fraud-scoring \
LEDGERML_DATASET_KIND=DoubleEntryLedgerSnapshot \
LEDGERML_DATASET_NAME=double-entry-sanitized-fixture \
LEDGERML_DATASET_PATH=training/tests/fixtures/double_entry_snapshot \
LEDGERML_DATASET_VERSION=double-entry-snapshot-v1 \
LEDGERML_OUTPUT_KIND=LocalPath \
LEDGERML_OUTPUT_NAME=training/local-output \
LEDGERML_OUTPUT_PATH=runs \
LEDGERML_OUTPUT_ARTIFACT_VERSION=double-entry-model-v1 \
LEDGERML_TRAINING_IMAGE_DIGEST=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
LEDGERML_CONFIGURATION_DIGEST=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
PYTHONPATH=. \
python training/train.py
```

### Docker build/run path

```bash
docker build -f training/Dockerfile -t ledger-ml-fraud-trainer:local .
```

Container execution uses the same `LEDGERML_*` contract values as above.

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
5. **Promotion is a separate handoff**: after approval and evaluation, the controller
   creates an immutable promotion record pointing to the exact artifact and lineage.
   This does not deploy serving traffic.

### Evidence boundaries

- Evidence records and Events capture operator decisions, reasons, and timestamps.
- Evidence intentionally excludes raw transaction payloads, labels, or other PII.
- This is operational governance evidence, **not** regulatory certification and **not**
  cryptographic/non-repudiation proof yet.
- A validating admission webhook now enforces the same create/update contract before
  reconciliation. The controller retains its checks as defense in depth. Production
  deployment assets are in `config/webhook`. They use cert-manager to issue a
  namespace-scoped self-signed certificate and inject its CA into the webhook
  configuration; replace the Issuer with an organization-managed issuer when required.
The kustomization also patches the conventional `ledger-ml-controller-manager`
Deployment to mount the generated certificate at controller-runtime's default path.
Install cert-manager before applying these resources, or replace the Issuer and
Certificate resources with the cluster's existing certificate provisioning system.

### Prometheus integration

`config/monitoring` contains Prometheus Operator resources for the serving path.
The serving Service exposes both the HTTP and metrics ports, while the
`ServiceMonitor` scrapes `/metrics` every 30 seconds. The included
`PrometheusRule` alerts when a model's inference error ratio exceeds 5% for 10
minutes. Install Prometheus Operator and adjust the namespace/selector labels to
match the cluster's monitoring stack.

### Resource-aware scheduling

`spec.scheduling` makes placement intent explicit without silently rewriting resource
requests. Supported profiles are `cpu-general`, `memory-optimized`, `gpu-training`,
and `latency-sensitive`. The API rejects GPU profiles without a GPU limit and rejects
latency-sensitive profiles unless serving is enabled. Optional node selectors,
tolerations, and priority classes are copied consistently to training, preparation,
evaluation, drift, and serving Pods. Resource bounds remain the safety control; queueing
and utilization-based recommendations are a later integration point.

### Batch queueing

Set `spec.scheduling.queueName` to opt training, preparation, evaluation, and drift
Jobs into a Kueue `LocalQueue`. Serving Deployments are never queued. The
`config/kueue` examples define a CPU/memory/GPU `ClusterQueue` and a default-namespace
`LocalQueue`; install Kueue first and tune quotas, flavors, and namespace selectors for
the cluster. Leaving `queueName` empty preserves direct Kubernetes scheduling.

Not implemented yet (later milestones): continuous retraining, real external data
integrations, feature store, GPU serving, or cryptographic attestation.

This milestone is a reproducible teaching model and governance baseline, not production
fraud detection or regulatory certification.

No real ledger records are committed, uploaded, or required for this repository.

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
make training-test # run trainer unit tests (requires Python deps)
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
- **Double-entry integrity**: accounting constraints become ML data-quality guarantees.
- **Leakage prevention**: proxy-label outcome fields are excluded from model features.
- **Temporal construction**: velocity features depend only on prior events.

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
