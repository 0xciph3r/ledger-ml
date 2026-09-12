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

### Included

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
