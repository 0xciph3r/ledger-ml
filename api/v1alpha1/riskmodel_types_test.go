package v1alpha1

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func validRiskModelSpec() RiskModelSpec {
	return RiskModelSpec{
		Task:          FraudScoringTask,
		TrainingImage: "ghcr.io/ledger-ml/trainer:latest",
		DatasetRef: DatasetReference{
			Kind: "ConfigMap",
			Name: "fraud-training-data",
			Path: "data/transactions.csv",
		},
		OutputRef: ArtifactReference{
			Kind: "PersistentVolumeClaim",
			Name: "model-artifacts",
			Path: "fraud/v1",
		},
		Lineage: RiskModelLineage{
			TrainingImageDigest:   "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			DatasetVersion:        "dataset-v1",
			OutputArtifactVersion: "model-v1",
			ConfigurationDigest:   "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		},
	}
}

func TestRiskModelDefault(t *testing.T) {
	model := &RiskModel{}
	model.Default()

	if model.Spec.Task != FraudScoringTask {
		t.Fatalf("expected task default %q, got %q", FraudScoringTask, model.Spec.Task)
	}
	if model.Spec.DatasetRef.Kind != "ConfigMap" {
		t.Fatalf("expected datasetRef.kind default ConfigMap, got %q", model.Spec.DatasetRef.Kind)
	}
	if model.Spec.OutputRef.Kind != "PersistentVolumeClaim" {
		t.Fatalf("expected outputRef.kind default PersistentVolumeClaim, got %q", model.Spec.OutputRef.Kind)
	}
	if model.Spec.Policy.ResourceBounds.MaxCPU != "2" {
		t.Fatalf("expected policy.resourceBounds.maxCPU default 2, got %q", model.Spec.Policy.ResourceBounds.MaxCPU)
	}
	if model.Spec.Policy.ResourceBounds.MaxMemory != "4Gi" {
		t.Fatalf("expected policy.resourceBounds.maxMemory default 4Gi, got %q", model.Spec.Policy.ResourceBounds.MaxMemory)
	}
	if model.Spec.Policy.ResourceBounds.MaxGPU != "0" {
		t.Fatalf("expected policy.resourceBounds.maxGPU default 0, got %q", model.Spec.Policy.ResourceBounds.MaxGPU)
	}
	if model.Spec.Policy.Approval.Required == nil || !*model.Spec.Policy.Approval.Required {
		t.Fatalf("expected policy.approval.required default true")
	}
	if model.Spec.Policy.Approval.MinimumApprovals != 1 {
		t.Fatalf("expected policy.approval.minimumApprovals default 1, got %d", model.Spec.Policy.Approval.MinimumApprovals)
	}
}

func TestRiskModelValidateCreate(t *testing.T) {
	model := &RiskModel{}
	model.Default()
	if err := model.ValidateCreate(); err == nil {
		t.Fatalf("expected validation error for empty required fields")
	}

	model.Spec = validRiskModelSpec()
	model.Default()

	if err := model.ValidateCreate(); err != nil {
		t.Fatalf("expected valid model, got validation error: %v", err)
	}
}

func TestRiskModelValidateServing(t *testing.T) {
	model := &RiskModel{
		Spec: validRiskModelSpec(),
	}
	model.Spec.Serving.Enabled = true
	model.Default()

	if err := model.ValidateCreate(); err == nil {
		t.Fatalf("expected validation error when serving is enabled without serving.image")
	}

	model.Spec.Serving.Image = "ghcr.io/ledger-ml/serving:latest"
	if err := model.ValidateCreate(); err != nil {
		t.Fatalf("expected valid model when serving image is present, got: %v", err)
	}
}

func TestRiskModelServingArtifactKindsAreTableDriven(t *testing.T) {
	cases := []struct {
		name       string
		outputKind string
		wantError  bool
	}{
		{name: "PVC is supported locally", outputKind: "PersistentVolumeClaim", wantError: false},
		{name: "object store requires adapter", outputKind: "ObjectStore", wantError: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := &RiskModel{Spec: validRiskModelSpec()}
			model.Spec.Serving.Enabled = true
			model.Spec.Serving.Image = "ghcr.io/ledger-ml/serving:latest"
			model.Spec.OutputRef.Kind = tc.outputKind
			model.Default()
			err := model.ValidateCreate()
			if (err != nil) != tc.wantError {
				t.Fatalf("expected validation error=%v, got %v", tc.wantError, err)
			}
		})
	}
}

func TestRiskModelValidateUpdateRejectsImmutableLineage(t *testing.T) {
	oldModel := &RiskModel{
		ObjectMeta: metav1.ObjectMeta{Name: "model-a"},
		Spec:       validRiskModelSpec(),
	}
	oldModel.Default()

	newModel := oldModel.DeepCopy()
	newModel.Spec.TrainingImage = "ghcr.io/ledger-ml/trainer:new-tag"

	if err := newModel.ValidateUpdate(oldModel); err == nil {
		t.Fatalf("expected immutable update validation error")
	}
}

func TestRiskModelValidateUpdateAllowsApprovalMutation(t *testing.T) {
	oldModel := &RiskModel{
		ObjectMeta: metav1.ObjectMeta{Name: "model-a"},
		Spec:       validRiskModelSpec(),
	}
	oldModel.Default()

	newModel := oldModel.DeepCopy()
	newModel.Spec.Approvals = []ModelApproval{
		{
			Approver:   "risk-owner",
			ApprovedAt: metav1.Now(),
			Reference:  "CHG-123",
		},
	}

	if err := newModel.ValidateUpdate(oldModel); err != nil {
		t.Fatalf("expected approval-only update to be valid, got: %v", err)
	}
}
