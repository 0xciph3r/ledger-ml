package v1alpha1

import (
	"testing"
)

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
	if model.Spec.Serving.Replicas == nil || *model.Spec.Serving.Replicas != 1 {
		t.Fatalf("expected serving.replicas default 1, got %#v", model.Spec.Serving.Replicas)
	}
	if model.Spec.Serving.Port != 8080 {
		t.Fatalf("expected serving.port default 8080, got %d", model.Spec.Serving.Port)
	}
}

func TestRiskModelValidateCreate(t *testing.T) {
	model := &RiskModel{}
	model.Default()
	if err := model.ValidateCreate(); err == nil {
		t.Fatalf("expected validation error for empty required fields")
	}

	model.Spec.TrainingImage = "ghcr.io/ledger-ml/trainer:latest"
	model.Spec.DatasetRef.Name = "fraud-training-data"
	model.Spec.DatasetRef.Path = "data/transactions.csv"
	model.Spec.OutputRef.Name = "model-artifacts"
	model.Spec.OutputRef.Path = "fraud/v1"

	if err := model.ValidateCreate(); err != nil {
		t.Fatalf("expected valid model, got validation error: %v", err)
	}
}

func TestRiskModelValidateServing(t *testing.T) {
	model := &RiskModel{
		Spec: RiskModelSpec{
			Task:          FraudScoringTask,
			TrainingImage: "ghcr.io/ledger-ml/trainer:latest",
			DatasetRef: DatasetReference{
				Name: "fraud-training-data",
				Path: "data/transactions.csv",
			},
			OutputRef: ArtifactReference{
				Name: "model-artifacts",
				Path: "fraud/v1",
			},
			Serving: ServingConfig{
				Enabled: true,
			},
		},
	}
	model.Default()

	if err := model.ValidateCreate(); err == nil {
		t.Fatalf("expected validation error when serving is enabled without serving.image")
	}

	model.Spec.Serving.Image = "ghcr.io/ledger-ml/serving:latest"
	if err := model.ValidateCreate(); err != nil {
		t.Fatalf("expected valid model when serving image is present, got: %v", err)
	}
}
