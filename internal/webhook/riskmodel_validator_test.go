package webhook

import (
	"context"
	"testing"

	ledgerv1alpha1 "github.com/ledger-ml/ledger-ml/api/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestRiskModelValidatorRejectsImmutableUpdatesTableDriven(t *testing.T) {
	cases := []struct {
		name      string
		mutate    func(*ledgerv1alpha1.RiskModel)
		wantError bool
	}{
		{
			name: "lineage mutation",
			mutate: func(model *ledgerv1alpha1.RiskModel) {
				model.Spec.Lineage.OutputArtifactVersion = "model-v2"
			},
			wantError: true,
		},
		{
			name: "approval mutation",
			mutate: func(model *ledgerv1alpha1.RiskModel) {
				model.Spec.Approvals = []ledgerv1alpha1.ModelApproval{{Approver: "risk-team", ApprovedAt: metav1.Now()}}
			},
			wantError: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			oldModel := validRiskModel()
			newModel := oldModel.DeepCopy()
			tc.mutate(newModel)
			_, err := (RiskModelValidator{}).ValidateUpdate(context.Background(), newModel, oldModel)
			if (err != nil) != tc.wantError {
				t.Fatalf("expected error=%v, got %v", tc.wantError, err)
			}
		})
	}
}

func validRiskModel() *ledgerv1alpha1.RiskModel {
	model := &ledgerv1alpha1.RiskModel{
		ObjectMeta: metav1.ObjectMeta{Name: "fraud-model", Namespace: "default"},
		Spec: ledgerv1alpha1.RiskModelSpec{
			Task:          ledgerv1alpha1.FraudScoringTask,
			TrainingImage: "trainer@sha256:" + "a" + "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			DatasetRef: ledgerv1alpha1.DatasetReference{
				Kind: "ConfigMap", Name: "dataset", Path: "dataset.csv",
			},
			OutputRef: ledgerv1alpha1.ArtifactReference{
				Kind: "PersistentVolumeClaim", Name: "artifacts", Path: "fraud",
			},
			Lineage: ledgerv1alpha1.RiskModelLineage{
				TrainingImageDigest:   "sha256:" + "a" + "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
				DatasetVersion:        "dataset-v1",
				OutputArtifactVersion: "model-v1",
				ConfigurationDigest:   "sha256:" + "b" + "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
			},
		},
	}
	model.Default()
	return model
}
