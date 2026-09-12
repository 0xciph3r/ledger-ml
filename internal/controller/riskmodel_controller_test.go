package controller

import (
	"context"
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	ledgerv1alpha1 "github.com/ledger-ml/ledger-ml/api/v1alpha1"
)

func newScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("add client-go scheme: %v", err)
	}
	if err := ledgerv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add Ledger ML scheme: %v", err)
	}
	return scheme
}

func newReconciler(t *testing.T, objs ...client.Object) (*RiskModelReconciler, client.Client) {
	t.Helper()
	scheme := newScheme(t)

	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&ledgerv1alpha1.RiskModel{}).
		WithObjects(objs...).
		Build()

	return &RiskModelReconciler{
		Client: c,
		Scheme: scheme,
	}, c
}

func validRiskModel() *ledgerv1alpha1.RiskModel {
	return &ledgerv1alpha1.RiskModel{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "fraud-model",
			Namespace:  "default",
			Generation: 3,
		},
		Spec: ledgerv1alpha1.RiskModelSpec{
			Task:          ledgerv1alpha1.FraudScoringTask,
			TrainingImage: "ghcr.io/ledger-ml/trainer:latest",
			DatasetRef: ledgerv1alpha1.DatasetReference{
				Kind: "ConfigMap",
				Name: "fraud-training-data",
				Path: "data/transactions.csv",
			},
			OutputRef: ledgerv1alpha1.ArtifactReference{
				Kind: "PersistentVolumeClaim",
				Name: "model-artifacts",
				Path: "fraud/v1",
			},
		},
	}
}

func TestBuildTrainingJobUsesWorkloadContract(t *testing.T) {
	model := validRiskModel()
	model.Spec.Resources = corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("500m"),
			corev1.ResourceMemory: resource.MustParse("512Mi"),
			gpuResourceName:       resource.MustParse("1"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("1"),
			corev1.ResourceMemory: resource.MustParse("1Gi"),
			gpuResourceName:       resource.MustParse("1"),
		},
	}

	job := buildTrainingJob(model)
	container := job.Spec.Template.Spec.Containers[0]

	if container.Image != model.Spec.TrainingImage {
		t.Fatalf("expected image %q, got %q", model.Spec.TrainingImage, container.Image)
	}
	if job.Name != "fraud-model-train" {
		t.Fatalf("expected deterministic job name fraud-model-train, got %q", job.Name)
	}

	env := map[string]string{}
	for _, e := range container.Env {
		env[e.Name] = e.Value
	}

	if env[envTask] != "fraud-scoring" {
		t.Fatalf("expected %s env var to be fraud-scoring, got %q", envTask, env[envTask])
	}
	if env[envDatasetName] != "fraud-training-data" || env[envDatasetPath] != "data/transactions.csv" {
		t.Fatalf("dataset env contract not set correctly: %#v", env)
	}
	if env[envOutputName] != "model-artifacts" || env[envOutputPath] != "fraud/v1" {
		t.Fatalf("output env contract not set correctly: %#v", env)
	}

	if _, ok := container.Resources.Requests[gpuResourceName]; ok {
		t.Fatalf("gpu should not be copied into requests")
	}
	if got := container.Resources.Limits[gpuResourceName]; got.String() != "1" {
		t.Fatalf("expected gpu limit to be copied, got %q", got.String())
	}
}

func TestTrainingStatusFromJob(t *testing.T) {
	testCases := []struct {
		name           string
		job            batchv1.Job
		expectedPhase  ledgerv1alpha1.RiskModelPhase
		expectedReady  metav1.ConditionStatus
		expectedTrain  metav1.ConditionStatus
		expectedReason string
	}{
		{
			name:           "pending",
			job:            batchv1.Job{},
			expectedPhase:  ledgerv1alpha1.RiskModelPhaseTrainingPending,
			expectedReady:  metav1.ConditionFalse,
			expectedTrain:  metav1.ConditionFalse,
			expectedReason: "TrainingPending",
		},
		{
			name: "running",
			job: batchv1.Job{
				Status: batchv1.JobStatus{Active: 1},
			},
			expectedPhase:  ledgerv1alpha1.RiskModelPhaseTrainingRunning,
			expectedReady:  metav1.ConditionFalse,
			expectedTrain:  metav1.ConditionFalse,
			expectedReason: "TrainingRunning",
		},
		{
			name: "succeeded",
			job: batchv1.Job{
				Status: batchv1.JobStatus{
					Conditions: []batchv1.JobCondition{
						{Type: batchv1.JobComplete, Status: corev1.ConditionTrue},
					},
				},
			},
			expectedPhase:  ledgerv1alpha1.RiskModelPhaseTrainingSucceeded,
			expectedReady:  metav1.ConditionTrue,
			expectedTrain:  metav1.ConditionTrue,
			expectedReason: "TrainingSucceeded",
		},
		{
			name: "failed",
			job: batchv1.Job{
				Status: batchv1.JobStatus{
					Conditions: []batchv1.JobCondition{
						{Type: batchv1.JobFailed, Status: corev1.ConditionTrue, Reason: "BackoffLimitExceeded"},
					},
				},
			},
			expectedPhase:  ledgerv1alpha1.RiskModelPhaseTrainingFailed,
			expectedReady:  metav1.ConditionFalse,
			expectedTrain:  metav1.ConditionFalse,
			expectedReason: "BackoffLimitExceeded",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			phase, trainingStatus, readyStatus, reason, _ := trainingStatusFromJob(&tc.job)
			if phase != tc.expectedPhase {
				t.Fatalf("expected phase %q, got %q", tc.expectedPhase, phase)
			}
			if trainingStatus != tc.expectedTrain {
				t.Fatalf("expected training condition %q, got %q", tc.expectedTrain, trainingStatus)
			}
			if readyStatus != tc.expectedReady {
				t.Fatalf("expected ready condition %q, got %q", tc.expectedReady, readyStatus)
			}
			if reason != tc.expectedReason {
				t.Fatalf("expected reason %q, got %q", tc.expectedReason, reason)
			}
		})
	}
}

func TestRiskModelReconcileCreatesTrainingJobAndSetsPendingStatus(t *testing.T) {
	model := validRiskModel()
	reconciler, c := newReconciler(t, model)

	if _, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: riskModelName(model.Namespace, model.Name)}); err != nil {
		t.Fatalf("reconcile returned error: %v", err)
	}

	var job batchv1.Job
	if err := c.Get(context.Background(), riskModelName(model.Namespace, trainingJobName(model.Name)), &job); err != nil {
		t.Fatalf("expected training job to be created: %v", err)
	}
	if !metav1.IsControlledBy(&job, model) {
		t.Fatalf("expected job to be owned by RiskModel")
	}

	var got ledgerv1alpha1.RiskModel
	if err := c.Get(context.Background(), riskModelName(model.Namespace, model.Name), &got); err != nil {
		t.Fatalf("get reconciled model: %v", err)
	}

	if got.Status.Phase != ledgerv1alpha1.RiskModelPhaseTrainingPending {
		t.Fatalf("expected phase %q, got %q", ledgerv1alpha1.RiskModelPhaseTrainingPending, got.Status.Phase)
	}
	if got.Status.ObservedGeneration != model.Generation {
		t.Fatalf("expected observedGeneration %d, got %d", model.Generation, got.Status.ObservedGeneration)
	}
	training := apimeta.FindStatusCondition(got.Status.Conditions, conditionTypeTraining)
	if training == nil || training.Status != metav1.ConditionFalse {
		t.Fatalf("expected Training condition to be false, got %#v", training)
	}
}
