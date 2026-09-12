package controller

import (
	"context"
	"strings"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
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

func newReconciler(t *testing.T, recorder record.EventRecorder, objs ...client.Object) (*RiskModelReconciler, client.Client) {
	t.Helper()
	scheme := newScheme(t)

	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&ledgerv1alpha1.RiskModel{}).
		WithObjects(objs...).
		Build()

	return &RiskModelReconciler{
		Client:   c,
		Scheme:   scheme,
		Recorder: recorder,
	}, c
}

func validRiskModel() *ledgerv1alpha1.RiskModel {
	model := &ledgerv1alpha1.RiskModel{
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
			Lineage: ledgerv1alpha1.RiskModelLineage{
				TrainingImageDigest:   "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
				DatasetVersion:        "dataset-v1",
				OutputArtifactVersion: "model-v1",
				ConfigurationDigest:   "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
			},
		},
	}
	model.Default()
	return model
}

func TestBuildTrainingJobUsesLineageAndContract(t *testing.T) {
	model := validRiskModel()
	model.Spec.Resources = corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("500m"),
			corev1.ResourceMemory: resource.MustParse("512Mi"),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse("1"),
			corev1.ResourceMemory: resource.MustParse("1Gi"),
			gpuResourceName:       resource.MustParse("1"),
		},
	}

	job := buildTrainingJob(model)
	container := job.Spec.Template.Spec.Containers[0]

	if job.Annotations[annotationLineageHash] == "" {
		t.Fatalf("expected lineage hash annotation on job")
	}
	if container.Image != model.Spec.TrainingImage {
		t.Fatalf("expected image %q, got %q", model.Spec.TrainingImage, container.Image)
	}

	env := map[string]string{}
	for _, e := range container.Env {
		env[e.Name] = e.Value
	}
	if env[envTrainingImageDigest] != model.Spec.Lineage.TrainingImageDigest {
		t.Fatalf("missing immutable training digest env, got %q", env[envTrainingImageDigest])
	}
	if env[envDatasetVersion] != model.Spec.Lineage.DatasetVersion {
		t.Fatalf("missing dataset version env, got %q", env[envDatasetVersion])
	}
	if env[envOutputVersion] != model.Spec.Lineage.OutputArtifactVersion {
		t.Fatalf("missing output version env, got %q", env[envOutputVersion])
	}
	if env[envConfigurationDigest] != model.Spec.Lineage.ConfigurationDigest {
		t.Fatalf("missing configuration digest env, got %q", env[envConfigurationDigest])
	}
}

func TestRiskModelReconcileRejectsUnsafeResourcesBeforeJobCreation(t *testing.T) {
	model := validRiskModel()
	model.Spec.Resources.Requests = corev1.ResourceList{
		corev1.ResourceCPU: resource.MustParse("3"),
	}
	recorder := record.NewFakeRecorder(10)

	reconciler, c := newReconciler(t, recorder, model)
	if _, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: riskModelName(model.Namespace, model.Name)}); err != nil {
		t.Fatalf("reconcile returned error: %v", err)
	}

	var job batchv1.Job
	if err := c.Get(context.Background(), riskModelName(model.Namespace, trainingJobName(model.Name)), &job); err == nil {
		t.Fatalf("expected no training job to be created for policy-violating spec")
	}

	var got ledgerv1alpha1.RiskModel
	if err := c.Get(context.Background(), riskModelName(model.Namespace, model.Name), &got); err != nil {
		t.Fatalf("get reconciled model: %v", err)
	}
	if got.Status.Phase != ledgerv1alpha1.RiskModelPhaseRejected {
		t.Fatalf("expected phase %q, got %q", ledgerv1alpha1.RiskModelPhaseRejected, got.Status.Phase)
	}
	if got.Status.Reason != "ResourcePolicyViolation" {
		t.Fatalf("expected reason ResourcePolicyViolation, got %q", got.Status.Reason)
	}
	if len(got.Status.Evidence) == 0 || got.Status.Evidence[len(got.Status.Evidence)-1].Decision != "Rejected" {
		t.Fatalf("expected rejection evidence to be recorded")
	}

	select {
	case event := <-recorder.Events:
		if !strings.Contains(event, "ResourcePolicyViolation") {
			t.Fatalf("expected policy violation event, got %q", event)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("expected policy rejection event to be emitted")
	}
}

func TestRiskModelReconcileAwaitsApprovalAfterSuccessfulTraining(t *testing.T) {
	model := validRiskModel()
	job := buildTrainingJob(model)
	if err := ctrl.SetControllerReference(model, job, newScheme(t)); err != nil {
		t.Fatalf("set owner reference: %v", err)
	}
	job.Status.Conditions = []batchv1.JobCondition{
		{Type: batchv1.JobComplete, Status: corev1.ConditionTrue},
	}

	reconciler, c := newReconciler(t, nil, model, job)
	if _, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: riskModelName(model.Namespace, model.Name)}); err != nil {
		t.Fatalf("reconcile returned error: %v", err)
	}

	var got ledgerv1alpha1.RiskModel
	if err := c.Get(context.Background(), riskModelName(model.Namespace, model.Name), &got); err != nil {
		t.Fatalf("get reconciled model: %v", err)
	}
	if got.Status.Phase != ledgerv1alpha1.RiskModelPhaseAwaitingApproval {
		t.Fatalf("expected phase %q, got %q", ledgerv1alpha1.RiskModelPhaseAwaitingApproval, got.Status.Phase)
	}
	approved := apimeta.FindStatusCondition(got.Status.Conditions, conditionTypeApproved)
	if approved == nil || approved.Status != metav1.ConditionFalse {
		t.Fatalf("expected Approved condition false, got %#v", approved)
	}
	ready := apimeta.FindStatusCondition(got.Status.Conditions, conditionTypeReady)
	if ready == nil || ready.Status != metav1.ConditionFalse {
		t.Fatalf("expected Ready condition false, got %#v", ready)
	}
}

func TestRiskModelReconcileObservesApprovalButNotProductionReady(t *testing.T) {
	model := validRiskModel()
	model.Spec.Policy.Approval.AllowedApprovers = []string{"risk-owner"}
	model.Spec.Approvals = []ledgerv1alpha1.ModelApproval{
		{
			Approver:   "risk-owner",
			ApprovedAt: metav1.Now(),
			Reference:  "CHG-456",
		},
	}

	job := buildTrainingJob(model)
	if err := ctrl.SetControllerReference(model, job, newScheme(t)); err != nil {
		t.Fatalf("set owner reference: %v", err)
	}
	job.Status.Conditions = []batchv1.JobCondition{
		{Type: batchv1.JobComplete, Status: corev1.ConditionTrue},
	}

	reconciler, c := newReconciler(t, nil, model, job)
	if _, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: riskModelName(model.Namespace, model.Name)}); err != nil {
		t.Fatalf("reconcile returned error: %v", err)
	}

	var got ledgerv1alpha1.RiskModel
	if err := c.Get(context.Background(), riskModelName(model.Namespace, model.Name), &got); err != nil {
		t.Fatalf("get reconciled model: %v", err)
	}
	if got.Status.Phase != ledgerv1alpha1.RiskModelPhaseApproved {
		t.Fatalf("expected phase %q, got %q", ledgerv1alpha1.RiskModelPhaseApproved, got.Status.Phase)
	}
	if got.Status.Reason != "ApprovalObserved" {
		t.Fatalf("expected reason ApprovalObserved, got %q", got.Status.Reason)
	}
	ready := apimeta.FindStatusCondition(got.Status.Conditions, conditionTypeReady)
	if ready == nil || ready.Reason != "PromotionNotImplemented" || ready.Status != metav1.ConditionFalse {
		t.Fatalf("expected Ready condition false with PromotionNotImplemented, got %#v", ready)
	}
	if len(got.Status.ApprovedBy) != 1 || got.Status.ApprovedBy[0] != "risk-owner" {
		t.Fatalf("expected approvedBy to list risk-owner, got %#v", got.Status.ApprovedBy)
	}
}

func TestRiskModelReconcileRejectsLineageMismatch(t *testing.T) {
	model := validRiskModel()
	job := buildTrainingJob(model)
	job.Annotations[annotationLineageHash] = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	if err := ctrl.SetControllerReference(model, job, newScheme(t)); err != nil {
		t.Fatalf("set owner reference: %v", err)
	}

	reconciler, c := newReconciler(t, nil, model, job)
	if _, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: riskModelName(model.Namespace, model.Name)}); err != nil {
		t.Fatalf("reconcile returned error: %v", err)
	}

	var got ledgerv1alpha1.RiskModel
	if err := c.Get(context.Background(), riskModelName(model.Namespace, model.Name), &got); err != nil {
		t.Fatalf("get reconciled model: %v", err)
	}
	if got.Status.Phase != ledgerv1alpha1.RiskModelPhaseRejected {
		t.Fatalf("expected phase %q, got %q", ledgerv1alpha1.RiskModelPhaseRejected, got.Status.Phase)
	}
	if got.Status.Reason != "LineageMismatch" {
		t.Fatalf("expected reason LineageMismatch, got %q", got.Status.Reason)
	}
}

func TestRiskModelReconcileRejectsImmutableLineageMutationAfterFirstReconcile(t *testing.T) {
	model := validRiskModel()
	recorder := record.NewFakeRecorder(20)
	reconciler, c := newReconciler(t, recorder, model)

	key := riskModelName(model.Namespace, model.Name)
	if _, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatalf("first reconcile returned error: %v", err)
	}

	var firstModel ledgerv1alpha1.RiskModel
	if err := c.Get(context.Background(), key, &firstModel); err != nil {
		t.Fatalf("get model after first reconcile: %v", err)
	}
	if firstModel.Status.LineageHash == "" {
		t.Fatalf("expected lineage hash to be persisted after first reconcile")
	}

	jobKey := riskModelName(model.Namespace, trainingJobName(model.Name))
	var firstJob batchv1.Job
	if err := c.Get(context.Background(), jobKey, &firstJob); err != nil {
		t.Fatalf("expected training job after first reconcile: %v", err)
	}
	originalJobLineage := firstJob.Annotations[annotationLineageHash]

	mutatedModel := firstModel.DeepCopy()
	mutatedModel.Spec.Lineage.ConfigurationDigest = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	mutatedModel.Generation = firstModel.Generation + 1
	if err := c.Update(context.Background(), mutatedModel); err != nil {
		t.Fatalf("update mutated model spec: %v", err)
	}

	if _, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatalf("second reconcile returned error: %v", err)
	}

	var got ledgerv1alpha1.RiskModel
	if err := c.Get(context.Background(), key, &got); err != nil {
		t.Fatalf("get model after second reconcile: %v", err)
	}
	if got.Status.Phase != ledgerv1alpha1.RiskModelPhaseRejected {
		t.Fatalf("expected phase %q, got %q", ledgerv1alpha1.RiskModelPhaseRejected, got.Status.Phase)
	}
	if got.Status.Reason != "LineageImmutable" {
		t.Fatalf("expected reason LineageImmutable, got %q", got.Status.Reason)
	}
	if !strings.Contains(got.Status.Message, "ImmutableLineage") {
		t.Fatalf("expected ImmutableLineage message, got %q", got.Status.Message)
	}
	if got.Status.LineageHash != firstModel.Status.LineageHash {
		t.Fatalf("expected persisted lineage hash to remain unchanged, old=%q new=%q", firstModel.Status.LineageHash, got.Status.LineageHash)
	}
	if len(got.Status.Evidence) == 0 || got.Status.Evidence[len(got.Status.Evidence)-1].Reason != "ImmutableLineage" {
		t.Fatalf("expected immutable-lineage rejection evidence, got %#v", got.Status.Evidence)
	}

	var secondJob batchv1.Job
	if err := c.Get(context.Background(), jobKey, &secondJob); err != nil {
		t.Fatalf("expected existing training job after second reconcile: %v", err)
	}
	if secondJob.Annotations[annotationLineageHash] != originalJobLineage {
		t.Fatalf("expected existing job lineage annotation unchanged, old=%q new=%q", originalJobLineage, secondJob.Annotations[annotationLineageHash])
	}
}
