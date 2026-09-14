package controller

import (
	"context"
	"encoding/json"
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
	if env[envOutputName] != artifactMountPath {
		t.Fatalf("expected PVC-mounted output path %q, got %q", artifactMountPath, env[envOutputName])
	}
	if len(container.VolumeMounts) != 1 || container.VolumeMounts[0].MountPath != artifactMountPath {
		t.Fatalf("expected artifact PVC mount at %q, got %#v", artifactMountPath, container.VolumeMounts)
	}
	if len(job.Spec.Template.Spec.Volumes) != 1 || job.Spec.Template.Spec.Volumes[0].PersistentVolumeClaim.ClaimName != model.Spec.OutputRef.Name {
		t.Fatalf("expected artifact PVC volume %q, got %#v", model.Spec.OutputRef.Name, job.Spec.Template.Spec.Volumes)
	}
}

func TestPreparationJobUsesSourceAndCuratedDatasetContract(t *testing.T) {
	model := validRiskModel()
	model.Spec.Preparation = ledgerv1alpha1.DatasetPreparationSpec{
		Enabled: true,
		Image:   "ghcr.io/ledger-ml/preparer:latest",
		OutputRef: ledgerv1alpha1.DatasetReference{
			Kind: "ObjectStore",
			Name: "curated-datasets",
			Path: "fraud/v1",
		},
	}
	model.Spec.Lineage.PreparedDatasetVersion = "curated-v1"

	job := buildPreparationJob(model)
	container := job.Spec.Template.Spec.Containers[0]
	env := map[string]string{}
	for _, entry := range container.Env {
		env[entry.Name] = entry.Value
	}
	if container.Image != model.Spec.Preparation.Image {
		t.Fatalf("expected preparation image %q, got %q", model.Spec.Preparation.Image, container.Image)
	}
	if env[envDatasetPath] != model.Spec.DatasetRef.Path {
		t.Fatalf("expected source dataset path %q, got %q", model.Spec.DatasetRef.Path, env[envDatasetPath])
	}
	if env[envPreparationOutputPath] != model.Spec.Preparation.OutputRef.Path {
		t.Fatalf("expected curated output path %q, got %q", model.Spec.Preparation.OutputRef.Path, env[envPreparationOutputPath])
	}
	if env[envPreparedDatasetVersion] != model.Spec.Lineage.PreparedDatasetVersion {
		t.Fatalf("expected prepared dataset version %q, got %q", model.Spec.Lineage.PreparedDatasetVersion, env[envPreparedDatasetVersion])
	}
}

func TestEvaluationJobCarriesQualityGatesAndArtifactLineage(t *testing.T) {
	model := validRiskModel()
	model.Spec.Evaluation = ledgerv1alpha1.EvaluationPolicy{
		Enabled:           true,
		Image:             "ghcr.io/ledger-ml/evaluator:latest",
		MinRecallBPS:      8000,
		MinPRAUCBPS:       3500,
		MaxFalseNegatives: 7,
	}

	job := buildEvaluationJob(model)
	container := job.Spec.Template.Spec.Containers[0]
	env := map[string]string{}
	for _, entry := range container.Env {
		env[entry.Name] = entry.Value
	}
	if container.Image != model.Spec.Evaluation.Image {
		t.Fatalf("expected evaluation image %q, got %q", model.Spec.Evaluation.Image, container.Image)
	}
	if env[envOutputVersion] != model.Spec.Lineage.OutputArtifactVersion {
		t.Fatalf("expected artifact version %q, got %q", model.Spec.Lineage.OutputArtifactVersion, env[envOutputVersion])
	}
	expectedEvaluationPath := artifactMountPath + "/" + model.Spec.OutputRef.Path + "/" + model.Spec.Lineage.OutputArtifactVersion + "/evaluation-lineage.json"
	if env[envEvaluationPath] != expectedEvaluationPath {
		t.Fatalf("expected evaluation path %q, got %q", expectedEvaluationPath, env[envEvaluationPath])
	}
	if env[envEvaluationMinRecall] != "0.8" || env[envEvaluationMinPRAUC] != "0.35" || env[envEvaluationMaxFalseNegatives] != "7" {
		t.Fatalf("expected quality gate environment, got %#v", env)
	}
}

func TestArtifactWorkloadContractsAreTableDriven(t *testing.T) {
	cases := []struct {
		name          string
		kind          string
		wantMounts    int
		wantVolumes   int
		wantModelPath string
	}{
		{
			name:          "persistent volume claim",
			kind:          "PersistentVolumeClaim",
			wantMounts:    1,
			wantVolumes:   1,
			wantModelPath: "/mnt/model-artifacts/fraud/v1/model-v1/model.joblib",
		},
		{
			name:          "s3 compatible object store",
			kind:          "ObjectStore",
			wantMounts:    0,
			wantVolumes:   0,
			wantModelPath: "fraud/v1/model-v1/model.joblib",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := validRiskModel()
			model.Spec.OutputRef.Kind = tc.kind
			deployment := buildShadowDeployment(model)
			container := deployment.Spec.Template.Spec.Containers[0]
			if len(container.VolumeMounts) != tc.wantMounts {
				t.Fatalf("expected %d volume mounts, got %#v", tc.wantMounts, container.VolumeMounts)
			}
			if len(deployment.Spec.Template.Spec.Volumes) != tc.wantVolumes {
				t.Fatalf("expected %d volumes, got %#v", tc.wantVolumes, deployment.Spec.Template.Spec.Volumes)
			}
			env := map[string]string{}
			for _, entry := range container.Env {
				env[entry.Name] = entry.Value
			}
			if env[envModelPath] != tc.wantModelPath {
				t.Fatalf("expected model path %q, got %q", tc.wantModelPath, env[envModelPath])
			}
		})
	}
}

func TestShadowServiceExposesHTTPAndMetricsPorts(t *testing.T) {
	model := validRiskModel()
	service := buildShadowService(model)
	if len(service.Spec.Ports) != 2 {
		t.Fatalf("expected HTTP and metrics ports, got %#v", service.Spec.Ports)
	}
	if service.Spec.Ports[0].Name != "http" || service.Spec.Ports[1].Name != "metrics" {
		t.Fatalf("unexpected service ports: %#v", service.Spec.Ports)
	}
}

func TestDriftCronJobUsesVersionedBaselineAndThresholdContract(t *testing.T) {
	cases := []struct {
		name       string
		psiBPS     int32
		missingBPS int32
		psi        string
		missing    string
	}{
		{name: "default thresholds", psiBPS: 2000, missingBPS: 1000, psi: "0.2", missing: "0.1"},
		{name: "strict thresholds", psiBPS: 500, missingBPS: 250, psi: "0.05", missing: "0.025"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := validRiskModel()
			model.Spec.DriftMonitoring = ledgerv1alpha1.DriftMonitoringSpec{
				Enabled:  true,
				Image:    "ghcr.io/ledger-ml/drift-detector:latest",
				Schedule: "*/15 * * * *",
				BaselineRef: ledgerv1alpha1.DatasetReference{
					Kind: "ObjectStore", Name: "baselines", Path: "fraud/model-v1.json",
				},
				CurrentFeaturesRef: ledgerv1alpha1.DatasetReference{
					Kind: "ObjectStore", Name: "feature-windows", Path: "fraud/current.csv",
				},
				PSIThresholdBPS:     tc.psiBPS,
				MissingRateDeltaBPS: tc.missingBPS,
			}
			cronJob := buildDriftCronJob(model)
			env := map[string]string{}
			container := cronJob.Spec.JobTemplate.Spec.Template.Spec.Containers[0]
			for _, entry := range container.Env {
				env[entry.Name] = entry.Value
			}
			if cronJob.Spec.Schedule != "*/15 * * * *" {
				t.Fatalf("expected schedule %q, got %q", "*/15 * * * *", cronJob.Spec.Schedule)
			}
			if env[envDriftModelVersion] != model.Spec.Lineage.OutputArtifactVersion {
				t.Fatalf("expected model version %q, got %q", model.Spec.Lineage.OutputArtifactVersion, env[envDriftModelVersion])
			}
			if env[envDriftPSIThreshold] != tc.psi || env[envDriftMissingRateThreshold] != tc.missing {
				t.Fatalf("expected thresholds %q/%q, got %q/%q", tc.psi, tc.missing, env[envDriftPSIThreshold], env[envDriftMissingRateThreshold])
			}
		})
	}
}

func TestObserveDriftReportIsTableDriven(t *testing.T) {
	cases := []struct {
		name              string
		status            string
		signals           int
		expectError       bool
		expectedCondition metav1.ConditionStatus
		expectedReason    string
	}{
		{name: "within baseline", status: "within_baseline", expectedCondition: metav1.ConditionTrue, expectedReason: "WithinBaseline"},
		{name: "drift detected", status: "drift_detected", signals: 2, expectedCondition: metav1.ConditionFalse, expectedReason: "DriftDetected"},
		{name: "unsupported status", status: "unknown", expectError: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := validRiskModel()
			model.Spec.DriftMonitoring.Enabled = true
			model.Spec.DriftMonitoring.ReportRef = ledgerv1alpha1.DatasetReference{
				Kind: "ConfigMap", Name: "drift-report", Path: "report.json",
			}
			signals := make([]string, tc.signals)
			for i := range signals {
				signals[i] = "signal"
			}
			reportData, err := json.Marshal(map[string]any{
				"model_version": model.Spec.Lineage.OutputArtifactVersion,
				"status":        tc.status,
				"signals":       signals,
			})
			if err != nil {
				t.Fatalf("marshal report: %v", err)
			}
			report := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "drift-report", Namespace: "default"},
				Data:       map[string]string{"report.json": string(reportData)},
			}
			reconciler, _ := newReconciler(t, nil, model, report)
			err = reconciler.observeDriftReport(context.Background(), model)
			if tc.expectError {
				if err == nil {
					t.Fatalf("expected invalid report error")
				}
				return
			}
			if err != nil {
				t.Fatalf("observe report: %v", err)
			}
			condition := apimeta.FindStatusCondition(model.Status.Conditions, conditionTypeDrift)
			if condition == nil || condition.Status != tc.expectedCondition || condition.Reason != tc.expectedReason {
				t.Fatalf("unexpected drift condition: %#v", condition)
			}
			if model.Status.DriftStatus != tc.status {
				t.Fatalf("expected drift status %q, got %q", tc.status, model.Status.DriftStatus)
			}
		})
	}
}

func TestObserveOutcomeReportIsTableDriven(t *testing.T) {
	cases := []struct {
		name           string
		coverage       float64
		recall         float64
		minCoverageBPS int32
		minRecallBPS   int32
		expectedStatus metav1.ConditionStatus
		expectedReason string
	}{
		{"healthy", 0.9, 0.85, 8000, 8000, metav1.ConditionTrue, "OutcomeQualityHealthy"},
		{"low coverage", 0.5, 0.85, 8000, 8000, metav1.ConditionFalse, "OutcomeQualityBelowPolicy"},
		{"low recall", 0.9, 0.4, 8000, 8000, metav1.ConditionFalse, "OutcomeQualityBelowPolicy"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := validRiskModel()
			model.Spec.OutcomeMonitoring = ledgerv1alpha1.OutcomeMonitoringSpec{
				Enabled: true,
				ReportRef: ledgerv1alpha1.DatasetReference{
					Kind: "ConfigMap", Name: "outcome-report", Path: "report.json",
				},
				MinimumCoverageBPS: tc.minCoverageBPS,
				MinimumRecallBPS:   tc.minRecallBPS,
			}
			raw, err := json.Marshal(map[string]any{
				"model_version":         model.Spec.Lineage.OutputArtifactVersion,
				"prediction_count":      100,
				"matched_outcome_count": int(tc.coverage * 100),
				"metrics": map[string]float64{
					"coverage": tc.coverage,
					"recall":   tc.recall,
				},
			})
			if err != nil {
				t.Fatalf("marshal report: %v", err)
			}
			report := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "outcome-report", Namespace: "default"},
				Data:       map[string]string{"report.json": string(raw)},
			}
			reconciler, _ := newReconciler(t, nil, model, report)
			if err := reconciler.observeOutcomeReport(context.Background(), model); err != nil {
				t.Fatalf("observe outcome report: %v", err)
			}
			condition := apimeta.FindStatusCondition(model.Status.Conditions, conditionTypeOutcome)
			if condition == nil || condition.Status != tc.expectedStatus || condition.Reason != tc.expectedReason {
				t.Fatalf("unexpected outcome condition: %#v", condition)
			}
		})
	}
}

func TestRiskModelReconcileBlocksApprovalUntilEvaluationCompletes(t *testing.T) {
	model := validRiskModel()
	required := true
	model.Spec.Policy.Approval.Required = &required
	model.Spec.Policy.Approval.AllowedApprovers = []string{"risk-owner"}
	model.Spec.Approvals = []ledgerv1alpha1.ModelApproval{
		{Approver: "risk-owner", ApprovedAt: metav1.Now()},
	}
	model.Spec.Evaluation = ledgerv1alpha1.EvaluationPolicy{
		Enabled:      true,
		Image:        "ghcr.io/ledger-ml/evaluator:latest",
		MinRecallBPS: 8000,
	}

	trainingJob := buildTrainingJob(model)
	if err := ctrl.SetControllerReference(model, trainingJob, newScheme(t)); err != nil {
		t.Fatalf("set training owner reference: %v", err)
	}
	trainingJob.Status.Conditions = []batchv1.JobCondition{
		{Type: batchv1.JobComplete, Status: corev1.ConditionTrue},
	}
	reconciler, c := newReconciler(t, nil, model, trainingJob)
	key := riskModelName(model.Namespace, model.Name)
	if _, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatalf("first reconcile returned error: %v", err)
	}

	var evaluationJob batchv1.Job
	if err := c.Get(context.Background(), riskModelName(model.Namespace, evaluationJobName(model.Name)), &evaluationJob); err != nil {
		t.Fatalf("expected evaluation job: %v", err)
	}
	var got ledgerv1alpha1.RiskModel
	if err := c.Get(context.Background(), key, &got); err != nil {
		t.Fatalf("get model after evaluation creation: %v", err)
	}
	if got.Status.Phase != ledgerv1alpha1.RiskModelPhaseEvaluationPending {
		t.Fatalf("expected evaluation pending, got %q", got.Status.Phase)
	}

	evaluationJob.Status.Conditions = []batchv1.JobCondition{
		{Type: batchv1.JobComplete, Status: corev1.ConditionTrue},
	}
	if err := c.Status().Update(context.Background(), &evaluationJob); err != nil {
		t.Fatalf("update evaluation status: %v", err)
	}
	if _, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatalf("second reconcile returned error: %v", err)
	}
	if err := c.Get(context.Background(), key, &got); err != nil {
		t.Fatalf("get model after evaluation completion: %v", err)
	}
	if got.Status.Phase != ledgerv1alpha1.RiskModelPhaseApproved {
		t.Fatalf("expected approval after evaluation, got %q", got.Status.Phase)
	}
}

func TestRiskModelReconcileBlocksTrainingUntilPreparationCompletes(t *testing.T) {
	model := validRiskModel()
	model.Spec.Preparation = ledgerv1alpha1.DatasetPreparationSpec{
		Enabled: true,
		Image:   "ghcr.io/ledger-ml/preparer:latest",
		OutputRef: ledgerv1alpha1.DatasetReference{
			Kind: "ObjectStore",
			Name: "curated-datasets",
			Path: "fraud/v1",
		},
	}
	model.Spec.Lineage.PreparedDatasetVersion = "curated-v1"

	reconciler, c := newReconciler(t, nil, model)
	key := riskModelName(model.Namespace, model.Name)
	if _, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatalf("first reconcile returned error: %v", err)
	}

	var preparationJob batchv1.Job
	if err := c.Get(context.Background(), riskModelName(model.Namespace, preparationJobName(model.Name)), &preparationJob); err != nil {
		t.Fatalf("expected preparation job: %v", err)
	}
	var trainingJob batchv1.Job
	if err := c.Get(context.Background(), riskModelName(model.Namespace, trainingJobName(model.Name)), &trainingJob); err == nil {
		t.Fatalf("training job must not exist before preparation succeeds")
	}

	preparationJob.Status.Conditions = []batchv1.JobCondition{
		{Type: batchv1.JobComplete, Status: corev1.ConditionTrue},
	}
	if err := c.Status().Update(context.Background(), &preparationJob); err != nil {
		t.Fatalf("update preparation status: %v", err)
	}
	if _, err := reconciler.Reconcile(context.Background(), ctrl.Request{NamespacedName: key}); err != nil {
		t.Fatalf("second reconcile returned error: %v", err)
	}
	if err := c.Get(context.Background(), riskModelName(model.Namespace, trainingJobName(model.Name)), &trainingJob); err != nil {
		t.Fatalf("expected training job after preparation: %v", err)
	}
	container := trainingJob.Spec.Template.Spec.Containers[0]
	env := map[string]string{}
	for _, entry := range container.Env {
		env[entry.Name] = entry.Value
	}
	if env[envDatasetPath] != model.Spec.Preparation.OutputRef.Path {
		t.Fatalf("expected training to consume curated path %q, got %q", model.Spec.Preparation.OutputRef.Path, env[envDatasetPath])
	}
	if env[envDatasetVersion] != model.Spec.Lineage.PreparedDatasetVersion {
		t.Fatalf("expected training to consume curated version %q, got %q", model.Spec.Lineage.PreparedDatasetVersion, env[envDatasetVersion])
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
	if ready == nil || ready.Reason != "PromotionRecorded" || ready.Status != metav1.ConditionTrue {
		t.Fatalf("expected Ready condition true with PromotionRecorded, got %#v", ready)
	}
	if len(got.Status.ApprovedBy) != 1 || got.Status.ApprovedBy[0] != "risk-owner" {
		t.Fatalf("expected approvedBy to list risk-owner, got %#v", got.Status.ApprovedBy)
	}
	var promotion corev1.ConfigMap
	if err := c.Get(context.Background(), riskModelName(model.Namespace, promotionRecordName(model.Name)), &promotion); err != nil {
		t.Fatalf("expected promotion record: %v", err)
	}
	if promotion.Immutable == nil || !*promotion.Immutable {
		t.Fatalf("expected immutable promotion record")
	}
	if got.Status.PromotionReference != model.Namespace+"/"+promotion.Name {
		t.Fatalf("expected promotion reference, got %q", got.Status.PromotionReference)
	}
}

func TestShadowServingUsesApprovedArtifactWithoutExternalRoute(t *testing.T) {
	model := validRiskModel()
	model.Spec.Serving.Enabled = true
	model.Spec.Serving.Image = "ghcr.io/ledger-ml/inference:latest"
	model.Spec.Serving.Mode = "Shadow"

	deployment := buildShadowDeployment(model)
	if deployment.Spec.Replicas == nil || *deployment.Spec.Replicas != 1 {
		t.Fatalf("expected one shadow replica, got %#v", deployment.Spec.Replicas)
	}
	if deployment.Spec.Template.Spec.Containers[0].Image != model.Spec.Serving.Image {
		t.Fatalf("expected serving image %q", model.Spec.Serving.Image)
	}
	if deployment.Spec.Template.Spec.Containers[0].Env[len(deployment.Spec.Template.Spec.Containers[0].Env)-1].Value != "shadow" {
		t.Fatalf("expected shadow serving mode")
	}

	service := buildShadowService(model)
	if service.Spec.Type != corev1.ServiceTypeClusterIP {
		t.Fatalf("expected internal ClusterIP service, got %q", service.Spec.Type)
	}
	if service.Spec.ExternalName != "" {
		t.Fatalf("shadow service must not use an external name")
	}
}

func TestCanaryRouteDefaultsToConfiguredWeightAndSupportsRollback(t *testing.T) {
	model := validRiskModel()
	model.Spec.Serving.Enabled = true
	model.Spec.Serving.Image = "ghcr.io/ledger-ml/inference:latest"
	model.Spec.Serving.Mode = "Canary"
	model.Spec.Serving.StableServiceName = "fraud-risk-model-stable"
	model.Spec.Serving.GatewayName = "ledger-gateway"
	model.Spec.Serving.RouteHost = "fraud.example.internal"
	model.Spec.Serving.CanaryWeightBPS = 1000

	route := buildCanaryRoute(model, serviceName(model.Name))
	spec, ok := route.Object["spec"].(map[string]any)
	if !ok {
		t.Fatalf("expected HTTPRoute spec")
	}
	rules, ok := spec["rules"].([]any)
	if !ok || len(rules) != 1 {
		t.Fatalf("expected one HTTPRoute rule, got %#v", spec["rules"])
	}
	backends := rules[0].(map[string]any)["backendRefs"].([]any)
	if backends[0].(map[string]any)["weight"] != int64(9000) || backends[1].(map[string]any)["weight"] != int64(1000) {
		t.Fatalf("expected 90/10 stable/canary split, got %#v", backends)
	}

	model.Annotations = map[string]string{annotationRollbackCanary: "true"}
	rollback := buildCanaryRoute(model, serviceName(model.Name))
	rollbackSpec := rollback.Object["spec"].(map[string]any)
	rollbackBackends := rollbackSpec["rules"].([]any)[0].(map[string]any)["backendRefs"].([]any)
	if rollbackBackends[0].(map[string]any)["weight"] != int64(10000) || rollbackBackends[1].(map[string]any)["weight"] != int64(0) {
		t.Fatalf("expected rollback to route 100%% stable and 0%% canary, got %#v", rollbackBackends)
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
