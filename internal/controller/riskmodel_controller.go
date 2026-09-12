package controller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/go-logr/logr"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	ledgerv1alpha1 "github.com/ledger-ml/ledger-ml/api/v1alpha1"
)

const (
	conditionTypeAccepted = "Accepted"
	conditionTypeTraining = "Training"
	conditionTypeApproved = "Approved"
	conditionTypeReady    = "Ready"

	trainingJobNameSuffix = "-train"
	trainingContainerName = "trainer"
	labelRiskModelName    = "ledger.ledgerml.io/riskmodel"
	labelComponent        = "ledger.ledgerml.io/component"
	componentTraining     = "training"

	annotationLineageHash = "ledger.ledgerml.io/lineage-hash"

	envTask                = "LEDGERML_TASK"
	envDatasetKind         = "LEDGERML_DATASET_KIND"
	envDatasetName         = "LEDGERML_DATASET_NAME"
	envDatasetPath         = "LEDGERML_DATASET_PATH"
	envDatasetVersion      = "LEDGERML_DATASET_VERSION"
	envOutputKind          = "LEDGERML_OUTPUT_KIND"
	envOutputName          = "LEDGERML_OUTPUT_NAME"
	envOutputPath          = "LEDGERML_OUTPUT_PATH"
	envOutputVersion       = "LEDGERML_OUTPUT_ARTIFACT_VERSION"
	envTrainingImageDigest = "LEDGERML_TRAINING_IMAGE_DIGEST"
	envConfigurationDigest = "LEDGERML_CONFIGURATION_DIGEST"

	gpuResourceName = corev1.ResourceName("nvidia.com/gpu")

	evidenceRetention = 32
)

// RiskModelReconciler reconciles a RiskModel object.
type RiskModelReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=ledger.ledgerml.io,resources=riskmodels,verbs=get;list;watch
// +kubebuilder:rbac:groups=ledger.ledgerml.io,resources=riskmodels/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=ledger.ledgerml.io,resources=riskmodels/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create
// +kubebuilder:rbac:groups=batch,resources=jobs/status,verbs=get
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch

func (r *RiskModelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues("riskModel", req.NamespacedName.String())

	model := &ledgerv1alpha1.RiskModel{}
	if err := r.Get(ctx, req.NamespacedName, model); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	if !model.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	model.Default()

	original := model.DeepCopy()
	model.Status.ObservedGeneration = model.Generation
	calculatedLineageHash := lineageHash(model)
	persistedLineageHash := strings.TrimSpace(model.Status.LineageHash)
	if persistedLineageHash != "" && persistedLineageHash != calculatedLineageHash {
		rejectModel(
			model,
			"LineageImmutable",
			fmt.Sprintf(
				"ImmutableLineage violation: stored lineage hash %q does not match current spec lineage hash %q. Create a new RiskModel for changed lineage/config.",
				persistedLineageHash,
				calculatedLineageHash,
			),
		)
		model.Status.LineageHash = persistedLineageHash
		r.recordEvidenceAndEvent(model, "Rejected", "ImmutableLineage", model.Status.Message, "", true)
		return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
	}
	if persistedLineageHash == "" {
		model.Status.LineageHash = calculatedLineageHash
	} else {
		model.Status.LineageHash = persistedLineageHash
	}

	if err := model.ValidateCreate(); err != nil {
		rejectModel(model, "InvalidSpec", err.Error())
		r.recordEvidenceAndEvent(model, "Rejected", "InvalidSpec", err.Error(), "", true)
		return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
	}

	if violation := evaluateResourcePolicy(model); violation != nil {
		rejectModel(model, violation.Reason, violation.Message)
		r.recordEvidenceAndEvent(model, "Rejected", violation.Reason, violation.Message, "", true)
		return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
	}

	setCondition(model, conditionTypeAccepted, metav1.ConditionTrue, "Accepted", "Spec and governance policy accepted.")
	r.recordEvidenceAndEvent(model, "Accepted", "Accepted", "Spec and governance policy accepted.", "", false)

	jobName := trainingJobName(model.Name)
	jobKey := types.NamespacedName{Name: jobName, Namespace: model.Namespace}
	job := &batchv1.Job{}
	if err := r.Get(ctx, jobKey, job); err != nil {
		if !apierrors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		job = buildTrainingJob(model)
		if err := ctrl.SetControllerReference(model, job, r.Scheme); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.Create(ctx, job); err != nil {
			setFailed(model, "TrainingJobCreateFailed", err.Error())
			r.recordEvidenceAndEvent(model, "Rejected", "TrainingJobCreateFailed", err.Error(), jobName, true)
			_ = r.patchStatusIfChanged(ctx, logger, original, model)
			return ctrl.Result{}, err
		}
		r.recordEvidenceAndEvent(model, "JobCreated", "TrainingJobCreated", "Training job created from accepted lineage.", jobName, false)
		logger.Info("created training job", "job", jobKey.String())
	}

	if !metav1.IsControlledBy(job, model) {
		rejectModel(model, "TrainingJobOwnershipConflict", fmt.Sprintf("job %q already exists but is not owned by RiskModel %q", job.Name, model.Name))
		r.recordEvidenceAndEvent(model, "Rejected", "TrainingJobOwnershipConflict", model.Status.Message, job.Name, true)
		return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
	}

	if job.Annotations[annotationLineageHash] != model.Status.LineageHash {
		rejectModel(model, "LineageMismatch", "existing training job lineage does not match immutable spec lineage")
		r.recordEvidenceAndEvent(model, "Rejected", "LineageMismatch", model.Status.Message, job.Name, true)
		return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
	}

	trainingPhase, trainingReason, trainingMessage, trainingDone, trainingFailed := trainingStatusFromJob(job)
	model.Status.Phase = trainingPhase
	model.Status.Reason = trainingReason
	model.Status.Message = trainingMessage
	setCondition(model, conditionTypeTraining, conditionStatusForTraining(trainingDone, trainingFailed), trainingReason, trainingMessage)

	if trainingFailed {
		setCondition(model, conditionTypeApproved, metav1.ConditionFalse, "ApprovalBlockedByTrainingFailure", "Approval is blocked because training failed.")
		setCondition(model, conditionTypeReady, metav1.ConditionFalse, trainingReason, trainingMessage)
		r.recordEvidenceAndEvent(model, "TrainingFailed", trainingReason, trainingMessage, job.Name, true)
		return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
	}

	if !trainingDone {
		setCondition(model, conditionTypeApproved, metav1.ConditionFalse, "ApprovalPendingTrainingCompletion", "Approval is evaluated after successful training.")
		setCondition(model, conditionTypeReady, metav1.ConditionFalse, trainingReason, trainingMessage)
		r.recordEvidenceAndEvent(model, "TrainingObserved", trainingReason, trainingMessage, job.Name, false)
		return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
	}

	r.recordEvidenceAndEvent(model, "JobCompleted", "TrainingSucceeded", "Training job completed successfully.", job.Name, false)

	approval := evaluateApprovals(model)
	model.Status.ApprovedBy = approval.ApprovedBy
	if approval.Approved {
		model.Status.Phase = ledgerv1alpha1.RiskModelPhaseApproved
		model.Status.Reason = "ApprovalObserved"
		model.Status.Message = approval.Message
		setCondition(model, conditionTypeApproved, metav1.ConditionTrue, "ApprovalObserved", approval.Message)
		setCondition(model, conditionTypeReady, metav1.ConditionFalse, "PromotionNotImplemented", "Model is approved, but promotion/deployment milestones are not implemented yet.")
		r.recordEvidenceAndEvent(model, "ApprovalObserved", "ApprovalObserved", approval.Message, job.Name, false)
	} else {
		model.Status.Phase = ledgerv1alpha1.RiskModelPhaseAwaitingApproval
		model.Status.Reason = approval.Reason
		model.Status.Message = approval.Message
		setCondition(model, conditionTypeApproved, metav1.ConditionFalse, approval.Reason, approval.Message)
		setCondition(model, conditionTypeReady, metav1.ConditionFalse, "AwaitingApproval", approval.Message)
		r.recordEvidenceAndEvent(model, "AwaitingApproval", approval.Reason, approval.Message, job.Name, false)
	}

	return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
}

func rejectModel(model *ledgerv1alpha1.RiskModel, reason string, message string) {
	model.Status.Phase = ledgerv1alpha1.RiskModelPhaseRejected
	model.Status.Reason = reason
	model.Status.Message = message
	setCondition(model, conditionTypeAccepted, metav1.ConditionFalse, reason, message)
	setCondition(model, conditionTypeTraining, metav1.ConditionFalse, "TrainingRejected", "Training job was not created due to governance rejection.")
	setCondition(model, conditionTypeApproved, metav1.ConditionFalse, "ApprovalNotEvaluated", "Approval is not evaluated for rejected models.")
	setCondition(model, conditionTypeReady, metav1.ConditionFalse, reason, message)
}

func setFailed(model *ledgerv1alpha1.RiskModel, reason string, message string) {
	model.Status.Phase = ledgerv1alpha1.RiskModelPhaseFailed
	model.Status.Reason = reason
	model.Status.Message = message
	setCondition(model, conditionTypeAccepted, metav1.ConditionTrue, "Accepted", "Spec and governance policy accepted.")
	setCondition(model, conditionTypeTraining, metav1.ConditionFalse, reason, message)
	setCondition(model, conditionTypeApproved, metav1.ConditionFalse, "ApprovalNotEvaluated", "Approval is evaluated after successful training.")
	setCondition(model, conditionTypeReady, metav1.ConditionFalse, reason, message)
}

func setCondition(
	model *ledgerv1alpha1.RiskModel,
	conditionType string,
	status metav1.ConditionStatus,
	reason string,
	message string,
) {
	apimeta.SetStatusCondition(&model.Status.Conditions, metav1.Condition{
		Type:               conditionType,
		Status:             status,
		ObservedGeneration: model.Generation,
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	})
}

func (r *RiskModelReconciler) patchStatusIfChanged(
	ctx context.Context,
	logger logr.Logger,
	original *ledgerv1alpha1.RiskModel,
	model *ledgerv1alpha1.RiskModel,
) error {
	if equality.Semantic.DeepEqual(original.Status, model.Status) {
		logger.V(1).Info("status already up to date")
		return nil
	}

	if err := r.Status().Patch(ctx, model, client.MergeFrom(original)); err != nil {
		return err
	}

	logger.Info("updated RiskModel status", "phase", model.Status.Phase, "reason", model.Status.Reason)
	return nil
}

func buildTrainingJob(model *ledgerv1alpha1.RiskModel) *batchv1.Job {
	labels := map[string]string{
		labelRiskModelName: model.Name,
		labelComponent:     componentTraining,
	}

	backoffLimit := int32(1)
	lineage := lineageHash(model)

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      trainingJobName(model.Name),
			Namespace: model.Namespace,
			Labels:    labels,
			Annotations: map[string]string{
				annotationLineageHash: lineage,
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						{
							Name:      trainingContainerName,
							Image:     model.Spec.TrainingImage,
							Resources: trainingContainerResources(model.Spec.Resources),
							Env: []corev1.EnvVar{
								{Name: envTask, Value: model.Spec.Task},
								{Name: envDatasetKind, Value: model.Spec.DatasetRef.Kind},
								{Name: envDatasetName, Value: model.Spec.DatasetRef.Name},
								{Name: envDatasetPath, Value: model.Spec.DatasetRef.Path},
								{Name: envDatasetVersion, Value: model.Spec.Lineage.DatasetVersion},
								{Name: envOutputKind, Value: model.Spec.OutputRef.Kind},
								{Name: envOutputName, Value: model.Spec.OutputRef.Name},
								{Name: envOutputPath, Value: model.Spec.OutputRef.Path},
								{Name: envOutputVersion, Value: model.Spec.Lineage.OutputArtifactVersion},
								{Name: envTrainingImageDigest, Value: model.Spec.Lineage.TrainingImageDigest},
								{Name: envConfigurationDigest, Value: model.Spec.Lineage.ConfigurationDigest},
							},
						},
					},
				},
			},
		},
	}
}

func trainingContainerResources(resources corev1.ResourceRequirements) corev1.ResourceRequirements {
	supportedRequests := []corev1.ResourceName{
		corev1.ResourceCPU,
		corev1.ResourceMemory,
		corev1.ResourceEphemeralStorage,
	}
	supportedLimits := []corev1.ResourceName{
		corev1.ResourceCPU,
		corev1.ResourceMemory,
		corev1.ResourceEphemeralStorage,
		gpuResourceName,
	}

	filtered := corev1.ResourceRequirements{}
	if len(resources.Requests) > 0 {
		filtered.Requests = make(corev1.ResourceList)
		for _, name := range supportedRequests {
			if qty, ok := resources.Requests[name]; ok {
				filtered.Requests[name] = qty.DeepCopy()
			}
		}
		if len(filtered.Requests) == 0 {
			filtered.Requests = nil
		}
	}

	if len(resources.Limits) > 0 {
		filtered.Limits = make(corev1.ResourceList)
		for _, name := range supportedLimits {
			if qty, ok := resources.Limits[name]; ok {
				filtered.Limits[name] = qty.DeepCopy()
			}
		}
		if len(filtered.Limits) == 0 {
			filtered.Limits = nil
		}
	}

	return filtered
}

type policyViolation struct {
	Reason  string
	Message string
}

func evaluateResourcePolicy(model *ledgerv1alpha1.RiskModel) *policyViolation {
	maxCPU, err := resource.ParseQuantity(model.Spec.Policy.ResourceBounds.MaxCPU)
	if err != nil {
		return &policyViolation{Reason: "InvalidPolicy", Message: "policy.resourceBounds.maxCPU is invalid"}
	}
	maxMemory, err := resource.ParseQuantity(model.Spec.Policy.ResourceBounds.MaxMemory)
	if err != nil {
		return &policyViolation{Reason: "InvalidPolicy", Message: "policy.resourceBounds.maxMemory is invalid"}
	}
	maxGPU, err := resource.ParseQuantity(model.Spec.Policy.ResourceBounds.MaxGPU)
	if err != nil {
		return &policyViolation{Reason: "InvalidPolicy", Message: "policy.resourceBounds.maxGPU is invalid"}
	}

	allowedRequests := map[corev1.ResourceName]struct{}{
		corev1.ResourceCPU:              {},
		corev1.ResourceMemory:           {},
		corev1.ResourceEphemeralStorage: {},
	}
	allowedLimits := map[corev1.ResourceName]struct{}{
		corev1.ResourceCPU:              {},
		corev1.ResourceMemory:           {},
		corev1.ResourceEphemeralStorage: {},
		gpuResourceName:                 {},
	}

	for name := range model.Spec.Resources.Requests {
		if _, ok := allowedRequests[name]; !ok {
			return &policyViolation{
				Reason:  "UnsupportedResourceRequest",
				Message: fmt.Sprintf("resource request %q is not allowed by governance policy", name),
			}
		}
	}
	for name := range model.Spec.Resources.Limits {
		if _, ok := allowedLimits[name]; !ok {
			return &policyViolation{
				Reason:  "UnsupportedResourceLimit",
				Message: fmt.Sprintf("resource limit %q is not allowed by governance policy", name),
			}
		}
	}

	if reqCPU, ok := model.Spec.Resources.Requests[corev1.ResourceCPU]; ok && reqCPU.Cmp(maxCPU) > 0 {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("cpu request %s exceeds policy maxCPU %s", reqCPU.String(), maxCPU.String()),
		}
	}
	if limitCPU, ok := model.Spec.Resources.Limits[corev1.ResourceCPU]; ok && limitCPU.Cmp(maxCPU) > 0 {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("cpu limit %s exceeds policy maxCPU %s", limitCPU.String(), maxCPU.String()),
		}
	}
	if reqMem, ok := model.Spec.Resources.Requests[corev1.ResourceMemory]; ok && reqMem.Cmp(maxMemory) > 0 {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("memory request %s exceeds policy maxMemory %s", reqMem.String(), maxMemory.String()),
		}
	}
	if limitMem, ok := model.Spec.Resources.Limits[corev1.ResourceMemory]; ok && limitMem.Cmp(maxMemory) > 0 {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("memory limit %s exceeds policy maxMemory %s", limitMem.String(), maxMemory.String()),
		}
	}
	if reqGPU, ok := model.Spec.Resources.Requests[gpuResourceName]; ok {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("gpu request %s is not supported; specify GPU only in limits and within maxGPU %s", reqGPU.String(), maxGPU.String()),
		}
	}
	if limitGPU, ok := model.Spec.Resources.Limits[gpuResourceName]; ok && limitGPU.Cmp(maxGPU) > 0 {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("gpu limit %s exceeds policy maxGPU %s", limitGPU.String(), maxGPU.String()),
		}
	}

	return nil
}

func conditionStatusForTraining(trainingDone bool, trainingFailed bool) metav1.ConditionStatus {
	if trainingFailed {
		return metav1.ConditionFalse
	}
	if trainingDone {
		return metav1.ConditionTrue
	}
	return metav1.ConditionFalse
}

func trainingStatusFromJob(job *batchv1.Job) (ledgerv1alpha1.RiskModelPhase, string, string, bool, bool) {
	if cond := findJobCondition(job, batchv1.JobFailed); cond != nil && cond.Status == corev1.ConditionTrue {
		reason := cond.Reason
		if reason == "" {
			reason = "TrainingFailed"
		}
		message := cond.Message
		if message == "" {
			message = "Training job failed."
		}
		return ledgerv1alpha1.RiskModelPhaseTrainingFailed, reason, message, false, true
	}

	if cond := findJobCondition(job, batchv1.JobComplete); cond != nil && cond.Status == corev1.ConditionTrue {
		return ledgerv1alpha1.RiskModelPhaseTrainingSucceeded, "TrainingSucceeded", "Training job completed successfully.", true, false
	}

	if job.Status.Active > 0 {
		return ledgerv1alpha1.RiskModelPhaseTrainingRunning, "TrainingRunning", fmt.Sprintf("Training job has %d active pod(s).", job.Status.Active), false, false
	}

	if job.Status.Succeeded > 0 {
		return ledgerv1alpha1.RiskModelPhaseTrainingSucceeded, "TrainingSucceeded", "Training job completed successfully.", true, false
	}

	if job.Status.Failed > 0 {
		return ledgerv1alpha1.RiskModelPhaseTrainingFailed, "TrainingFailed", "Training job failed.", false, true
	}

	return ledgerv1alpha1.RiskModelPhaseTrainingPending, "TrainingPending", "Training job created; waiting to start.", false, false
}

func findJobCondition(job *batchv1.Job, conditionType batchv1.JobConditionType) *batchv1.JobCondition {
	for i := range job.Status.Conditions {
		if job.Status.Conditions[i].Type == conditionType {
			return &job.Status.Conditions[i]
		}
	}
	return nil
}

type approvalResult struct {
	Approved   bool
	ApprovedBy []string
	Reason     string
	Message    string
}

func evaluateApprovals(model *ledgerv1alpha1.RiskModel) approvalResult {
	if !model.ApprovalRequired() {
		return approvalResult{
			Approved:   true,
			ApprovedBy: nil,
			Reason:     "ApprovalNotRequired",
			Message:    "Approval policy does not require explicit approval for this model lineage.",
		}
	}

	allowed := map[string]struct{}{}
	for _, approver := range model.Spec.Policy.Approval.AllowedApprovers {
		allowed[strings.TrimSpace(approver)] = struct{}{}
	}

	approvedBySet := map[string]struct{}{}
	ignoredApprovers := map[string]struct{}{}
	for _, approval := range model.Spec.Approvals {
		approver := strings.TrimSpace(approval.Approver)
		if approver == "" {
			continue
		}
		if len(allowed) > 0 {
			if _, ok := allowed[approver]; !ok {
				ignoredApprovers[approver] = struct{}{}
				continue
			}
		}
		approvedBySet[approver] = struct{}{}
	}

	approvedBy := make([]string, 0, len(approvedBySet))
	for approver := range approvedBySet {
		approvedBy = append(approvedBy, approver)
	}
	sort.Strings(approvedBy)

	minimum := int(model.Spec.Policy.Approval.MinimumApprovals)
	if minimum < 1 {
		minimum = 1
	}

	if len(approvedBy) < minimum {
		ignored := make([]string, 0, len(ignoredApprovers))
		for approver := range ignoredApprovers {
			ignored = append(ignored, approver)
		}
		sort.Strings(ignored)
		if len(ignored) > 0 {
			return approvalResult{
				Approved:   false,
				ApprovedBy: approvedBy,
				Reason:     "ApprovalPolicyMismatch",
				Message: fmt.Sprintf(
					"Awaiting approvals: %d/%d valid approvals. Ignored approvers not in policy: %s.",
					len(approvedBy), minimum, strings.Join(ignored, ", "),
				),
			}
		}
		return approvalResult{
			Approved:   false,
			ApprovedBy: approvedBy,
			Reason:     "AwaitingApproval",
			Message:    fmt.Sprintf("Awaiting approvals: %d/%d valid approvals recorded.", len(approvedBy), minimum),
		}
	}

	return approvalResult{
		Approved:   true,
		ApprovedBy: approvedBy,
		Reason:     "ApprovalObserved",
		Message:    fmt.Sprintf("Approval requirement satisfied with %d/%d approvals.", len(approvedBy), minimum),
	}
}

func (r *RiskModelReconciler) recordEvidenceAndEvent(
	model *ledgerv1alpha1.RiskModel,
	decision string,
	reason string,
	message string,
	jobName string,
	warning bool,
) {
	record := ledgerv1alpha1.RiskModelEvidence{
		Time:               metav1.Now(),
		Decision:           decision,
		Reason:             reason,
		Message:            message,
		JobName:            jobName,
		ObservedGeneration: model.Generation,
	}

	if !appendEvidenceRecord(&model.Status.Evidence, record) {
		return
	}

	if r.Recorder == nil {
		return
	}

	eventType := corev1.EventTypeNormal
	if warning {
		eventType = corev1.EventTypeWarning
	}
	r.Recorder.Eventf(model, eventType, reason, "%s: %s", decision, message)
}

func appendEvidenceRecord(evidence *[]ledgerv1alpha1.RiskModelEvidence, next ledgerv1alpha1.RiskModelEvidence) bool {
	if evidence == nil {
		return false
	}

	records := *evidence
	if len(records) > 0 {
		last := records[len(records)-1]
		if last.Decision == next.Decision &&
			last.Reason == next.Reason &&
			last.Message == next.Message &&
			last.JobName == next.JobName &&
			last.ObservedGeneration == next.ObservedGeneration {
			return false
		}
	}

	records = append(records, next)
	if len(records) > evidenceRetention {
		records = records[len(records)-evidenceRetention:]
	}
	*evidence = records
	return true
}

func (r *RiskModelReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&ledgerv1alpha1.RiskModel{}).
		Owns(&batchv1.Job{}).
		Named("riskmodel").
		Complete(r)
}

func riskModelName(namespace, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: namespace, Name: name}
}

func trainingJobName(modelName string) string {
	const maxNameLength = 63
	name := modelName + trainingJobNameSuffix
	if len(name) <= maxNameLength {
		return name
	}
	return name[:maxNameLength]
}

func lineageHash(model *ledgerv1alpha1.RiskModel) string {
	payload, _ := json.Marshal(struct {
		Task          string                           `json:"task"`
		TrainingImage string                           `json:"trainingImage"`
		DatasetRef    ledgerv1alpha1.DatasetReference  `json:"datasetRef"`
		OutputRef     ledgerv1alpha1.ArtifactReference `json:"outputRef"`
		Lineage       ledgerv1alpha1.RiskModelLineage  `json:"lineage"`
		Resources     corev1.ResourceRequirements      `json:"resources"`
		Serving       ledgerv1alpha1.ServingConfig     `json:"serving"`
		Policy        ledgerv1alpha1.GovernancePolicy  `json:"policy"`
	}{
		Task:          model.Spec.Task,
		TrainingImage: model.Spec.TrainingImage,
		DatasetRef:    model.Spec.DatasetRef,
		OutputRef:     model.Spec.OutputRef,
		Lineage:       model.Spec.Lineage,
		Resources:     model.Spec.Resources,
		Serving:       model.Spec.Serving,
		Policy:        model.Spec.Policy,
	})
	sum := sha256.Sum256(payload)
	return "sha256:" + hex.EncodeToString(sum[:])
}
