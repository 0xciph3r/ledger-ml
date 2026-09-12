package controller

import (
	"context"
	"fmt"

	"github.com/go-logr/logr"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	ledgerv1alpha1 "github.com/ledger-ml/ledger-ml/api/v1alpha1"
)

const (
	conditionTypeTraining = "Training"
	conditionTypeReady    = "Ready"

	trainingJobNameSuffix = "-train"
	trainingContainerName = "trainer"
	labelRiskModelName    = "ledger.ledgerml.io/riskmodel"
	labelComponent        = "ledger.ledgerml.io/component"
	componentTraining     = "training"

	envTask        = "LEDGERML_TASK"
	envDatasetKind = "LEDGERML_DATASET_KIND"
	envDatasetName = "LEDGERML_DATASET_NAME"
	envDatasetPath = "LEDGERML_DATASET_PATH"
	envOutputKind  = "LEDGERML_OUTPUT_KIND"
	envOutputName  = "LEDGERML_OUTPUT_NAME"
	envOutputPath  = "LEDGERML_OUTPUT_PATH"

	gpuResourceName = corev1.ResourceName("nvidia.com/gpu")
)

// RiskModelReconciler reconciles a RiskModel object.
type RiskModelReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=ledger.ledgerml.io,resources=riskmodels,verbs=get;list;watch
// +kubebuilder:rbac:groups=ledger.ledgerml.io,resources=riskmodels/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=ledger.ledgerml.io,resources=riskmodels/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create
// +kubebuilder:rbac:groups=batch,resources=jobs/status,verbs=get

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

	if err := model.ValidateCreate(); err != nil {
		setModelStatus(
			model,
			ledgerv1alpha1.RiskModelPhaseFailed,
			metav1.ConditionFalse,
			metav1.ConditionFalse,
			"InvalidSpec",
			err.Error(),
		)
		return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
	}

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
			setModelStatus(
				model,
				ledgerv1alpha1.RiskModelPhaseFailed,
				metav1.ConditionFalse,
				metav1.ConditionFalse,
				"TrainingJobCreateFailed",
				err.Error(),
			)
			_ = r.patchStatusIfChanged(ctx, logger, original, model)
			return ctrl.Result{}, err
		}
		logger.Info("created training job", "job", jobKey.String())
	}

	if !metav1.IsControlledBy(job, model) {
		setModelStatus(
			model,
			ledgerv1alpha1.RiskModelPhaseFailed,
			metav1.ConditionFalse,
			metav1.ConditionFalse,
			"TrainingJobOwnershipConflict",
			fmt.Sprintf("job %q already exists but is not owned by RiskModel %q", job.Name, model.Name),
		)
		return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
	}

	phase, trainingConditionStatus, readyConditionStatus, reason, message := trainingStatusFromJob(job)
	setModelStatus(model, phase, trainingConditionStatus, readyConditionStatus, reason, message)

	if err := r.patchStatusIfChanged(ctx, logger, original, model); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

func setModelStatus(
	model *ledgerv1alpha1.RiskModel,
	phase ledgerv1alpha1.RiskModelPhase,
	trainingStatus metav1.ConditionStatus,
	readyStatus metav1.ConditionStatus,
	reason string,
	message string,
) {
	model.Status.Phase = phase
	model.Status.Reason = reason
	model.Status.Message = message

	setCondition(model, conditionTypeTraining, trainingStatus, reason, message)
	setCondition(model, conditionTypeReady, readyStatus, reason, message)
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

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      trainingJobName(model.Name),
			Namespace: model.Namespace,
			Labels:    labels,
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
								{Name: envOutputKind, Value: model.Spec.OutputRef.Kind},
								{Name: envOutputName, Value: model.Spec.OutputRef.Name},
								{Name: envOutputPath, Value: model.Spec.OutputRef.Path},
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

func trainingStatusFromJob(job *batchv1.Job) (
	ledgerv1alpha1.RiskModelPhase,
	metav1.ConditionStatus,
	metav1.ConditionStatus,
	string,
	string,
) {
	if cond := findJobCondition(job, batchv1.JobFailed); cond != nil && cond.Status == corev1.ConditionTrue {
		reason := cond.Reason
		if reason == "" {
			reason = "TrainingFailed"
		}
		message := cond.Message
		if message == "" {
			message = "Training job failed."
		}
		return ledgerv1alpha1.RiskModelPhaseTrainingFailed, metav1.ConditionFalse, metav1.ConditionFalse, reason, message
	}

	if cond := findJobCondition(job, batchv1.JobComplete); cond != nil && cond.Status == corev1.ConditionTrue {
		return ledgerv1alpha1.RiskModelPhaseTrainingSucceeded, metav1.ConditionTrue, metav1.ConditionTrue, "TrainingSucceeded", "Training job completed successfully."
	}

	if job.Status.Active > 0 {
		return ledgerv1alpha1.RiskModelPhaseTrainingRunning, metav1.ConditionFalse, metav1.ConditionFalse, "TrainingRunning", fmt.Sprintf("Training job has %d active pod(s).", job.Status.Active)
	}

	if job.Status.Succeeded > 0 {
		return ledgerv1alpha1.RiskModelPhaseTrainingSucceeded, metav1.ConditionTrue, metav1.ConditionTrue, "TrainingSucceeded", "Training job completed successfully."
	}

	if job.Status.Failed > 0 {
		return ledgerv1alpha1.RiskModelPhaseTrainingFailed, metav1.ConditionFalse, metav1.ConditionFalse, "TrainingFailed", "Training job failed."
	}

	return ledgerv1alpha1.RiskModelPhaseTrainingPending, metav1.ConditionFalse, metav1.ConditionFalse, "TrainingPending", "Training job created; waiting to start."
}

func findJobCondition(job *batchv1.Job, conditionType batchv1.JobConditionType) *batchv1.JobCondition {
	for i := range job.Status.Conditions {
		if job.Status.Conditions[i].Type == conditionType {
			return &job.Status.Conditions[i]
		}
	}
	return nil
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
