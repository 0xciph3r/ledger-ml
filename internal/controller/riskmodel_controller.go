package controller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/go-logr/logr"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	ledgerv1alpha1 "github.com/ledger-ml/ledger-ml/api/v1alpha1"
)

const (
	conditionTypeAccepted    = "Accepted"
	conditionTypePreparation = "Preparation"
	conditionTypeTraining    = "Training"
	conditionTypeEvaluation  = "Evaluation"
	conditionTypeServing     = "Serving"
	conditionTypeDrift       = "Drift"
	conditionTypeOutcome     = "OutcomeQuality"
	conditionTypeApproved    = "Approved"
	conditionTypeReady       = "Ready"

	trainingJobNameSuffix     = "-train"
	preparationJobNameSuffix  = "-prepare"
	evaluationJobNameSuffix   = "-evaluate"
	promotionRecordNameSuffix = "-promotion"
	trainingContainerName     = "trainer"
	preparationContainerName  = "preparer"
	evaluationContainerName   = "evaluator"
	labelRiskModelName        = "ledger.ledgerml.io/riskmodel"
	labelComponent            = "ledger.ledgerml.io/component"
	labelServingName          = "ledger.ledgerml.io/serving"
	componentTraining         = "training"
	componentPreparation      = "preparation"
	componentEvaluation       = "evaluation"
	componentServing          = "shadow-serving"

	annotationLineageHash    = "ledger.ledgerml.io/lineage-hash"
	annotationPromotionHash  = "ledger.ledgerml.io/promotion-lineage-hash"
	annotationRollbackCanary = "ledger.ledgerml.io/rollback-canary"

	envTask                        = "LEDGERML_TASK"
	envDatasetKind                 = "LEDGERML_DATASET_KIND"
	envDatasetName                 = "LEDGERML_DATASET_NAME"
	envDatasetPath                 = "LEDGERML_DATASET_PATH"
	envDatasetVersion              = "LEDGERML_DATASET_VERSION"
	envOutputKind                  = "LEDGERML_OUTPUT_KIND"
	envOutputName                  = "LEDGERML_OUTPUT_NAME"
	envOutputPath                  = "LEDGERML_OUTPUT_PATH"
	envOutputVersion               = "LEDGERML_OUTPUT_ARTIFACT_VERSION"
	envModelPath                   = "LEDGERML_MODEL_PATH"
	envLineageHash                 = "LEDGERML_LINEAGE_HASH"
	envDriftBaselineKind           = "LEDGERML_DRIFT_BASELINE_KIND"
	envDriftBaselineName           = "LEDGERML_DRIFT_BASELINE_NAME"
	envDriftBaselinePath           = "LEDGERML_DRIFT_BASELINE_PATH"
	envDriftCurrentKind            = "LEDGERML_DRIFT_CURRENT_FEATURES_KIND"
	envDriftCurrentName            = "LEDGERML_DRIFT_CURRENT_FEATURES_NAME"
	envDriftCurrentPath            = "LEDGERML_DRIFT_CURRENT_FEATURES_PATH"
	envDriftModelVersion           = "LEDGERML_DRIFT_MODEL_VERSION"
	envDriftPSIThreshold           = "LEDGERML_DRIFT_PSI_THRESHOLD"
	envDriftMissingRateThreshold   = "LEDGERML_DRIFT_MISSING_RATE_DELTA_THRESHOLD"
	envPreparationOutputKind       = "LEDGERML_PREPARATION_OUTPUT_KIND"
	envPreparationOutputName       = "LEDGERML_PREPARATION_OUTPUT_NAME"
	envPreparationOutputPath       = "LEDGERML_PREPARATION_OUTPUT_PATH"
	envTrainingImageDigest         = "LEDGERML_TRAINING_IMAGE_DIGEST"
	envConfigurationDigest         = "LEDGERML_CONFIGURATION_DIGEST"
	envPreparedDatasetVersion      = "LEDGERML_PREPARED_DATASET_VERSION"
	envEvaluationMinRecall         = "LEDGERML_EVALUATION_MIN_RECALL"
	envEvaluationMinPRAUC          = "LEDGERML_EVALUATION_MIN_PR_AUC"
	envEvaluationMaxFalseNegatives = "LEDGERML_EVALUATION_MAX_FALSE_NEGATIVES"
	envEvaluationPath              = "LEDGERML_EVALUATION_PATH"

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
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;create
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;create
// +kubebuilder:rbac:groups="",resources=services,verbs=get;create
// +kubebuilder:rbac:groups=gateway.networking.k8s.io,resources=httproutes,verbs=get;create;update
// +kubebuilder:rbac:groups=batch,resources=cronjobs,verbs=get;create
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
	if model.Spec.Preparation.Enabled {
		if violation := evaluateResourcePolicyFor(model.Spec.Preparation.Resources, model.Spec.Policy.ResourceBounds, "preparation"); violation != nil {
			rejectModel(model, violation.Reason, violation.Message)
			r.recordEvidenceAndEvent(model, "Rejected", violation.Reason, violation.Message, "", true)
			return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
		}
	}
	if model.Spec.Evaluation.Enabled {
		if violation := evaluateResourcePolicyFor(model.Spec.Evaluation.Resources, model.Spec.Policy.ResourceBounds, "evaluation"); violation != nil {
			rejectModel(model, violation.Reason, violation.Message)
			r.recordEvidenceAndEvent(model, "Rejected", violation.Reason, violation.Message, "", true)
			return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
		}
	}
	if model.Spec.DriftMonitoring.Enabled {
		if violation := evaluateResourcePolicyFor(model.Spec.DriftMonitoring.Resources, model.Spec.Policy.ResourceBounds, "drift monitoring"); violation != nil {
			rejectModel(model, violation.Reason, violation.Message)
			r.recordEvidenceAndEvent(model, "Rejected", violation.Reason, violation.Message, "", true)
			return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
		}
	}

	setCondition(model, conditionTypeAccepted, metav1.ConditionTrue, "Accepted", "Spec and governance policy accepted.")
	r.recordEvidenceAndEvent(model, "Accepted", "Accepted", "Spec and governance policy accepted.", "", false)

	if model.Spec.Preparation.Enabled {
		preparationName := preparationJobName(model.Name)
		preparationKey := types.NamespacedName{Name: preparationName, Namespace: model.Namespace}
		preparationJob := &batchv1.Job{}
		if err := r.Get(ctx, preparationKey, preparationJob); err != nil {
			if !apierrors.IsNotFound(err) {
				return ctrl.Result{}, err
			}
			preparationJob = buildPreparationJob(model)
			if err := ctrl.SetControllerReference(model, preparationJob, r.Scheme); err != nil {
				return ctrl.Result{}, err
			}
			if err := r.Create(ctx, preparationJob); err != nil {
				setFailed(model, "PreparationJobCreateFailed", err.Error())
				r.recordEvidenceAndEvent(model, "Rejected", "PreparationJobCreateFailed", err.Error(), preparationName, true)
				_ = r.patchStatusIfChanged(ctx, logger, original, model)
				return ctrl.Result{}, err
			}
			r.recordEvidenceAndEvent(model, "PreparationJobCreated", "PreparationJobCreated", "Dataset preparation job created from accepted lineage.", preparationName, false)
		}
		if !metav1.IsControlledBy(preparationJob, model) {
			rejectModel(model, "PreparationJobOwnershipConflict", fmt.Sprintf("job %q already exists but is not owned by RiskModel %q", preparationJob.Name, model.Name))
			r.recordEvidenceAndEvent(model, "Rejected", "PreparationJobOwnershipConflict", model.Status.Message, preparationJob.Name, true)
			return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
		}
		if preparationJob.Annotations[annotationLineageHash] != model.Status.LineageHash {
			rejectModel(model, "LineageMismatch", "existing preparation job lineage does not match immutable spec lineage")
			r.recordEvidenceAndEvent(model, "Rejected", "LineageMismatch", model.Status.Message, preparationJob.Name, true)
			return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
		}

		preparationPhase, preparationReason, preparationMessage, preparationDone, preparationFailed := preparationStatusFromJob(preparationJob)
		setCondition(model, conditionTypePreparation, conditionStatusForTraining(preparationDone, preparationFailed), preparationReason, preparationMessage)
		if preparationFailed {
			model.Status.Phase = preparationPhase
			model.Status.Reason = preparationReason
			model.Status.Message = preparationMessage
			setCondition(model, conditionTypeTraining, metav1.ConditionFalse, "PreparationFailed", "Training is blocked because dataset preparation failed.")
			setCondition(model, conditionTypeApproved, metav1.ConditionFalse, "ApprovalBlockedByPreparationFailure", "Approval is blocked because dataset preparation failed.")
			setCondition(model, conditionTypeReady, metav1.ConditionFalse, preparationReason, preparationMessage)
			r.recordEvidenceAndEvent(model, "PreparationFailed", preparationReason, preparationMessage, preparationJob.Name, true)
			return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
		}
		if !preparationDone {
			model.Status.Phase = preparationPhase
			model.Status.Reason = preparationReason
			model.Status.Message = preparationMessage
			setCondition(model, conditionTypeTraining, metav1.ConditionFalse, "PreparationPending", "Training is blocked until dataset preparation succeeds.")
			setCondition(model, conditionTypeApproved, metav1.ConditionFalse, "ApprovalPendingPreparation", "Approval is evaluated after preparation and training complete.")
			setCondition(model, conditionTypeReady, metav1.ConditionFalse, preparationReason, preparationMessage)
			r.recordEvidenceAndEvent(model, "PreparationObserved", preparationReason, preparationMessage, preparationJob.Name, false)
			return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
		}
		r.recordEvidenceAndEvent(model, "PreparationCompleted", "PreparationSucceeded", "Dataset preparation job completed successfully.", preparationJob.Name, false)
	} else {
		setCondition(model, conditionTypePreparation, metav1.ConditionTrue, "PreparationNotRequired", "Dataset preparation is disabled for this RiskModel.")
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
		setCondition(model, conditionTypeEvaluation, metav1.ConditionFalse, "EvaluationBlockedByTrainingFailure", "Evaluation is blocked because training failed.")
		setCondition(model, conditionTypeReady, metav1.ConditionFalse, trainingReason, trainingMessage)
		r.recordEvidenceAndEvent(model, "TrainingFailed", trainingReason, trainingMessage, job.Name, true)
		return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
	}

	if !trainingDone {
		setCondition(model, conditionTypeApproved, metav1.ConditionFalse, "ApprovalPendingTrainingCompletion", "Approval is evaluated after successful training.")
		setCondition(model, conditionTypeEvaluation, metav1.ConditionFalse, "EvaluationPendingTraining", "Evaluation starts after training completes successfully.")
		setCondition(model, conditionTypeReady, metav1.ConditionFalse, trainingReason, trainingMessage)
		r.recordEvidenceAndEvent(model, "TrainingObserved", trainingReason, trainingMessage, job.Name, false)
		return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
	}

	r.recordEvidenceAndEvent(model, "JobCompleted", "TrainingSucceeded", "Training job completed successfully.", job.Name, false)

	if model.Spec.Evaluation.Enabled {
		evaluationName := evaluationJobName(model.Name)
		evaluationKey := types.NamespacedName{Name: evaluationName, Namespace: model.Namespace}
		evaluationJob := &batchv1.Job{}
		if err := r.Get(ctx, evaluationKey, evaluationJob); err != nil {
			if !apierrors.IsNotFound(err) {
				return ctrl.Result{}, err
			}
			evaluationJob = buildEvaluationJob(model)
			if err := ctrl.SetControllerReference(model, evaluationJob, r.Scheme); err != nil {
				return ctrl.Result{}, err
			}
			if err := r.Create(ctx, evaluationJob); err != nil {
				setFailed(model, "EvaluationJobCreateFailed", err.Error())
				r.recordEvidenceAndEvent(model, "Rejected", "EvaluationJobCreateFailed", err.Error(), evaluationName, true)
				_ = r.patchStatusIfChanged(ctx, logger, original, model)
				return ctrl.Result{}, err
			}
			r.recordEvidenceAndEvent(model, "EvaluationJobCreated", "EvaluationJobCreated", "Evaluation job created from the immutable model artifact.", evaluationName, false)
		}
		if !metav1.IsControlledBy(evaluationJob, model) {
			rejectModel(model, "EvaluationJobOwnershipConflict", fmt.Sprintf("job %q already exists but is not owned by RiskModel %q", evaluationJob.Name, model.Name))
			r.recordEvidenceAndEvent(model, "Rejected", "EvaluationJobOwnershipConflict", model.Status.Message, evaluationJob.Name, true)
			return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
		}
		if evaluationJob.Annotations[annotationLineageHash] != model.Status.LineageHash {
			rejectModel(model, "LineageMismatch", "existing evaluation job lineage does not match immutable spec lineage")
			r.recordEvidenceAndEvent(model, "Rejected", "LineageMismatch", model.Status.Message, evaluationJob.Name, true)
			return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
		}

		evaluationPhase, evaluationReason, evaluationMessage, evaluationDone, evaluationFailed := evaluationStatusFromJob(evaluationJob)
		setCondition(model, conditionTypeEvaluation, conditionStatusForTraining(evaluationDone, evaluationFailed), evaluationReason, evaluationMessage)
		if evaluationFailed {
			model.Status.Phase = evaluationPhase
			model.Status.Reason = evaluationReason
			model.Status.Message = evaluationMessage
			setCondition(model, conditionTypeApproved, metav1.ConditionFalse, "ApprovalBlockedByEvaluationFailure", "Approval is blocked because evaluation failed or a quality gate did not pass.")
			setCondition(model, conditionTypeReady, metav1.ConditionFalse, evaluationReason, evaluationMessage)
			r.recordEvidenceAndEvent(model, "EvaluationFailed", evaluationReason, evaluationMessage, evaluationJob.Name, true)
			return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
		}
		if !evaluationDone {
			model.Status.Phase = evaluationPhase
			model.Status.Reason = evaluationReason
			model.Status.Message = evaluationMessage
			setCondition(model, conditionTypeApproved, metav1.ConditionFalse, "ApprovalPendingEvaluation", "Approval is evaluated only after all model quality gates pass.")
			setCondition(model, conditionTypeReady, metav1.ConditionFalse, evaluationReason, evaluationMessage)
			r.recordEvidenceAndEvent(model, "EvaluationObserved", evaluationReason, evaluationMessage, evaluationJob.Name, false)
			return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
		}
		r.recordEvidenceAndEvent(model, "EvaluationCompleted", "EvaluationSucceeded", "All configured model quality gates passed.", evaluationJob.Name, false)
	} else {
		setCondition(model, conditionTypeEvaluation, metav1.ConditionTrue, "EvaluationNotRequired", "Model evaluation gates are disabled for this RiskModel.")
	}

	approval := evaluateApprovals(model)
	model.Status.ApprovedBy = approval.ApprovedBy
	if approval.Approved {
		model.Status.Phase = ledgerv1alpha1.RiskModelPhaseApproved
		model.Status.Reason = "ApprovalObserved"
		model.Status.Message = approval.Message
		setCondition(model, conditionTypeApproved, metav1.ConditionTrue, "ApprovalObserved", approval.Message)
		promotion, err := r.ensurePromotionRecord(ctx, model)
		if err != nil {
			model.Status.Reason = "PromotionRecordFailed"
			model.Status.Message = err.Error()
			setCondition(model, conditionTypeReady, metav1.ConditionFalse, "PromotionRecordFailed", err.Error())
			r.recordEvidenceAndEvent(model, "PromotionFailed", "PromotionRecordFailed", err.Error(), "", true)
			_ = r.patchStatusIfChanged(ctx, logger, original, model)
			return ctrl.Result{}, err
		}
		model.Status.PromotionReference = promotion.Namespace + "/" + promotion.Name
		if model.Status.PromotedAt == nil {
			now := metav1.Now()
			model.Status.PromotedAt = &now
		}
		setCondition(model, conditionTypeReady, metav1.ConditionTrue, "PromotionRecorded", "Approved immutable model artifact recorded as a promotion candidate; serving rollout remains a separate milestone.")
		if model.Spec.Serving.Enabled {
			deployment, service, available, err := r.ensureShadowServing(ctx, model)
			if err != nil {
				model.Status.Reason = "ServingDeploymentFailed"
				model.Status.Message = err.Error()
				setCondition(model, conditionTypeServing, metav1.ConditionFalse, "ServingDeploymentFailed", err.Error())
				setCondition(model, conditionTypeReady, metav1.ConditionFalse, "ServingDeploymentFailed", err.Error())
				r.recordEvidenceAndEvent(model, "ServingFailed", "ServingDeploymentFailed", err.Error(), deploymentName(model.Name), true)
				_ = r.patchStatusIfChanged(ctx, logger, original, model)
				return ctrl.Result{}, err
			}
			model.Status.ServingReference = model.Namespace + "/" + deployment.Name
			if !available {
				setCondition(model, conditionTypeServing, metav1.ConditionFalse, "ShadowServingPending", "Shadow serving Deployment exists but has no available replicas yet.")
				setCondition(model, conditionTypeReady, metav1.ConditionFalse, "ShadowServingPending", "Waiting for shadow serving replicas to become available.")
				r.recordEvidenceAndEvent(model, "ServingObserved", "ShadowServingPending", "Shadow serving Deployment exists but has no available replicas yet.", service.Name, false)
				return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
			}
			setCondition(model, conditionTypeServing, metav1.ConditionTrue, "ShadowServingReady", "Shadow serving is available internally; no production traffic route has been created.")
			setCondition(model, conditionTypeReady, metav1.ConditionTrue, "ShadowServingReady", "Approved model is promoted and available for shadow traffic.")
			if model.Spec.Serving.Mode == "Canary" {
				route, err := r.ensureCanaryRoute(ctx, model, service)
				if err != nil {
					model.Status.Reason = "CanaryRouteFailed"
					model.Status.Message = err.Error()
					setCondition(model, conditionTypeServing, metav1.ConditionFalse, "CanaryRouteFailed", err.Error())
					setCondition(model, conditionTypeReady, metav1.ConditionFalse, "CanaryRouteFailed", err.Error())
					r.recordEvidenceAndEvent(model, "CanaryRouteFailed", "CanaryRouteFailed", err.Error(), routeName(model.Name), true)
					_ = r.patchStatusIfChanged(ctx, logger, original, model)
					return ctrl.Result{}, err
				}
				model.Status.TrafficReference = model.Namespace + "/" + route.GetName()
				setCondition(model, conditionTypeServing, metav1.ConditionTrue, "CanaryRouteReady", "Gateway API canary route is installed; traffic weight is explicitly controlled.")
				setCondition(model, conditionTypeReady, metav1.ConditionTrue, "CanaryRouteReady", "Approved model is promoted and canary routing is available.")
			}
			if model.Spec.DriftMonitoring.Enabled {
				driftJob, err := r.ensureDriftMonitoring(ctx, model)
				if err != nil {
					model.Status.Reason = "DriftMonitoringFailed"
					model.Status.Message = err.Error()
					setCondition(model, conditionTypeReady, metav1.ConditionFalse, "DriftMonitoringFailed", err.Error())
					r.recordEvidenceAndEvent(model, "DriftMonitoringFailed", "DriftMonitoringFailed", err.Error(), driftCronJobName(model.Name), true)
					_ = r.patchStatusIfChanged(ctx, logger, original, model)
					return ctrl.Result{}, err
				}
				model.Status.DriftMonitoringReference = model.Namespace + "/" + driftJob.Name
				r.recordEvidenceAndEvent(model, "DriftMonitoringScheduled", "DriftMonitoringScheduled", "Scheduled drift monitoring is active; drift signals do not automatically retrain or promote models.", driftJob.Name, false)
				if err := r.observeDriftReport(ctx, model); err != nil {
					setCondition(model, conditionTypeDrift, metav1.ConditionFalse, "DriftReportInvalid", err.Error())
					r.recordEvidenceAndEvent(model, "DriftReportInvalid", "DriftReportInvalid", err.Error(), driftJob.Name, true)
					return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
				}
			}
			if model.Spec.OutcomeMonitoring.Enabled {
				if err := r.observeOutcomeReport(ctx, model); err != nil {
					setCondition(model, conditionTypeOutcome, metav1.ConditionFalse, "OutcomeReportInvalid", err.Error())
					r.recordEvidenceAndEvent(model, "OutcomeReportInvalid", "OutcomeReportInvalid", err.Error(), "", true)
					return ctrl.Result{}, r.patchStatusIfChanged(ctx, logger, original, model)
				}
			} else {
				setCondition(model, conditionTypeOutcome, metav1.ConditionTrue, "OutcomeMonitoringNotRequired", "Delayed outcome monitoring is disabled for this RiskModel.")
			}
		} else {
			setCondition(model, conditionTypeServing, metav1.ConditionTrue, "ServingNotEnabled", "Serving is disabled for this RiskModel.")
		}
		r.recordEvidenceAndEvent(model, "PromotionRecorded", "PromotionRecorded", "Approved immutable model artifact recorded as a promotion candidate.", promotion.Name, false)
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
	setCondition(model, conditionTypePreparation, metav1.ConditionFalse, "PreparationRejected", "Dataset preparation was not started due to governance rejection.")
	setCondition(model, conditionTypeTraining, metav1.ConditionFalse, "TrainingRejected", "Training job was not created due to governance rejection.")
	setCondition(model, conditionTypeEvaluation, metav1.ConditionFalse, "EvaluationRejected", "Evaluation was not started due to governance rejection.")
	setCondition(model, conditionTypeApproved, metav1.ConditionFalse, "ApprovalNotEvaluated", "Approval is not evaluated for rejected models.")
	setCondition(model, conditionTypeReady, metav1.ConditionFalse, reason, message)
}

func setFailed(model *ledgerv1alpha1.RiskModel, reason string, message string) {
	model.Status.Phase = ledgerv1alpha1.RiskModelPhaseFailed
	model.Status.Reason = reason
	model.Status.Message = message
	setCondition(model, conditionTypeAccepted, metav1.ConditionTrue, "Accepted", "Spec and governance policy accepted.")
	setCondition(model, conditionTypePreparation, metav1.ConditionFalse, reason, message)
	setCondition(model, conditionTypeEvaluation, metav1.ConditionFalse, reason, message)
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

func (r *RiskModelReconciler) ensurePromotionRecord(
	ctx context.Context,
	model *ledgerv1alpha1.RiskModel,
) (*corev1.ConfigMap, error) {
	key := types.NamespacedName{
		Name:      promotionRecordName(model.Name),
		Namespace: model.Namespace,
	}
	record := &corev1.ConfigMap{}
	if err := r.Get(ctx, key, record); err != nil {
		if !apierrors.IsNotFound(err) {
			return nil, err
		}
		record = buildPromotionRecord(model)
		if err := ctrl.SetControllerReference(model, record, r.Scheme); err != nil {
			return nil, err
		}
		if err := r.Create(ctx, record); err != nil {
			return nil, err
		}
		return record, nil
	}
	if !metav1.IsControlledBy(record, model) {
		return nil, fmt.Errorf("promotion record %q already exists but is not owned by RiskModel %q", record.Name, model.Name)
	}
	if record.Annotations[annotationPromotionHash] != model.Status.LineageHash {
		return nil, fmt.Errorf("promotion record %q lineage does not match the approved RiskModel", record.Name)
	}
	return record, nil
}

func buildPromotionRecord(model *ledgerv1alpha1.RiskModel) *corev1.ConfigMap {
	payload := map[string]any{
		"task":                     model.Spec.Task,
		"artifact_kind":            model.Spec.OutputRef.Kind,
		"artifact_name":            model.Spec.OutputRef.Name,
		"artifact_path":            model.Spec.OutputRef.Path,
		"artifact_version":         model.Spec.Lineage.OutputArtifactVersion,
		"lineage_hash":             model.Status.LineageHash,
		"training_image_digest":    model.Spec.Lineage.TrainingImageDigest,
		"dataset_version":          model.Spec.Lineage.DatasetVersion,
		"prepared_dataset_version": model.Spec.Lineage.PreparedDatasetVersion,
	}
	raw, _ := json.Marshal(payload)
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      promotionRecordName(model.Name),
			Namespace: model.Namespace,
			Annotations: map[string]string{
				annotationPromotionHash: model.Status.LineageHash,
			},
		},
		Immutable: boolPtr(true),
		Data: map[string]string{
			"promotion.json": string(raw),
		},
	}
}

func boolPtr(value bool) *bool {
	return &value
}

func (r *RiskModelReconciler) ensureShadowServing(
	ctx context.Context,
	model *ledgerv1alpha1.RiskModel,
) (*appsv1.Deployment, *corev1.Service, bool, error) {
	deployment := &appsv1.Deployment{}
	deploymentKey := types.NamespacedName{Name: deploymentName(model.Name), Namespace: model.Namespace}
	if err := r.Get(ctx, deploymentKey, deployment); err != nil {
		if !apierrors.IsNotFound(err) {
			return nil, nil, false, err
		}
		deployment = buildShadowDeployment(model)
		if err := ctrl.SetControllerReference(model, deployment, r.Scheme); err != nil {
			return nil, nil, false, err
		}
		if err := r.Create(ctx, deployment); err != nil {
			return nil, nil, false, err
		}
	} else if err := validateServingOwnership(deployment, model); err != nil {
		return nil, nil, false, err
	}

	service := &corev1.Service{}
	serviceKey := types.NamespacedName{Name: serviceName(model.Name), Namespace: model.Namespace}
	if err := r.Get(ctx, serviceKey, service); err != nil {
		if !apierrors.IsNotFound(err) {
			return nil, nil, false, err
		}
		service = buildShadowService(model)
		if err := ctrl.SetControllerReference(model, service, r.Scheme); err != nil {
			return nil, nil, false, err
		}
		if err := r.Create(ctx, service); err != nil {
			return nil, nil, false, err
		}
	} else if err := validateServingOwnership(service, model); err != nil {
		return nil, nil, false, err
	}

	return deployment, service, deployment.Status.AvailableReplicas >= servingReplicas(model), nil
}

func (r *RiskModelReconciler) ensureCanaryRoute(
	ctx context.Context,
	model *ledgerv1alpha1.RiskModel,
	candidateService *corev1.Service,
) (*unstructured.Unstructured, error) {
	route := buildCanaryRoute(model, candidateService.Name)
	key := types.NamespacedName{Name: routeName(model.Name), Namespace: model.Namespace}
	existing := &unstructured.Unstructured{}
	existing.SetGroupVersionKind(schema.GroupVersionKind{
		Group: "gateway.networking.k8s.io", Version: "v1", Kind: "HTTPRoute",
	})
	if err := r.Get(ctx, key, existing); err != nil {
		if !apierrors.IsNotFound(err) {
			return nil, err
		}
		if err := ctrl.SetControllerReference(model, route, r.Scheme); err != nil {
			return nil, err
		}
		if err := r.Create(ctx, route); err != nil {
			return nil, err
		}
		return route, nil
	}
	if err := validateServingOwnership(existing, model); err != nil {
		return nil, err
	}
	desiredSpec := route.Object["spec"]
	if existingSpec := existing.Object["spec"]; !equality.Semantic.DeepEqual(existingSpec, desiredSpec) {
		route.SetResourceVersion(existing.GetResourceVersion())
		if err := r.Update(ctx, route); err != nil {
			return nil, err
		}
		return route, nil
	}
	return existing, nil
}

func (r *RiskModelReconciler) ensureDriftMonitoring(
	ctx context.Context,
	model *ledgerv1alpha1.RiskModel,
) (*batchv1.CronJob, error) {
	key := types.NamespacedName{Name: driftCronJobName(model.Name), Namespace: model.Namespace}
	cronJob := &batchv1.CronJob{}
	if err := r.Get(ctx, key, cronJob); err != nil {
		if !apierrors.IsNotFound(err) {
			return nil, err
		}
		cronJob = buildDriftCronJob(model)
		if err := ctrl.SetControllerReference(model, cronJob, r.Scheme); err != nil {
			return nil, err
		}
		if err := r.Create(ctx, cronJob); err != nil {
			return nil, err
		}
		return cronJob, nil
	}
	if !metav1.IsControlledBy(cronJob, model) {
		return nil, fmt.Errorf("drift CronJob %q already exists but is not owned by RiskModel %q", cronJob.Name, model.Name)
	}
	if cronJob.Annotations[annotationLineageHash] != model.Status.LineageHash {
		return nil, fmt.Errorf("drift CronJob %q lineage does not match the promoted RiskModel", cronJob.Name)
	}
	return cronJob, nil
}

func (r *RiskModelReconciler) observeDriftReport(
	ctx context.Context,
	model *ledgerv1alpha1.RiskModel,
) error {
	reportRef := model.Spec.DriftMonitoring.ReportRef
	report := &corev1.ConfigMap{}
	if err := r.Get(ctx, types.NamespacedName{Name: reportRef.Name, Namespace: model.Namespace}, report); err != nil {
		if apierrors.IsNotFound(err) {
			setCondition(model, conditionTypeDrift, metav1.ConditionFalse, "DriftReportPending", "No drift report has been published for the promoted model yet.")
			return nil
		}
		return err
	}
	raw := report.Data[reportRef.Path]
	if strings.TrimSpace(raw) == "" {
		return fmt.Errorf("drift report ConfigMap %q has no data key %q", report.Name, reportRef.Path)
	}
	var payload struct {
		ModelVersion string `json:"model_version"`
		Status       string `json:"status"`
		Signals      []any  `json:"signals"`
	}
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return fmt.Errorf("drift report ConfigMap %q contains invalid JSON: %w", report.Name, err)
	}
	if payload.ModelVersion != model.Spec.Lineage.OutputArtifactVersion {
		return fmt.Errorf("drift report model version %q does not match promoted artifact version %q", payload.ModelVersion, model.Spec.Lineage.OutputArtifactVersion)
	}
	if payload.Status != "within_baseline" && payload.Status != "drift_detected" {
		return fmt.Errorf("drift report has unsupported status %q", payload.Status)
	}

	reference := report.Namespace + "/" + report.Name
	changed := model.Status.DriftReportReference != reference || model.Status.DriftStatus != payload.Status
	model.Status.DriftReportReference = reference
	model.Status.DriftStatus = payload.Status
	if changed || model.Status.DriftObservedAt == nil {
		now := metav1.Now()
		model.Status.DriftObservedAt = &now
	}

	if payload.Status == "drift_detected" {
		message := fmt.Sprintf("Drift detected for model %s with %d signal(s); investigation is required.", payload.ModelVersion, len(payload.Signals))
		setCondition(model, conditionTypeDrift, metav1.ConditionFalse, "DriftDetected", message)
		if changed {
			r.recordEvidenceAndEvent(model, "DriftDetected", "DriftDetected", message, report.Name, true)
		}
		return nil
	}

	message := fmt.Sprintf("Latest drift report for model %s is within baseline.", payload.ModelVersion)
	setCondition(model, conditionTypeDrift, metav1.ConditionTrue, "WithinBaseline", message)
	if changed {
		r.recordEvidenceAndEvent(model, "DriftWithinBaseline", "WithinBaseline", message, report.Name, false)
	}
	return nil
}

func (r *RiskModelReconciler) observeOutcomeReport(
	ctx context.Context,
	model *ledgerv1alpha1.RiskModel,
) error {
	ref := model.Spec.OutcomeMonitoring.ReportRef
	report := &corev1.ConfigMap{}
	if err := r.Get(ctx, types.NamespacedName{Name: ref.Name, Namespace: model.Namespace}, report); err != nil {
		if apierrors.IsNotFound(err) {
			setCondition(model, conditionTypeOutcome, metav1.ConditionFalse, "OutcomeReportPending", "No delayed-outcome report has been published for the promoted model yet.")
			return nil
		}
		return err
	}
	raw := report.Data[ref.Path]
	if strings.TrimSpace(raw) == "" {
		return fmt.Errorf("outcome report ConfigMap %q has no data key %q", report.Name, ref.Path)
	}
	var payload struct {
		ModelVersion        string `json:"model_version"`
		MatchedOutcomeCount int64  `json:"matched_outcome_count"`
		PredictionCount     int64  `json:"prediction_count"`
		Metrics             struct {
			Coverage float64 `json:"coverage"`
			Recall   float64 `json:"recall"`
		} `json:"metrics"`
	}
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return fmt.Errorf("outcome report ConfigMap %q contains invalid JSON: %w", report.Name, err)
	}
	if payload.ModelVersion != model.Spec.Lineage.OutputArtifactVersion {
		return fmt.Errorf("outcome report model version %q does not match promoted artifact version %q", payload.ModelVersion, model.Spec.Lineage.OutputArtifactVersion)
	}
	if payload.PredictionCount <= 0 || payload.MatchedOutcomeCount < 0 || payload.MatchedOutcomeCount > payload.PredictionCount {
		return fmt.Errorf("outcome report has invalid prediction coverage counts")
	}
	if payload.Metrics.Coverage < 0 || payload.Metrics.Coverage > 1 || payload.Metrics.Recall < 0 || payload.Metrics.Recall > 1 {
		return fmt.Errorf("outcome report metrics must be between 0 and 1")
	}

	coverageBPS := int32(payload.Metrics.Coverage * 10000)
	recallBPS := int32(payload.Metrics.Recall * 10000)
	changed := model.Status.OutcomeReportReference != report.Namespace+"/"+report.Name ||
		model.Status.OutcomeCoverage == nil || *model.Status.OutcomeCoverage != coverageBPS ||
		model.Status.OutcomeRecall == nil || *model.Status.OutcomeRecall != recallBPS
	model.Status.OutcomeReportReference = report.Namespace + "/" + report.Name
	model.Status.OutcomeCoverage = &coverageBPS
	model.Status.OutcomeRecall = &recallBPS
	if changed || model.Status.OutcomeObservedAt == nil {
		now := metav1.Now()
		model.Status.OutcomeObservedAt = &now
	}

	minCoverage := model.Spec.OutcomeMonitoring.MinimumCoverageBPS
	minRecall := model.Spec.OutcomeMonitoring.MinimumRecallBPS
	if (minCoverage > 0 && coverageBPS < minCoverage) || (minRecall > 0 && recallBPS < minRecall) {
		message := fmt.Sprintf("Delayed outcome quality below policy: coverage=%d bps recall=%d bps.", coverageBPS, recallBPS)
		setCondition(model, conditionTypeOutcome, metav1.ConditionFalse, "OutcomeQualityBelowPolicy", message)
		if changed {
			r.recordEvidenceAndEvent(model, "OutcomeQualityBelowPolicy", "OutcomeQualityBelowPolicy", message, report.Name, true)
		}
		return nil
	}
	message := fmt.Sprintf("Delayed outcome quality satisfies policy: coverage=%d bps recall=%d bps.", coverageBPS, recallBPS)
	setCondition(model, conditionTypeOutcome, metav1.ConditionTrue, "OutcomeQualityHealthy", message)
	if changed {
		r.recordEvidenceAndEvent(model, "OutcomeQualityHealthy", "OutcomeQualityHealthy", message, report.Name, false)
	}
	return nil
}

func buildDriftCronJob(model *ledgerv1alpha1.RiskModel) *batchv1.CronJob {
	labels := map[string]string{
		labelRiskModelName: model.Name,
		labelComponent:     "drift-monitoring",
	}
	backoffLimit := int32(1)
	psiThreshold := float64(model.Spec.DriftMonitoring.PSIThresholdBPS) / 10000
	missingThreshold := float64(model.Spec.DriftMonitoring.MissingRateDeltaBPS) / 10000
	return &batchv1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      driftCronJobName(model.Name),
			Namespace: model.Namespace,
			Labels:    labels,
			Annotations: map[string]string{
				annotationLineageHash: model.Status.LineageHash,
			},
		},
		Spec: batchv1.CronJobSpec{
			Schedule: model.Spec.DriftMonitoring.Schedule,
			JobTemplate: batchv1.JobTemplateSpec{
				Spec: batchv1.JobSpec{
					BackoffLimit: &backoffLimit,
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: labels},
						Spec: corev1.PodSpec{
							RestartPolicy: corev1.RestartPolicyNever,
							Containers: []corev1.Container{{
								Name:      "drift-detector",
								Image:     model.Spec.DriftMonitoring.Image,
								Resources: trainingContainerResources(model.Spec.DriftMonitoring.Resources),
								Env: []corev1.EnvVar{
									{Name: envDriftBaselineKind, Value: model.Spec.DriftMonitoring.BaselineRef.Kind},
									{Name: envDriftBaselineName, Value: model.Spec.DriftMonitoring.BaselineRef.Name},
									{Name: envDriftBaselinePath, Value: model.Spec.DriftMonitoring.BaselineRef.Path},
									{Name: envDriftCurrentKind, Value: model.Spec.DriftMonitoring.CurrentFeaturesRef.Kind},
									{Name: envDriftCurrentName, Value: model.Spec.DriftMonitoring.CurrentFeaturesRef.Name},
									{Name: envDriftCurrentPath, Value: model.Spec.DriftMonitoring.CurrentFeaturesRef.Path},
									{Name: envDriftModelVersion, Value: model.Spec.Lineage.OutputArtifactVersion},
									{Name: envDriftPSIThreshold, Value: strconv.FormatFloat(psiThreshold, 'f', -1, 64)},
									{Name: envDriftMissingRateThreshold, Value: strconv.FormatFloat(missingThreshold, 'f', -1, 64)},
								},
							}},
						},
					},
				},
			},
		},
	}
}

func buildCanaryRoute(model *ledgerv1alpha1.RiskModel, candidateServiceName string) *unstructured.Unstructured {
	canaryWeight := int64(model.Spec.Serving.CanaryWeightBPS)
	if model.Annotations[annotationRollbackCanary] == "true" {
		canaryWeight = 0
	}
	stableWeight := int64(10000 - canaryWeight)
	route := &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "gateway.networking.k8s.io/v1",
		"kind":       "HTTPRoute",
		"metadata": map[string]any{
			"name":      routeName(model.Name),
			"namespace": model.Namespace,
			"annotations": map[string]any{
				annotationLineageHash: model.Status.LineageHash,
			},
		},
		"spec": map[string]any{
			"parentRefs": []any{
				map[string]any{"name": model.Spec.Serving.GatewayName},
			},
			"hostnames": []any{model.Spec.Serving.RouteHost},
			"rules": []any{
				map[string]any{
					"matches": []any{map[string]any{
						"path": map[string]any{"type": "PathPrefix", "value": "/"},
					}},
					"backendRefs": []any{
						map[string]any{
							"name":   model.Spec.Serving.StableServiceName,
							"port":   int64(model.Spec.Serving.Port),
							"weight": stableWeight,
						},
						map[string]any{
							"name":   candidateServiceName,
							"port":   int64(model.Spec.Serving.Port),
							"weight": canaryWeight,
						},
					},
				},
			},
		},
	}}
	route.SetGroupVersionKind(schema.GroupVersionKind{
		Group: "gateway.networking.k8s.io", Version: "v1", Kind: "HTTPRoute",
	})
	return route
}

func validateServingOwnership(object client.Object, model *ledgerv1alpha1.RiskModel) error {
	if !metav1.IsControlledBy(object, model) {
		return fmt.Errorf("serving resource %q already exists but is not owned by RiskModel %q", object.GetName(), model.Name)
	}
	if object.GetAnnotations()[annotationLineageHash] != model.Status.LineageHash {
		return fmt.Errorf("serving resource %q lineage does not match the approved RiskModel", object.GetName())
	}
	return nil
}

func buildShadowDeployment(model *ledgerv1alpha1.RiskModel) *appsv1.Deployment {
	labels := map[string]string{
		labelRiskModelName: model.Name,
		labelComponent:     componentServing,
		labelServingName:   model.Name,
	}
	replicas := servingReplicas(model)
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      deploymentName(model.Name),
			Namespace: model.Namespace,
			Labels:    labels,
			Annotations: map[string]string{
				annotationLineageHash: model.Status.LineageHash,
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{labelServingName: model.Name}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name:  "inference",
					Image: model.Spec.Serving.Image,
					Ports: []corev1.ContainerPort{{Name: "http", ContainerPort: model.Spec.Serving.Port}},
					Env: []corev1.EnvVar{
						{Name: envOutputKind, Value: model.Spec.OutputRef.Kind},
						{Name: envOutputName, Value: model.Spec.OutputRef.Name},
						{Name: envOutputPath, Value: model.Spec.OutputRef.Path},
						{Name: envOutputVersion, Value: model.Spec.Lineage.OutputArtifactVersion},
						{Name: envModelPath, Value: model.Spec.OutputRef.Path + "/" + model.Spec.Lineage.OutputArtifactVersion + "/model.joblib"},
						{Name: envLineageHash, Value: model.Status.LineageHash},
						{Name: envTrainingImageDigest, Value: model.Spec.Lineage.TrainingImageDigest},
						{Name: envConfigurationDigest, Value: model.Spec.Lineage.ConfigurationDigest},
						{Name: "LEDGERML_SERVING_MODE", Value: "shadow"},
					},
				}}},
			},
		},
	}
}

func buildShadowService(model *ledgerv1alpha1.RiskModel) *corev1.Service {
	labels := map[string]string{
		labelRiskModelName: model.Name,
		labelComponent:     componentServing,
	}
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName(model.Name),
			Namespace: model.Namespace,
			Labels:    labels,
			Annotations: map[string]string{
				annotationLineageHash: model.Status.LineageHash,
			},
		},
		Spec: corev1.ServiceSpec{
			Type:     corev1.ServiceTypeClusterIP,
			Selector: map[string]string{labelServingName: model.Name},
			Ports:    []corev1.ServicePort{{Name: "http", Port: model.Spec.Serving.Port, TargetPort: intstr.FromInt32(model.Spec.Serving.Port)}},
		},
	}
}

func servingReplicas(model *ledgerv1alpha1.RiskModel) int32 {
	if model.Spec.Serving.Replicas == nil {
		return 1
	}
	return *model.Spec.Serving.Replicas
}

func buildPreparationJob(model *ledgerv1alpha1.RiskModel) *batchv1.Job {
	labels := map[string]string{
		labelRiskModelName: model.Name,
		labelComponent:     componentPreparation,
	}
	backoffLimit := int32(1)
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      preparationJobName(model.Name),
			Namespace: model.Namespace,
			Labels:    labels,
			Annotations: map[string]string{
				annotationLineageHash: lineageHash(model),
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{{
						Name:      preparationContainerName,
						Image:     model.Spec.Preparation.Image,
						Resources: trainingContainerResources(model.Spec.Preparation.Resources),
						Env: []corev1.EnvVar{
							{Name: envDatasetKind, Value: model.Spec.DatasetRef.Kind},
							{Name: envDatasetName, Value: model.Spec.DatasetRef.Name},
							{Name: envDatasetPath, Value: model.Spec.DatasetRef.Path},
							{Name: envDatasetVersion, Value: model.Spec.Lineage.DatasetVersion},
							{Name: envPreparationOutputKind, Value: model.Spec.Preparation.OutputRef.Kind},
							{Name: envPreparationOutputName, Value: model.Spec.Preparation.OutputRef.Name},
							{Name: envPreparationOutputPath, Value: model.Spec.Preparation.OutputRef.Path},
							{Name: envPreparedDatasetVersion, Value: model.Spec.Lineage.PreparedDatasetVersion},
							{Name: envTrainingImageDigest, Value: model.Spec.Lineage.TrainingImageDigest},
							{Name: envConfigurationDigest, Value: model.Spec.Lineage.ConfigurationDigest},
						},
					}},
				},
			},
		},
	}
}

func buildEvaluationJob(model *ledgerv1alpha1.RiskModel) *batchv1.Job {
	labels := map[string]string{
		labelRiskModelName: model.Name,
		labelComponent:     componentEvaluation,
	}
	backoffLimit := int32(1)
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      evaluationJobName(model.Name),
			Namespace: model.Namespace,
			Labels:    labels,
			Annotations: map[string]string{
				annotationLineageHash: lineageHash(model),
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{{
						Name:      evaluationContainerName,
						Image:     model.Spec.Evaluation.Image,
						Resources: trainingContainerResources(model.Spec.Evaluation.Resources),
						Env: []corev1.EnvVar{
							{Name: envOutputKind, Value: model.Spec.OutputRef.Kind},
							{Name: envOutputName, Value: model.Spec.OutputRef.Name},
							{Name: envOutputPath, Value: model.Spec.OutputRef.Path},
							{Name: envOutputVersion, Value: model.Spec.Lineage.OutputArtifactVersion},
							{Name: envEvaluationPath, Value: model.Spec.OutputRef.Path + "/" + model.Spec.Lineage.OutputArtifactVersion + "/evaluation-lineage.json"},
							{Name: envTrainingImageDigest, Value: model.Spec.Lineage.TrainingImageDigest},
							{Name: envConfigurationDigest, Value: model.Spec.Lineage.ConfigurationDigest},
							{Name: envEvaluationMinRecall, Value: strconv.FormatFloat(float64(model.Spec.Evaluation.MinRecallBPS)/10000, 'f', -1, 64)},
							{Name: envEvaluationMinPRAUC, Value: strconv.FormatFloat(float64(model.Spec.Evaluation.MinPRAUCBPS)/10000, 'f', -1, 64)},
							{Name: envEvaluationMaxFalseNegatives, Value: strconv.Itoa(int(model.Spec.Evaluation.MaxFalseNegatives))},
						},
					}},
				},
			},
		},
	}
}

func buildTrainingJob(model *ledgerv1alpha1.RiskModel) *batchv1.Job {
	labels := map[string]string{
		labelRiskModelName: model.Name,
		labelComponent:     componentTraining,
	}

	backoffLimit := int32(1)
	lineage := lineageHash(model)
	datasetKind := model.Spec.DatasetRef.Kind
	datasetName := model.Spec.DatasetRef.Name
	datasetPath := model.Spec.DatasetRef.Path
	datasetVersion := model.Spec.Lineage.DatasetVersion
	if model.Spec.Preparation.Enabled {
		datasetKind = model.Spec.Preparation.OutputRef.Kind
		datasetName = model.Spec.Preparation.OutputRef.Name
		datasetPath = model.Spec.Preparation.OutputRef.Path
		datasetVersion = model.Spec.Lineage.PreparedDatasetVersion
	}

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
								{Name: envDatasetKind, Value: datasetKind},
								{Name: envDatasetName, Value: datasetName},
								{Name: envDatasetPath, Value: datasetPath},
								{Name: envDatasetVersion, Value: datasetVersion},
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
	return evaluateResourcePolicyFor(model.Spec.Resources, model.Spec.Policy.ResourceBounds, "training")
}

func evaluateResourcePolicyFor(
	resources corev1.ResourceRequirements,
	bounds ledgerv1alpha1.ResourceBoundsPolicy,
	workload string,
) *policyViolation {
	maxCPU, err := resource.ParseQuantity(bounds.MaxCPU)
	if err != nil {
		return &policyViolation{Reason: "InvalidPolicy", Message: "policy.resourceBounds.maxCPU is invalid"}
	}
	maxMemory, err := resource.ParseQuantity(bounds.MaxMemory)
	if err != nil {
		return &policyViolation{Reason: "InvalidPolicy", Message: "policy.resourceBounds.maxMemory is invalid"}
	}
	maxGPU, err := resource.ParseQuantity(bounds.MaxGPU)
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

	for name := range resources.Requests {
		if _, ok := allowedRequests[name]; !ok {
			return &policyViolation{
				Reason:  "UnsupportedResourceRequest",
				Message: fmt.Sprintf("resource request %q is not allowed by governance policy", name),
			}
		}
	}
	for name := range resources.Limits {
		if _, ok := allowedLimits[name]; !ok {
			return &policyViolation{
				Reason:  "UnsupportedResourceLimit",
				Message: fmt.Sprintf("resource limit %q is not allowed by governance policy", name),
			}
		}
	}

	if reqCPU, ok := resources.Requests[corev1.ResourceCPU]; ok && reqCPU.Cmp(maxCPU) > 0 {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("%s cpu request %s exceeds policy maxCPU %s", workload, reqCPU.String(), maxCPU.String()),
		}
	}
	if limitCPU, ok := resources.Limits[corev1.ResourceCPU]; ok && limitCPU.Cmp(maxCPU) > 0 {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("%s cpu limit %s exceeds policy maxCPU %s", workload, limitCPU.String(), maxCPU.String()),
		}
	}
	if reqMem, ok := resources.Requests[corev1.ResourceMemory]; ok && reqMem.Cmp(maxMemory) > 0 {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("%s memory request %s exceeds policy maxMemory %s", workload, reqMem.String(), maxMemory.String()),
		}
	}
	if limitMem, ok := resources.Limits[corev1.ResourceMemory]; ok && limitMem.Cmp(maxMemory) > 0 {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("%s memory limit %s exceeds policy maxMemory %s", workload, limitMem.String(), maxMemory.String()),
		}
	}
	if reqGPU, ok := resources.Requests[gpuResourceName]; ok {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("gpu request %s is not supported; specify GPU only in limits and within maxGPU %s", reqGPU.String(), maxGPU.String()),
		}
	}
	if limitGPU, ok := resources.Limits[gpuResourceName]; ok && limitGPU.Cmp(maxGPU) > 0 {
		return &policyViolation{
			Reason:  "ResourcePolicyViolation",
			Message: fmt.Sprintf("%s gpu limit %s exceeds policy maxGPU %s", workload, limitGPU.String(), maxGPU.String()),
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

func preparationStatusFromJob(job *batchv1.Job) (ledgerv1alpha1.RiskModelPhase, string, string, bool, bool) {
	if cond := findJobCondition(job, batchv1.JobFailed); cond != nil && cond.Status == corev1.ConditionTrue {
		reason := cond.Reason
		if reason == "" {
			reason = "PreparationFailed"
		}
		message := cond.Message
		if message == "" {
			message = "Dataset preparation job failed."
		}
		return ledgerv1alpha1.RiskModelPhasePreparationFailed, reason, message, false, true
	}
	if cond := findJobCondition(job, batchv1.JobComplete); cond != nil && cond.Status == corev1.ConditionTrue {
		return ledgerv1alpha1.RiskModelPhasePreparationSucceeded, "PreparationSucceeded", "Dataset preparation job completed successfully.", true, false
	}
	if job.Status.Active > 0 {
		return ledgerv1alpha1.RiskModelPhasePreparationRunning, "PreparationRunning", fmt.Sprintf("Dataset preparation job has %d active pod(s).", job.Status.Active), false, false
	}
	if job.Status.Succeeded > 0 {
		return ledgerv1alpha1.RiskModelPhasePreparationSucceeded, "PreparationSucceeded", "Dataset preparation job completed successfully.", true, false
	}
	if job.Status.Failed > 0 {
		return ledgerv1alpha1.RiskModelPhasePreparationFailed, "PreparationFailed", "Dataset preparation job failed.", false, true
	}
	return ledgerv1alpha1.RiskModelPhasePreparationPending, "PreparationPending", "Dataset preparation job created; waiting to start.", false, false
}

func evaluationStatusFromJob(job *batchv1.Job) (ledgerv1alpha1.RiskModelPhase, string, string, bool, bool) {
	if cond := findJobCondition(job, batchv1.JobFailed); cond != nil && cond.Status == corev1.ConditionTrue {
		reason := cond.Reason
		if reason == "" {
			reason = "EvaluationFailed"
		}
		message := cond.Message
		if message == "" {
			message = "Evaluation job failed or a quality gate did not pass."
		}
		return ledgerv1alpha1.RiskModelPhaseEvaluationFailed, reason, message, false, true
	}
	if cond := findJobCondition(job, batchv1.JobComplete); cond != nil && cond.Status == corev1.ConditionTrue {
		return ledgerv1alpha1.RiskModelPhaseEvaluationSucceeded, "EvaluationSucceeded", "All configured model quality gates passed.", true, false
	}
	if job.Status.Active > 0 {
		return ledgerv1alpha1.RiskModelPhaseEvaluationRunning, "EvaluationRunning", fmt.Sprintf("Evaluation job has %d active pod(s).", job.Status.Active), false, false
	}
	if job.Status.Succeeded > 0 {
		return ledgerv1alpha1.RiskModelPhaseEvaluationSucceeded, "EvaluationSucceeded", "All configured model quality gates passed.", true, false
	}
	if job.Status.Failed > 0 {
		return ledgerv1alpha1.RiskModelPhaseEvaluationFailed, "EvaluationFailed", "Evaluation job failed or a quality gate did not pass.", false, true
	}
	return ledgerv1alpha1.RiskModelPhaseEvaluationPending, "EvaluationPending", "Evaluation job created; waiting to start.", false, false
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

func preparationJobName(modelName string) string {
	const maxNameLength = 63
	name := modelName + preparationJobNameSuffix
	if len(name) <= maxNameLength {
		return name
	}
	return name[:maxNameLength]
}

func evaluationJobName(modelName string) string {
	const maxNameLength = 63
	name := modelName + evaluationJobNameSuffix
	if len(name) <= maxNameLength {
		return name
	}
	return name[:maxNameLength]
}

func promotionRecordName(modelName string) string {
	const maxNameLength = 63
	name := modelName + promotionRecordNameSuffix
	if len(name) <= maxNameLength {
		return name
	}
	return name[:maxNameLength]
}

func deploymentName(modelName string) string {
	const maxNameLength = 63
	name := modelName + "-shadow"
	if len(name) <= maxNameLength {
		return name
	}
	return name[:maxNameLength]
}

func serviceName(modelName string) string {
	const maxNameLength = 63
	name := modelName + "-shadow"
	if len(name) <= maxNameLength {
		return name
	}
	return name[:maxNameLength]
}

func routeName(modelName string) string {
	const maxNameLength = 63
	name := modelName + "-canary"
	if len(name) <= maxNameLength {
		return name
	}
	return name[:maxNameLength]
}

func driftCronJobName(modelName string) string {
	const maxNameLength = 63
	name := modelName + "-drift"
	if len(name) <= maxNameLength {
		return name
	}
	return name[:maxNameLength]
}

func lineageHash(model *ledgerv1alpha1.RiskModel) string {
	payload, _ := json.Marshal(struct {
		Task            string                                `json:"task"`
		TrainingImage   string                                `json:"trainingImage"`
		DatasetRef      ledgerv1alpha1.DatasetReference       `json:"datasetRef"`
		Preparation     ledgerv1alpha1.DatasetPreparationSpec `json:"preparation"`
		Evaluation      ledgerv1alpha1.EvaluationPolicy       `json:"evaluation"`
		DriftMonitoring ledgerv1alpha1.DriftMonitoringSpec    `json:"driftMonitoring"`
		OutputRef       ledgerv1alpha1.ArtifactReference      `json:"outputRef"`
		Lineage         ledgerv1alpha1.RiskModelLineage       `json:"lineage"`
		Resources       corev1.ResourceRequirements           `json:"resources"`
		Serving         ledgerv1alpha1.ServingConfig          `json:"serving"`
		Policy          ledgerv1alpha1.GovernancePolicy       `json:"policy"`
	}{
		Task:            model.Spec.Task,
		TrainingImage:   model.Spec.TrainingImage,
		DatasetRef:      model.Spec.DatasetRef,
		Preparation:     model.Spec.Preparation,
		Evaluation:      model.Spec.Evaluation,
		DriftMonitoring: model.Spec.DriftMonitoring,
		OutputRef:       model.Spec.OutputRef,
		Lineage:         model.Spec.Lineage,
		Resources:       model.Spec.Resources,
		Serving:         model.Spec.Serving,
		Policy:          model.Spec.Policy,
	})
	sum := sha256.Sum256(payload)
	return "sha256:" + hex.EncodeToString(sum[:])
}
