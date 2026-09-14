package v1alpha1

import (
	"fmt"
	"reflect"
	"slices"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const (
	// FraudScoringTask is the first supported Ledger ML task.
	FraudScoringTask = "fraud-scoring"
)

var supportedTasks = map[string]struct{}{
	FraudScoringTask: {},
}

// RiskModelPhase represents where a model is in the operator lifecycle.
type RiskModelPhase string

const (
	// RiskModelPhaseRejected indicates the model was rejected before training.
	RiskModelPhaseRejected RiskModelPhase = "Rejected"
	// RiskModelPhasePreparationPending indicates the preparation Job exists but has not started.
	RiskModelPhasePreparationPending RiskModelPhase = "PreparationPending"
	// RiskModelPhasePreparationRunning indicates the preparation Job has active pods.
	RiskModelPhasePreparationRunning RiskModelPhase = "PreparationRunning"
	// RiskModelPhasePreparationSucceeded indicates preparation completed successfully.
	RiskModelPhasePreparationSucceeded RiskModelPhase = "PreparationSucceeded"
	// RiskModelPhasePreparationFailed indicates preparation reached a failed terminal state.
	RiskModelPhasePreparationFailed RiskModelPhase = "PreparationFailed"
	// RiskModelPhaseEvaluationPending indicates the evaluation Job exists but has not started.
	RiskModelPhaseEvaluationPending RiskModelPhase = "EvaluationPending"
	// RiskModelPhaseEvaluationRunning indicates the evaluation Job has active pods.
	RiskModelPhaseEvaluationRunning RiskModelPhase = "EvaluationRunning"
	// RiskModelPhaseEvaluationSucceeded indicates all configured quality gates passed.
	RiskModelPhaseEvaluationSucceeded RiskModelPhase = "EvaluationSucceeded"
	// RiskModelPhaseEvaluationFailed indicates a quality gate or evaluation workload failed.
	RiskModelPhaseEvaluationFailed RiskModelPhase = "EvaluationFailed"
	// RiskModelPhaseTrainingPending indicates the training Job exists but has not started.
	RiskModelPhaseTrainingPending RiskModelPhase = "TrainingPending"
	// RiskModelPhaseTrainingRunning indicates the training Job has active pods.
	RiskModelPhaseTrainingRunning RiskModelPhase = "TrainingRunning"
	// RiskModelPhaseTrainingSucceeded indicates training completed and awaits governance outcome.
	RiskModelPhaseTrainingSucceeded RiskModelPhase = "TrainingSucceeded"
	// RiskModelPhaseTrainingFailed indicates the training Job reached a failed terminal state.
	RiskModelPhaseTrainingFailed RiskModelPhase = "TrainingFailed"
	// RiskModelPhaseAwaitingApproval indicates training succeeded but approvals are still pending.
	RiskModelPhaseAwaitingApproval RiskModelPhase = "AwaitingApproval"
	// RiskModelPhaseApproved indicates governance approval was observed.
	RiskModelPhaseApproved RiskModelPhase = "Approved"
	// RiskModelPhaseFailed is set when reconciliation hits a terminal controller failure.
	RiskModelPhaseFailed RiskModelPhase = "Failed"
)

// DatasetReference identifies training data inputs.
type DatasetReference struct {
	// Kind identifies how the dataset should be resolved.
	// +kubebuilder:validation:Enum=ConfigMap;PersistentVolumeClaim;ObjectStore
	// +kubebuilder:default=ConfigMap
	Kind string `json:"kind,omitempty"`
	// Name is the resource/bucket identifier.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`
	// Path is the key/path inside the dataset source.
	// +kubebuilder:validation:MinLength=1
	Path string `json:"path"`
}

// DatasetPreparationSpec defines an optional validation and curation Job that
// runs before model training.
type DatasetPreparationSpec struct {
	// Enabled controls whether the preparation Job is required.
	Enabled bool `json:"enabled,omitempty"`
	// Image is the immutable preparation workload image.
	Image string `json:"image,omitempty"`
	// OutputRef identifies the curated dataset location.
	OutputRef DatasetReference `json:"outputRef,omitempty"`
	// Resources declares resources for the preparation workload.
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`
}

// EvaluationPolicy defines measurable quality gates for a trained artifact.
type EvaluationPolicy struct {
	// Enabled controls whether an evaluator Job must pass before approval.
	Enabled bool `json:"enabled,omitempty"`
	// Image is the immutable evaluator workload image.
	Image string `json:"image,omitempty"`
	// Resources declares resources for the evaluator workload.
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`
	// MinRecallBPS is the minimum required recall in basis points (0-10000).
	// Zero disables the gate.
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=10000
	MinRecallBPS int32 `json:"minRecallBPS,omitempty"`
	// MinPRAUCBPS is the minimum required PR-AUC in basis points (0-10000).
	// Zero disables the gate.
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=10000
	MinPRAUCBPS int32 `json:"minPRAUCBPS,omitempty"`
	// MaxFalseNegatives is the maximum allowed false-negative count. Zero disables the gate.
	// +kubebuilder:validation:Minimum=0
	MaxFalseNegatives int32 `json:"maxFalseNegatives,omitempty"`
}

// DriftMonitoringSpec defines an optional scheduled runtime drift check.
type DriftMonitoringSpec struct {
	// Enabled controls whether drift monitoring is scheduled.
	Enabled bool `json:"enabled,omitempty"`
	// Image is the immutable drift detector workload image.
	Image string `json:"image,omitempty"`
	// Schedule is a Kubernetes CronJob schedule.
	Schedule string `json:"schedule,omitempty"`
	// BaselineRef points to the immutable training baseline JSON.
	BaselineRef DatasetReference `json:"baselineRef,omitempty"`
	// CurrentFeaturesRef points to the current feature window location.
	CurrentFeaturesRef DatasetReference `json:"currentFeaturesRef,omitempty"`
	// ReportRef points to the ConfigMap containing the latest detector report.
	ReportRef DatasetReference `json:"reportRef,omitempty"`
	// Resources declares resources for the detector workload.
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`
	// PSIThresholdBPS is the PSI alert threshold in basis points.
	// +kubebuilder:validation:Minimum=0
	PSIThresholdBPS int32 `json:"psiThresholdBPS,omitempty"`
	// MissingRateDeltaBPS is the missing-rate delta threshold in basis points.
	// +kubebuilder:validation:Minimum=0
	MissingRateDeltaBPS int32 `json:"missingRateDeltaBPS,omitempty"`
}

// OutcomeMonitoringSpec defines an optional delayed-label quality report.
type OutcomeMonitoringSpec struct {
	// Enabled controls whether outcome reports are consumed.
	Enabled bool `json:"enabled,omitempty"`
	// ReportRef identifies the ConfigMap and data key containing the report.
	ReportRef DatasetReference `json:"reportRef,omitempty"`
	// MinimumCoverageBPS is the minimum fraction of predictions with observed outcomes.
	// Zero disables the coverage condition.
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=10000
	MinimumCoverageBPS int32 `json:"minimumCoverageBPS,omitempty"`
	// MinimumRecallBPS is the minimum observed recall in basis points.
	// Zero disables the recall condition.
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=10000
	MinimumRecallBPS int32 `json:"minimumRecallBPS,omitempty"`
}

// ArtifactReference identifies where trained artifacts should be written.
type ArtifactReference struct {
	// Kind identifies the output backend.
	// +kubebuilder:validation:Enum=PersistentVolumeClaim;ObjectStore
	// +kubebuilder:default=PersistentVolumeClaim
	Kind string `json:"kind,omitempty"`
	// Name is the output target identifier.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`
	// Path is the output key/path prefix.
	// +kubebuilder:validation:MinLength=1
	Path string `json:"path"`
}

// ServingConfig captures runtime-serving intent for later milestones.
type ServingConfig struct {
	// Enabled toggles inference deployment.
	Enabled bool `json:"enabled,omitempty"`
	// Mode controls traffic exposure.
	// +kubebuilder:validation:Enum=Shadow;Canary
	// +kubebuilder:default=Shadow
	Mode string `json:"mode,omitempty"`
	// StableServiceName is the existing stable backend used by a canary route.
	StableServiceName string `json:"stableServiceName,omitempty"`
	// GatewayName is the Gateway API parent for a canary HTTPRoute.
	GatewayName string `json:"gatewayName,omitempty"`
	// RouteHost is the hostname matched by the canary HTTPRoute.
	RouteHost string `json:"routeHost,omitempty"`
	// CanaryWeightBPS is candidate traffic in basis points (0-10000).
	// Zero is the safe default and disables candidate traffic.
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=10000
	CanaryWeightBPS int32 `json:"canaryWeightBPS,omitempty"`
	// Image is the serving container image.
	Image string `json:"image,omitempty"`
	// Replicas is the desired serving replica count.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	Replicas *int32 `json:"replicas,omitempty"`
	// Port is the serving container port.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +kubebuilder:default=8080
	Port int32 `json:"port,omitempty"`
}

// RiskModelLineage captures immutable lineage identity.
type RiskModelLineage struct {
	// TrainingImageDigest is the immutable digest for spec.trainingImage content.
	// +kubebuilder:validation:Pattern="^sha256:[a-f0-9]{64}$"
	TrainingImageDigest string `json:"trainingImageDigest"`
	// DatasetVersion identifies an immutable dataset snapshot (commit/version/object generation).
	// +kubebuilder:validation:MinLength=1
	DatasetVersion string `json:"datasetVersion"`
	// OutputArtifactVersion identifies the intended immutable artifact version.
	// +kubebuilder:validation:MinLength=1
	OutputArtifactVersion string `json:"outputArtifactVersion"`
	// ConfigurationDigest identifies immutable training configuration.
	// +kubebuilder:validation:Pattern="^sha256:[a-f0-9]{64}$"
	ConfigurationDigest string `json:"configurationDigest"`
	// PreparedDatasetVersion identifies the immutable curated dataset produced by preparation.
	PreparedDatasetVersion string `json:"preparedDatasetVersion,omitempty"`
}

// ResourceBoundsPolicy defines allowed training resources.
type ResourceBoundsPolicy struct {
	// MaxCPU is the maximum allowed CPU request/limit.
	// +kubebuilder:default:="2"
	MaxCPU string `json:"maxCPU,omitempty"`
	// MaxMemory is the maximum allowed memory request/limit.
	// +kubebuilder:default:="4Gi"
	MaxMemory string `json:"maxMemory,omitempty"`
	// MaxGPU is the maximum allowed GPU limit (`nvidia.com/gpu`).
	// +kubebuilder:default:="0"
	MaxGPU string `json:"maxGPU,omitempty"`
}

// ApprovalPolicy defines approval semantics before promotion.
type ApprovalPolicy struct {
	// Required indicates if approval is needed after successful training.
	// +kubebuilder:default:=true
	Required *bool `json:"required,omitempty"`
	// MinimumApprovals is the number of unique approvals required when Required is true.
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default:=1
	MinimumApprovals int32 `json:"minimumApprovals,omitempty"`
	// AllowedApprovers limits approval identities when set.
	AllowedApprovers []string `json:"allowedApprovers,omitempty"`
}

// GovernancePolicy defines resource and approval controls.
type GovernancePolicy struct {
	ResourceBounds ResourceBoundsPolicy `json:"resourceBounds,omitempty"`
	Approval       ApprovalPolicy       `json:"approval,omitempty"`
}

// ModelApproval captures a human approval statement.
type ModelApproval struct {
	// Approver is the identity of the approver (user/group alias).
	// +kubebuilder:validation:MinLength=1
	Approver string `json:"approver"`
	// ApprovedAt is the approval timestamp.
	ApprovedAt metav1.Time `json:"approvedAt"`
	// Reference is an optional ticket or change request identifier.
	Reference string `json:"reference,omitempty"`
	// Comment is optional, non-PII approval context.
	Comment string `json:"comment,omitempty"`
}

// RiskModelSpec defines the desired state of RiskModel.
type RiskModelSpec struct {
	// Task defines the ML workflow to execute.
	// +kubebuilder:validation:Enum=fraud-scoring
	// +kubebuilder:default=fraud-scoring
	Task string `json:"task,omitempty"`
	// TrainingImage is the container image for the training job.
	// Mutable tags are not production-safe; pair with lineage.trainingImageDigest.
	// +kubebuilder:validation:MinLength=1
	TrainingImage string `json:"trainingImage"`
	// DatasetRef points to training dataset input.
	DatasetRef DatasetReference `json:"datasetRef"`
	// Preparation optionally materializes a validated curated dataset before training.
	Preparation DatasetPreparationSpec `json:"preparation,omitempty"`
	// Evaluation optionally gates approval on measurable model quality.
	Evaluation EvaluationPolicy `json:"evaluation,omitempty"`
	// DriftMonitoring optionally schedules runtime drift checks after promotion.
	DriftMonitoring DriftMonitoringSpec `json:"driftMonitoring,omitempty"`
	// OutcomeMonitoring optionally consumes delayed-label quality reports.
	OutcomeMonitoring OutcomeMonitoringSpec `json:"outcomeMonitoring,omitempty"`
	// OutputRef points to model artifact output location.
	OutputRef ArtifactReference `json:"outputRef"`
	// Lineage captures immutable identity (image digest, dataset version, artifact version, config digest).
	Lineage RiskModelLineage `json:"lineage"`
	// Resources declares container resource requests/limits for training.
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`
	// Serving captures deployment intent used by later milestones.
	Serving ServingConfig `json:"serving,omitempty"`
	// Policy controls resource safety and approval semantics.
	Policy GovernancePolicy `json:"policy,omitempty"`
	// Approvals are explicit human approvals observed by the controller.
	Approvals []ModelApproval `json:"approvals,omitempty"`
}

// RiskModelEvidence records governance-relevant lifecycle decisions.
type RiskModelEvidence struct {
	// Time is when the decision was observed.
	Time metav1.Time `json:"time"`
	// Decision is a machine-readable decision name.
	Decision string `json:"decision"`
	// Reason is a short decision reason.
	Reason string `json:"reason"`
	// Message provides operator-facing context without raw transaction data.
	Message string `json:"message,omitempty"`
	// JobName links a decision to a training Job when applicable.
	JobName string `json:"jobName,omitempty"`
	// ObservedGeneration links evidence to the reconciled generation.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// RiskModelStatus defines the observed state of RiskModel.
type RiskModelStatus struct {
	// Phase provides a high-level workflow stage.
	Phase RiskModelPhase `json:"phase,omitempty"`
	// Conditions contain granular reconciliation signals.
	Conditions []metav1.Condition `json:"conditions,omitempty"`
	// ObservedGeneration is the most recent reconciled metadata generation.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
	// LineageHash is the controller-calculated hash of immutable identity fields.
	LineageHash string `json:"lineageHash,omitempty"`
	// ModelVersion is populated once an immutable model artifact is produced.
	ModelVersion string `json:"modelVersion,omitempty"`
	// PromotionReference identifies the immutable promotion record after approval.
	PromotionReference string `json:"promotionReference,omitempty"`
	// PromotedAt records when the approved artifact was handed off for serving.
	PromotedAt *metav1.Time `json:"promotedAt,omitempty"`
	// ServingReference identifies the shadow Deployment after it is created.
	ServingReference string `json:"servingReference,omitempty"`
	// TrafficReference identifies the Gateway API route after canary configuration.
	TrafficReference string `json:"trafficReference,omitempty"`
	// DriftMonitoringReference identifies the scheduled drift CronJob.
	DriftMonitoringReference string `json:"driftMonitoringReference,omitempty"`
	// DriftReportReference identifies the latest observed drift report.
	DriftReportReference string `json:"driftReportReference,omitempty"`
	// DriftStatus is the latest report status: within_baseline or drift_detected.
	DriftStatus string `json:"driftStatus,omitempty"`
	// DriftObservedAt records when the latest report was observed.
	DriftObservedAt *metav1.Time `json:"driftObservedAt,omitempty"`
	// OutcomeReportReference identifies the latest observed outcome report.
	OutcomeReportReference string `json:"outcomeReportReference,omitempty"`
	// OutcomeCoverage and OutcomeRecall are the latest observed quality metrics.
	OutcomeCoverage *int32 `json:"outcomeCoverage,omitempty"`
	OutcomeRecall   *int32 `json:"outcomeRecall,omitempty"`
	// OutcomeObservedAt records when the latest outcome report was observed.
	OutcomeObservedAt *metav1.Time `json:"outcomeObservedAt,omitempty"`
	// ApprovedBy lists approvers counted toward policy.
	ApprovedBy []string `json:"approvedBy,omitempty"`
	// Evidence captures key governance decisions.
	Evidence []RiskModelEvidence `json:"evidence,omitempty"`
	// Reason is a machine-readable summary of the current phase.
	Reason string `json:"reason,omitempty"`
	// Message is a human-readable explanation of the current phase.
	Message string `json:"message,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=rm
// +kubebuilder:printcolumn:name="Task",type=string,JSONPath=`.spec.task`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Version",type=string,JSONPath=`.status.modelVersion`

// RiskModel is the Schema for the riskmodels API.
type RiskModel struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   RiskModelSpec   `json:"spec,omitempty"`
	Status RiskModelStatus `json:"status,omitempty"`
}

// Default sets defaults that mirror CRD defaults for local validation/testing.
func (r *RiskModel) Default() {
	if r.Spec.Task == "" {
		r.Spec.Task = FraudScoringTask
	}
	if r.Spec.DatasetRef.Kind == "" {
		r.Spec.DatasetRef.Kind = "ConfigMap"
	}
	if r.Spec.OutputRef.Kind == "" {
		r.Spec.OutputRef.Kind = "PersistentVolumeClaim"
	}
	if r.Spec.Serving.Replicas == nil {
		var replicas int32 = 1
		r.Spec.Serving.Replicas = &replicas
	}
	if r.Spec.Serving.Port == 0 {
		r.Spec.Serving.Port = 8080
	}
	if r.Spec.Serving.Mode == "" {
		r.Spec.Serving.Mode = "Shadow"
	}
	if r.Spec.Policy.ResourceBounds.MaxCPU == "" {
		r.Spec.Policy.ResourceBounds.MaxCPU = "2"
	}
	if r.Spec.Policy.ResourceBounds.MaxMemory == "" {
		r.Spec.Policy.ResourceBounds.MaxMemory = "4Gi"
	}
	if r.Spec.Policy.ResourceBounds.MaxGPU == "" {
		r.Spec.Policy.ResourceBounds.MaxGPU = "0"
	}
	if r.Spec.Policy.Approval.Required == nil {
		required := true
		r.Spec.Policy.Approval.Required = &required
	}
	if r.Spec.Policy.Approval.MinimumApprovals == 0 {
		r.Spec.Policy.Approval.MinimumApprovals = 1
	}
}

// ApprovalRequired indicates whether approvals are required for this model.
func (r *RiskModel) ApprovalRequired() bool {
	if r.Spec.Policy.Approval.Required == nil {
		return true
	}
	return *r.Spec.Policy.Approval.Required
}

// ValidateCreate validates a newly created RiskModel.
func (r *RiskModel) ValidateCreate() error {
	return r.validate()
}

// ValidateUpdate validates an updated RiskModel.
func (r *RiskModel) ValidateUpdate(old runtime.Object) error {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, r.validateErrorList()...)
	allErrs = append(allErrs, r.validateImmutableFields(old)...)
	if len(allErrs) == 0 {
		return nil
	}
	groupKind := schema.GroupKind{Group: GroupVersion.Group, Kind: "RiskModel"}
	return apierrors.NewInvalid(groupKind, r.Name, allErrs)
}

// ValidateDelete validates a deleted RiskModel.
func (r *RiskModel) ValidateDelete() error {
	return nil
}

func (r *RiskModel) validate() error {
	allErrs := r.validateErrorList()
	if len(allErrs) == 0 {
		return nil
	}

	groupKind := schema.GroupKind{Group: GroupVersion.Group, Kind: "RiskModel"}
	return apierrors.NewInvalid(groupKind, r.Name, allErrs)
}

func (r *RiskModel) validateErrorList() field.ErrorList {
	allErrs := field.ErrorList{}
	specPath := field.NewPath("spec")

	task := strings.TrimSpace(r.Spec.Task)
	if task == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("task"), "task is required"))
	} else {
		if _, ok := supportedTasks[task]; !ok {
			allErrs = append(allErrs, field.NotSupported(specPath.Child("task"), task, []string{FraudScoringTask}))
		}
	}

	if strings.TrimSpace(r.Spec.TrainingImage) == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("trainingImage"), "trainingImage is required"))
	}

	if strings.TrimSpace(r.Spec.DatasetRef.Name) == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("datasetRef", "name"), "datasetRef.name is required"))
	}
	if strings.TrimSpace(r.Spec.DatasetRef.Path) == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("datasetRef", "path"), "datasetRef.path is required"))
	}
	if r.Spec.Preparation.Enabled {
		if strings.TrimSpace(r.Spec.Preparation.Image) == "" {
			allErrs = append(allErrs, field.Required(specPath.Child("preparation", "image"), "preparation.image is required when preparation.enabled=true"))
		}
		if r.Spec.Evaluation.Enabled {
			if strings.TrimSpace(r.Spec.Evaluation.Image) == "" {
				allErrs = append(allErrs, field.Required(specPath.Child("evaluation", "image"), "evaluation.image is required when evaluation.enabled=true"))
			}
			if r.Spec.DriftMonitoring.Enabled {
				if strings.TrimSpace(r.Spec.DriftMonitoring.Image) == "" {
					allErrs = append(allErrs, field.Required(specPath.Child("driftMonitoring", "image"), "driftMonitoring.image is required when driftMonitoring.enabled=true"))
				}
				if r.Spec.OutcomeMonitoring.Enabled {
					if r.Spec.OutcomeMonitoring.ReportRef.Kind != "ConfigMap" ||
						strings.TrimSpace(r.Spec.OutcomeMonitoring.ReportRef.Name) == "" ||
						strings.TrimSpace(r.Spec.OutcomeMonitoring.ReportRef.Path) == "" {
						allErrs = append(allErrs, field.Invalid(specPath.Child("outcomeMonitoring", "reportRef"), r.Spec.OutcomeMonitoring.ReportRef, "reportRef must identify a ConfigMap name and data key when outcomeMonitoring.enabled=true"))
					}
					if r.Spec.OutcomeMonitoring.MinimumCoverageBPS < 0 || r.Spec.OutcomeMonitoring.MinimumCoverageBPS > 10000 ||
						r.Spec.OutcomeMonitoring.MinimumRecallBPS < 0 || r.Spec.OutcomeMonitoring.MinimumRecallBPS > 10000 {
						allErrs = append(allErrs, field.Invalid(specPath.Child("outcomeMonitoring"), r.Spec.OutcomeMonitoring, "outcome thresholds must be between 0 and 10000 basis points"))
					}
				}
				if strings.TrimSpace(r.Spec.DriftMonitoring.Schedule) == "" {
					allErrs = append(allErrs, field.Required(specPath.Child("driftMonitoring", "schedule"), "driftMonitoring.schedule is required when driftMonitoring.enabled=true"))
				}
				if strings.TrimSpace(r.Spec.DriftMonitoring.BaselineRef.Name) == "" || strings.TrimSpace(r.Spec.DriftMonitoring.BaselineRef.Path) == "" {
					allErrs = append(allErrs, field.Required(specPath.Child("driftMonitoring", "baselineRef"), "baselineRef.name and baselineRef.path are required when driftMonitoring.enabled=true"))
				}
				if strings.TrimSpace(r.Spec.DriftMonitoring.CurrentFeaturesRef.Name) == "" || strings.TrimSpace(r.Spec.DriftMonitoring.CurrentFeaturesRef.Path) == "" {
					allErrs = append(allErrs, field.Required(specPath.Child("driftMonitoring", "currentFeaturesRef"), "currentFeaturesRef.name and currentFeaturesRef.path are required when driftMonitoring.enabled=true"))
				}
				if r.Spec.DriftMonitoring.ReportRef.Kind != "ConfigMap" || strings.TrimSpace(r.Spec.DriftMonitoring.ReportRef.Name) == "" || strings.TrimSpace(r.Spec.DriftMonitoring.ReportRef.Path) == "" {
					allErrs = append(allErrs, field.Invalid(specPath.Child("driftMonitoring", "reportRef"), r.Spec.DriftMonitoring.ReportRef, "reportRef must identify a ConfigMap name and data key when driftMonitoring.enabled=true"))
				}
				if r.Spec.DriftMonitoring.PSIThresholdBPS < 0 || r.Spec.DriftMonitoring.MissingRateDeltaBPS < 0 {
					allErrs = append(allErrs, field.Invalid(specPath.Child("driftMonitoring"), r.Spec.DriftMonitoring, "drift thresholds must be >= 0"))
				}
			}
			if r.Spec.Evaluation.MinRecallBPS < 0 || r.Spec.Evaluation.MinRecallBPS > 10000 {
				allErrs = append(allErrs, field.Invalid(specPath.Child("evaluation", "minRecallBPS"), r.Spec.Evaluation.MinRecallBPS, "minRecallBPS must be between 0 and 10000"))
			}
			if r.Spec.Evaluation.MinPRAUCBPS < 0 || r.Spec.Evaluation.MinPRAUCBPS > 10000 {
				allErrs = append(allErrs, field.Invalid(specPath.Child("evaluation", "minPRAUCBPS"), r.Spec.Evaluation.MinPRAUCBPS, "minPRAUCBPS must be between 0 and 10000"))
			}
			if r.Spec.Evaluation.MaxFalseNegatives < 0 {
				allErrs = append(allErrs, field.Invalid(specPath.Child("evaluation", "maxFalseNegatives"), r.Spec.Evaluation.MaxFalseNegatives, "maxFalseNegatives must be >= 0"))
			}
		}
		if strings.TrimSpace(r.Spec.Preparation.OutputRef.Name) == "" {
			allErrs = append(allErrs, field.Required(specPath.Child("preparation", "outputRef", "name"), "preparation.outputRef.name is required when preparation.enabled=true"))
		}
		if strings.TrimSpace(r.Spec.Preparation.OutputRef.Path) == "" {
			allErrs = append(allErrs, field.Required(specPath.Child("preparation", "outputRef", "path"), "preparation.outputRef.path is required when preparation.enabled=true"))
		}
		if strings.TrimSpace(r.Spec.Lineage.PreparedDatasetVersion) == "" {
			allErrs = append(allErrs, field.Required(specPath.Child("lineage", "preparedDatasetVersion"), "lineage.preparedDatasetVersion is required when preparation.enabled=true"))
		}
	}

	if strings.TrimSpace(r.Spec.OutputRef.Name) == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("outputRef", "name"), "outputRef.name is required"))
	}
	if strings.TrimSpace(r.Spec.OutputRef.Path) == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("outputRef", "path"), "outputRef.path is required"))
	}

	if strings.TrimSpace(r.Spec.Lineage.TrainingImageDigest) == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("lineage", "trainingImageDigest"), "lineage.trainingImageDigest is required"))
	}
	if strings.TrimSpace(r.Spec.Lineage.DatasetVersion) == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("lineage", "datasetVersion"), "lineage.datasetVersion is required"))
	}
	if strings.TrimSpace(r.Spec.Lineage.OutputArtifactVersion) == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("lineage", "outputArtifactVersion"), "lineage.outputArtifactVersion is required"))
	}
	if strings.TrimSpace(r.Spec.Lineage.ConfigurationDigest) == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("lineage", "configurationDigest"), "lineage.configurationDigest is required"))
	}

	if r.Spec.Serving.Enabled && strings.TrimSpace(r.Spec.Serving.Image) == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("serving", "image"), "serving.image is required when serving.enabled=true"))
	}
	if r.Spec.Serving.Enabled && r.Spec.OutputRef.Kind != "PersistentVolumeClaim" {
		allErrs = append(allErrs, field.NotSupported(specPath.Child("outputRef", "kind"), r.Spec.OutputRef.Kind, []string{"PersistentVolumeClaim"}))
	}
	if r.Spec.Serving.Mode != "Shadow" && r.Spec.Serving.Mode != "Canary" {
		allErrs = append(allErrs, field.NotSupported(specPath.Child("serving", "mode"), r.Spec.Serving.Mode, []string{"Shadow", "Canary"}))
	}
	if r.Spec.Serving.CanaryWeightBPS < 0 || r.Spec.Serving.CanaryWeightBPS > 10000 {
		allErrs = append(allErrs, field.Invalid(specPath.Child("serving", "canaryWeightBPS"), r.Spec.Serving.CanaryWeightBPS, "canaryWeightBPS must be between 0 and 10000"))
	}
	if r.Spec.Serving.Enabled && r.Spec.Serving.Mode == "Canary" {
		if strings.TrimSpace(r.Spec.Serving.StableServiceName) == "" {
			allErrs = append(allErrs, field.Required(specPath.Child("serving", "stableServiceName"), "stableServiceName is required for canary serving"))
		}
		if strings.TrimSpace(r.Spec.Serving.GatewayName) == "" {
			allErrs = append(allErrs, field.Required(specPath.Child("serving", "gatewayName"), "gatewayName is required for canary serving"))
		}
		if strings.TrimSpace(r.Spec.Serving.RouteHost) == "" {
			allErrs = append(allErrs, field.Required(specPath.Child("serving", "routeHost"), "routeHost is required for canary serving"))
		}
	}

	if _, err := resource.ParseQuantity(r.Spec.Policy.ResourceBounds.MaxCPU); err != nil {
		allErrs = append(allErrs, field.Invalid(specPath.Child("policy", "resourceBounds", "maxCPU"), r.Spec.Policy.ResourceBounds.MaxCPU, "maxCPU must be a valid Kubernetes quantity"))
	}
	if _, err := resource.ParseQuantity(r.Spec.Policy.ResourceBounds.MaxMemory); err != nil {
		allErrs = append(allErrs, field.Invalid(specPath.Child("policy", "resourceBounds", "maxMemory"), r.Spec.Policy.ResourceBounds.MaxMemory, "maxMemory must be a valid Kubernetes quantity"))
	}
	if _, err := resource.ParseQuantity(r.Spec.Policy.ResourceBounds.MaxGPU); err != nil {
		allErrs = append(allErrs, field.Invalid(specPath.Child("policy", "resourceBounds", "maxGPU"), r.Spec.Policy.ResourceBounds.MaxGPU, "maxGPU must be a valid Kubernetes quantity"))
	}

	if r.ApprovalRequired() && r.Spec.Policy.Approval.MinimumApprovals < 1 {
		allErrs = append(allErrs, field.Invalid(specPath.Child("policy", "approval", "minimumApprovals"), r.Spec.Policy.Approval.MinimumApprovals, "minimumApprovals must be >= 1 when approval is required"))
	}

	approvers := slices.Clone(r.Spec.Policy.Approval.AllowedApprovers)
	slices.Sort(approvers)
	for i := 1; i < len(approvers); i++ {
		if approvers[i-1] == approvers[i] {
			allErrs = append(allErrs, field.Duplicate(specPath.Child("policy", "approval", "allowedApprovers"), approvers[i]))
		}
	}

	for i, approval := range r.Spec.Approvals {
		if strings.TrimSpace(approval.Approver) == "" {
			allErrs = append(allErrs, field.Required(specPath.Child("approvals").Index(i).Child("approver"), "approver is required"))
		}
	}

	return allErrs
}

func (r *RiskModel) validateImmutableFields(old runtime.Object) field.ErrorList {
	oldModel, ok := old.(*RiskModel)
	if !ok || oldModel == nil {
		return nil
	}

	allErrs := field.ErrorList{}
	specPath := field.NewPath("spec")

	immutableChecks := []struct {
		path string
		old  any
		new  any
	}{
		{path: "task", old: oldModel.Spec.Task, new: r.Spec.Task},
		{path: "trainingImage", old: oldModel.Spec.TrainingImage, new: r.Spec.TrainingImage},
		{path: "datasetRef", old: oldModel.Spec.DatasetRef, new: r.Spec.DatasetRef},
		{path: "preparation", old: oldModel.Spec.Preparation, new: r.Spec.Preparation},
		{path: "evaluation", old: oldModel.Spec.Evaluation, new: r.Spec.Evaluation},
		{path: "driftMonitoring", old: oldModel.Spec.DriftMonitoring, new: r.Spec.DriftMonitoring},
		{path: "outcomeMonitoring", old: oldModel.Spec.OutcomeMonitoring, new: r.Spec.OutcomeMonitoring},
		{path: "outputRef", old: oldModel.Spec.OutputRef, new: r.Spec.OutputRef},
		{path: "lineage", old: oldModel.Spec.Lineage, new: r.Spec.Lineage},
		{path: "resources", old: oldModel.Spec.Resources, new: r.Spec.Resources},
		{path: "serving", old: oldModel.Spec.Serving, new: r.Spec.Serving},
		{path: "policy", old: oldModel.Spec.Policy, new: r.Spec.Policy},
	}

	for _, check := range immutableChecks {
		if !reflect.DeepEqual(check.old, check.new) {
			allErrs = append(allErrs, field.Forbidden(specPath.Child(check.path), fmt.Sprintf("%s is immutable; create a new RiskModel for a new lineage", check.path)))
		}
	}

	return allErrs
}

// +kubebuilder:object:root=true

// RiskModelList contains a list of RiskModel.
type RiskModelList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []RiskModel `json:"items"`
}

func (r RiskModelPhase) String() string {
	return string(r)
}

func (r RiskModelStatus) Summary() string {
	if r.Reason == "" {
		return r.Phase.String()
	}
	if r.Message == "" {
		return fmt.Sprintf("%s: %s", r.Phase, r.Reason)
	}
	return fmt.Sprintf("%s: %s - %s", r.Phase, r.Reason, r.Message)
}
