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
