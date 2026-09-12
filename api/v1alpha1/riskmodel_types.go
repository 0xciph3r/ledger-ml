package v1alpha1

import (
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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
	// RiskModelPhaseTrainingPending indicates the training Job exists but has not started.
	RiskModelPhaseTrainingPending RiskModelPhase = "TrainingPending"
	// RiskModelPhaseTrainingRunning indicates the training Job has active pods.
	RiskModelPhaseTrainingRunning RiskModelPhase = "TrainingRunning"
	// RiskModelPhaseTrainingSucceeded indicates the training Job completed successfully.
	RiskModelPhaseTrainingSucceeded RiskModelPhase = "TrainingSucceeded"
	// RiskModelPhaseTrainingFailed indicates the training Job reached a failed terminal state.
	RiskModelPhaseTrainingFailed RiskModelPhase = "TrainingFailed"
	// RiskModelPhaseFailed is set when reconciliation reaches a terminal failure.
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

// RiskModelSpec defines the desired state of RiskModel.
type RiskModelSpec struct {
	// Task defines the ML workflow to execute.
	// +kubebuilder:validation:Enum=fraud-scoring
	// +kubebuilder:default=fraud-scoring
	Task string `json:"task,omitempty"`
	// TrainingImage is the container image for the training job.
	// +kubebuilder:validation:MinLength=1
	TrainingImage string `json:"trainingImage"`
	// DatasetRef points to training dataset input.
	DatasetRef DatasetReference `json:"datasetRef"`
	// OutputRef points to model artifact output location.
	OutputRef ArtifactReference `json:"outputRef"`
	// Resources declares container resource requests/limits for training.
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`
	// Serving captures deployment intent used by later milestones.
	Serving ServingConfig `json:"serving,omitempty"`
}

// RiskModelStatus defines the observed state of RiskModel.
type RiskModelStatus struct {
	// Phase provides a high-level workflow stage.
	Phase RiskModelPhase `json:"phase,omitempty"`
	// Conditions contain granular reconciliation signals.
	Conditions []metav1.Condition `json:"conditions,omitempty"`
	// ObservedGeneration is the most recent reconciled metadata generation.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
	// ModelVersion is populated once an immutable model artifact is produced.
	ModelVersion string `json:"modelVersion,omitempty"`
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
}

// ValidateCreate validates a newly created RiskModel.
func (r *RiskModel) ValidateCreate() error {
	return r.validate()
}

// ValidateUpdate validates an updated RiskModel.
func (r *RiskModel) ValidateUpdate(_ runtime.Object) error {
	return r.validate()
}

// ValidateDelete validates a deleted RiskModel.
func (r *RiskModel) ValidateDelete() error {
	return nil
}

func (r *RiskModel) validate() error {
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

	if r.Spec.Serving.Enabled && strings.TrimSpace(r.Spec.Serving.Image) == "" {
		allErrs = append(allErrs, field.Required(specPath.Child("serving", "image"), "serving.image is required when serving.enabled=true"))
	}

	if len(allErrs) == 0 {
		return nil
	}

	groupKind := schema.GroupKind{Group: GroupVersion.Group, Kind: "RiskModel"}
	return apierrors.NewInvalid(groupKind, r.Name, allErrs)
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
