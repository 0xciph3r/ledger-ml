package webhook

import (
	"context"
	"fmt"

	ledgerv1alpha1 "github.com/ledger-ml/ledger-ml/api/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// RiskModelValidator exposes the API validation contract through admission.
type RiskModelValidator struct{}

func (RiskModelValidator) ValidateCreate(_ context.Context, obj runtime.Object) (admission.Warnings, error) {
	model, ok := obj.(*ledgerv1alpha1.RiskModel)
	if !ok {
		return nil, fmt.Errorf("expected RiskModel, got %T", obj)
	}
	return nil, model.ValidateCreate()
}

func (RiskModelValidator) ValidateUpdate(_ context.Context, obj, oldObj runtime.Object) (admission.Warnings, error) {
	model, ok := obj.(*ledgerv1alpha1.RiskModel)
	if !ok {
		return nil, fmt.Errorf("expected RiskModel, got %T", obj)
	}
	return nil, model.ValidateUpdate(oldObj)
}

func (RiskModelValidator) ValidateDelete(_ context.Context, obj runtime.Object) (admission.Warnings, error) {
	model, ok := obj.(*ledgerv1alpha1.RiskModel)
	if !ok {
		return nil, fmt.Errorf("expected RiskModel, got %T", obj)
	}
	return nil, model.ValidateDelete()
}
