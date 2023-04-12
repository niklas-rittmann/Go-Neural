package net

import "testing"

func TestNameFromFunc(t *testing.T) {
	got := nameFromFunc(ReLU.ActivationFunc)
	want := "neural-go/net.relu"
	if got != want {
		t.Errorf("got %s but want %s", got, want)
	}
}

func TestBuildStore(t *testing.T) {
	CostPersistMap = map[string]Cost{}
	CostBackPersistMap = map[string]BackProp{}
	ActPersistMap = map[string]Activation{}
	ActBackPersistMap = map[string]Activation{}

	BuildStore()
	_, ok := ActPersistMap["neural-go/net.relu"]
	if !ok {
		t.Error("Element not initalized correctly")
	}
}
