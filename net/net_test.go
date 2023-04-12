package net

import (
	"reflect"
	"testing"
)

func TestNetCreation(t *testing.T) {
	layers := []LayerDef{{Input_dim: 1, Output_dim: 1, Activation: ActivationFunction{}}}
	t.Run("Net creation learning rate", func(t *testing.T) {
		got := NewNet(0.1, QuadraticLoss, layers)
		want := &Net{Lr: 0.1}
		if got.Lr != want.Lr {
			t.Errorf("got %f but wanted %f", got.Lr, want.Lr)
		}
	})
}

func TestPersistenseOfModel(t *testing.T) {

	layers := []LayerDef{{Input_dim: 1, Output_dim: 1, Activation: ReLU}}
	model := NewNet(0.5, QuadraticLoss, layers)
	model.ToFile("test.json")
	x := Matrix{{1}}
	want := model.Predict(x)
	loaded := NewNetFromFile("test.json")
	got := loaded.Predict(x)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Got %#v but wanted %#v", got, want)
	}

}
