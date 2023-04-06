package net

import (
	"reflect"
	"testing"
)

type Act struct{}

func (a Act) calcActivation(matrix Matrix) Matrix {
	return make([][]float64, 1)
}
func (a Act) calcBackProp(matrix Matrix) Matrix {
	return make([][]float64, 1)
}

func TestNetCreation(t *testing.T) {
	layers := []LayerDef{{Input_dim: 1, Output_dim: 1, Activation: Act{}}}
	t.Run("Net creation learning rate", func(t *testing.T) {
		got := NewNet(0.1, MSE, layers)
		want := &Net{lr: 0.1}
		if got.lr != want.lr {
			t.Errorf("got %f but wanted %f", got.lr, want.lr)
		}
	})
}

func TestLayerForward(t *testing.T) {
	weights := Matrix{{1.0, 1.0}, {1.0, 1.0}}
	bias := Matrix{{1.0}, {1.0}}
	layers := Layer{weights, bias, ReLU}
	cache := map[int]CacheVal{}
	t.Run("Single Layer forward with 1 Layer", func(t *testing.T) {
		net := &Net{0.1, []Layer{layers}, cache, MSE}
		in := Matrix{{1.0}, {1.0}}
		got := net.forward(in)
		want := Matrix{{3.0}, {3.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
	t.Run("Layer forward with 2 Layer", func(t *testing.T) {
		net := &Net{0.1, []Layer{layers, layers}, cache, MSE}
		in := Matrix{{1.0}, {1.0}}
		got := net.forward(in)
		want := Matrix{{7.0}, {7.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
	t.Run("Cache forward with 2 Layer", func(t *testing.T) {
		net := &Net{0.1, []Layer{layers, layers}, cache, MSE}
		in := Matrix{{1.0}, {1.0}}
		net.forward(in)
		got := len(net.cache)
		want := 2
		if got != want {
			t.Errorf("got %d but wanted %d", got, want)
		}
	})
}

func TestSingleLayerForward(t *testing.T) {
	weights := Matrix{{1.0, 1.0}, {1.0, 1.0}}
	bias := Matrix{{1.0}, {1.0}}
	layers := Layer{weights, bias, ReLU}
	t.Run("Single Layer forward activated func", func(t *testing.T) {
		in := Matrix{{1.0}, {1.0}}
		got, _ := single_layer_forward(in, layers)
		want := Matrix{{3.0}, {3.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
	t.Run("Single Layer forward neg value", func(t *testing.T) {
		in := Matrix{{-1.0}, {1.0}}
		got, _ := single_layer_forward(in, layers)
		want := Matrix{{1.0}, {1.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
	t.Run("Single Layer forward z curr", func(t *testing.T) {
		in := Matrix{{1.0}, {1.0}}
		_, got := single_layer_forward(in, layers)
		want := Matrix{{3.0}, {3.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
}
