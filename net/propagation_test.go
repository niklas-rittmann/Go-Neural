package net

import (
	"reflect"
	"testing"
)

func TestLayerForward(t *testing.T) {
	weights := Matrix{{1.0, 1.0}, {1.0, 1.0}}
	bias := Matrix{{1.0}, {1.0}}
	layers := Layer{weights, bias, ReLU}
	cache := map[int]CacheVal{}
	t.Run("Single Layer forward with 1 Layer", func(t *testing.T) {
		net := &Net{0.1, []Layer{layers}, cache, QuadraticCost}
		in := Matrix{{1.0}, {1.0}}
		got := net.forward(in)
		want := Matrix{{3.0}, {3.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
	t.Run("Layer forward with 2 Layer", func(t *testing.T) {
		net := &Net{0.1, []Layer{layers, layers}, cache, QuadraticCost}
		in := Matrix{{1.0}, {1.0}}
		got := net.forward(in)
		want := Matrix{{7.0}, {7.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
	t.Run("Cache forward with 2 Layer", func(t *testing.T) {
		net := &Net{0.1, []Layer{layers, layers}, cache, QuadraticCost}
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

func TestTrain(t *testing.T) {
	layers := []LayerDef{{Input_dim: 2, Output_dim: 1, Activation: Sigmoid}}
	x := Matrix{{1, 1}, {1, 1}}
	y := Matrix{{0, 1}}
	net := NewNet(20, QuadraticCost, layers)
	net_copy_layers := net.Layers[0].Weights
	net.Train(x, y, 1)
	net_layers := net.Layers[0].Weights
	if reflect.DeepEqual(net_layers, net_copy_layers) {
		t.Errorf("Train did not have an effect %v, %v", net, net_copy_layers)
	}
}

func TestPredict(t *testing.T) {
	x := Matrix{{1, 1}, {1, 1}}
	layers := []Layer{{Weights: Matrix{{1.0, 1.0}}, Bias: Matrix{{1.0, 1.0}}, Activation: ReLU}}
	cache := make(map[int]CacheVal)
	net := &Net{Lr: 1.0, Layers: layers, cache: cache, Cost: QuadraticCost}
	got := net.Predict(x)
	want := Matrix{{3, 3}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Got %v but wanted %v", got, want)
	}
}
