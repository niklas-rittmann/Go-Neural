package net

import (
	"reflect"
	"testing"
)

func TestActivationFunctions(t *testing.T) {
	t.Run("Sigmoid Activation", func(t *testing.T) {
		got := sigmoid(0)
		want := 0.5
		if got != want {
			t.Errorf("got %f but wanted %f", got, want)
		}
	})
	t.Run("Relu Function negative", func(t *testing.T) {
		got := relu(-1)
		want := 0.0
		if got != want {
			t.Errorf("got %f but wanted %f", got, want)
		}
	})
	t.Run("Relu Function positive", func(t *testing.T) {
		got := relu(1)
		want := 1.0
		if got != want {
			t.Errorf("got %f but wanted %f", got, want)
		}
	})
}
func TestDerivFunctions(t *testing.T) {
	t.Run("Sigmoid Deriv", func(t *testing.T) {
		got := sigmoid_deriv(0.0)
		want := 0.25
		if got != want {
			t.Errorf("got %f but wanted %f", got, want)
		}
	})
	t.Run("Relu Function Deriv", func(t *testing.T) {
		got := relu_back(-1)
		want := 0.0
		if got != want {
			t.Errorf("got %f but wanted %f", got, want)
		}
	})
}

func TestActivationWrapper(t *testing.T) {
	arr := Matrix{{0.0}, {0.0}, {0.0}}
	test_func := func(val float64) float64 { return 1.0 }
	got := wrapper(arr, test_func)
	want := Matrix{{1.0}, {1.0}, {1.0}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v but wanted %v", got, want)
	}

}

func TestProvidedActivations(t *testing.T) {
	t.Run("Sigmoid Activation", func(t *testing.T) {
		arr := Matrix{{0.0}, {0.0}, {0.0}}
		got := Sigmoid.calcActivation(arr)
		want := Matrix{{0.5}, {0.5}, {0.5}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
	t.Run("Relu Activation", func(t *testing.T) {
		arr := Matrix{{-1.0}, {0.0}, {1.0}}
		got := ReLU.calcActivation(arr)
		want := Matrix{{0.0}, {0.0}, {1.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
}

func TestProvidedDerivs(t *testing.T) {
	t.Run("Sigmoid Activation", func(t *testing.T) {
		arr := Matrix{{0.0}, {0.0}, {0.0}}
		got := Sigmoid.calcBackProp(arr)
		want := Matrix{{0.25}, {0.25}, {0.25}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
	t.Run("Relu Activation", func(t *testing.T) {
		arr := Matrix{{-1.0}, {0.0}, {1.0}}
		got := ReLU.calcBackProp(arr)
		want := Matrix{{0.0}, {0.0}, {1.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v but wanted %v", got, want)
		}
	})
}
