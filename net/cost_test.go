package main

import (
	"reflect"
	"testing"
)

func TestCostFunctions(t *testing.T) {
	arr1 := Matrix{{1.0}, {2.0}}
	arr2 := Matrix{{2.0}, {1.0}}
	t.Run("SqaredError Activation", func(t *testing.T) {
		got := mse(arr1, arr2)
		want := Matrix{{1.0}, {1.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %f but wanted %f", got, want)
		}
	})
	t.Run("SqaredError Backprop", func(t *testing.T) {
		got := mse_back(arr1, arr2)
		want := Matrix{{2.0}, {-2.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %f but wanted %f", got, want)
		}
	})
}

func TestCostFuncWrapper(t *testing.T) {
	arr1 := Matrix{{1.0}, {2.0}}
	arr2 := Matrix{{2.0}, {1.0}}
	fun := CostFunction{mse, mse_back}
	t.Run("SqaredError Activation", func(t *testing.T) {
		got := fun.calcCost(arr1, arr2)
		want := Matrix{{1.0}, {1.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %f but wanted %f", got, want)
		}
	})
	t.Run("SqaredError Activation", func(t *testing.T) {
		got := fun.calcBackProp(arr1, arr2)
		want := Matrix{{2.0}, {-2.0}}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %f but wanted %f", got, want)
		}
	})

}
