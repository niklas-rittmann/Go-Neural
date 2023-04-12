package net

import (
	"log"
	"reflect"
	"runtime"
)

var CostPersistMap = map[string]Cost{}
var CostBackPersistMap = map[string]BackProp{}
var ActPersistMap = map[string]Activation{}
var ActBackPersistMap = map[string]Activation{}

// This func returns the name of a function to store and retrieve the func from json
func nameFromFunc(fn interface{}) string {
	name := runtime.FuncForPC(reflect.ValueOf(fn).Pointer()).Name()
	return name
}

//Wrapper for storing Funtions in PersistMap
func StoreCostInMap(fns []CostFunction) {
	for _, fn := range fns {
		cost_func_name := nameFromFunc(fn.CostFunc)
		back_func_name := nameFromFunc(fn.BackpropFunc)
		CostPersistMap[cost_func_name] = fn.CostFunc
		CostBackPersistMap[back_func_name] = fn.BackpropFunc
	}
}
func StoreActInMap(fns []ActivationFunction) {
	for _, fn := range fns {
		act_func_name := nameFromFunc(fn.ActivationFunc)
		back_func_name := nameFromFunc(fn.BackpropFunc)
		ActPersistMap[act_func_name] = fn.ActivationFunc
		ActBackPersistMap[back_func_name] = fn.BackpropFunc
	}
}

// Build store
func BuildStore() {
	StoreCostInMap([]CostFunction{QuadraticCost})
	StoreActInMap([]ActivationFunction{ReLU, Sigmoid})
}

func init() {
	BuildStore()
}
