package net

import (
	"encoding/json"
	"io/ioutil"
)

// Struct for creating a Neural-Network
type Net struct {
	Lr     float64
	Layers []Layer
	cache  map[int]CacheVal
	Cost   CostFunction
}

type CacheVal struct {
	a Matrix
	z Matrix
}

func (n *Net) MarshalJSON() ([]byte, error) {
	type Alias Net
	return json.Marshal(&struct {
		Cost         string
		CostBackprop string
		*Alias
	}{
		Cost:         nameFromFunc(n.Cost.CostFunc),
		CostBackprop: nameFromFunc(n.Cost.BackpropFunc),
		Alias:        (*Alias)(n),
	})
}

func (n *Net) UnmarshalJSON(data []byte) error {
	type Alias Net
	aux := &struct {
		Cost         string
		CostBackprop string
		*Alias
	}{
		Alias: (*Alias)(n),
	}
	json.Unmarshal(data, &aux)
	n.Cost.CostFunc = CostPersistMap[aux.Cost]
	n.Cost.BackpropFunc = CostBackPersistMap[aux.CostBackprop]
	return nil
}

// Save net to file
func (n *Net) ToFile(filename string) {
	file, _ := json.MarshalIndent(n, "", " ")
	_ = ioutil.WriteFile(filename, file, 0644)
}

// Create a new Net based on the provided input
func NewNet(lr float64, cost CostFunction, layer_defs []LayerDef) *Net {
	layers := InitLayers(layer_defs)
	cache := make(map[int]CacheVal)
	return &Net{Lr: lr, Layers: layers, cache: cache, Cost: cost}
}

// Create a new Net based on the provided input
func NewNetFromFile(filename string) *Net {
	file, _ := ioutil.ReadFile(filename)
	newNet := &Net{}
	_ = json.Unmarshal([]byte(file), &newNet)
	cache := make(map[int]CacheVal)
	newNet.cache = cache
	return newNet
}
