
# Go-Neural :dna:
Go-Neural is a simple implementation of a Neural-Network written purely in Go. This implementation does not claim to be state of the art, but rather was a nice exercise to practice Go. 

## Getting started
The setup for Go-Neural is fairly simple. You just have to follow these steps:
1. Define the layers your network is supposed to have. Make sure to provide an activation function for each layer. Basic implementations for ReLU and Sigmoid can be found in the `net` package. 
2. Initialize a new network using `NewNet`. Make sure to provide a sufficient learning rate and loss-function (Mean-Squared-Error is already implemented).
3. Initialize your training data and labels. Make sure they are provided in the following format: 

|Sample 1 |Sample ...|Sample n| 
|--|--|--| 
|Feature 1|Feature 1|Feature 1|
|Feature ...|Feature ...|Feature ...|
|Feature m|Feature m|Feature m| 
>If rows and columns are switched, you can use the `Transpose` method. 

4. Train your model by setting a number of epochs to train.
5. Save your model in json Format via ´model.ToFile`
6. Predict new samples with your newly trained model. 

## To-Dos:
- [ ] Add a "sanity check for the matrix dimensions
- [x] Add methods to persist models on disk
- [ ] Add a visualizer for the training process (e.g. a loss curve) 

### Disclaimer
Use at your own risk. Implementation with the best knowledge and conscience.
