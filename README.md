# HD_maintenance

# What is HD_maintenance?
HD_maintenance is a service for Hard Drive maintenance that issues HDs' replacement orders if an HD's remaining useful life time is expected to be over. This is accomplished by calculating a major fault prediction probability. HDs replacement orders are issued if the fault probability is higher than a threshold configured by the maintenance manager. This maitenance service is useful for large data storage providers as a means to reduce operational costs and optimise maintenance work by antecipating the HD's failure and thus enabling its premature replacement.

# Machine Learning for fault probability
A supervised learning approach is used where a neural network learns how to calculate a major fault probability by regression. The training data sets are attached in this repo.

# Neural network approach
A three layer Multi-Layer Perceptron is used. The MLP has:
- 6 input neurons: smart_1_raw, smart_5_raw, smart_9_raw, smart_194_raw, smart_197_raw, failure 
- 2 hidden layers with 4 neurons each (no under fitting, no over fitting)
- 1 output layer with only one neuron containing a probability of failure

# Training
The training is done in batches of size 1 by evaluating the Mean Squared Error of the difference between the actual value of the failure field (0 or 1) in the data set and the predicted percentage by MLP, for each sample of the training data set.

# Running instructions
Run the script XXX one time. Then use the last command, testNN(4,0.3) [testNN(HD_ID,fault_threshold)], on the console to test different values of HD_ID and fault_threshold. The output is the fault prediction probability.
