# This code gives you a class for a neural network with one hidden layer. 
# You can specify the number of hidden neurons as well as the training and
# validation data for the network. When finishing training the code returns
# the weights and thresholds through csv files.

# Written by: Albin Steen
# Date: 21-09-2023

import numpy as np
import pandas as pd
import os

class NetworkWithOneHiddenLayer:

    def __init__(self, M, training_data, validation_data):
        # Initialize weights and thresholds
        self.w1 = np.random.normal(0, 1 / np.sqrt(2), [M, 2])
        self.w2 = np.random.normal(0, 1 / np.sqrt(M), [M, 1])
        self.thresholds1 = np.zeros((M, 1))
        self.thresholds2 = 0

        # Load and preprocess training and validation data
        self.x_train, self.t_train, self.x_test, self.t_test = self.readData(training_data, validation_data)

    def readData(self, training_data, validation_data):
        # Read CSV data
        raw_train_data = pd.read_csv(training_data, header=None)
        raw_validation_data = pd.read_csv(validation_data, header=None)

        # Extract features and labels
        x_train_data = np.array(raw_train_data)[:, :2]
        t_train_data = np.array(raw_train_data)[:, -1]
        x_validation_data = np.array(raw_validation_data)[:, :2]
        t_validation_data = np.array(raw_validation_data)[:, -1]

        # Normalize training and validation data
        x_train_mean = np.mean(x_train_data, axis=0)
        x_train_std = np.std(x_train_data, axis=0)
        x_train_normalized = (x_train_data - x_train_mean) / x_train_std
        x_validation_mean = np.mean(x_validation_data, axis=0)
        x_validation_std = np.std(x_validation_data, axis=0)
        x_validation_normalized = (x_validation_data - x_validation_mean) / x_validation_std

        return x_train_normalized, t_train_data, x_validation_normalized, t_validation_data

    def returnData(self):
        # Save weights and thresholds to CSV files
        df_w1 = pd.DataFrame(self.w1)
        df_w1.to_csv('w1.csv', header=False, index=False)

        df_w2 = pd.DataFrame(self.w2)
        df_w2.to_csv('w2.csv', header=False, index=False)

        df_O1 = pd.DataFrame(self.thresholds1)
        df_O1.to_csv('t1.csv', header=False, index=False)

        df_O2 = pd.DataFrame(np.array([self.thresholds2]).reshape(1, -1))
        df_O2.to_csv('t2.csv', header=False, index=False)

    def activationFunction(self, x):
        return np.tanh(x)

    def derivativeOfActivationFunction(self, x):
        return 1 - self.activationFunction(x) ** 2

    def forwardPropagation(self, x_values):
        # Calculate forward propagation
        output_one = self.w1 @ x_values - self.thresholds1
        output_one = self.activationFunction(output_one)
        output_two = self.w2.T @ output_one - self.thresholds2
        output_two = self.activationFunction(output_two)
        return [output_one, output_two]

    def backwardsPropagation(self, forw_prop_values, x_values, t_val):
        # Calculate backward propagation
        delta_output_layer = self.derivativeOfActivationFunction(self.w2.T @ forw_prop_values[0] - self.thresholds2) * (t_val - forw_prop_values[1])
        delta_hidden_layer = (self.w2 * delta_output_layer) * self.derivativeOfActivationFunction(self.w1 @ x_values - self.thresholds1)
        return [delta_output_layer, delta_hidden_layer]

    def updatingParameters(self, x_values, forw_prop_values, back_prop_values, learning_rate):
        # Update weights and thresholds
        self.w2 += learning_rate * (back_prop_values[0] * forw_prop_values[0])
        self.w1 += learning_rate * (back_prop_values[1] * x_values.T)
        self.thresholds2 -= learning_rate * back_prop_values[0]
        self.thresholds1 -= learning_rate * back_prop_values[1]

    def trainNetwork(self, learning_rate=0.005):
        C = 1
        epoch = 0
        for _ in range(1000):
            epoch += 1
            for _ in range(len(self.x_train)):
                index = np.random.randint(len(self.x_train))
                x_vals, t_value = self.x_train[index].reshape(2, 1), self.t_train[index]
                outputs = self.forwardPropagation(x_vals)
                deltas = self.backwardsPropagation(outputs, x_vals, t_value)
                self.updatingParameters(x_vals, outputs, deltas, learning_rate)

            p_val = len(self.t_test)
            error_value = 0

            # Calculating accuracy on validation data
            for i in range(len(self.x_test)):
                pattern = self.x_test[i].reshape(2,1)
                target = self.t_test[i]
                output = self.forwardPropagation(pattern)[1]
                error_value += np.abs(np.sign(output) - target)

            C = error_value / (2 * p_val)

            print(f"Epoch number: {epoch}, Error: {C*100}%")
            if C < 0.12:
                self.returnData()
                break


def main(hidden_neurons = 16, learning_rate = 0.007):
    # Execution part of the code
    neural_net = NetworkWithOneHiddenLayer(hidden_neurons, 'training_set.csv', 'validation_set.csv') # 15 hidden neurons worked good for me
    neural_net.trainNetwork(learning_rate) # Learning rate of 0.01 worked best for me


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    main() # Enter other values for learning rate and number of hidden neurons here if you want to

