import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, hid_lay_size):

        super(MultiLayerPerceptron, self).__init__()

        ######

        # 3.3 YOUR CODE HERE
        self.fc_layer1 = nn.Linear(input_size, hid_lay_size)  #hidden layer has x inputs to each of its 20 neurons
        self.fc_layer2 = nn.Linear(hid_lay_size, 1)   #Output layer has 1 outputs, each connected to the 20 inputs from hidden layer

        self.underfit = nn.Linear(input_size, 1)
        self.overfit = nn.Linear(hid_lay_size, hid_lay_size)
        ######

    def forward(self, features):
        '''Features: A PyTorch tensor of row samples and y featuretypes '''
        pass
        ######

        # 3.3 YOUR CODE HERE
        features = features.float()
        x = self.fc_layer1(features)  #Input:raw input data. Output: raw values for 20 output neurons
        x = F.tanh(x)
        x = self.fc_layer2(x)  #Input - activation data from 20 hidden neurons. Ouptut - raw values for 2 output neruons
        x = F.sigmoid(x)  #Input: raw NN output data. Output: output neuron values with sigmoid function
        return x

        #Underfit Version
        # features = features.float()
        # x = self.underfit(features)
        # x = F.sigmoid(x)
        # return x

        #Overfit Numbers
        # features = features.float()
        # x = self.fc_layer1(features)
        # x = F.tanh(x)
        # x = self.overfit(x)
        # x = F.tanh(x)
        # x = self.overfit(x)
        # x = F.tanh(x)
        # x = self.fc_layer2(x)
        # x = F.sigmoid(x)
        # return x

        ######
