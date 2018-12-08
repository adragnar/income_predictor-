import argparse
from time import time

import scipy.signal as sig
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *

import matplotlib.pyplot as plt

""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0

# =================================== LOAD DATASET =========================================== #

######

# 2.1 YOUR CODE HERE
data = pd.read_csv("./data/adult.csv")

######

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

######

# 2.2 YOUR CODE HERE

print("shape is ", data.shape)
print("columns are:", data.columns)
print("first 5 rows: \n", data.head())
print(data["income"].value_counts())

######


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
for feature in col_names:
    ######

    # 2.3 YOUR CODE HERE
    print("For column", feature, data[feature].isin(["?"]).sum())
    ######
print(3)
# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

    ######

    # 2.3 YOUR CODE HERE
for feature in col_names:
    try:
        data = data.loc[data[str(feature)] != "?"]
        print(data.shape)
    except:  #If the column is all numbers and str comparisons dont work
        pass
print("cleaned shape: ", data.shape)
    ######

# =================================== BALANCE DATASET =========================================== #

    ######

    # 2.4 YOUR CODE HERE
seed = 1738
overrep_class = data.loc[data["income"] == "<=50K" ]
underrep_class = data.loc[data["income"] == ">50K" ]

overrep_class = overrep_class.sample(underrep_class.shape[0], random_state=seed)
data = pd.concat([overrep_class, underrep_class])
    ######

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######

# 2.5 YOUR CODE HERE
print(data["age"].describe())
print(data["hours-per-week"].describe())
######

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    ######

    # 2.5 YOUR CODE HERE
    freq_list = data[feature].value_counts()
    print(feature, ": ", freq_list)
    ######

# visualize the first 3 features using pie and bar graphs

######

# 2.5 YOUR CODE HERE
    #pie_chart(data,feature)
    #binary_bar_chart(data, feature)
######

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# NORMALIZE CONTINUOUS FEATURES
######

# 2.6 YOUR CODE HERE
continous_feats_names = ["age","fnlwgt","educational-num","capital-gain", "capital-loss", "hours-per-week"]
continous_feats_norm = data[continous_feats_names]

for feature in continous_feats_names :
    if feature != "income":
        continous_feats_norm[feature] = continous_feats_norm.loc[:,feature] - continous_feats_norm.loc[:,feature].mean()
        continous_feats_norm[feature] = continous_feats_norm.loc[:,feature]/continous_feats_norm.loc[:,feature].std()

continous_feats_norm_vals = continous_feats_norm.values
######

# ENCODE CATEGORICAL FEATURES
label_encoder = LabelEncoder()
######

# 2.6 YOUR CODE HERE
categorical_feats_vals = data[categorical_feats]
for feature in categorical_feats :
    categorical_feats_vals[feature] = label_encoder.fit_transform(categorical_feats_vals[feature])
######

oneh_encoder = OneHotEncoder()
######

# 2.6 YOUR CODE HERE

######
income_vals_data =categorical_feats_vals[["income"]].values
del categorical_feats_vals["income"]

categorical_feats_vals= oneh_encoder.fit_transform(categorical_feats_vals).toarray()  #convert np cat_feats int to np cat_feats one_h (cols get expanded)
final_features_data = np.concatenate((continous_feats_norm_vals, categorical_feats_vals), axis=1)

# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######

# 2.7 YOUR CODE HERE
train_data, train_labels, val_data, val_labels = train_test_split(final_features_data, income_vals_data, test_size=0.2, random_state=seed)  #seed defined earlier
######
# =================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size, train_data, val_data, train_labels, val_labels):
    ######  DO I GIVE THE VALIDATION SET A BATCH SIZE?

    # 3.2 YOUR CODE HERE
    train_data = AdultDataset(train_data, train_labels)
    val_data = AdultDataset(val_data, val_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    ######


    return train_loader, val_loader


def load_model(lr, hid_lay_size):

    ######

    # 3.4 YOUR CODE HERE
    model = MultiLayerPerceptron(final_features_data.shape[1], hid_lay_size)  #initialize with #of features it takes in
    loss_fnc = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ######

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    ######

    # 3.6 YOUR CODE HERE
    for i, batch in enumerate(val_loader) :
        feats, label = batch
        predictions = model.forward(feats)
        corr_num = (predictions > 0.5).squeeze().long().reshape(-1, 1) == label
        corr_num = int(corr_num.sum())
        total_corr += corr_num

    print("Total validation accurracy over last batches is ", (float(total_corr)/len(val_loader.dataset)), "\n")
    ######

    return float(total_corr)/len(val_loader.dataset)


##Plotting
def plot_graph(path, type, config, train_data, val_data):
    """
    Plot the training loss/error curve given the data from CSV
    """
    plt.figure()
    type_title = "Error" if type == "err" else "Loss"
    plt.title("{} over training epochs".format(type_title))
    plt.plot(train_data["epoch"], train_data["train_{}".format(type)], label="Train")
    plt.plot(val_data["epoch"], val_data["val_{}".format(type)], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(type_title)
    plt.legend(loc='best')
    plt.savefig("{}_{}.png".format(type, path))

    return



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--hidden_layer_size', type=int)

    args = parser.parse_args()

    ######

    # 3.5 YOUR CODE HERE

    batch_size = 64#args.batch_size
    learn_rate = 0.1#args.lr
    MaxEpochs = 100#args.epochs
    hidden_layer_size = 64#args.hidden_layer_size
    eval_every = 1000#args.eval_every

    train_loader, val_loader = load_data(batch_size, train_data, train_labels, val_data, val_labels)
    model, loss_fnc, optimizer = load_model(learn_rate, hidden_layer_size)

    ##Variable lists for plotting
    leftover = 0
    train_step_list = []
    val_step_list = []
    train_data_list = []
    val_data_list = []
    time_step_list = []

    for counter, epoch in enumerate(range(MaxEpochs)) :
        accum_loss = 0
        tot_corr = 0
        tot_time_train = 0

        prev_accum_loss = 0  #for measuring total loss on moving window
        prev_tot_corr = 0
        prev_time_train = 0

        for i, batch in enumerate(train_loader) :
            start = time()

            feats, label = batch
            optimizer.zero_grad()

            predictions = model.forward(feats)
            batch_loss = loss_fnc(input=predictions.squeeze(), target=label.float())
            accum_loss += float(batch_loss)
            batch_loss.backward()
            optimizer.step()

            a = (predictions > 0.5).squeeze().long().view(-1,1)
            b = (a == label)
            corr_num = int(b.sum())
            tot_corr += corr_num

            end = time()
            tot_time_train += end-start

            # Print training stats
            if ((i % eval_every == 0) and (i!=0)):
                print("Total correct in last", eval_every, "batches at ", i , "batches is", tot_corr- prev_tot_corr, "out of ", eval_every*batch_size)
                #print("Total loss in last 10 batches after", i, "batches is", accum_loss-prev_accum_loss)
                val_acc = evaluate(model, val_loader)

                #Record relevant values
                if len(train_step_list) == 0:
                    train_step_list.append(0)
                else:
                    train_step_list.append(train_step_list[-1] + leftover + eval_every)
                train_data_list.append(float((tot_corr-prev_tot_corr)/(eval_every*batch_size)))

                if len(val_step_list) == 0:
                    val_step_list.append(0)
                else:
                    val_step_list.append(val_step_list[-1] + leftover + eval_every)
                val_data_list.append(val_acc)

                if len(time_step_list) == 0:
                    time_step_list.append(tot_time_train-prev_time_train)
                else:
                    time_step_list.append(time_step_list[-1] + (tot_time_train-prev_time_train))

                # prev_accum_loss = accum_loss
                prev_tot_corr = tot_corr
                prev_time_train = tot_time_train
            leftover = i % eval_every
        print("epoch ", counter, " complete \n")

    plt.figure()
    plt.title("Train Accuracy vs Batch Number")
    smoothed_train_data_list = sig.savgol_filter(train_data_list, 11, 5)
    plt.plot(train_step_list, smoothed_train_data_list, label="Train")
    plt.xlabel("Number of batches")
    plt.ylabel("Accuracy")
    plt.savefig("{}_{}_{}.png".format("bat", str(batch_size), "train"))

    plt.figure()
    plt.title("Validation Error vs Batch Number")
    smoothed_val_data_list = sig.savgol_filter(val_data_list, 11, 5)
    plt.plot(val_step_list, smoothed_val_data_list, label="Validation")
    plt.xlabel("Number of batches")
    plt.ylabel("Accuracy")
    plt.savefig("{}_{}_{}.png".format("bat", str(batch_size), "val"))


    plt.figure()
    plt.title("Train Accuracy vs Time")
    smoothed_train_data_list = sig.savgol_filter(train_data_list,11, 5)
    plt.plot(time_step_list, smoothed_train_data_list, label="Train")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.savefig("{}_{}_{}_{}.png".format("bat", str(batch_size), "train", "time"))

    plt.figure()
    plt.title("Validation Error vs Time")
    smoothed_val_data_list = sig.savgol_filter(val_data_list, 11, 5)
    plt.plot(time_step_list, smoothed_val_data_list, label="Validation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.savefig("{}_{}_{}_{}.png".format("bat", str(batch_size), "val", "time"))

    plt.figure()
    plt.title("Train Accuracy vs Time")
    smoothed_train_data_list = train_data_list
    plt.plot(time_step_list, smoothed_train_data_list, label="Train")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.savefig("{}_{}_{}_{}_{}.png".format("bat", str(batch_size), "train", "time", "no_smooth"))

    plt.figure()
    plt.title("Validation Error vs Time")
    smoothed_val_data_list = val_data_list
    plt.plot(time_step_list, smoothed_val_data_list, label="Validation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.savefig("{}_{}_{}_{}_{}.png".format("bat", str(batch_size), "val", "time", "no_smooth"))

    smoothed_val_data_list = sig.savgol_filter(val_data_list, 11, 5)
    smoothed_train_data_list = sig.savgol_filter(train_data_list, 11, 5)
    txtfile_name = "bat_" + str(batch_size) + "_maxacc.txt"
    file = open(txtfile_name, "w")
    file.write(str(max(smoothed_val_data_list)))
    file.write("\n training:")
    file.write(str(max(smoothed_train_data_list)))
    file.close()

    #for 4.6 write csv
    csv_data = pd.DataFrame({"tanh_time": time_step_list, "tanh_steps":train_step_list, "tanh_train": smoothed_train_data_list, "tanh_val": smoothed_val_data_list })
    csv_data.to_csv("tanh")


if __name__ == "__main__":
    main()