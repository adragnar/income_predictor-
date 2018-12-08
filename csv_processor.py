from util import *

import matplotlib.pyplot as plt


def csv_processor(filepath) :
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()

    time_list = []
    step_list = []
    train_data_list = []
    val_data_list = []
    for i, line in enumerate(lines):
        line = line.split(",")
        time_list.append(float(line[1]))
        step_list.append(float(line[2]))
        train_data_list.append(float(line[3]))
        val_data_list.append(float(line[4]))


    return time_list, step_list, train_data_list, val_data_list


def plot():

    relu_time, relu_step_list, relu_train_data_list, relu_val_data_list = csv_processor("/Users/RobertAdragna/Desktop/relu.txt")
    sig_time, sig_step_list, sig_train_data_list, sig_val_data_list = csv_processor("/Users/RobertAdragna/Desktop/sig.txt")
    tanh_time, tanh_step_list, tanh_train_data_list, tanh_val_data_list = csv_processor("/Users/RobertAdragna/Desktop/tanh.txt")



    plt.figure()
    plt.title("Train Accuracy vs Time")
    plt.plot(relu_step_list, relu_train_data_list, label="Relu")
    plt.plot(relu_step_list, sig_train_data_list, label="Sig")
    plt.plot(relu_step_list, tanh_train_data_list, label="Tanh")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("{}_{}.png".format("train", "4.6"))

    plt.figure()
    plt.title("Validation Accuracy vs Time")
    plt.plot(relu_step_list, relu_val_data_list, label="Relu")
    plt.plot(relu_step_list, sig_val_data_list, label="Sig")
    plt.plot(relu_step_list, tanh_val_data_list, label="Tanh")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("{}_{}.png".format("val", "4.6"))

if __name__ == "__main__":
    plot()