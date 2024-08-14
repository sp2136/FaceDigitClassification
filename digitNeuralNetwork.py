import math
import numpy as np
import time
import matplotlib.pyplot as plt
from digitsloader import load_label, load_sample

def one_hot(data):
    for i in range(len(data)):
        sr = [0,0,0,0,0,0,0,0,0,0]
        sr[int(data[i])] = 1
        data[i] = np.array(sr)
    data=np.array(data)
    return data

def process_data(data_file, label_file):
    label, sample_num = load_label(label_file)
    data = load_sample(data_file, sample_num)
    label = one_hot(label)
    new_data=[]
    for i in range(len(data)):
        new_data.append(data[i].flatten())
    idx = np.random.shuffle(np.arange(int(len(new_data))))
    return np.squeeze(np.array(new_data)[idx]), np.squeeze(np.array(label)[idx])

def optimize(w, b, x, y, iterations, lr):
    for i in range(iterations):
        dw, db ,cost = propagation(w, b, x, y)
        w = w - lr*dw 
        b = b - lr*db
    return w, b, dw, db

def propagation(w, b, x, y):
   m = x.shape[0]
   atv = np.squeeze(sigmoid(np.dot(x,w)+b))
   cost = -(1/m)*np.sum(y*np.log(atv)+(1-y)*np.log(1-atv)) # loss function
   dw = (1/m)*np.dot(x.T,(atv-y)).reshape(w.shape[0],10)
   db = (1/m)*np.sum(atv-y)
   return dw, db, cost

def sigmoid(z): 
    s = 1 / (1 + np.exp(-z)) 
    return s   


def predict(w, b, x ):
    w = w.reshape(x.shape[1], 10)
    y_pred = sigmoid(np.dot(x, w) + b)
    for i in range(y_pred.shape[0]):
        init = [0]*10
        idx_max = np.argmax(y_pred[i]) 
        init[idx_max] = 1
        y_pred[i] = init
    return y_pred

def acc(pred, label):
    cut = pred - label
    count = 0
    for i in range(cut.shape[0]):
        if((cut[i] == 1.0).any()): 
            count += 1
    acc = 1-count/pred.shape[0]
    return acc

def model(x_train, y_train, iterations = 2000, lr = 0.6):
    w = np.zeros((x_train.shape[1],10));b = [0]*10
    w, b, dw, db = optimize(w, b, x_train, y_train, iterations, lr)
    return w, b

def plot(var, title, color, ylabel):
    x = np.arange(0.1, 1.1, 0.1)
    plt.plot(x, var, label = 'time', color=color)
    plt.xlabel('Percentage of Training Data')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def main():
    train = "digitdata/trainingimages"
    train_label = "digitdata/traininglabels"
    test = "digitdata/testimages"
    test_label = "digitdata/testlabels"
    x_train, y_train = process_data(train, train_label)
    x_test, y_test = process_data(test, test_label)
    amount = int(x_train.shape[0]/10)
    time_consume = []
    test_acc = []
    iter_acc=[]
    test_std=[]
    
    for i in range(10):
        totalAcc = 0
        totalTime = 0
        print('Training using',amount*(i+1))
        for j in range(3):
            x_train, y_train = process_data(train, train_label)
            x_test, y_test = process_data(test, test_label)
            start = time.time()
            w, b = model(x_train[0:amount*(i+1)],y_train[0:amount*(i+1)])
            end = time.time()
            y_pred_test = predict(w, b, x_test)
            test_accuracy = acc(np.squeeze(y_pred_test), y_test)
            totalAcc += test_accuracy
            totalTime += round(end-start, 3)
            iter_acc.append(test_accuracy)
        avgAcc = totalAcc/3
        avgTime = totalTime/3
        print("Test accuracy:{}".format(round(avgAcc, 3)))
        print("Time taken:{}".format(round(avgTime, 3)))
        time_consume.append(avgTime)
        test_acc.append(avgAcc)
        stdDev = np.std(iter_acc)
        test_std.append(stdDev)
        print ("Standard deviation of accuracy: %0.4f" % stdDev)
    plot(time_consume, title='Neural Network Classifier for Digits', color='blue', ylabel="Time(s)")
    plot(test_acc, title='Neural Network Classifier for Digits', color='green', ylabel='Accuracy')
    plot(test_std, title='Neural Network Classifier for Digits', color='red', ylabel="Standard Deviation")

main()
