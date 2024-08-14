import math
import numpy as np
import time
import matplotlib.pyplot as plt
from digitsloader import load_label, load_sample

def sigmoid(z):
    s = 1 / (1 + np.exp(-z)) 
    return s   

def model(x_train, y_train, lr = 0.5, iterations = 50):
    w = np.random.rand(x_train.shape[1], 10)
    for iter in range(iterations):
        error = 0
        for i in range(y_train.shape[0]):
            temp = np.squeeze(np.dot(x_train[i], w)) # loss function
            idx_temp = np.argmax(temp)  
            if( idx_temp != y_train[i]):
                w[:,y_train[i]] += lr*x_train[i,y_train[i]]
                error += 1
            else:
                pass
        if(error == 0):
            break
    return w

def predict(w, x_test):
    temp =np.dot(x_test, w)
    y_pred = np.zeros(temp.shape[0])
    w = w.reshape(x_test.shape[1],10)
    for i in range(x_test.shape[0]):                                  
        idx_max = np.argmax(temp[i])
        y_pred[i] = idx_max
    return y_pred

def plot(var, title, color, ylabel):
    x = np.arange(0.1, 1.1, 0.1)
    plt.plot(x, var, label = 'time', color=color)
    plt.xlabel('Percentage of Training Data')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def process_data(data_file, label_file):
    label, sample_num = load_label(label_file)
    data = load_sample(data_file, sample_num)
    new_data=[]
    for i in range(len(data)):
        new_data.append(data[i].flatten())
    idx = np.random.shuffle(np.arange(int(len(new_data))))
    return np.squeeze(np.array(new_data)[idx]), np.squeeze(np.array(label)[idx])

def acc(pred, label):
    count=0
    for i in range(pred.shape[0]):
        if(pred[i]!=label[i]):
            count+=1
    acc = count/pred.shape[0]
    return acc

def main():
    train = "digitdata/trainingimages"
    train_label = "digitdata/traininglabels"
    test = "digitdata/testimages"
    test_label = "digitdata/testlabels"
    x_train, y_train = process_data(train, train_label)
    x_test, y_test = process_data(test, test_label)    
    amount = int(x_train.shape[0]/10)
    time_consume = []
    test_std = []
    test_acc = []
    iter_acc = []
    print('We will be training every data set three times')
    for i in range(10):
        totalAcc = 0
        totalTime = 0
        print('Training using',amount*(i+1))
        for j in range(3):
            x_train, y_train = process_data(train, train_label)
            x_test, y_test = process_data(test, test_label)
            start = time.time()
            w = model(x_train[0:amount*(i+1)],y_train[0:amount*(i+1)])
            end = time.time()
            y_pred_test = predict(w, x_test)
            accuracy = acc(np.squeeze(y_pred_test), y_test)
            iter_acc.append(accuracy)
            totalAcc += accuracy
            totalTime += round(end-start, 3)        
        avgAcc = totalAcc/3
        avgTime = totalTime/3
        time_consume.append(avgTime)
        test_acc.append(avgAcc)
        print("Test accuracy:{}".format(round(avgAcc, 3)))
        print("Time taken:{}".format(round(avgTime,3)))
        stdDev = np.std(iter_acc)
        test_std.append(stdDev)
        print ("Standard deviation of accuracy: %0.4f" % stdDev)
    plot(time_consume, title='Perceptron Classifier for Digits', color='blue', ylabel="Time(s)")
    plot(test_acc, title='Perceptron Classifier for Digits', color='green', ylabel='Accuracy')
    plot(test_std, title='Perceptron Classifier for Digits', color='red', ylabel="Standard Deviation")
main()

