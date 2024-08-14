import math
import numpy as np
import random
import time
import matplotlib.pyplot as plt
def load_label(label_file):
    f = open(label_file)
    line = f.readlines()
    line = [int(item.strip()) for item in line]
    sample_num = len(line)
    return line, sample_num

def load_sample(sample_file, sample_num, pool):
    f = open(sample_file)
    line = f.readlines()
    file_length = int(len(line)) 
    width = int(len(line[0]))  
    length = int(file_length/sample_num) 
    all_image = []
    for i in range(sample_num):
        single_image = np.zeros((length,width))
        count=0
        for j in range(length*i,length*(i+1)): 
            single_line=line[j]
            for k in range(len(single_line)):
                if(single_line[k] == "+" or single_line[k] == "#"):
                    single_image[count, k] = 1 
            count+=1        
        all_image.append(single_image) 
    new_row = int(length/pool)
    new_col = int(width/pool)
    new_all_image = np.zeros((sample_num, new_row, new_col))
    for i in range(sample_num):
        for j in range(new_row):
            for k in range(new_col):
                new_pixel = 0
                for row in range(pool*j,pool*(j+1)):
                    for col in range(pool*k,pool*(k+1)):
                        new_pixel += all_image[i][row,col]
                new_all_image[i,j,k] = new_pixel
    return new_all_image

def process_data(data_file, label_file, pool):
    label, sample_num = load_label(label_file)
    data = load_sample(data_file, sample_num, pool)
    new_data=[]
    for i in range(len(data)):
        new_data.append(data[i].flatten())
    idx = np.random.shuffle(np.arange(int(len(new_data))))
    return np.squeeze(np.array(new_data)[idx]), np.squeeze(np.array(label)[idx])


def optimization(w, b, x, y, iterations, lr):
    for i in range(iterations):
        dw, db, cost = propagation(w, b, x, y)
        w = w - lr*dw
        b = b - lr*db
    return w, b, dw ,db

def propagation(w, b, x,y):
    m = x.shape[0]
    atv = np.squeeze(sigmoid(np.dot(x,w)+b)) 
    y = np.array([int(item) for item in y])
    cost = -(1/m)*np.sum(y*np.log(atv)+(1-y)*np.log(1-atv)) # loss function
    dw = (1/m)*np.dot(x.T,(atv-y)).reshape(w.shape[0],1)
    db = (1/m)*np.sum(atv-y)
    return dw, db, cost

def sigmoid(z):
    s = 1 / (1 + np.exp(-z)) 
    return s   

def predict(w, b, x ):
    w = w.reshape(x.shape[1], 1)
    y_pred = sigmoid(np.dot(x, w) + b)
    for i in range(y_pred.shape[0]):
        if(y_pred[i] > 0.5):

            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred

def model(x_train, y_train, iterations = 2000, lr = 0.5):
    w = np.zeros((x_train.shape[1],1));b = 0
    w,b, dw, db = optimization(w, b, x_train, y_train, iterations, lr)
    return w, b

def plot(var, title, color, ylabel):
    x = np.arange(0.1, 1.1, 0.1)
    plt.plot(x, var, label = 'time', color=color)
    plt.xlabel('Percentage of Training Data')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def acc(pred, label):
    acc = 1 - np.mean(np.abs(pred-label))
    return acc

def main():
    pool = 3
    train = "facedata/facedatatrain"
    train_label = "facedata/facedatatrainlabels"
    test = "facedata/facedatatest"
    test_label = "facedata/facedatatestlabels"
    x_train, y_train = process_data(train, train_label, pool)
    x_test, y_test = process_data(test, test_label, pool)
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
            x_train, y_train = process_data(train, train_label, pool)
            x_test, y_test = process_data(test, test_label, pool)
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
    plot(time_consume, title='Neural Network Classifier for Faces', color='blue', ylabel="Time(s)")
    plot(test_acc, title='Neural Network Classifier for Faces', color='green', ylabel='Accuracy')
    plot(test_std, title='Neural Network Classifier for Faces', color='red', ylabel="Standard Deviation")
main()