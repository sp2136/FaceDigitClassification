import math
import numpy as np
import time
import matplotlib.pyplot as plt

def load_label(label_file):
    f = open(label_file)
    line = f.readlines()
    line = [int(item.strip()) for item in line]
    for i in range(len(line)):
        if(line[i] <= 0):
            line[i] = -1
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
    for i in range(len(all_image)):
        for j in range(new_row):
            for k in range(new_col):
                new_pixel = 0
                for row in range(pool*j,pool*(j+1)):
                    for col in range(pool*k,pool*(k+1)):
                        new_pixel += all_image[i][row,col]
                new_all_image[i,j,k] = new_pixel

    return new_all_image

def model(x_train, y_train, lr = 0.5, iterations = 2000):
    w = np.random.rand(x_train.shape[1]); b = 0
    start = time.time()
    for iterations in range(iterations):
        error = 0
        for i in range(y_train.shape[0]):
            y = np.dot(x_train[i], w) + b # loss function
            if( y*y_train[i]<=0):
                w += lr*x_train[i]*y_train[i]
                b += lr*y_train[i]
                error += 1
            else:
                pass
        end = time.time()
        if(error == 0):
            break
    return w, b 

def predict(w, b, x_test):
    y_pred = np.zeros(x_test.shape[0]) - 1
    for i in range(x_test.shape[0]):
        if(np.dot(x_test[i], w) + b > 0):
            y_pred[i] = 1
    return y_pred

def plot(var, title, color, ylabel):
    x = np.arange(0.1, 1.1, 0.1)
    plt.plot(x, var, label = 'time(s)', color=color)
    plt.xlabel('Percentage of Training Data')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def process_data(data_file, label_file, pool):
    label, sample_num = load_label(label_file)
    data = load_sample(data_file, sample_num, pool)
    new_data=[]
    for i in range(len(data)):
        new_data.append(data[i].flatten())
    
    idx = np.random.shuffle(np.arange(int(len(new_data))))
    return np.squeeze(np.array(new_data)[idx]), np.squeeze(np.array(label)[idx])

def acc(pred, label):
    acc = 1 - np.mean(np.abs(pred-label))/2
    return acc

def main():
    pool = 2
    train = "facedata/facedatatrain"
    train_label = "facedata/facedatatrainlabels"
    test = "facedata/facedatatest"
    test_label = "facedata/facedatatestlabels"
    x_train, y_train = process_data(train, train_label, pool)
    x_test, y_test = process_data(test, test_label, pool)
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
            x_train, y_train = process_data(train, train_label, pool)
            x_test, y_test = process_data(test, test_label, pool)
            start = time.time()
            w, b = model(x_train[0:amount*(i+1)],y_train[0:amount*(i+1)])
            end = time.time()
            y_pred_test = predict(w, b, x_test)
            accuracy = acc(np.squeeze(y_pred_test), y_test)
            iter_acc.append(accuracy)
            totalAcc += accuracy
            # print(accuracy, totalAcc)
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
    plot(time_consume, title='Perceptron Classifier for Faces', color='blue', ylabel="Time(s)")
    plot(test_acc, title='Perceptron Classifier for Faces', color='green', ylabel='Accuracy')
    plot(test_std, title='Perceptron Classifier for Faces', color='red', ylabel="Standard Deviation")
main()

