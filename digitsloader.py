import numpy as np

def load_label(label_file):
    f = open(label_file)
    line = f.readlines()
    line = [int(item.strip()) for item in line]
    sample_num = len(line)
    return line, sample_num

def load_sample(sample_file, sample_num):
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
    return all_image
