# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 16:20:18 2016

@author: dell
"""
#TERData
import numpy as np
words_num=20
dim=200*words_num
test_data_path="data\\questions_vector_all.txt"
train_data_path="data\\question_entity_vector.txt"
question_path="data\\question_entity.txt"

#加载测试数据
def load_test_data(path=test_data_path):
    f=open(path)
    X_train=[]
    #y_train=[]
    for line in f:
        words=line.strip().split(" ")
        #bound=words[0].split(":")
        temp=[float(words[j]) for j in range(len(words)) if words[j]!='' and words[j]!='\n']
        x=[0]*dim
        length=len(temp)
        if length<dim:
            x[0:length]=temp
        else:
            x[0:dim]=temp[0:dim]
        X_train.append(np.reshape(np.array(x),(20,200)))
    X_train=np.array(X_train)
    return  X_train

#加载训练数据
def load_data(path=train_data_path,samples_num=10000):
    f=open(path)
    lines=f.readlines()
    print len(lines)
    X_train=[]
    y_train=[]
    bounds=[]
    for i in range(min(samples_num,len(lines))):
        line=lines[i]
        words=line.strip().split(" ")
        #print i,bound
        bound=words[0].split(":")
        #print bound,i
        bounds.append(words[0])
        words=words[1:]
        temp=[float(words[j]) for j in range(len(words)) if words[j]!='' and words[j]!='\n']
        x=[0]*dim
        length=len(temp)
        if length<dim:
            x[0:length]=temp
        else:
            x[0:dim]=temp[0:dim]
        y=[0]*20
        start=int(bound[0])
        end=int(bound[1])
        if end>20:
            end=20
            #continue
        y[start:end]=[1]*(end-start)
        X_train.append(np.reshape(np.array(x),(20,200)))
        y_train.append(y)
    f.close()
    
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    print 'len',len(X_train)
    return X_train,y_train,bounds

#加载问题文本
def load_questions(path=question_path,samples_num=10000):
    f=open(path)
    questions=[]
    for i in range(samples_num):
        line=f.readline()
        line=line.strip().split(' ')
        if line[0].find(':')!=-1:
            questions.append(line[1:])
        else:
            questions.append(line)
    return questions
