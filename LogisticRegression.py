# -*- coding: utf-8 -*-
#LogistRegress
from __future__ import print_function

import six.moves.cPickle as pickle
import theano
import theano.tensor as T
import numpy as np
import os
import gzip
import timeit

rng=np.random

#多类分类逻辑回归
class LogisticRegression(object):
    def __init__(self,input,n_in,n_out):
        self.W=theano.shared(name='W',
                             value=np.zeros((n_in,n_out),dtype=theano.config.floatX),
                             borrow=True)
                             
        self.b=theano.shared(name='b',
                             value=np.zeros(n_out,dtype=theano.config.floatX),
                             borrow=True)
        #计算输出
        out=T.dot(input,self.W)+self.b
        #softmax
        self.p_y_given_x=T.nnet.softmax(out)
        
        #概率最大的维作为类别
        self.y_pred=T.argmax(self.p_y_given_x,axis=1)
        
        #模型的参数
        self.parms=[self.W,self.b]
        
        self.input=input
        
    def negative_log_likelihood(self,y):
        '''
            返回负log似然损失函数        
        '''
        #y.shape[0]是y的行数，i.e,minibatch中样本的个数。
        #T.arange(y.shape[0])是一个符号向量，将包含[0,1,2,...,n-1]
        #T.log(self.p_y_given_x)是一个矩阵，每一行是一个example,每一列是一个类别的log概率。
        #LP[T.arrange(y.shape[0]),y]是一个向量v，包含[LP[0,y[0]],LP[1,y[1]],...,LP[n-1,y[n-1]]]
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
        
    def errors(self,y):
        """
            返回minibatch中错误样本的个数        
        """
        #检查y与y_pred的维度是否相同
        if y.ndim !=self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y',y.type,'y_pred',self.y_pred.type)
            )
        #检查y是否是正确的数据集
        if y.dtype.startswith('int'):
            #T.neq操作返回0和1的向量，1表示有一个错误
            return T.mean(T.neq(self.y_pred,y))
        else:
            raise NotImplementedError()

def load_data(dataset):
    """
        加载数据集    
    """
    #如果没有就下载MNIST数据集
    data_dir,data_file=os.path.split(dataset)
    if data_dir=="" and not os.path.isfile(dataset):
        #检查dataset是否在data目录下
        new_path=os.path.join(os.path.split(__file__)[0],
                              "..",
                              "data",
                              dataset)
        if os.path.isfile(new_path) or data_file=='mnist.pkl.gz':
            dataset=new_path
            
    if (not os.path.isfile(dataset) and data_file=='mnist.pkl.gz'):
        from six.moves import urllib
        origin=('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print('Downloading data from %s' %origin)
        urllib.requestion.urlretrieve(origin,dataset)
        
    print('... loading data')
    
    #加载数据集
    with gzip.open(dataset,'rb') as f:
        try:
            train_set,valid_set,test_set=pickle.load(f,encoding='latin1')
        except:
            train_set,valid_set,test_set=pickle.load(f)
            
    #train_set,valid_set,test_set的格式：tuple(input,target)
    #input是一个numpy.ndarray类型的2维矩阵
    #每一行对应一个样本。
    #target是一个numpy.ndarray类型的1维向量。
    def shared_dataset(data_xy,borrow=True):
        """把数据集加载为shared变量
        这样做的理由是为了允许Theano把数据复制到GPU的内存中(当代码是在GPU上运行时)。
        因为复制数据到GPU很慢，每次复制一个minibatch会使性能下降。
        """
        data_x,data_y=data_xy
        shared_x=theano.shared(np.asarray(data_x,
                                          dtype=theano.config.floatX),
                                          borrow=borrow)
        shared_y=theano.shared(np.asarray(data_y,
                                          dtype=theano.config.floatX),
                                          borrow=borrow)
        #当把数据存储在GPU时，必须存储为float类型的，因此要把labels也存储为floatX
        #但是在计算的时候我们需要int类型的label作为index,因此我们要把shared_y转换为int返回。
        return shared_x,T.cast(shared_y,'int32')
    test_set_x,test_set_y=shared_dataset(test_set)
    valid_set_x,valid_set_y=shared_dataset(valid_set)
    train_set_x,train_set_y=shared_dataset(train_set)
    
    rval=[(train_set_x,train_set_y),
          (valid_set_x,valid_set_y),
            (test_set_x,test_set_y)]
    return rval
        
def sgd_optimization_mnist(learning_rate=0.13,n_epochs=1000,dataset='mnist.pkl.gz',batch_size=600):
    
    datasets=load_data(dataset)
    
    train_set_x,train_set_y=datasets[0]
    valid_set_x,valid_set_y=datasets[1]
    test_set_x,test_set_y=datasets[2]
    
    #计算minibatch数,get_value得到的是数值而不是Theano符号
    n_train_batches=train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches=valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches=test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    ############################
    #Build actual Model        #
    ############################
    
    #生成输入数据的符号变量(x,y表示一个minibatch)
    x=T.matrix('x') #data,presented as rasterized images
    y=T.ivector('y') #labels,presented as 1D vector of [int] labels
    
    #minibatch的索引
    index=T.lscalar('index')
    
    #构造logistic regression类
    #每个MNIST图像的大小是28*28
    classifier=LogisticRegression(input=x,n_in=28*28,n_out=10)
    
    #计算损失函数
    cost=classifier.negative_log_likelihood(y)
    
    #编译test_model,计算一个minibatch的错误数
    test_model=theano.function(inputs=[index],
                               outputs=classifier.errors(y),
                               givens={
                                   x:test_set_x[index*batch_size:(index+1)*batch_size],
                                   y:test_set_y[index*batch_size:(index+1)*batch_size]
                               })
                               
    validate_model=theano.function(inputs=[index],
                               outputs=classifier.errors(y),
                               givens={
                                   x:valid_set_x[index*batch_size:(index+1)*batch_size],
                                   y:valid_set_y[index*batch_size:(index+1)*batch_size]
                               })
    
    
    g_W=T.grad(cost=cost,wrt=classifier.W)
    g_b=T.grad(cost=cost,wrt=classifier.b)
    
    updates=[(classifier.W,classifier.W-learning_rate*g_W),
             (classifier.b,classifier.b-learning_rate*g_b)]
             
    #编译Theano function 'train_model',返回cost，同时根据`updates`的规则更新参数
    train_model=theano.function(inputs=[index],
                                outputs=[cost],
                                updates=updates,
                                givens={x:train_set_x[index*batch_size:(index+1)*batch_size],
                                        y:train_set_y[index*batch_size:(index+1)*batch_size]})
    #Train Model#
    print('... train the model')
    #early-stopping参数
    patience=5000  #look as this many examples regardless
    patience_increase=2 #当找到一个新的最好的epoch后再等待patience_increse个epoch
    
    improvement_threshold=0.995 #有这么多的相对提高就看作是显著的提高
    #验证的频率
    validation_frequency=min(n_train_batches,patience/2)
    
    best_validation_loss=np.inf 
    test_score=0.
    start_time=timeit.default_timer()

    done_looping=False
    epoch=0
    while(epoch<n_epochs) and (not done_looping):
        epoch=epoch+1
        for minibatch_index in range(n_train_batches):
            #训练模型
            minibatch_avg_cost=train_model(minibatch_index)
            #迭代数
            it=(epoch-1)*n_train_batches+minibatch_index
            
            if(it+1)%validation_frequency==0:
                #计算验证集上的0-1loss
                validation_losses=[validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss=np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i,validation error %f %%' %(epoch,minibatch_index+1,n_train_batches,this_validation_loss))
                
                #如果目前是最好的score
                if this_validation_loss<best_validation_loss:
                    #如果loss改善明显就提高patience
                    if this_validation_loss<best_validation_loss*improvement_threshold:
                        patience=max(patience,it*patience_increase)
                    best_validation_loss=this_validation_loss
                    
                    #在测试集上测试
                    test_losses=[test_model(i) for i in range(n_test_batches)]
                    
                    test_score=np.mean(test_losses)
                    
                    print('epoch %i, minibatch %i/%i, test error of best model %f %%' %
                            (epoch,minibatch_index+1,n_train_batches,test_score*100))
                    
                    #保存最好的模型
                    with open('best_model.pkl','wb') as f:
                        pickle.dump(classifier,f)
                if patience<=it:
                    done_looping=True
                    break
    end_time=timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))

def predict(start,end):
    """
        一个如何加载训练模型并使用它预测labels的例子    
    """
    #加载保存的模型
    classifier=pickle.load(open('best_model.pkl'))
    
    #编译predicator函数
    predict_model=theano.function(inputs=[classifier.input],outputs=classifier.y_pred)
    
    #测试
    dataset='mnist.pkl.gz'
    datasets=load_data(dataset)
    test_set_x,test_set_y=datasets[2]
    test_set_x=test_set_x.get_value()
    
    predicted_values=predict_model(test_set_x[start:end])
    print ('Predicted values for the first 10 examples in test set:')
    print(predicted_values)
    
if __name__=='__main__':
    sgd_optimization_mnist()