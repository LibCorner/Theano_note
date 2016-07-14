#ConvNet
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d,conv

import numpy
import pylab
from PIL import Image
from theano.tensor.signal import downsample

rng=numpy.random.RandomState(23455)


#初始化输入的4D张量
input = T.tensor4(name='input')

#初始化权重的shared变量,输入3个特征图（一张RGB）
#size为120X160,使用9X9的感受野卷积过滤器
#输出为2个特征图，所以权重的shape为(2,3,9,9)-->(output map,input map,weidth,height)
w_shp=(2,3,9,9)
w_bound=numpy.sqrt(3*9*9)
W=theano.shared(numpy.asarray(
                rng.uniform(low=-1.0/w_bound,
                            high=1.0/w_bound,
                            size=w_shp),
                            dtype=input.dtype),name='W')
                            
#初始化偏置向量
b_shp=(2,)
b=theano.shared(numpy.asarray(
                rng.uniform(low=-.5,high=.5,size=b_shp),
                dtype=input.dtype),name='b')
                
#建造符号表达式，使用w中的过滤器计算输入的卷积
conv_out=conv2d(input,W)

#建立符号表达式添加偏置，并应用激活函数
#dimshuffle是reshape张量的工具
#dimshuffle可以对dim进行重新洗牌，也可以插入可广播的新的dim,
#例如：dimshuffle('x',2,'x',0,1): 可以作用在没有boradcatable维的3d张量上，
#'x'表示插入新的维，这样第一维将会是可广播的。如果原来的tensor的shape是(1,40,1,20,30)，
#dimshuffle后的tensor为(1,40,1,20,30)。(AxBxC的张量变成1xCx1xAxB)
#更多的例子：
# dimshuffle('x') -> 0d(scalar)转换成1d向量
# dimshuffle(1,0) -> 交换第一维和第二维
# dimshuffle('x',0) -> N to 1xN
output=T.nnet.sigmoid(conv_out+b.dimshuffle('x',0,'x','x'))

#创建计算过滤的图像的函数
f=theano.function([input],output)

#打开一个639*516的图像
img=Image.open(open('3wolfmoon.jpg','rb'))
#维度是（height,width,channel）
img=numpy.asarray(img,dtype='float64')/256

print img.shape
print img

#输入图像，以4D张量的形式(1,3,height,width)
#使用transpose转换维度顺序
img_=img.transpose(2,0,1).reshape(1,3,639,516)
filtered_img=f(img_)

#先画原始图像，然后输出
pylab.subplot(1,3,1)
pylab.axis('off')
pylab.imshow(img)
pylab.gray()

pylab.subplot(1,3,2)
pylab.axis('off')
pylab.imshow(filtered_img[0,0,:,:])

pylab.subplot(1,3,3)
pylab.axis('off')
pylab.imshow(filtered_img[0,1,:,:])
pylab.show()

#最大池化,在张量的最后2个维度上执行
maxpool_shape=(2,2)
pool_out=downsample.max_pool_2d(input,maxpool_shape,ignore_border=True) #ignore_boreder设为True
f=theano.function([input],pool_out)

invals=numpy.random.RandomState(1).rand(3,2,5,5)
print 'With ignore_border set to True:'
print 'invals[0,0,:,:]=\n',invals[0,0,:,:]
print 'output[0,0,:,:]=\n', f(invals)[0,0,:,:]

pool_out=downsample.max_pool_2d(input,maxpool_shape,ignore_border=False)
f=theano.function([input],pool_out)
print 'With ignore_border set to False:'
print 'invals[1,0,:,:]=\n',invals[1,0,:,:]
print 'output[0,0,:,:]=\n',f(invals)[1,0,:,:]


#LeNetConvPoolLayer类，用来实现卷积+最大池化层
class LeNetConvPoolLayer(object):
    '''卷积神经网络的池化层'''
    def __init__(self,rng,input,filter_shape,image_shape):
        '''
        type rng：numpy.random.RandomState
        :param rng:用来初始化权重的随机数生成器
        ：type input：theano.tensor.dtensor4
        :param input: 符号image张量，
        :type filter_shape:(number of filters, num input feature maps, filter height,filter width) 
                            (过滤器个数，输入特征图个数，过滤器高度(感受野高度)，过滤器宽度)
        :type image_shape: tuple or list of lenth 4
        :param image_shape: (batch size, num input feature maps, image heigth,image width)
        :type poolsize: tuple or list of length 2
        :param poolsize: 采样池化因子（#rows,#cols）
        '''
        assert image_shape[1]==filter_shape[1]
        self.input=input
        
        #每个隐层单元有“num input feature maps * filter height * filter width”个输入
        fan_in=numpy.prod(filter_shape[1:])
        #在低层的每个单元接受一个梯度,来自：
        #“输出特征图数 * filter height * filter width”/pooling size
        fan_out=(filter_shape[0]*numpy.prod(filter_shape[2:]))
         #随机初始化权重
        W_bound=numpy.sqrt(6. / (fan_in+fan_out))
        self.W=theano.shared(
                            numpy.asarray(rng.uniform(low=-W_bound,high=W_bound,size=filter_shape),
                                          dtype=theano.config.floatX),
                                          borrow=True)

        #偏置是1D的tensor, 每个输入特征图一个偏置
        b_values=numpy.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b=theano.shared(value=b_values,borrow=True)
        
        #对输入进行卷积
        conv_out=conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape)
        #使用maxpooling采样
        '''
        pooled_out=downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=False)
            '''
        #激活输出
        self.output=T.nnet.sigmoid(conv_out+self.b.dimshuffle('x',0,'x','x'))
        
        #存储这层的参数
        self.params=[self.W,self.b]

class MaxPoolingLayer(object):
    def __init__(self,rng,input,poolsize=(2,2),ignore_boreder=True):
        pooled_out=downsample.max_pool_2d(input=input,
                                          ds=poolsize,
                                          ignore_border=ignore_boreder)
                                          
        self.output=pooled_out
 
if __name__=='__main__':
    #打开一个639*516的图像
    #img=Image.open(open('3wolfmoon.jpg','rb'))
    #输入图像，以4D张量的形式(1,3,height,width)
    #使用transpose转换维度顺序
    #img_=img.transpose(2,0,1).reshape(1,3,639,516)
    
    #初始化输入的4D张量
    input = T.tensor4(name='input')
    conv_0=LeNetConvPoolLayer(rng=rng,input=input,filter_shape=(10,3,9,9),image_shape=(1,3,639,516))
    maxpool_0=MaxPoolingLayer(rng=rng,input=conv_0.output,poolsize=(2,2))
    conv_1=LeNetConvPoolLayer(rng=rng,input=maxpool_0.output,filter_shape=(5,10,9,9),image_shape=(1,10,315,254))
    
    conv_2=LeNetConvPoolLayer(rng=rng,input=conv_1.output,filter_shape=(2,5,5,5),image_shape=(1,5,307,246))
    
    
    output_0=maxpool_0.output
    output=conv_2.output
    
    f0=theano.function([input],output_0)    
    f=theano.function([input],output)
            
    filtered_img_0=f0(img_)
    filtered_img=f(img_)
    #先画原始图像，然后输出
    pylab.subplot(1,3,1);pylab.axis('off');pylab.imshow(img);pylab.gray()
    pylab.subplot(1,3,2);pylab.axis('off');pylab.imshow(filtered_img[0,0,:,:])
    pylab.subplot(1,3,3);pylab.axis('off');pylab.imshow(filtered_img[0,1,:,:])
    pylab.show()
    
    pylab.subplot(1,4,1);pylab.axis('off');pylab.imshow(img);pylab.gray()
    pylab.subplot(1,4,2);pylab.axis('off');pylab.imshow(filtered_img_0[0,0,:,:])
    pylab.subplot(1,4,3);pylab.axis('off');pylab.imshow(filtered_img_0[0,1,:,:])
    pylab.subplot(1,4,4);pylab.axis('off');pylab.imshow(filtered_img_0[0,2,:,:])
    pylab.show()
    
    