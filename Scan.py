# -*- coding: utf-8 -*-
#Scan
import theano
import theano.tensor as T
import numpy as np
'''
scan 函数提供了Theano循环的基本功能
'''
#1. 简单的累积：计算A的k次方:results=A**k
'''
我们需要处理三个东西：分配给result的初始值，result的累积结果，和不变的变量A。
不变的变量A传入到scan里作为non_sequences,在outputs_info初始化result,累积会自动进行。
'''
k=T.iscalar('k')
A=T.vector('A')
'''
1.我们首先使用lambda表达式构造了一个函数：给定prior_result和A,返回prior_result*A。
参数的顺序是由scan固定了的：fn函数的前一个输出（或初始化的值）是第一个参数，后面是所有的
非序列参数。
2.然后我们初始化output为与A类型相同的tensor,初始值为1。把A作为scan的非序列参数。
3.设置迭代的次数n_steps为k.
4.scan 返回一个元组，包含我们的result和一个updates字典（在本例中是空的）。注意，result不是一个
矩阵，而是一个3D的tensor包含每次迭代的A**k的值，我们需要最后的值，所以我们编译的函数仅仅返回
'''
#result的符号描述
result,updates=theano.scan(fn=lambda prior_result,A:prior_result*A,
                           outputs_info=T.ones_like(A),
                            non_sequences=A,
                            n_steps=k)
                            
'''
我们只关心A**k,但是scan提供了A**1到A**k,丢掉我们不关心的值。scan可以做到，并且不浪费内存。
'''
final_result=result[-1]

#编译返回A**k的函数
power=theano.function(inputs=[A,k],outputs=final_result,updates=updates)

print(power(range(10),2))
print(power(range(10),4))



# 2. 沿着张量的第一维进行迭代：计算多项式
'''
scan不仅可以循环一个固定的次数，还可以沿着tensor的维度进行迭代（与python的for x in a_list类似）。
scan使用`sequence`参数提供这个tensor，下面是一个使用符号计算多项式的例子。
'''
coefficients=T.vector("coefficients")
x=T.scalar("x")

max_coefficients_supported=10000

#生成多项式的组件:系数*(自变量**幂指数)
components,updates=theano.scan(fn=lambda coefficients,power,free_variable:coefficients*(free_variable**power),
                              outputs_info=None,
                              sequences=[coefficients,T.arange(max_coefficients_supported)],
                              non_sequences=x)

show_components=theano.function(inputs=[coefficients,x],outputs=components)
#Sum theam up
polynomial=components.sum()   

#编译函数
calculate_polynomial=theano.function(inputs=[coefficients,x],outputs=polynomial)

#Test
test_coefficients=np.asarray([1,0,2],dtype=np.float32)
test_value=3
print(calculate_polynomial(test_coefficients,test_value))
print(1.0*(3**0)+0.0*(3**1)+2.0*(3**2))
print(show_components(test_coefficients,test_value))

'''
注意以下几点：
1. 我们计算多项式时，首先生成了每个系数项，然后把它们加起来。（也可以每次迭代时加起来，然后取最后一项）
2. 这里没有results的计算，我们可以把`outputs_info`设为`None`，这就告诉scan不需要把之前的结果传给`fn`。
一般`fn`参数的顺序是：
`sequences (if any), prior results(s) (if needed), non_sequences(if any)`
3. 这里使用一个方便的技巧用来模拟python的`emumerate`:简单的在sequences中包含theano.tensor.arange
4. 如果有不等长的多个sequences, scan会把所有的sequence缩短为它们中最短的。这就使得传递一个很长的arange变得安全
'''     
# 3. 抛弃lambda，简单的累积得到一个标量  
'''
注意：初始的output状态outputs_info必须与输出变量的shape一致，并且不能有下转型
'''                 
up_to=T.iscalar("up_to")

#定义一个函数而不是使用lambda
def accumulate_by_adding(arange_val,sum_to_date):
    return sum_to_date+arange_val
seq=T.arange(up_to)

#T.as_tensor_variable(0)的默认的type是int8, 而scan的output是int64
#因此用下面的函数设置变量的类型
outputs_info=T.as_tensor_variable(np.asarray(0,seq.dtype))
scan_result,scan_updates=theano.scan(fn=accumulate_by_adding,
                                     outputs_info=outputs_info,
                                     sequences=seq)
triangular_sequence=theano.function(inputs=[up_to],outputs=scan_result)
#test
some_num=15
print(triangular_sequence(some_num))
print([n*(n+1)//2 for n in range(some_num)])

#4. 另一个简单的例子
'''
与之前的例子不同，这个例子如果不使用scan很难实现。

这个例子输入是：一个数组下标序列，和一个替换值，以及一个model output(它的shape和dtype会被模仿)
产生的输出是一个数组序列，shape和dtype与model output一致，除了给定的下标值以外的值都设置为0
'''
location=T.imatrix("location")
values=T.vector("values")
output_model=T.matrix("output_model")

def set_value_at_position(a_location,a_value,output_model):
    zeros=T.zeros_like(output_model)
    zeros_subtensor=zeros[a_location[0],a_location[1]]
    return T.set_subtensor(zeros_subtensor,a_value)  #T.set_suntensor函数为Tensor数组赋值
    
result,updates=theano.scan(fn=set_value_at_position,
                           outputs_info=None,
                           sequences=[location,values],
                           non_sequences=output_model)

assign_values_at_positions=theano.function(inputs=[location,values,output_model],outputs=result)
#test
test_locations=np.asarray([[1,1],[2,3]],dtype=np.int32)
test_values=np.asarray([42,50],dtype=np.float32)
test_output_model=np.zeros((5,5),dtype=np.float32)
print(assign_values_at_positions,test_values,test_output_model) 

#5. 使用shared 变量-Gibbs sampling
'''
scan的另一个有用的特征是它可以处理shared变量。
比如，如果我们想实现长度为10的Gibbs chain,可以按下面方法做
'''   
W=theano.shared(W_values)

bvis=theano.shared(bvis_values)
bhid=theano.shared(bhid_values)

trng=T.shared_randomstreams.RandomStreams(1234)

def OneStep(vsample):
    hmean=T.nnet.sigmoid(theano.dot(vsample,W)+bhid)
    hsample=trng.binomial(size=hmean.shape,n=1,p=hmean)
    vmean=T.nnet.sigmoid(theano.dot(hsample,W.T)+bvis)
    return trng.binomial(size=vsample.shape,n=1,p=vmean,dtype=theano.config.floatX)
    
sample=T.vector()
values,updates=theano.scan(fn=OneStep,outputs_info=sample,n_steps=10)

gibbs10=theano.function([sample],values[-1],updates=updates)

'''
1. 首先，这里最重要的是updates字典，它把shared变量和该变量的k steps后的updated值联系在一起。
这样，它就会告诉random streams 在10次迭代之后如何获取updated。
如果你不把这个update字典传到function里，就会总是得到同样的10个随机数集合。
你也可以在scan函数之后使用updates字典，如下面的例子：
'''


a=theano.shared(1)
values,updates=theano.scan(lambda:{a:a+1},n_steps=10)

b=a+1  #使用更新前的a
c=updates[a]+1 #使用updates字典,会使用更新后的a
f=theano.function([],[b,c],updates=updates)  #如果这里不传入updates参数，a的值总是1
print f()
print a.get_value()

'''
2. 其次，如果我们使用shared 变量(W,bvis,bhid)但是我们不对他们进行迭代，你不需要把他们作为参数传递。
Scan会自己找到他们并加到计算图中。然而，把他们传递个Scan函数会更好，因为那样可以避免Scan Op调用更早的Op。
这样就会使计算图更简单，可以提高运行和优化的速度。要传递变量给Scan需要把他们放到一个list里，并传递给non_sequences参数。
下面是一个更新的Gibbs smapling代码。
'''
W=theano.shared(W_values)

bvis=theano.shared(bvis_value)
bhid=theano.shared(bhid_values)

trng=T.shared_randomstreams.RandomStreams(1234)

#OneStep,显式的使用shared变量
def OneStep(vsample,W,bvis,bhid):
    hmean=T.nnet.sigmoid(theano.dot(vsample,W),bhid)
    hsample=trng.binomial(size=hmean.shape,n=1,p=hmean)
    vmean=T.nnet.sigmoid(theano.dot(hsample,W.T)+bvis)
    return trng.binomial(size=vsample.shape,n=1,p=vmean,dtype=theano.config.floatX)
    
sample=T.vector()
#The new scan,shared变量作为non_sequeces
values,updates=theano.scan(fn=OneStep,
                           outputs_info=sample,
                           non_sequences=[W,bvis,bhid],
                           n_steps=10)
                           
gibbs10=theano.function([sample],values[-1],updates=updates)