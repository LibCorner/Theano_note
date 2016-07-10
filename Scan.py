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