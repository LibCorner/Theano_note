# -*- coding: utf-8 -*-
#Scan
import theano
import theano.tensor as T
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

