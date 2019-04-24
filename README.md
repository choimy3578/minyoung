import tensorflow as tf          tensorflow를 실행한다.
tf.enable_eager_execution()      활성화시켜서 즉시 실행시킨다.

# Data
x_data = [1, 2, 3, 4, 5]         데이터 준비
y_data = [1, 2, 3, 4, 5]

# W, b initialize
W = tf.Variable(2.9)             임의로 2.9라는 값을 지정한다.
b = tf.Variable(0.5)             임의로 0.5라는 값을 지정한다.
w와 b 변수 두개를 준비한다.

learning_rate = 0.01             learning_rate를 산수로 지정한다.

# W, b update
for i in range(100+1):                      i는 0부터 101까지 변해간다.(101번 반복한다.)
    # Gradient descent                      경사 하강법 : cost가 최소화되는 w와 b를 찾는다.
    with tf.GradientTape() as tape:     
tensorflow에서 gradient descent를 구현한다. 변수들의 기록을 tape에 저장한다.
        hypothesis = W * x_data + b             변수 w와 b를 tape에 저장한다.   
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W_grad, b_grad = tape.gradient(cost, [W, b]) 
cost function에서 w에 대한 기울기는 W_grade, b에 대한 기울기는 b_grade에 할당된다.
    W.assign_sub(learning_rate * W_grad               learning rate을 곱해서 할당한다.                    
    b.assign_sub(learning_rate * b_grad)              assign_sub : 뺀 값을 다시 그 값에 할당해준다.
 
    if i % 10 == 0:                                   10의 배수가 될 때마다 print해준다.
      print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

print()

# predict
print(W * 5 + b)                  우리 모델을 통해 직접 예측해본다.
실제 결과값이 5와 가깝다.
print(W * 2.5 + b)
실제 결과값이 2.5와 가깝다.

# 결과값
    i       w           b       cost
    0|    2.4520|    0.3760| 45.660004
   10|    1.1036|    0.0034|  0.206336
   20|    1.0128|   -0.0209|  0.001026
   30|    1.0065|   -0.0218|  0.000093        
   40|    1.0059|   -0.0212|  0.000083     
   50|    1.0057|   -0.0205|  0.000077
   60|    1.0055|   -0.0198|  0.000072
   70|    1.0053|   -0.0192|  0.000067
   80|    1.0051|   -0.0185|  0.000063
   90|    1.0050|   -0.0179|  0.000059
w값은 처음에 2.9부터 시작해서 1.0으로 수렴한다.
b값은 0.5로 시작해서 대략 0으로 수렴한다.
cost값은 처음에 45.6으로 시작해서 마지막에는 대략 0으로 작아졌다.(이 모델이 실제 데이터와 유사하다.)
=> 오차가 적고 실제 값을 예상하는데 잘 맞는 모델이다.
