#20180915: tesorflow学习2   数据流图（Dataflow Graph）

# 引入 tensorflow 模块
import tensorflow as tf

### 常量计算: 用tf.constant()创建的 Tensor 都是常量，一旦创建后其中的值就不能改变了

# 创建一个整型常量，结点0
node0 = tf.constant(3, dtype=tf.int32)

# 创建一个整型常量，结点1
node1 = tf.constant(3, dtype=tf.int32)

# 创建汇总结点2
node2 = node0 + node1

# 创建汇总结点3
node3 = node0 * node1

# 打印上面创建的几个 Tensor
print(node0)
print(node1)
print(node2)
print(node3)

# 打印上面创建的几个 Tensor的值 [Session是连接前端python和后端C++之间的枢纽]
sess = tf.Session()
print(sess.run(node0))
print(sess.run(node1))
print(sess.run(node2))
print(sess.run(node3))


### 变量计算: 用tf.placeholder 创建占位 Tensor，占位 Tensor 的值可以在运行的时候输入

# 创建一个整型变量，结点0
v0 = tf.placeholder(dtype=tf.int32)

# 创建一个整型变量，结点1
v1 = tf.placeholder(dtype=tf.int32)

# 创建汇总结点2
v2 = v0 + v1

# 创建汇总结点3
v3 = v0 * v1

# 打印上面创建的几个 Tensor
print(v0)
print(v1)
print(v2)
print(v3)

# 打印上面创建的几个 Tensor的值 [Session是连接前端python和后端C++之间的枢纽]
sess = tf.Session()
print(sess.run(v0, {v0:3}))
print(sess.run(v1, {v1:4}))
print(sess.run(v2, {v0:3, v1:4}))
print(sess.run(v3, {v0:[3], v1:[4,4]}))
