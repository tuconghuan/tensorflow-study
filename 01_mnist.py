# 20190701
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE =784
OUTPUT_NODE =10

LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.8

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 网络前向传播：输入权重和网络参数，输出前向结果
def inference(input_tensor, avg_class, weights1,biases1,weights2,biases2):
    if avg_class==None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型过程
def train(mnist):
    x = tf.placeholder(tf.float32,[None, INPUT_NODE],name = 'x-input')
    y_ = tf.placeholder(tf.float32,[None, OUTPUT_NODE],name = 'y-input')

    # 生成隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE]), stddev = 0.1)
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    # 生成输出层参数
    weights2 = tf.Variable(tf.truncated_normal[LAYER1_NODE, OUTPUT_NODE], stddev = 0.1)
    biases2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))

    # 计算前向传播
    y = inference(x, None, weights1, biases1, weights2, biases2)
    # 定义存储训练轮数的变量
    global_step = tf.Variable(0, trainable=False)

    # 初始化华东平均类？
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    # 对神经网络参数变量使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用滑动平均的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵作为loss1
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels = tf.arg_max(y_,1))
    # 计算当前batch所有样例loss1平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总loss
    loss=cross_entropy_mean+regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)

    # 梯度下降优化法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

    # 参数更新
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name = 'train')

    # ??
    correct_prediction = tf.equal(tf.argmax(average_y,1), tf.argmax(y_,1))
    accuracy  = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    # 迭代训练
    for i in range(TRAINING_STEPS):
        if i%1000 == 0:
            validate_acc = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d training step(s), validation accuracy using average model is %g" % (i,validate_acc))

        xs,ys = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_op, feed_dict = {x:xs,y_:ys})

    test_acc = sess.run(accuracy,feed_dict=test_feed)
    print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS,test_acc))

# 主程序入口
def main(avgv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
