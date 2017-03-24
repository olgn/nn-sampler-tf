import tensorflow as tf
import numpy as np

class Network(object):

    def __init__(self, layer_sizes, activation_function=None):
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.build_network()
        self.start_session()

    #this function allows us to add a layer to our network
    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        return outputs

    def start_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_network(self):
        layer_sizes = self.layer_sizes
        input_size = layer_sizes[0]
        output_size = layer_sizes[len(layer_sizes) - 1]
        self.xs = tf.placeholder(tf.float32, shape=[1, input_size], name='x_sample')
        self.ys = tf.placeholder(tf.float32, shape=[1], name='y_sample')

        layer = self.add_layer(self.xs, input_size, layer_sizes[1], activation_function=self.activation_function)

        # add the layers to the network
        for _ in range(2,len(layer_sizes)):
            if _ == (len(layer_sizes) - 1):
                a_function = None
            else:
                a_function = self.activation_function
            layer = self.add_layer(layer, layer_sizes[_ - 1], layer_sizes[_], activation_function=a_function)

        self.net = layer
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.net), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

    def train(self, input_sample, output_sample):
        self.sess.run(self.train_step, feed_dict={self.xs: input_sample, self.ys: output_sample})

    def predict(self, input_sample):
        output = self.sess.run(self.net, feed_dict={self.xs:input_sample})
        return output

    # def loss(self, loss):
    #     loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.net), reduction_indices=[1]))
    #     return loss

    def print_loss(self, input_sample, output_sample):
        print('loss:', self.sess.run(self.loss, feed_dict ={self.xs: input_sample, self.ys: output_sample}))
#
# #build a network with 1 input node, one hidden layer of 10 nodes, and an output layer with one node
# network = Network([1,10,1], activation_function=tf.nn.relu)
#
# #initialize all of the variables
# init = tf.global_variables_initializer()
#
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1000):
#     sess.run(network.train, feed_dict={network.xs: x_data, network.ys: y_data})
#     if i % 50 == 0:
#         print(sess.run(network.loss, feed_dict={network.xs: x_data, network.ys: y_data}))
#
# sess.close()