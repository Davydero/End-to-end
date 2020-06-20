from __future__ import print_function

import numpy as np

import tensorflow as tf


def weight_ones(shape, name):
    initial = tf.constant(1.0, shape=shape, name=name)
    return tf.Variable(initial)


def weight_xavi_init(shape, name):
    initial = tf.get_variable(name=name, shape=shape,
                              initializer=tf.contrib.layers.xavier_initializer())# un inicializador conocido de tensorflow
    return initial #retorna la matriz de pesos iniciales


def bias_variable(shape, name): #el bias sirve para poder variar el punto de activacion de cada neurona
    initial = tf.constant(0.1, shape=shape, name=name) #se hace un vector de longitud shape llena de 0.1s
    return tf.Variable(initial)


class Network(object):

    def __init__(self, dropout, image_shape):
        """ We put a few counters to see how many times we called each function """
        self._dropout_vec = dropout
        self._image_shape = image_shape
        self._count_conv = 0
        self._count_pool = 0
        self._count_bn = 0
        self._count_activations = 0
        self._count_dropouts = 0
        self._count_fc = 0
        self._count_lstm = 0
        self._count_soft_max = 0
        self._conv_kernels = []
        self._conv_strides = []
        self._weights = {}
        self._features = {}

    """ Our conv is currently using bias """

    def conv(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_' + str(self._count_conv))#se inicializan los pesos con un inicializador predefinido
        bias = bias_variable([output_size], name='B_c_' + str(self._count_conv)) #devuelve el vector bias

        self._weights['W_conv' + str(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)

        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,#en el parametro weights ya viene la dimension de los filtros
                                       name='conv2d_' + str(self._count_conv)), bias,
                          name='add_' + str(self._count_conv))

        self._features['conv_block' + str(self._count_conv - 1)] = conv_res

        return conv_res #se devuelve el resultado despues de la convolucion

    def max_pool(self, x, ksize=3, stride=2):
        self._count_pool += 1
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name='max_pool' + str(self._count_pool))

    def bn(self, x):
        self._count_bn += 1
        return tf.contrib.layers.batch_norm(x, is_training=False,
                                            updates_collections=None,
                                            scope='bn' + str(self._count_bn))

    def activation(self, x):
        self._count_activations += 1
        return tf.nn.relu(x, name='relu' + str(self._count_activations))

    def dropout(self, x):
        print("Dropout", self._count_dropouts)
        self._count_dropouts += 1
        output = tf.nn.dropout(x, self._dropout_vec[self._count_dropouts - 1],
                               name='dropout' + str(self._count_dropouts))

        return output

    def fc(self, x, output_size):
        self._count_fc += 1
        filters_in = x.get_shape()[-1] #devuelve la dimension -1 de la imagen,a mi entender es la ultima dimension
        shape = [filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_f_' + str(self._count_fc))#se inicializa los pesos de esta capa
        bias = bias_variable([output_size], name='B_f_' + str(self._count_fc))#devuelve el vector bias

        return tf.nn.xw_plus_b(x, weights, bias, name='fc_' + str(self._count_fc)) #matmul(x, weights) + biases.

    def conv_block(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        print(" === Conv", self._count_conv, "  :  ", kernel_size, stride, output_size)
        with tf.name_scope("conv_block" + str(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)#se hace uso del bloque convolucional, retorna la convolucion de la imagen 
            x = self.bn(x) #batch normalization normalizacion de las salidas de las neuronas para que no haya desbalance
            x = self.dropout(x) 

            return self.activation(x)#se hace la activacion relu

    def fc_block(self, x, output_size):
        print(" === FC", self._count_fc, "  :  ", output_size)
        with tf.name_scope("fc" + str(self._count_fc + 1)):
            x = self.fc(x, output_size)# re realiza la multiplicacion de x por los pesos y la suma de los bias
            x = self.dropout(x)
            self._features['fc_block' + str(self._count_fc + 1)] = x
            return self.activation(x)#relu

    def get_weigths_dict(self):
        return self._weights

    def get_feat_tensors_dict(self):
        return self._features


def load_imitation_learning_network(input_image, input_data, input_size, dropout):#el input image tiene 4 dimensiones [n, 88,200,3]
    branches = []

    x = input_image##o se puede imprimir esta tomando en cuenta un canal

    network_manager = Network(dropout, tf.shape(x))#las dimensiones de la imagen (3)## se crea un nuevo objeto tipo Network
    #hasta la linea anterior solo se ha creado el objeto network definiendo contadores, nada mas 
    """conv1"""  # kernel sz, stride, num feature maps
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID')# cada conv_block hace llamado a una convolucion, batch normalization, dropout y una activacion relu
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID')
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID')
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID')
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID')
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID')
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')#se aplican 8 convoluciones 
    print(xc)
    """mp3 (default values)"""

    """ reshape """
    x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')#se redimensiona la imagen, -1 se hace para mantener una forma constante
    print(x)

    """ fc1 """
    x = network_manager.fc_block(x, 512)#se realiza la multiplicacion de x por los pesos, dropout y activacion  relu
    print(x)
    """ fc2 """
    x = network_manager.fc_block(x, 512)#mismo que el anterior 512 neuronas de salida

    """Process Control"""

    """ Speed (measurements)"""
    with tf.name_scope("Speed"):
        speed = input_data[1]  # get the speed from input data
        speed = network_manager.fc_block(speed, 128)#capa fully conected, multiplicacion de speed por los pesos
        speed = network_manager.fc_block(speed, 128)#lo mismo que el anterior

    """ Joint sensory """
    j = tf.concat([x, speed], 1) #se concatena la convolucion de la imagen con la salida de velocidad
    j = network_manager.fc_block(j, 512) #y se la hace pasar por una capa fc, 512 neuronas de salida

    """Start BRANCHING"""
    branch_config = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], \
                     ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Speed"]]

    for i in range(0, len(branch_config)):
        with tf.name_scope("Branch_" + str(i)):
            if branch_config[i][0] == "Speed":
                # we only use the image as input to speed prediction
                branch_output = network_manager.fc_block(x, 256)#cuando es speed solo se alimenta con la imagen 
                branch_output = network_manager.fc_block(branch_output, 256)
            else:
                branch_output = network_manager.fc_block(j, 256) #se alimenta con la concatenacion de la imagen y velocidad
                branch_output = network_manager.fc_block(branch_output, 256) #dos capas de 256

            branches.append(network_manager.fc(branch_output, len(branch_config[i])))#branches esta definido anteriormente como un arreglo vacio
                                                                #len(branch config) tiene una longitud de 3 para los primeros 4 casos y 1 para el ultimo
        print(branch_output)                                    #es una capa que para los primeros 4 casos tendra una salida de 3 neuronas
                                                                #y para el ultimo caso una salida de una neurona
    return branches                                             #basicamente se muestra una salida de la forma branch_config
