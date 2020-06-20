from __future__ import print_function

import os

import scipy

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

from carla.agent import Agent
from carla.carla_server_pb2 import Control
from agents.imitation.imitation_learning_network import load_imitation_learning_network


class ImitationLearning(Agent):

    def __init__(self, city_name, avoid_stopping, memory_fraction=0.25, image_cut=[115, 510]):

        Agent.__init__(self)

        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5#? dropout.. algunos nodos se eliminan durante el entrenamiento para evitar overfitting

        config_gpu = tf.ConfigProto()

        # GPU a seleccionar

        config_gpu.gpu_options.visible_device_list = '0'

        config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction

        self._image_size = (88, 200, 3) #tamano de la imagen que se adquiere por la camara?
        self._avoid_stopping = avoid_stopping

        self._sess = tf.Session(config=config_gpu)

        with tf.device('/gpu:0'):
            self._input_images = tf.placeholder("float", shape=[None, self._image_size[0], ##se definen todos los parametros del objeto que se define 
                                                                self._image_size[1],      ##placeholder es como una variable (lugar en la memoria) para trabajar con tensores
                                                                self._image_size[2]],   #los placeholders son tipo variables a los cuales no es necesario asignarles un valor inicial, puede ser asignado luego
                                                name="input_image") #se le asigna el valor cuando se corre la sesion  #en shaoe tiene el valor none, que viene a ser como el batch size, o numero de muestras con las que se

            self._input_data = []

            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 4], name="input_control"))#todo es como un instanciado de variables que seran usadas

            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 1], name="input_speed"))

            self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

        with tf.name_scope("Network"):#   scope=extension, anade la extension "network" a los operadores
            self._network_tensor = load_imitation_learning_network(self._input_images, #se retorna el arreglo de 5 branches, los 4 primeros tienen aceleracion freno y angulo 
                                                                   self._input_data,#control y speed
                                                                   self._image_size, self._dout)#metodo que esta en ILN.py---retorna la variable branches
                #network_tensor seria de una forma branch_config
        import os
        dir_path = os.path.dirname(__file__)#Return the directory name of pathname path. 

        self._models_path = dir_path + '/model/'#path del modelo entrenado

        # tf.reset_default_graph()
        self._sess.run(tf.global_variables_initializer())#inicializa las variables, para correr el modelo

        self.load_model()#se recupera las variables del modelo desde donde fue guardado

        self._image_cut = image_cut #[115,510]

    def load_model(self):

        variables_to_restore = tf.global_variables()#el conjunto de todas las variables globales, es como el instanciado antes de iniciar?

        saver = tf.train.Saver(variables_to_restore, max_to_keep=0)#se restauran las variables en variables_to_restore

        if not os.path.exists(self._models_path): #si es path de model no existe manda error
            raise RuntimeError('failed to find the models path')

        ckpt = tf.train.get_checkpoint_state(self._models_path)#-------->Retorna el checkpoint state 
        if ckpt:
            print('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)#restaura las variables previamente guardadas en ese path, requiere la sesion en la que el grafo ha sido lanzado
        else:
            ckpt = 0

        return ckpt #retorna el checkpoint state

    def run_step(self, measurements, sensor_data, directions, target): #se llama en driving_benchmark.py

        control = self._compute_action(sensor_data['CameraRGB'].data,
                                       measurements.player_measurements.forward_speed, directions)

        return control

    def _compute_action(self, rgb_image, speed, direction=None):

        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        steer, acc, brake = self._control_function(image_input, speed, direction, self._sess)

  

        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit speed to 35 km/h to avoid
        if speed > 10.0 and brake == 0.0:
            acc = 0.0

        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0

        return control

    def _control_function(self, image_input, speed, control_input, sess):

        branches = self._network_tensor
        x = self._input_images
        dout = self._dout
        input_speed = self._input_data[1]

        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))###imagen a imprimir

        # Normalize with the maximum speed from the training set ( 90 km/h)
        speed = np.array(speed / 25.0)

        speed = speed.reshape((1, 1))

        if control_input == 2 or control_input == 0.0:# la red procesa todos los branches, pero solo tomar'a en cuenta la ramificacion que corresponde a la entrada de control
            all_net = branches[0]#se decide cargar la red y especificar con que branch trabajar'a
        elif control_input == 3:
            all_net = branches[2]
        elif control_input == 4:
            all_net = branches[3]
        else:
            all_net = branches[1]

        feedDict = {x: image_input, input_speed: speed, dout: [1] * len(self.dropout_vec)}

        output_all = sess.run(all_net, feed_dict=feedDict)#se corre la sesion de tensorflow, y se obtiene la salida, se selecciona la ramificacion con la que se desea trabajar

        predicted_steers = (output_all[0][0])

        predicted_acc = (output_all[0][1])

        predicted_brake = (output_all[0][2])

        if self._avoid_stopping:
            predicted_speed = sess.run(branches[4], feed_dict=feedDict)
            predicted_speed = predicted_speed[0][0]
            real_speed = speed * 25.0

            real_predicted = predicted_speed * 25.0
            if real_speed < 2.0 and real_predicted > 3.0:
                # If (Car Stooped) and
                #  ( It should not have stopped, use the speed prediction branch for that)

                predicted_acc = 1 * (5.6 / 25.0 - speed) + predicted_acc

                predicted_brake = 0.0

                predicted_acc = predicted_acc[0][0]

        return predicted_steers, predicted_acc, predicted_brake
