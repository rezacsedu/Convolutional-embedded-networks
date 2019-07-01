'''
Created on Jul 11, 2017
'''
#from keras.utils import plot_model
from datetime import datetime
import logging
import tensorflow as tf
import math
#from lasagne import layers
from keras import layers
from sklearn.cluster import KMeans
import time
import keras
from keras import regularizers
import signal
from keras.models import Sequential
from customlayers import ClusteringLayer, Unpool2DLayer, getSoftAssignments
from misc import evaluateKMeans, visualizeData, rescaleReshapeAndSaveImage
import numpy as np
#import theano.tensor as T
#from keras.backend import tensor_array_ops as tensor
from keras.layers import BatchNormalization
import keras.backend as K
from keras.layers import LSTM, Dense, Dropout, GaussianDropout
import re
# Logging utilities - logs get saved in folder logs named by date and time, and also output
# at standard output
logFormatter = logging.Formatter("[%(asctime)s]  %(message)s", datefmt='%m/%d %I:%M:%S')

rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(datetime.now().strftime('logs/dcjc_%H_%M_%d_%m.log'))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

class LossHistory(keras.callbacks.Callback):
    def on_epoch_begin(self,epoch,logs=None):
        self.losses = []

    def on_epoch_end(self,epoch, logs={}):
        self.losses.append(logs.get('loss'))
class DCJC(object):

    # Main class holding autoencoder network and training functions
    def __init__(self, network_description):
        signal.signal(signal.SIGINT, self.signal_handler)
        self.name = network_description['name']
        netbuilder = NetworkBuilder(network_description)
        self.shouldStopNow  = False
        # Get the lasagne network using the network builder class that creates autoencoder with the specified architecture
        self.model,self.encoder_model = netbuilder.buildNetwork()
        self.network = self.model
        self.encode_layer, self.encode_size = netbuilder.getEncodeLayerAndSize()
        self.input_type = netbuilder.getInputType()
        self.batch_size = netbuilder.getBatchSize()
        rootLogger.info("Network: " + self.networkToStr())
        # Latent/Encoded space is the output of the bottleneck/encode layer
        encode_prediction_expression = self.encode_layer
        # Loss for autoencoder = reconstruction loss + weight decay regularizer
        recon_prediction_expression = self.network.layers[-1].output       
        # SGD with momentum + Decaying learning rate
        self.learning_rate = 0.0001
        self.trainAutoencoder = keras.models.Model(inputs=[self.network.layers[0].input], outputs=[self.network.layers[-1].output])
        #plot_model(self.trainAutoencoder, to_file='self.trainAutoencoder.png', show_shapes=True)
        #Image(filename='self.trainAutoencoder.png') 
        adam = keras.optimizers.Adam(lr=0.0001)
        self.trainAutoencoder.compile(loss="mean_squared_error",optimizer=adam)
        # Reconstruction is just output of the network
        self.predictReconstruction = keras.models.Model(inputs=[self.network.layers[0].input],outputs=[recon_prediction_expression])
        #plot_model(self.predictReconstruction, to_file='self.predictReconstruction.png', show_shapes=True)
        #Image(filename='self.predictReconstruction.png') 
        # encoded is just output of predictEncoding for input, plot model save in png image to see the structure of model created.
        self.predictEncoding = keras.models.Model(inputs=[self.network.layers[0].input],outputs=[encode_prediction_expression])
        #plot_model(self.predictEncoding, to_file='self.predictEncoding.png', show_shapes=True)
        #Image(filename='self.predictEncoding.png') 

    def getReconstructionLossExpression(self, prediction_expression, t_target):
        '''
        Reconstruction loss = means square error between input and reconstructed input
        '''
        loss = keras.losses.mean_squared_error(prediction_expression, t_target)
        return loss

    def signal_handler(self,signal, frame):

        command = raw_input('\nWhat is your command?')
        if str(command).lower()=="stop":
            self.shouldStopNow  = True
        else:
            exec(command)
    
    def pretrainWithData(self, dataset, epochs, continue_training=False):
        '''
        Pretrains the autoencoder on the given dataset
        :param dataset: Data on which the autoencoder is trained
        :param epochs: number of training epochs
        :param continue_training: Resume training if saved params available
        :return: None - (side effect) saves the trained network params and latent space in appropriate location
        '''
        batch_size = self.batch_size
        # array for holding the latent space representation of input
        Z = np.zeros((dataset.input.shape[0], int(self.encode_size)), dtype=np.int32)
        # in case we're continuing training load the network params
        if continue_training:
            with np.load('saved_params/%s/m_%s.npz' % (dataset.name, self.name),encoding=='latin1') as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                self.trainAutoencoder.set_weights(param_values)
        for epoch in range(epochs):
            error = 0
            total_batches = 0
            for batch in dataset.iterate_minibatches(self.input_type, batch_size, shuffle=True):
                inputs, targets = batch
                history  = self.trainAutoencoder.fit(inputs, targets,steps_per_epoch=1)
                error += history.history['loss'][0]
                total_batches += 1
            # learning rate decay
            self.learning_rate = self.learning_rate * float(0.9999)
            # For every 20th iteration, print the clustering accuracy and nmi - for checking if the network
            # is actually doing something meaningful - the labels are never used for training
            if (epoch + 1) % 2 == 0:
                for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                    batch_reverse = inputs.reshape(tuple([batch[0].shape[0]])+tuple(reversed(list(batch[0].shape[1:]))))
                    Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding.predict(batch[0])
                rootLogger.info(evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), "%d/%d [%.4f]" % (epoch + 1, epochs, error / total_batches))[0])
            else:
                # Just report the training loss
                rootLogger.info("%-30s     %8s     %8s" % ("%d/%d [%.4f]" % (epoch + 1, epochs, error / total_batches), "", ""))
            if self.shouldStopNow:
            	break
        # The inputs in latent space after pretraining
        for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
            #batch_reverse = inputs.reshape(tuple([batch[0].shape[0]])+tuple(reversed(list(batch[0].shape[1:]))))
            Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding.predict(batch[0],steps=1)
        # Save network params and latent space
        np.save('saved_params/%s/z_%s.npy' % (dataset.name, self.name), Z)
        # Borrowed from mnist lasagne example
        print(np.array(self.trainAutoencoder.get_weights())[0].shape)
        np.savez('saved_params/%s/m_%s.npz' % (dataset.name, self.name), *self.trainAutoencoder.get_weights())
    
    def doClusteringWithKLdivLoss(self, dataset, combined_loss, epochs):
        '''
        Trains the autoencoder with combined kldivergence loss and reconstruction loss, or just the kldivergence loss
        At the moment does not give good results
        :param dataset: Data on which the autoencoder is trained
        :param combined_loss: boolean - whether to use both reconstruction and kl divergence loss or just kldivergence loss
        :param epochs: Number of training epochs
        :return: None - (side effect) saves the trained network params and latent space in appropriate location
        '''
        batch_size = self.batch_size
        # Load saved network params and inputs in latent space obtained after pretraining
        with np.load('saved_params/%s/m_%s.npz' % (dataset.name, self.name),encoding="latin1") as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            self.network.set_weights(param_values)	
        Z = np.load('saved_params/%s/z_%s.npy' % (dataset.name, self.name))
        #just for tracing.
        #Z_reshaped = Z.reshape((Z.shape[0],Z.shape[-1]))
        #print(self.model.layers[1].output)
        #print(Z.shape)
        #print(dataset.labels) 
        #print(dataset.getClusterCount())
        #Find initial cluster centers
        quality_desc, cluster_centers = evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), 'Initial')
        rootLogger.info(quality_desc)
        #Extend the network so it calculates soft assignment cluster distribution for the inputs in latent space
        clustering_network = ClusteringLayer(dataset.getClusterCount(),batch_size,int(self.encode_size))
        clustering_network.build(cluster_centers.shape)
        cluster_output = clustering_network(self.encode_layer)
        reconstructed_output_exp = self.network.layers[-1].output
        #get soft assignments model and plot model
        soft_model = keras.models.Model(inputs=self.encoder_model.layers[0].input,outputs=cluster_output)
        #plot_model(soft_model, to_file='soft_model.png', show_shapes=True)
        #Image(filename='soft_model.png') 
        # SGD with momentum, LR = 0.01, Momentum = 0.9
        adam = keras.optimizers.Adam(0.0001)
        trainFunction = None
       
        if combined_loss:
            trainFunction = keras.models.Model(inputs=self.encoder_model.layers[0].input,outputs=[reconstructed_output_exp,cluster_output])
            #plot_model(trainFunction, to_file='train.png', show_shapes=True)
            #Image(filename='train.png')
            trainFunction.compile(loss=['mse','kld'],loss_weights=[1.0, 0.1], optimizer=adam)
        else:
            trainFunction = keras.models.Model(inputs=[self.network.layers[0].input],outputs=[cluster_output])
            trainFunction.compile(loss='kld', optimizer=adam)
        
        for epoch in range(epochs):
            # Get the current distribution
            qij = np.zeros((dataset.input.shape[0], dataset.getClusterCount()), dtype=np.float32)
            for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                qij[i * batch_size: (i + 1) * batch_size] = soft_model.predict(batch[0],steps=1)
            # Calculate the desired distribution
            pij = self.calculateP(qij)
            error = 0
            total_batches = 0
            history=LossHistory()
            for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, pij, shuffle=True)):
                if (combined_loss):
                    history = trainFunction.fit(x=[batch[0]], y=[batch[0], batch[1]],steps_per_epoch=1)
                    error +=history.history['loss'][0]
                else:
                    history = trainFunction.fit(batch[0], batch[1],steps_per_epoch=1)
                    error += history.history['loss'][0]
                total_batches += 1
            for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding.predict(batch[0])
            # For every 10th iteration, print the clustering accuracy and nmi - for checking if the network
            # is actually doing something meaningful - the labels are never used for training
            if (epoch + 1) % 10 == 0:
                rootLogger.info(evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), "%d [%.4f]" % (
                    epoch, error / total_batches))[0])
            if self.shouldStopNow:
           	   break
        # Save the inputs in latent space and the network parameters
        for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
            Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding.predict(batch[0])
        np.save('saved_params/%s/pc_z_%s.npy' % (dataset.name, self.name), Z)
        np.savez('saved_params/%s/pc_m_%s.npz' % (dataset.name, self.name),
                 *trainFunction.get_weights())

    def calculateP(self, Q):
        # Function to calculate the desired distribution Q^2, for more details refer to DEC paper
        f = Q.sum(axis=0)
        pij_numerator = Q * Q
        pij_numerator = pij_numerator / f
        normalizer_p = pij_numerator.sum(axis=1).reshape((Q.shape[0], 1))
        P = pij_numerator / normalizer_p
        return P

    def doClusteringWithKMeansLoss(self, dataset, epochs):
        '''
        Trains the autoencoder with combined kMeans loss and reconstruction loss
        At the moment does not give good results
        :param dataset: Data on which the autoencoder is trained
        :param epochs: Number of training epochs
        :return: None - (side effect) saves the trained network params and latent space in appropriate location
        '''
        batch_size = self.batch_size
        # Load the inputs in latent space produced by the pretrained autoencoder and use it to initialize cluster centers
        Z = np.load('saved_params/%s/z_%s.npy' % (dataset.name, self.name),encoding='latin1')
        quality_desc, cluster_centers = evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), 'Initial')
        rootLogger.info(quality_desc)
        #Load network parameters - code borrowed from mnist lasagne example
        with np.load('saved_params/%s/m_%s.npz' % (dataset.name, self.name),encoding='latin1') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            model=self.network
            model.set_weights(param_values)
        #extent the network to do soft cluster assignments
        clustering_network = ClusteringLayer(dataset.getClusterCount(),batch_size, int(self.encode_size),name='cluster')
        clustering_network.build(cluster_centers.shape)
        soft_assignments =  clustering_network(self.encode_layer)
        weights_cluster  =  clustering_network.get_config()['W']
        #Parameters help in the custom loss of KMeans
        self.soft_assignments = soft_assignments 
        self.num_clusters     = dataset.getClusterCount()
        self.latent_space_dim = int(self.encode_size)
        self.num_samples      = batch_size   
        weight_reconstruction = 1
        weight_kmeans = 0.1
        #Optimizer SGD And Model
        sgd = keras.optimizers.Adam(0.0001)
        trainKMeansWithAE = keras.models.Model(inputs=[model.layers[0].input] ,outputs=[model.layers[-1].output,soft_assignments])
        trainKMeansWithAE.compile(loss=['mse',self.KMeansLoss],loss_weights =[weight_reconstruction,weight_kmeans] ,optimizer=sgd)
        #plot_model(trainKMeansWithAE, to_file='trainKMeansWithAE.png', show_shapes=True)
        #Image(filename='trainKMeansWithAE.png')
        #Tensorboard To Visualizations Gradients.
        tensorboard1 = keras.callbacks.TensorBoard(log_dir="logs/{}".format(time.time()),  write_grads=True, write_images=True, histogram_freq=0)
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=20)
        for epoch in range(epochs):
            error = 0
            total_batches = 0
            for batch in dataset.iterate_minibatches(self.input_type, batch_size, shuffle=True):
                inputs, targets = batch
                encoded=self.predictEncoding.predict(inputs)
                self.y_pred = kmeans.fit_predict(encoded)
                trainKMeansWithAE.get_layer(name='cluster').set_weights([kmeans.cluster_centers_])
                history=trainKMeansWithAE.fit(inputs, [targets,encoded],steps_per_epoch=1,callbacks=[tensorboard1])
                error += history.history['loss'][0] 
                total_batches += 1
            # For every 10th epoch, update the cluster centers and print the clustering accuracy and nmi - for checking if the network
            # is actually doing something meaningful - the labels are never used for training
            if (epoch + 1) % 10 == 0:
                for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                    batch_reverse = batch[0].reshape(tuple([batch[0].shape[0]])+tuple(reversed(list(batch[0].shape[1:]))))
                    Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding.predict(batch[0])
                quality_desc, cluster_centers = evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(), "%d/%d [%.4f]" % (epoch + 1, epochs, error / total_batches))
                rootLogger.info(quality_desc)
            else:
                # Just print the training loss
                rootLogger.info("%-30s     %8s     %8s" % ("%d/%d [%.4f]" % (epoch + 1, epochs, error / total_batches), "", ""))
            if self.shouldStopNow:
            	break

        # Save the inputs in latent space and the network parameters
        for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
            Z[i * batch_size:(i + 1) * batch_size] = self.predictEncoding.predict(batch[0])
        np.save('saved_params/%s/pc_km_z_%s.npy' % (dataset.name, self.name), Z)
        np.savez('saved_params/%s/pc_km_m_%s.npz' % (dataset.name, self.name),
                 *trainKMeansWithAE.get_weights())
     
    def KMeansLoss(self,y_true,y_pred,soft_loss=False):
        # Kmeans loss = weighted sum of latent space representation of inputs from the cluster centers
        z = K.reshape(y_true,(self.num_samples, 1, self.latent_space_dim))
        z = K.tile(z, (1, self.num_clusters, 1))
        y_pred = K.expand_dims(y_pred,0)
        u = K.reshape(y_pred,(self.num_samples,self.num_clusters,1))
        u = K.tile(u,[1,1,self.latent_space_dim])
        distances = K.sqrt(K.sum((z - u)**2,axis=2))
        distances = K.reshape(distances,(self.num_samples,self.num_clusters))
        if soft_loss:
            weighted_distances = distances * self.soft_assignments
            loss = K.sum(weighted_distances,axis=1)
            loss = K.mean(loss)
        else:
            loss = K.min(distances,axis=1)
            loss = K.mean(loss)
        return loss
      

    def networkToStr(self):
        # Utility method for printing the network structure in a shortened form
        result=''
        layers = self.network.layers
        for layer in layers:
            t = type(layer)
            if t is keras.layers.InputLayer:
                pass
            else:
                result += ' ' + layer.name
        return result.strip()

  
    def get_output(self):
        return network
   
class NetworkBuilder(object):
    '''
    Class that handles parsing the architecture dictionary and creating an autoencoder out of it
    '''
    def __init__(self, network_description):
        '''
        :param network_description: python dictionary specifying the autoencoder architecture
        '''
        # Populate the missing values in the dictionary with defaults, also add the missing decoder part
        # of the autoencoder which is missing in the dictionary
        self.network_description = self.populateMissingDescriptions(network_description)
        # Create theano variables for input and output - would be of different types for simple and convolutional autoencoders
        if self.network_description['network_type'] == 'CAE':
            self.input_type = "IMAGE"
        else:
            self.input_type = "FLAT"
        self.network_type = self.network_description['network_type']
        self.BatchNormalizationm = bool(self.network_description["use_batch_norm"])
        self.layer_list = []
        self.model=Sequential()

    def getBatchSize(self):
        return self.network_description["batch_size"]

    def getInputAndTargetVars(self):
        return self.t_input, self.t_target

    def getInputType(self):
        return self.input_type

    def buildNetwork(self):
        '''
        :return: Lasagne autoencoder network based on the network decription dictionary
        '''
        model = Sequential()
        encoder_model = Sequential()
        f= 0
        i = 0
        layers_list=[]
        for layer in self.network_description['layers']:
                    self.processLayer(layer,model)
                    if(not re.search("conv2d_transpose",str(model.layers[i].output))):
                          if(f!=1):
                             encoder_model.add(model.layers[i])
                             i = i+1
                    else:
                         f = 1  
                 
        return model,encoder_model

    def getEncodeLayerAndSize(self):
        '''
        :return: The encode layer - layer between encoder and decoder (bottleneck)
        '''
        return self.encode_layer, self.encode_size

    def populateDecoder(self, encode_layers):
        '''
        Creates a specification for the mirror of encode layers - which completes the autoencoder specification
        '''
        decode_layers = []
        for i, layer in reversed(list(enumerate(encode_layers))):
            if (layer["type"] == "MaxPool2D*"):
                # Inverse max pool doesn't upscale the input, but does reverse of what happened when maxpool
                # operation was performed
                decode_layers.append({
                    "type": "InverseMaxPool2D",
                    "layer_index": i,
                    'filter_size': layer['filter_size']
                })
            elif (layer["type"] == "MaxPool2D"):
                # Unpool just upscales the input back
                decode_layers.append({
                    "type": "Unpool2D",
                    'filter_size': layer['filter_size']
                })
            elif (layer["type"] == "Conv2D"):
                # Inverse convolution = deconvolution
                decode_layers.append({
                    'type': 'Deconv2D',
                    'conv_mode': layer['conv_mode'],
                    'non_linearity': layer['non_linearity'],
                    'filter_size': layer['filter_size'],
                    'num_filters': encode_layers[i - 1]['output_shape'][0]
                })
            elif (layer["type"] == "Dense" and not layer["is_encode"]):
                # Inverse of dense layers is just a dense layer, though we dont create an inverse layer corresponding to bottleneck layer
                decode_layers.append({
                    'type': 'Dense',
                    'num_units': encode_layers[i]['output_shape'][2],
                    'non_linearity': encode_layers[i]['non_linearity']
                })
                # if the layer following the dense layer is one of these, we need to reshape the output
                if (encode_layers[i - 1]['type'] in ("Conv2D", "MaxPool2D", "MaxPool2D*")):
                    decode_layers.append({
                        "type": "Reshape",
                        "output_shape": encode_layers[i - 1]['output_shape']
                    })
        encode_layers.extend(decode_layers)

    def populateShapes(self, layers):
        # Fills the dictionary with shape information corresponding to each layer, which will be used in creating the decode layers
        last_layer_dimensions = layers[0]['output_shape']
        for layer in layers[1:]:
            if (layer['type'] == 'MaxPool2D' or layer['type'] == 'MaxPool2D*'):
                layer['output_shape'] = [last_layer_dimensions[0], last_layer_dimensions[1] / layer['filter_size'][0],
                                         last_layer_dimensions[2] / layer['filter_size'][1]]
            elif (layer['type'] == 'Conv2D'):
                multiplier = 1
                if (layer['conv_mode'] == "same"):
                    multiplier = 0
                layer['output_shape'] = [layer['num_filters'],
                                         last_layer_dimensions[1] - (layer['filter_size'][0] - 1) * multiplier,
                                         last_layer_dimensions[2] - (layer['filter_size'][1] - 1) * multiplier]
              
            elif (layer['type'] == 'Dense'):
                layer['output_shape'] = [1, 1, layer['num_units']]
            last_layer_dimensions = layer['output_shape']

    def populateMissingDescriptions(self, network_description):
        # Complete the architecture dictionary by filling in default values and populating description for decoder
        if 'network_type' not in network_description:
            if (network_description['name'].split('_')[0].split('-')[0] == 'fc'):
                network_description['network_type'] = 'AE'
            else:
                network_description['network_type'] = 'CAE'
        for layer in network_description['layers']:
            if 'conv_mode' not in layer:
                layer['conv_mode'] = 'valid'
            layer['is_encode'] = False
        network_description['layers'][-1]['is_encode'] = True
        if 'output_non_linearity' not in network_description:
            network_description['output_non_linearity'] = network_description['layers'][1]['non_linearity']
        self.populateShapes(network_description['layers'])
        self.populateDecoder(network_description['layers'])
        if 'use_batch_norm' not in network_description:
            network_description['use_batch_norm'] = False
        for layer in network_description['layers']:
            if 'is_encode' not in layer:
                layer['is_encode'] = False
            layer['is_output'] = False
        network_description['layers'][-1]['is_output'] = True
        network_description['layers'][-1]['non_linearity'] = network_description['output_non_linearity']
        return network_description
        
    def getInitializationFct(self):
         return keras.initializers.glorot_uniform()

    def processLayer(self, layer_definition,model):
        '''
        Create a lasagne layer corresponding to the "layer definition"
        '''
        #Regularizer Adjustment For Layers
        self.l2 = 0.01
        if (layer_definition["type"] == "Input"):
            if self.network_type == 'CAE':
               #print(layer_definition['output_shape'])
               model.add(keras.layers.InputLayer(tuple(list(reversed(layer_definition['output_shape'])),)))
               print("CAE")
            elif self.network_type == 'AE':
               print("AE")
               model.add(keras.layers.InputLayer((layer_definition['output_shape'][2],)))
        elif (layer_definition['type'] == 'Dense'): model.add(keras.layers.Dense(input_shape=self.model.layers[-1].output.shape,units=layer_definition['num_units'],activation='relu', name=self.getLayerName(layer_definition),kernel_initializer=self.getInitializationFct()))
       
        elif (layer_definition['type'] == 'Conv2D'):
            print("Conv2D")
            model.add(keras.layers.Conv2D(filters=layer_definition["num_filters"],kernel_size=tuple(layer_definition["filter_size"]), padding=layer_definition['conv_mode'],activation='relu'))
        elif (layer_definition['type'] == 'MaxPool2D' or layer_definition['type'] == 'MaxPool2D*'):
            print("max")
            model.add(keras.layers.MaxPool2D(pool_size=tuple(layer_definition["filter_size"])))
        elif (layer_definition['type'] == 'InverseMaxPool2D'):
                    if re.search("max_pooling",str(self.layer_list[layer_definition['layer_index']])):
                        print("Unpooling")
                        #Another method to inverse layer.Using_Upsampling
                        #model.add(keras.layers.UpSampling2D(size=tuple(layer_definition['filter_size'])))
                        UnpoolLayer=Unpool2DLayer(model.layers[-1].output,tuple(layer_definition['filter_size']))
                        UnpoolLayer.call(model.layers[-1].output)
                        model.add(UnpoolLayer)  
                        print(model.layers[-1].output.shape)           
        elif (layer_definition['type'] == 'Unpool2D'):
            UnpoolLayer=Unpool2DLayer(model.layers[-1].output,tuple(layer_definition['filter_size']))
            UnpoolLayer.call(model.layers[-1].output)
            model.add(UnpoolLayer)                     
        elif (layer_definition['type'] == 'Reshape'):
            print("RSH")   
            model.add(keras.layers.Reshape(model.layers[-1].output, input_shape=([-1] + layer_definition["output_shape"])))
        elif (layer_definition['type'] == 'Deconv2D'):
            print("Deconv2D")            
            model.add(keras.layers.Conv2DTranspose(filters=layer_definition['num_filters'],kernel_size=tuple(layer_definition['filter_size']),padding=layer_definition['conv_mode'], activation='relu'))
         
        self.layer_list.append(model.layers[-1].output)
        # Batch normalization on all convolutional layers except if at output
        if (self.BatchNormalizationm and (not layer_definition["is_output"]) and layer_definition['type'] in ("Conv2D", "Deconv2D")):
            print("BatchNorm")
            model.add(BatchNormalization())
            model.add(GaussianDropout(0.7))
        # Save the encode layer separately
        if (layer_definition['is_encode']):
            self.encode_layer= keras.layers.Flatten()(model.layers[-1].output)
            self.encode_size = layer_definition['output_shape'][0] * layer_definition['output_shape'][1] * layer_definition['output_shape'][2]
    
    def getNonLinearity(self, non_linearity):
        return {
            'rectify':keras.activations.relu,
            'linear': keras.activations.linear,
            'elu':keras.activations.elu
        }[non_linearity]
    

