'''
Created on Jul 11, 2017
'''

import _pickle as cPickle
import _pickle 
import gzip

import numpy as np
from PIL import Image
import matplotlib

# For plotting graphs via ssh with no display
# Ref: https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from numpy import float32
from sklearn import metrics
from sklearn.cluster.k_means_ import KMeans
from sklearn import manifold
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn import preprocessing
import os
from keras.preprocessing.image import load_img

import _pickle as cPickle
import _pickle 
import gzip
from skimage import transform 
import numpy as np
from PIL import Image
import matplotlib
import os
# For plotting graphs via ssh with no display
# Ref: https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg')
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
from numpy import float32
from sklearn import metrics
from sklearn.cluster.k_means_ import KMeans
from sklearn import manifold
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn import preprocessing

import tensorflow as tf
import keras.backend as K
K.set_image_dim_ordering('tf')


class DatasetHelper(object):
    '''
    Utility class for handling different datasets
    '''

    def __init__(self, name):
        '''
        A dataset instance keeps dataset name, the input set, the flat version of input set
        and the cluster labels
        '''
        self.name = name
        if name == 'MNIST':
            self.dataset = MNISTDataset()
        elif name == 'STL':
            self.dataset = STLDataset()
        elif name == 'COIL20':
            self.dataset = COIL20Dataset()
        elif name == 'cancer':    # added by Sher
            self.dataset = CANCERDataset()    

    def loadDataset(self):
        '''
        Load the appropriate dataset based on the dataset name
        '''
        self.input, self.labels, self.input_flat = self.dataset.loadDataset()

    def getClusterCount(self):
        '''
        Number of clusters in the dataset - e.g 10 for mnist, 20 for coil20
        '''
        return self.dataset.cluster_count

    def iterate_minibatches(self, set_type, batch_size, targets=None, shuffle=False):
        '''
        Utility method for getting batches out of a dataset
        :param set_type: IMAGE - suitable input for CNNs or FLAT - suitable for DNN
        :param batch_size: Size of minibatches
        :param targets: None if the output should be same as inputs (autoencoders), otherwise takes a target array from which batches can be extracted. Must have the same order as the dataset, e.g, dataset inputs nth sample has output at target's nth element
        :param shuffle: If the dataset needs to be shuffled or not
        :return: generates a batches of size batch_size from the dataset, each batch is the pair (input, output)
        '''
        inputs = None
        if set_type == 'IMAGE':
            inputs = self.input
            if targets is None:
                targets = self.input
        elif set_type == 'FLAT':
            inputs = self.input_flat
            if targets is None:
                targets = self.input_flat
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]


class MNISTDataset(object):
    '''
    Class for reading and preparing MNIST dataset
    '''

    def __init__(self):
        self.cluster_count = 10

    def loadDataset(self):
        f = gzip.open('mnist/mnist.pkl.gz', 'rb')
        train_set, _, test_set = cPickle.load(f,encoding='latin1')
        train_input, train_input_flat, train_labels = self.prepareDatasetForAutoencoder(train_set[0], train_set[1])
        test_input, test_input_flat, test_labels = self.prepareDatasetForAutoencoder(test_set[0], test_set[1])
        f.close()
        # combine test and train samples
        return [np.concatenate((train_input, test_input)), np.concatenate((train_labels, test_labels)),
                np.concatenate((train_input_flat, test_input_flat))]

    def prepareDatasetForAutoencoder(self, inputs, targets):
        '''
        Returns the image, flat and labels as a tuple
        '''
        X = inputs
        X = X.reshape((-1,28, 28,1))
        return (X, X.reshape((-1, 28 * 28)), targets)


class CANCERDataset1(object):
    '''
    Class for reading and preparing MNIST dataset
    '''

    def __init__(self):
        self.cluster_count = 5

    def loadDataset(self):
        import pandas as pd
        import pandas as pd
        
        trainDF = pd.read_csv('cancer/TCGA_train.csv')     
        train_labels = trainDF[trainDF.columns[-1]]
        train_labels = np.asarray(train_labels)
                
        train_features = trainDF.drop(trainDF.columns[-1],axis=1)        
        train_features = train_features.as_matrix().astype(np.float32)
        train_features = np.asarray([[train_features[row][col] for col in range(1,16130)] for row in range(599)])
        train_features = np.asarray(train_features)
        
        testDF = pd.read_csv('cancer/TCGA_test.csv')
        test_labels = testDF[testDF.columns[-1]]
        test_labels = np.asarray(test_labels)
        
        test_features = testDF.drop(testDF.columns[-1],axis=1)        
        test_features = test_features.as_matrix().astype(np.float32)
        test_features = np.asarray([[test_features[row][col] for col in range(1,16130)] for row in range(200)])
        test_features = np.asarray(test_features)        
        
        train_input, train_input_flat, train_labels = self.prepareDatasetForAutoencoder(train_features, train_labels)
        test_input, test_input_flat, test_labels = self.prepareDatasetForAutoencoder(test_features, test_labels)

        # combine test and train samples
        return [np.concatenate((train_input, test_input)), np.concatenate((train_labels, test_labels)),
                np.concatenate((train_input_flat, test_input_flat))]

    def prepareDatasetForAutoencoder(self, inputs, targets):
        '''
        Returns the image, flat and labels as a tuple
        '''
        X = inputs
        X = X.reshape((-1, 127, 127, 1))
        return (X, X.reshape((-1, 127 * 127)), targets)        

class CANCERDataset(object):
    '''
    Class for reading and preparing CANCER dataset
    '''
    def __init__(self):
        self.cluster_count = 4

    def loadDataset(self):
          root ='/home/rkarim/Training_data/'
          features = []
          features_flat = []
          for rootName,dirName,fileNames in os.walk(root):
            if(not rootName == root):
               for fileName in fileNames:
                  imgGray = load_img(rootName+'/'+fileName,color_mode='grayscale')
                  transformed=transform.resize(np.array(imgGray),(512,512))
                  features += [transformed.reshape((transformed.shape[0],transformed.shape[1]))]
                  features_flat+=[transformed.reshape((transformed.shape[0]*transformed.shape[1]*1))]        
          features=np.stack(features)
          features_flat = np.stack(features_flat)
          labels= features
          return [np.concatenate((features, features),axis=0), np.concatenate((labels, labels),axis=0),
                np.concatenate((features_flat,features_flat),axis=0)]     
         
    def loadDataset1(self):
        import pandas as pd
        import pandas as pd
		 
        df = pd.read_csv('cancer/TCGA_train.csv')
        print(len(df.columns))
        
        labels = df[df.columns[-1]]
        features = df.drop(df.columns[-1],axis=1)        
        features = features.as_matrix().astype(np.float32)
        features = np.asarray([[features[row][col] for col in range(1,16130)] for row in range(599)])
        print("Is there any NaN value?")
        print(np.count_nonzero(np.isnan(features)))
        
        min_max_scaler = preprocessing.MinMaxScaler()
        train_input = min_max_scaler.fit_transform(features)
        print(np.isfinite(train_input))
        
        train_input_flat = train_input
        train_input = train_input.reshape((-1, 127, 127, 1))
        train_input_flat = np.reshape(train_input, (-1, 127 * 127))
        train_labels = np.asarray(labels)
        
        df2 = pd.read_csv('cancer/TCGA_test.csv')
        labels2 = df2[df.columns[-1]]
        features2 = df2.drop(df2.columns[-1],axis=1)
        
        features2 = features2.as_matrix().astype(np.float32)
        features2 = np.asarray([[features2[row][col] for col in range(1,16130)] for row in range(200)])

        test_input = np.asarray(features2)
        print(np.isfinite(test_input))
        
        test_input = min_max_scaler.fit_transform(test_input)
        test_input_flat = test_input
        test_input = test_input.reshape((-1, 127, 127, 1))
        test_input_flat = np.reshape(test_input, (-1, 127 * 127))
        test_labels = np.asarray(labels2)
        
        # combine test and train samples
        return [np.concatenate((train_input, test_input)), np.concatenate((train_labels, test_labels)),
                np.concatenate((train_input_flat, test_input_flat))]    
    

class STLDataset(object):
    '''
    Class for preparing and reading the STL dataset
    '''

    def __init__(self):
        self.cluster_count = 10

    def loadDataset(self):
        train_x = np.fromfile('stl/train_X.bin', dtype=np.uint8)
        train_y = np.fromfile('stl/train_y.bin', dtype=np.uint8)
        test_x = np.fromfile('stl/train_X.bin', dtype=np.uint8)
        test_y = np.fromfile('stl/train_y.bin', dtype=np.uint8)
        train_input = np.reshape(train_x, (-1, 3, 96, 96))
        train_labels = train_y
        train_input_flat = np.reshape(test_x, (-1, 1, 3 * 96 * 96))
        test_input = np.reshape(test_x, (-1, 3, 96, 96))
        test_labels = test_y
        test_input_flat = np.reshape(test_x, (-1, 1, 3 * 96 * 96))
        return [np.concatenate(train_input, test_input), np.concatenate(train_labels, test_labels),
                np.concatenate(train_input_flat, test_input_flat)]


class COIL20Dataset(object):
    '''
    Class for reading and preparing the COIL20Dataset
    '''

    def __init__(self):
        self.cluster_count = 20

    def loadDataset(self):
        train_x = np.load('coil/coil_X.npy').astype(np.float32) / 256.0
        train_y = np.load('coil/coil_y.npy')
        train_x_flat = np.reshape(train_x, (-1, 128 * 128))
        return [train_x, train_y, train_x_flat]


def rescaleReshapeAndSaveImage(image_sample, out_filename):
    '''
    For saving the reconstructed output as an image
    :param image_sample: output of the autoencoder
    :param out_filename: filename for the saved image
    :return: None (side effect) Image saved
    '''
    image_sample = ((image_sample - np.amin(image_sample)) / (np.amax(image_sample) - np.amin(image_sample))) * 255;
    image_sample = np.rint(image_sample).astype(int)
    image_sample = np.clip(image_sample, a_min=0, a_max=255).astype('uint8')
    img = Image.fromarray(image_sample, 'L')
    img.save(out_filename)


def cluster_acc(y_true, y_pred):
    '''
    Uses the hungarian algorithm to find the best permutation mapping and then calculates the accuracy wrt
    Implementation inpired from https://github.com/piiswrong/dec, since scikit does not implement this metric
    this mapping and true labels
    :param y_true: True cluster labels
    :param y_pred: Predicted cluster labels
    :return: accuracy score for the clustering
    '''
    D = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((D, D), dtype=np.int32)
    for i in range(y_pred.size):
        idx1 = int(y_pred[i])
        idx2 = int(y_true[i])
        w[idx1, idx2] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def getClusterMetricString(method_name, labels_true, labels_pred):
    '''
    Creates a formatted string containing the method name and acc, nmi metrics - can be used for printing
    :param method_name: Name of the clustering method (just for printing)
    :param labels_true: True label for each sample
    :param labels_pred: Predicted label for each sample
    :return: Formatted string containing metrics and method name
    '''
    acc = cluster_acc(labels_true, labels_pred)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    return '%-50s     %8.3f     %8.3f' % (method_name, acc, nmi)


def evaluateKMeans(data, labels, nclusters, method_name):
    '''
    Clusters data with kmeans algorithm and then returns the string containing method name and metrics, and also the evaluated cluster centers
    :param data: Points that need to be clustered as a numpy array
    :param labels: True labels for the given points
    :param nclusters: Total number of clusters
    :param method_name: Name of the method from which the clustering space originates (only used for printing)
    :return: Formatted string containing metrics and method name, cluster centers
    '''
    kmeans = KMeans(n_clusters=nclusters, n_init=5)
    kmeans.fit(data)
    return getClusterMetricString(method_name, labels, kmeans.labels_), kmeans.cluster_centers_


def visualizeData(Z, labels, num_clusters, title):
    '''
    TSNE visualization of the points in latent space Z
    :param Z: Numpy array containing points in latent space in which clustering was performed
    :param labels: True labels - used for coloring points
    :param num_clusters: Total number of clusters
    :param title: filename where the plot should be saved
    :return: None - (side effect) saves clustering visualization plot in specified location
    '''
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(Z)
    fig = plt.figure()
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=fig.dpi)
