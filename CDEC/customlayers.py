'''
Created on Jul 25, 2017
'''

#from lasagne import layers
from keras.models import Sequential
from keras import backend as K
from keras import layers
from keras.engine.topology import Layer
class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    Layer borrowed from: https://swarbrickjones.wordpress.com/2015/04/29/convolutional-autoencoders-in-pythontheanolasagne/
    """

    def __init__(self, incoming, ds, **kwargs):
        self.ds = ds
        super(Unpool2DLayer, self).__init__(**kwargs)
 
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = input_shape[1] * self.ds[0]
        output_shape[2] = input_shape[2] * self.ds[1]
        return tuple(output_shape)
        
    def call(self,incoming,**kwargs):  
        '''
        Just repeats the input element the upscaled image
        '''
        repaxis2 =  K.repeat_elements(incoming,self.ds[0], axis=1)  
        Unpool_layer =  K.repeat_elements(repaxis2,self.ds[1], axis=2)
        return Unpool_layer


class ClusteringLayer(layers.Layer):
    '''
    This layer gives soft assignments for the clusters based on distance from k-means based
    cluster centers. The weights of the layers are the cluster centers so that they can be learnt
    while optimizing for loss
    '''
    def __init__(self,num_of_clusters, num_samples,latent_space_dim,**kwargs):
        self.num_of_clusters = num_of_clusters
        #self.alpha = alpha
        #self.cluster_centers = cluster_centers
        self.num_samples = num_samples
        self.latent_space_dim = latent_space_dim
        #self.intial_clusters = intial_clusters     
        super(ClusteringLayer, self).__init__(**kwargs)  
    def build(self,intial_clusters_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W', 
                                 shape=intial_clusters_shape,
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(ClusteringLayer, self).build(intial_clusters_shape)  # Be sure to call this at the end
          
      
    def call(self,incoming,**kwargs):
           
       return  getSoftAssignments(incoming,self.W,self.num_of_clusters,self.num_samples, self.latent_space_dim)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_of_clusters)
   
    def get_config(self):
        config = {'W': self.W}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def getSoftAssignments(latent_space, cluster_centers, num_clusters,num_samples,latent_space_dim):
    '''
    Returns cluster membership distribution for each sample
    :param latent_space: latent space representation of inputs
    :param cluster_centers: the coordinates of cluster centers in latent space
    :param num_clusters: total number of clusters
    :param latent_space_dim: dimensionality of latent space
    :param num_samples: total number of input samples
    :return: soft assigment based on the equation qij = (1+|zi - uj|^2)^(-1)/sum_j'((1+|zi - uj'|^2)^(-1))
    ''' 
    z_expanded = K.reshape(latent_space,shape=(num_samples,1,latent_space_dim,))
    z_expanded = K.tile(z_expanded, (1,num_clusters,1))
    u_expanded = K.tile(K.expand_dims(cluster_centers,0), [num_samples, 1, 1])#[1, 10,120] after expand_dims #[100,10,120] after tile
    distances_from_cluster_centers = K.sqrt(K.sum((z_expanded - u_expanded)**2,axis=2))#K.norm((z_expanded - u_expanded),2,axis=2)
    qij_numerator = 1 + distances_from_cluster_centers**2
    qij_numerator = 1 / qij_numerator
    normalizer_q  =  K.sum(qij_numerator, axis=1)
    normalizer_q  =  K.reshape(normalizer_q,(num_samples, 1))
    #print((qij_numerator/normalizer_q).shape)
    return qij_numerator/normalizer_q




