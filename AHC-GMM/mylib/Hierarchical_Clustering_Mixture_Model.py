# library mathematic & statistic 
import numpy as np
from numpy import linalg as LA
import math
import random
import itertools as it
from datetime import datetime

# data visualization and graphical plotting library
import matplotlib
import matplotlib . pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import PrintfTickFormatter
from ipywidgets import interact, IntSlider, Dropdown

# library for machine learning
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

output_notebook(hide_banner=True)

class Hierarchical_Mixture_Model :
    """
    Hierarchical Clustering of a Mixture Model
    Parameters
    ----------
    k : int
        Number of clusters/mixture components in which the data will be
        partitioned into.
    m : Number of clusters for reducing a large mixture of Gaussians
    n_iters : int
        Maximum number of iterations to run the algorithm default is 100.
    tol : float default is 0.01
    Tolerance. If the kl distance  between GMM and reduce GMM (two iterations) is smaller than
        the specified tolerance level, the algorithm will stop 
    seed : int
        Seed / random state used to initialize the parameters.
        
    mean_F : means of data  if data under law GMM with mean_F is known
    
    cov_F : covariances of data if  under law GMM with  cov_F is known
    
    weights_F : means of data if data under law GMM with weights_F  is known
    """
    
    def __init__(self,k,m, n_iters = 100 , tol = 0.01, seed =0   , mean_F = None, cov_F= None, weight_F= None ):
    
        self.k         = k                              
        self.m         = m                               
        self.n_iters   = n_iters
        self.tol       = tol
        self.seed      = seed
        
        self.mean_F = mean_F
        self.cov_F  = cov_F
        self.weight_F = weight_F
        
        if n_iters < 1:
            raise ValueError('GMM estimation requires at least one run')
        if tol < 0.:
            raise ValueError('Invalid value for covariance_type: %s' %
                             tol)

    def fit(self , X) :
        
        """ data's dimensionality and responsibility vector """
        self.X           =  X                         # X_train data dataset 
        self.N           =  X.shape[0]                # number of observations
        self.d           =  X.shape[1]                # dimension of variables
        self.KL_distance = []
        
        
        if self.N  < self.k:
            raise ValueError(
                'GMM estimation with %s components, but got only %s samples' %
                (self.n_components, self.N))
        
        """ call large function F to estimate params of dataset """
        if self.mean_F is None :
            self.large_GMM()
            
        """ initialize parameters"""
        self._init_G()                                 # call _init_G function to initialize params of G function
        self._init_PI()                                # call _init_PI function to initialize params of PI

        ''' update params of G '''
        self.update_param()
        
        return self 
    
    def large_GMM(self) :
        
        ''' get F function '''
        
        gmm            = GaussianMixture(n_components=self.k, random_state=0).fit(self.X)
        self.mean_F    = gmm.means_        # mean of F
        self.weight_F  = gmm.weights_      # weight of F
        self.cov_F     = gmm.covariances_  # covariance of F 
           
    def _init_G(self):
        
        """ randomly initialize the starting GMM parameters of G """
        
        np.random.seed(self.seed)
        # cluster means to. initialize mean of G
        kmeans = KMeans(n_clusters=self.m, random_state=0).fit(self.X)
        self.mean_G = dict(enumerate( kmeans.cluster_centers_))
        # cluster covariance matrices
        shape = self.m, self.d, self.d
        self.cov_G = dict(enumerate( np.full(shape, np.cov(self.X, rowvar = False))))
        # cluster priors
        W= np.random.rand(self.m)
        self.weights_G = dict(enumerate(W / W.sum() , 0))
        
    def _init_PI(self):
        
        """ randomly initialize the starting PI """ 
        
        lst = list(range(self.m))
        for i in range(self.k-self.m):
            lst.append(random.choice(lst))
            random.shuffle(lst)
        self.P = lst
        
    def kl_mvn(self, m0, S0, m1, S1):
        
        """ Kullback-Liebler divergence from Gaussian 
            KL( (m0, S0) || (m1, S1)) = .5 * ( tr(S1^{-1} S0) 
            + log |S1|/|S0| + (m1 - m0)^T S1^{-1} (m1 - m0) - N )
        """
        # store inv diag covariance of S1 and diff between means
        N = m0.shape[0]
        iS1 = np.linalg.inv(S1)
        diff = m1 - m0
        # kl is made of three terms
        tr_term   = np.trace(np.dot(iS1 , S0))
        np.seterr(divide='ignore', invalid='ignore') 
        det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0))
        quad_term = np.dot(np.dot(diff.T , np.linalg.inv(S1) ), diff) 
        return .5 * (tr_term + det_term + quad_term - N) 
    
    
    def update_G( self):
        
        '''   update params [mean_G ,cov_G ,weights_G] of G function  '''
        d_weight = dict()
        d_mean = dict()
        d_sigma=dict()
        #********calcul mean & cov & weights of G ******
        weight_G=dict()
        mean_G=dict()
        cov_G=dict()
        for u in np.unique(self.P):
            
            index = np.where(self.P==u) 
            d_weight[u]=self.weight_F[index ]
            d_mean[u]=self.mean_F[ index ]
            d_sigma[u]=self.cov_F[ index ]
            #******** calcul weights of G ********
            weight_G[u]=np.sum(d_weight[u] , axis=0)
            #******** calcul mean  of G **********
            mean_G[u]=1/weight_G[u]*np.sum(d_mean[u]*d_weight[u].reshape(-1,1),axis=0) 
            #******** calcul cov of G ************
            prod    = [ np.dot( np.asmatrix(row).T , np.asmatrix(row)) for row in d_mean[u]-mean_G[u]]
            cov_G[u] = 1/weight_G[u]*np.sum(d_sigma[u]*d_weight[u].reshape(-1,1,1)
                                            + prod*d_weight[u].reshape(-1,1,1)
                                            , axis = 0)
        # params of G 
        self.mean_G   = mean_G 
        self.cov_G    = cov_G
        self.weight_G = weight_G
             
    def update_PI(self ):
        
        ''' update PI  '''
        
        args= [[self.kl_mvn(self.mean_F[u] ,
                            self.cov_F[u] , self.mean_G[v]
                            , self.cov_G[v] ) for v in self.mean_G.keys()] for u in range(self.k)  ]
        self.P = np.argmin(np.asarray(args), axis=1)
        
    def distance(self):
        
        '''Compute the distance between two GMM: F and G'''
        mean_G_PI = np.stack(list(self.mean_G.values()) , axis=0 )[self.P]
        cov_G_PI  = np.stack(list(self.cov_G.values())  , axis=0 )[self.P]
        args=np.sum([ self.weight_F[i]*self.kl_mvn(self.mean_F[i], 
                                                   self.cov_F[i], mean_G_PI[i] , 
                                                   cov_G_PI[i]) for i in range(self.k)])
        return args
    
    def update_param(self ):
        
        it               = 0
        KL_distance_old  = self.distance()
        self.converged   = False
        self.KL_distance = []
        self.KL_distance.append(KL_distance_old)
        for it in range(self.n_iters): 
            # update PI
            self.update_PI() 
            # update G
            self.update_G()
            # calculate convergence stability
            KL_distance_new = self.distance()
            if abs(KL_distance_new - KL_distance_old) <= self.tol :
                self.converged = True
                break
            KL_distance_old = KL_distance_new 
            self.KL_distance.append(KL_distance_old )
            
     
    def predict_likelihood(self , X_test):
        
        likelihood = np.zeros( (X_test.shape[0] , len(self.mean_G)) )
        for it in self.mean_G.keys():
            distribution = multivariate_normal(
                self.mean_G[it], 
                self.cov_G[it])
            likelihood[:,it] = distribution.pdf(X_test)
        return likelihood
    
    def predict(self, X_test):
        
        likelihood       = self.predict_likelihood( X_test)
        numerator        = likelihood * np.asarray(list(self.weights_G.values()))
        denominator      = numerator.sum(axis=1)[:, np.newaxis]
        np.seterr(divide ='ignore', invalid='ignore') 
        args             = numerator / denominator
        return np.argmax(args , axis=1) 