# library mathematic & statistic 
import numpy as np
from numpy import linalg as LA
import math
import random
from functools import reduce
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
        Hierarchical Clustering of a Mixture Model.
        
        Parametres  : 
        =============
        k            : The number of mixture components of a large mixture of Gaussians
        m            : The number of mixture components of a a smaller mixture of Gaussians
        X            : data to training 
        n_iters      : Number of iterations. Default to 10000
        tol.         : tolerance for  convergence. Default to 0.000001
        mean_F       : Default to None , means of  data under Gaussians mixture model
        cov_F        : Default to None , covaraince of  data under Gaussians mixture model
        weight_F     : Default to None , weights of  data under Gaussians mixture model
        
        Attributes :
        ============
        mean_G       : Estimate means of smaller mixture of Gaussians  model
        cov_G        : Estimate covariances of smaller mixture of Gaussians  model
        weights_G    : Estimate weights of smaller mixture of Gaussians  model
        N            : Number of samples of training data
        d            : n_features of training data
        KL_distance. : computing distances 
        converged.   : True once converged False otherwise.
        """
    
    def __init__(self , k :  int ,m : int , n_iters :  int = 10000 , tol : float = 0.000001, seed =  0   , mean_F = None, cov_F= None, weight_F= None ):
        self.k         = k                              
        self.m         = m                               
        self.n_iters   = n_iters
        self.tol       = tol
        self.seed      = seed
        self.mean_F = mean_F
        self.cov_F  = cov_F
        self.weight_F = weight_F
        if n_iters < 1: raise ValueError('GMM estimation requires at least one run')
        if tol < 0.: raise ValueError('Invalid value for tol try with value positive: %s' %tol)

    def fit(self , X) :
        self.X           =  X                        
        self.N           =  X.shape[0]                
        self.d           =  X.shape[1]
        self.KL_distance = []
        if self.N  < self.k : raise ValueError('GMM estimation with %s components, but got only %s samples' %(self.n_components, self.N))
        if (self.mean_F and self.cov_F and self.weight_F) is None :
            gmm            = GaussianMixture(n_components=self.k, random_state=0).fit(self.X)
            self.mean_F    = gmm.means_       
            self.weight_F  = gmm.weights_      
            self.cov_F     = gmm.covariances_
        # initialization
        np.random.seed(self.seed)
        kmeans = KMeans(n_clusters=self.m, random_state=0).fit(self.X)
        self.mean_G = kmeans.cluster_centers_
        self.cov_G =  np.full((self.m, self.d, self.d),np.cov( self.X , rowvar=False ) )
        self.weights_G =  np.asarray([1/self.m]*self.m )
        KL_distance =[0.]
        self.converged   = False
        for it in range(self.n_iters):
            self.update_PI()
            KL_distance.append(self.dis_KL_GMM())
            if abs(KL_distance[it+1]-KL_distance[it]) <= self.tol : 
                self.converged = True
                break
            self.update_G()
        self.KL_distance = KL_distance[1:]
        return self
    
    def kl_mvn(self, gaussinne_1, gaussinne_2):
        """ 
        Kullback-Liebler divergence from Gaussian 
        KL( (m0, S0) || (m1, S1)) = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + (m1 - m0)^T S1^{-1} (m1 - m0) - N )
        """
        m0, S0 = gaussinne_1 ;  m1, S1 = gaussinne_2
        N = m0.shape[0] ; iS1 = np.linalg.pinv(S1) ; diff = m1 - m0
        tr_term   = np.trace(np.dot(iS1 , S0)) ; det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0))
        quad_term = np.dot(np.dot(diff.T , np.linalg.pinv(S1) ), diff) 
        return .5 * (tr_term + det_term + quad_term - N)
    
    def update_PI(self ):
        """  the optimal mapping function between the components of F and G  """
        PI = np.array([list(map(lambda  gauss_G :  self.kl_mvn(gauss_F , gauss_G ) , zip(self.mean_G , self.cov_G) )) for gauss_F in  zip( self.mean_F ,self.cov_F)  ] ).argmin(axis =1 )
        PI_inv = dict(list(map(lambda x : (x,np.where(PI == x)[0]) , range(self.m))))
        self.PI = PI ; self.PI_inv = PI_inv
    
    def update_G(self) :
        """ parametres of G obtained by collapsing F according to π   """
        weight_G = np.empty((self.m,)) ; mean_G = np.empty((self.m,self.d)) ; cov_G = np.empty((self.m,self.d , self.d))
        PI_inv = self.PI_inv
        for i , index in PI_inv.items():
            weight_G[i]=np.sum(self.weight_F[index ] , axis=0)
            mean_G[i]=1/weight_G[i]*np.sum(self.mean_F[ index ]*self.weight_F[index ][:, np.newaxis],axis=0) 
            cov_G[i] = 1/weight_G[i]*np.sum(self.cov_F[index]*self.weight_F[index ][:, np.newaxis , np.newaxis] +list(map(lambda row : row[:, np.newaxis] *row , 
                            self.mean_F[index]-mean_G[i] ))*self.weight_F[index ][:, np.newaxis , np.newaxis] , axis = 0)
            self.mean_G   = mean_G ; self.cov_G    = cov_G ; self.weight_G = weight_G
            
    def dis_KL_GMM(self):
        """  distance between large mixture of Gaussians & smaller mixture of Gaussians  """
        return sum(list(map(lambda gauss :  gauss[-1]*self.kl_mvn(gauss[0:2], gauss[2:4]) ,zip(self.mean_F, self.cov_F ,self.mean_G[self.PI] ,self.cov_G[self.PI] 
                                                                                               , self.weight_F  ))))
    
    def predict(self , X_test):
        """ Labels of each point """
        likelihood = np.zeros( (X_test.shape[0] , self.m) )
        for i in range(self.m):
            likelihood[:,i] = multivariate_normal(self.mean_G[i],self.cov_G[i] , allow_singular= True ).pdf(X_test)
        numerator        = likelihood * self.weights_G
        denominator      = numerator.sum(axis=1)[:, np.newaxis]
        return np.argmax(numerator / denominator , axis=1) 
