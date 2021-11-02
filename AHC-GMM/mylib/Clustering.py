# library mathematic & statistic 
import numpy as np
from scipy.stats import multivariate_normal
from scipy import linalg
from numpy.linalg import multi_dot
from sklearn.mixture.base import BaseMixture, _check_shape
from sklearn.utils import check_array, check_random_state
import warnings
# library for machine learning
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
np.seterr(divide='ignore', invalid='ignore')

def Kullback_Leibler_Distance(GMM1 , GMM2 ):
  Means_1 , Covars_1 = GMM1
  Means_2 , Covars_2 = GMM2
  Covars_1 =  numpy_cholesky(Covars_1)
  Covars_2 =  numpy_cholesky(Covars_2)
  Det = np.log(np.abs( det_cholesky(Covars_2)/det_cholesky(Covars_1)[:, np.newaxis]))
  inv_S = inv_cholesky(Covars_2 )
  n_clusters_2 , n_features = Means_2.shape ;  n_clusters_1 = Means_1.shape[0] ; dis = np.empty((n_clusters_1, n_clusters_2))
  for k in range(n_clusters_2):
    dif_mu  = Means_1-Means_2[k]
    dis[:,k] = 0.5*( Det[:, k]+ np.trace(np.dot( Covars_1 , inv_S[k] ) , axis1 = 1 , axis2 = 2 ) + (dif_mu.dot(inv_S[k])*dif_mu ).sum(axis=-1)  )
  return dis 

def numpy_cholesky(covariances):
  n_components, n_features, _ = covariances.shape
  chol = np.empty((covariances.shape))
  for k, covariance in enumerate(covariances):
    try:
      chol[k] =  linalg.cholesky(covariance, lower=True) 
    except linalg.LinAlgError:
      raise("_cholesky not work")
  return chol

def inv_cholesky(covariances):
  n_components, n_features, _ = covariances.shape
  inv_chol = np.empty((covariances.shape))
  for k, covariance in enumerate(covariances):
    try:
      inv_chol[k] =  linalg.solve_triangular( covariance , np.eye(n_features), lower=True ).T
    except linalg.LinAlgError:
      pass 
  return inv_chol

def det_cholesky( covariances ):
  n_components, n_features , _ = covariances.shape
  return np.multiply.reduce(   covariances.reshape(n_components, -1)[:, :: n_features + 1] , axis = 1    )


class Hierarchical_Mixture_Model :
    """
        Hierarchical Clustering of a Mixture Model.

        Parametres  : 
        =============
        n_composents      : The number of mixture components of a large mixture of Gaussians
        reduce_composents : The number of mixture components of a a smaller mixture of Gaussians
        X                 : data to training 
        n_iters           :  int, defaults to 100. The number of EM iterations to perform.
        tol.              :  float, defaults to 1e-3. The convergence threshold. EM iterations will stop when the
                             lower bound average gain is below this threshold.
        Weights_init      : array-like, shape (n_components, ), optional
                            The user-provided initial weights, defaults to None.
        Means_init        : array-like, shape (n_components, n_features), optional
                            The user-provided initial means, defaults to None,
        Covars_init       : array-like, optional.The user-provided initial Covars , defaults to None shape (n_components, n_features, n_features) 
        cov_reg           : float, defaults to 0. The strength of the penalty on covariance matrix.
        
        Attributes :
        ============
        Means          : array-like, shape (n_components, n_features), means of  data under Gaussians mixture model
        Covars         : array-like, (n_components, n_features, n_features) , covaraince of  data under Gaussians mixture model
        Weights        : array-like, shape (n_components,) , weights of  data under Gaussians mixture model

        Means_reduce   : array-like, shape (reduce_composents  , n_features), Estimate means of smaller mixture of Gaussians  model
        Covars_reduce  : array-like, (reduce_composents  , n_features, n_features) , Estimate covariances of smaller mixture of Gaussians  model
        Weights_reduce : array-like, shape (reduce_composents  ,) , Estimate weights of smaller mixture of Gaussians  model
        N              : Number of samples of training data
        d              : n_features of training data
        KL_distance.   : computing distances 
        converged.     : bool True when convergence was reached in fit(), False otherwise.
        """

    def __init__(self ,  n_composents :  int , reduce_composents : int   ):
        self.n_composents        = n_composents                             
        self.reduce_composents   = reduce_composents  
        if self.n_composents   < self.reduce_composents  : 
          raise ValueError('n_composents  %s components, should be bigger than reduce_composents %s ' 
                           %(self.n_components, self.reduce_composents ))    
        
                               
       
    def fit(self  , X , n_iters :  int = 10000 , tol : float = 0.000001, seed =  0   ,
            Means_init = None, Covars_init= None, Weights_init= None , reg_covar=1e-6):  
        self.X         = X                         
        self.n_iters   = n_iters
        self.tol       = tol
        self.seed      = seed
        self.Means     = Means_init
        self.Covars    = Covars_init
        self.Weights   = Weights_init
        self.reg_covar = reg_covar
        self.n_samples , self.n_features = self.X.shape

        if n_iters < 1: 
          raise ValueError('GMM estimation requires at least one run')
        if tol < 0.: 
          raise ValueError('Invalid value for tol try with value positive: %s' %tol)
        if self.reg_covar < 0.:
            raise ValueError("Invalid value for ' reg_covar': %.5f "
                             "regularization on covariance must be "
                             "non-negative"
                             % self.reg_covar)
            
        self.Means_reduce = np.empty((self.reduce_composents ,self.n_features ))
        self.Covars_reduce = np.empty((self.reduce_composents ,self.n_features ,
                                       self.n_features )) 
        self.Weights_reduce = np.empty((self.reduce_composents,)) 

        if None in  [self.Means, self.Covars, self.Weights]:
          gmm            = GaussianMixture(n_components= self.n_composents,
                                           random_state=self.seed ).fit(self.X)
          self.Means    = gmm.means_       
          self.Weights  = gmm.weights_      
          self.Covars     = gmm.covariances_
        else :
          self.Weights = check_array(self.Weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
          _check_shape(self.Weights, (self.n_components,), 'weights')
          # check range
          if (any(np.less(self.Weights, 0.)) or any(np.greater(self.Weights, 1.))):
            raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(self.Weights), np.max(self.Weights)))

          # check normalization
          if not np.allclose(np.abs(1. - np.sum(self.Weights)), 0.):
            raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(self.Weights))
          # check means
          self.Means = check_array(self.Means, dtype=[np.float64, np.float32], ensure_2d=False)
          _check_shape(self.Means, (self.n_components, self.n_features), 'means')
          # check Covars 
          self.Covars = check_array(self.Covars, dtype=[np.float64, np.float32],
                             ensure_2d=False)

          _check_shape(self.Covars, (self.n_components, self.n_features, self.n_features),'Covars ')
        
        # initialization
        kmeans = KMeans(n_clusters = self.reduce_composents  , random_state=self.seed).fit(self.Means )
        self.Means_reduce = kmeans.cluster_centers_
        self.Covars_reduce =  np.full((self.reduce_composents, self.n_features, self.n_features ), np.identity(self.n_features) )
        self.Weights_reduce =  np.asarray([1/self.reduce_composents]*self.reduce_composents )

        KL_distance =[np.inf]
        self.converged   = False
        for it in range(self.n_iters):
            self.update_mapping_func()
            KL_distance.append(self.convergence_distance)
            if abs(KL_distance[it+1]-KL_distance[it]) <= self.tol : 
                self.converged = True
                break
            self.update_reduce_func()

        self.KL_distance = KL_distance[1:]

        if not self.converged:
            warnings.warn('Initialization did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          , ConvergenceWarning)
        return self


    def update_mapping_func(self ):
        # the optimal mapping function between the components of F and G  
        dis_gaussinne = Kullback_Leibler_Distance( (self.Means  ,self.Covars) , (self.Means_reduce , self.Covars_reduce) )
        PI = dis_gaussinne.argmin(axis = 1)
        PI_inv = dict([(k, np.where(PI == k)[0] ) for k in range(self.reduce_composents) ])
        self.convergence_distance = (dis_gaussinne.min(1)*self.Weights[PI]).sum()
        self.PI_inv = PI_inv 

    
    def update_reduce_func(self) :
        """ parametres of G obtained by collapsing F according to π   """
        PI_inv = self.PI_inv
        for i , index in PI_inv.items():
            
            self.Weights_reduce[i] = np.sum(self.Weights[index ], axis=0)

            self.Means_reduce[i]=1/(self.Weights_reduce[i])*np.sum(self.Means[ index ]
                                                                   *self.Weights[index ][:, np.newaxis],axis=0) 
            
            self.Covars_reduce[i] = 1/(self.Weights_reduce[i])*np.sum(self.Covars[index]
                                    *self.Weights[index ][:, np.newaxis , np.newaxis] 
                                    +  np.einsum('ij , im ->ijm', self.Means[index]-self.Means_reduce[i] ,
                                                 self.Means[index]-self.Means_reduce[i] )
                                    *self.Weights[index ][:, np.newaxis , np.newaxis] , axis = 0)
            self.Covars_reduce[i].flat[:: self.n_features + 1] += self.reg_covar
            
    def predict(self , data ):
        """ Labels of each point """
        likelihood = np.zeros( (data.shape[0] , self.reduce_composents) )
        for i in range(self.reduce_composents):
            likelihood[:,i] = multivariate_normal(self.Means_reduce[i],self.Covars_reduce[i] , allow_singular= True ).pdf(data)
        likelihood = likelihood*self.Weights_reduce
        return  likelihood.argmax(1)
