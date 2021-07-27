# library mathematic & statistic 
import numpy as np
from numpy import linalg as LA
import math
import random
import itertools as it
from sklearn.datasets import make_sparse_spd_matrix as mat
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
from ipywidgets import FloatSlider
from bokeh.core.properties import Instance, String
from bokeh.io import show
from bokeh.models import ColumnDataSource, LayoutDOM
from bokeh.util.compiler import TypeScript

# library for machine learning
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

output_notebook(hide_banner=True)


'''______________________________<<<class : generator data under gaussian mixture model >>>_____________________'''

class sampling_gaussian_mixture_model():
    
    def __init__(self, n_components  ,dim ):
        ''' params :
                     n_components : nbre de commponents of the models to generate 
                     dim la dim of data '''
        self.n_components = n_components    
        self.dim          = dim 
        
    def fit( self, samples : int , option_weights = 'equal_weights' , edge = 5 , interval = 100, pas = 100 , float_noise = 10.):
        
        ''' fit fcn : input : 
                             samples        : numbers of data to generate 
                             option_weights : tow option all components have the same probabilités a posteriori'
                                              or have random probabilités a posteriori 
                             edge           : the number of classes that to separate all classes of the models 
                      output:        
                               X            : sampling (data)
                               labels       : labels of sampling (data)
                                              
        '''
        self.samples                = samples
        self.option_weights         = option_weights
        self.interval               = interval              # [-interval, interval ] interval to discretize 
        self.pas                    = pas                   # number of discretization
        self.edge                   = edge                  # global clusters 
        self.float_noise            = float_noise           # interval of noise to get random means around global clusters
        self.X                      = None
        self.labels                 = None
        #_________generte weights___________________
        
        if self.option_weights == 'equal_weights':
            # Default equally likely clusters
            self.weights  = [1./self.n_components]*self.n_components
        
        if self.option_weights   == 'random_weights' :
            # randomly likely clusters
            self.weights  = list(np.random.dirichlet(np.ones(self.n_components),size=1).reshape(self.n_components))
            
        # _________generte means and cov ____________
        self.get_means()                                                   # to get mean params 
        self.cov     = [ mat(self.dim) for n in range(self.n_components)]  # to get cov params
        labels=[]                                                          # to get labels params
        # Check if the mixing coefficients sum to one:
        EPS = 1E-15
        
        if np.abs(1-np.sum(self.weights)) > EPS:
            raise ValueError( "The sum of mc must be 1.0")
            
        # Cluster selection
        cs_mc = np.cumsum(self.weights)
        cs_mc = np.concatenate(([0], cs_mc))
        sel_idx = np.random.rand(self.samples)
        # Draw samples
        X = np.zeros((self.samples, self.dim))
        labels= np.zeros((self.samples, ))
        
        for k in range(self.n_components):
            idx = (sel_idx >= cs_mc[k]) * (sel_idx < cs_mc[k+1])
            ksamples = np.sum(idx)
            drawn_samples = np.random.multivariate_normal(self.means[k], self.cov[k], ksamples)
            X[idx,:] = drawn_samples
            labels[idx]= [k] * ksamples
        self.X = X
        self.labels = labels.astype(int)
        return self
    
    def get_means( self) :
        ''' 
        discretize an interval to get means
        input : 
        [-a,a] : intervale
        n     : number of discretization 
                   ch.   
        '''
        a   = self.interval
        n   = self.pas
        h   = (2*a)/n
        ch  = self.edge
        n_c = self.n_components
        l   = self.float_noise 
        X_axis = []
        for i in range(n):
            X_axis.append(i*h)
                
        if self.dim  == 2 :
            args = np.around(np.asarray(list(it.product(X_axis, X_axis))),self.dim)
                
        elif self.dim == 3 :
            args = np.around(np.asarray(list(it.product(X_axis, X_axis, X_axis))),self.dim)
            
        else :
            raise ValueError("dim is not in 2D or 3D ") 
        args.shape[0]
        index= np.random.choice(range(args.shape[0]), ch)
        args = args[index , :]
        tas  = np.random.choice(ch,n_c )
        args = args[tas ,:]
        if self.dim  == 2 :
            d = args.shape[0]*args.shape[1]
                
        elif  self.dim  == 3 :
            d = args.shape[0]*args.shape[1]
                
        args = args + np.random.uniform(-l, l, d).reshape(args.shape)
        self.means = args
            
    def visualization_samling_data_GMM(self ,plot_spherical_cluster = None ) :
        
        def gauss_ellipse_2d( centroid, ccov, sdwidth=1, points=100):
            
            """Returns x,y vectors corresponding to ellipsoid at standard deviation sdwidth """
            mean = np.c_[centroid]
            tt = np.c_[np.linspace(0, 2*np.pi, points)]
            x = np.cos(tt); y=np.sin(tt);
            ap = np.concatenate((x,y), axis=1).T
            d, v = np.linalg.eig(ccov);
            d = np.diag(d)
            d = sdwidth * np.sqrt(d);                                     
            bp = np.dot(v, np.dot(d, ap)) + np.tile(mean, (1, ap.shape[1])) 
            return bp[0,:], bp[1,:] 

        colors     = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        colors_name =list(colors.keys())
        colors_name.remove('w')
        n_colors   = len(colors_name)
        plt.figure() #figsize=(15, 6))
        
        if self.n_components <= n_colors :
            if self.X.shape[1] == 2 :
                plt.scatter(self.X[:,0], self.X[:,1],c= self.labels ) #np.array(colors_name)[self.labels] )
                if plot_spherical_cluster == ' spherical cluster ' :
                    for i in range(self.n_components):
                        x1, x2 = gauss_ellipse_2d(self.means[i], self.cov[i])
                        plt.plot(x1, x2, 'k', linewidth=2)
                    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
                    plt.show() 
            elif self.X.shape[1] == 3 :
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(self.X[:,0], self.X[:,1], self.X[:,2] , c= self.labels ) # np.array(colors_name)[self.labels] )
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()
            else :
                raise ValueError("dim is not in 2D or 3D ")  
        else :
            if self.X.shape[1] == 3 : 
                plt.scatter(self.X[:,0], self.X[:,1], self.X[:2] ,  c = self.labels )
                plt.show()
            elif self.X.shape[1] == 2 : 
                plt.scatter(self.X[:,0], self.X[:,1] ,  c = self.labels )
                plt.show()
            else :
                raise ValueError("dim is not in 2D or 3D " )
       
    def plot_cluster_animation_parametre(self , edge ,interval , pas, noise ):
        
        def fcn_animation( method , edge ,interval , pas, noise) :
            
            edge_ ,lst_ , pas_, noise_ =  edge , interval , pas, noise
            N              = self.n_components 
            d              = self.dim
            n_samples      = self.samples
            object_HCMM    = self

            if method   == 'random_weights' :
                object_HCMM    = sampling_gaussian_mixture_model( N,d).fit(n_samples
                                                                       , option_weights = method
                                                                       ,edge       = edge_ ,
                                                                       interval    = lst_,
                                                                       pas         = pas_ ,
                                                                       float_noise = noise_)
            elif method   == 'equal_weights'  :
                object_HCMM    = sampling_gaussian_mixture_model( N,d).fit(n_samples
                                                                       , option_weights= method 
                                                                       ,edge       = edge_ ,
                                                                       interval    = lst_,
                                                                       pas         = pas_ ,
                                                                       float_noise = noise_)
            X              = object_HCMM.X
            labels         = object_HCMM.labels 
            return X , labels
        
        method = self.option_weights
        X ,labels          = fcn_animation(method , edge ,interval , pas, noise)
        if self.dim == 2:
            tini= X[:,0].min() -1000
            tend=X[:,1].max() +1000
            p = figure(x_range=(tini, tend), plot_height=300, plot_width=900, title="edge_evaluation")
            plt_p = p.x(X[:,0], X[:,1], line_width=2 , legend_label="data")
            show(column(p), notebook_handle=True)
            
        else :
            print("ValueError : dim is not in 2D ")
            pass
        lst_min = []
        lst_max = []
        
        method = ['random_weights','eqaul_weights'][1]
        def update(method , edge, pas , interval , noise):
            
            X ,labels = fcn_animation(method , edge ,interval , pas, noise) 
            tini=X[:,0].min() 
            tend=X[:,1].max()
            lst_min.append(tini)
            lst_max.append(tend)
            if self.dim == 2:
                plt_p.data_source.data = dict(x=X[:,0], y=X[:,1] )
                push_notebook()
        interact(update ,
                edge            = IntSlider(min=1, max=self.n_components,value = edge  , step=1, continuous_update=False ),
                interval        = FloatSlider(min=-100., max=100.,value = interval  , step=10, continuous_update=False ),
                pas             = IntSlider(min=1, max=100,value = pas  , step=5, continuous_update=False ),
                noise           = FloatSlider(min=-100., max=100.,value = noise  , step=10, continuous_update=False ),
                method  = Dropdown(options=['random_weights','eqaul_weights'], value = method, description='Method:'))