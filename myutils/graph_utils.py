import numpy as np
from .utils import preprocess_adj_tensor_with_identity
import logging
logger = logging.getLogger(__name__)

def simple_distance_filters(A,D):
   """Calculate distance matrices up to fifth neighbor"""
   logger.debug("Calculating distance matrices up to fifth neighbor...")
   A=preprocess_adj_tensor_with_identity(A, SYM_NORM)
   A=A.astype(np.bool)
   second = np.where(D==2,True,False)
   third = np.where(D==3,True,False)
   fourth = np.where(D==4,True,False)
   fifth = np.where(D==5,True,False)
   graph_conv_filters = np.concatenate((A,second,third,fourth,fifth),axis=1)
   graph_conv_filters = graph_conv_filters.astype(np.float64)
   return graph_conv_filters


def first_order(A, D, normalize=True):
   """Use first-order approximation"""
   if normalize:
      logger.debug('Normalizing adjacency matrix')
      deg = A.sum(axis=1)
      deg = np.where(deg>0,deg**(-0.5),1)
      deg = np.array([np.diag(x) for x in deg])
      A=np.matmul(np.matmul(deg,A),deg)
   
   I = np.array([np.identity(x.shape[0]) for x in A])
   graph_conv_filters = np.concatenate((I,A),axis=1)

   return graph_conv_filters


def human_readable_size(size_in_bytes):
   power = int(np.log2(size_in_bytes)/10)
   divisor = 1024**power
   unit = [' B', 'KB', 'MB', 'GB']
   return '%8.1f %s'%(size_in_bytes/divisor, unit[power])


def multi_cheb(A, D, k=2):
    """Calculate Chebyshev polynomials up to order k. Works on 3D arrays."""
    logger.debug("Calculating Chebyshev polynomials up to order {}...".format(k))
    
    logger.debug('Stage 1: calculation of normalized laplacian matrices')
    del D
    L = [np.diag(x.sum(axis=1)) - x for x in A]
    del A
    L = np.array([x/np.linalg.eigvalsh(x).max() for x in L])
    size_in_mem = L.itemsize*L.size
    logger.debug('Laplacians size in memory: %s', human_readable_size(size_in_mem))
    
    T_k = list()
    T_k.append(np.array([np.identity(L.shape[1]) for _ in range(L.shape[0])]))
    T_k.append(L)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        return 2 * np.matmul(X,T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], L))
    del L
    T_k = np.concatenate(T_k, axis=1)
    size_in_mem = T_k.itemsize*T_k.size
    logger.debug('T_k size in memory: %s', human_readable_size(size_in_mem))
    return T_k

FILTERS={'cheb':multi_cheb, 'first_order':first_order, 'simple_distance':simple_distance_filters}

