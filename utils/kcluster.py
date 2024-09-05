############################################################
#
#  Library for k-cluster filtration
# 
############################################################

# Reqirements numpy, scipy, and typing (so it works for python before 3.9)

import numpy as np
from scipy.sparse import csr_matrix 
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
import statsmodels.stats.multitest as mt
import scipy.stats as st

# for readability
from typing import Dict, List, Tuple, Union
import numpy.typing as npt

Edge = Tuple[int,int,float]



############################################################
# helper function
# - this should never be called directly
# - follows the 
############################################################
def source(C:Dict[int,int],v:int):
    while (v!=C[v]):
        v=C[v]
    return v

############################################################
# Compute persistence diagram as efficiently as possible
# from MST
############################################################
def computeDiagramFromMST(edges:List[Edge],verts:List[int],k:int)->np.array:
    """
     This is one of two internal functions. This takes an MST (list of (int,int,float))
     and returns the persistence diagram only. 
     
     - input edges: list of tuples (int,int,float) which gives the MST
             verts: a list with vertex ids (should match edge ids)
             k: parameter for filtration

     - returns:  PD: numpy array with 1st column birth second column death

      Notes: 1. this does not return points
                on the diagonal,
             2. this does not return anything else - if you want to cluster call
                computeFiltrationfromMST
    """
     # TODO: Add checks and errorchecking
    # Initialization 
    conn = dict(zip(verts,verts))          # simplified union find 
                                           # connected components look up table

    weights = dict.fromkeys(verts,1)       # weight function
    F = dict.fromkeys(verts,0)             # vertex filtration function
    PD = []                                # initialize empty persistence diagram

    edges.sort(key = lambda x: x[2])       # make sure edges are sorted
 
    #------------------------------------------
    # Main loop - iterate over edges in order
    # - this is massively simplified since 
    #   all edges are negative 
    #------------------------------------------
    for x,y,f in edges:  # edges is (x,y) with edge weight f
       
        #  find representative vertices
        x_source = source(conn,x)
        y_source = source(conn,y)
        
        # apply lexicographical ordering (to prevent further tests)
        # NOTE: since these are representative vertices, we cannot 
        #       control the ordering a priori
        s = x_source if x_source<y_source else y_source
        t = y_source if x_source<y_source else x_source

        total_weight = weights[s]+weights[t]     # total weight of new component

        #---------------------------------------
        # Cases: 
        #---------------------------------------
        
        # neither component active
        if (weights[s]<k) and (weights[t]<k):
            # update weights to new root
            weights[s] = total_weight
            weights[t] = 0                               # for debugging purposes

            conn[t] = s                                  # update union find root
            
            # if it has become active then record 
            # function of root vertex 
            if (weights[s]>=k):
                 F[s] = f                                # update only root 

        # s component active, tnot
        elif (weights[s]>=k) and (weights[t]<k):
            weights[s] += weights[t]
            weights[t] = 0                               # for debugging purposes
            
            conn[t] = s                                  # update union find root
 
        # t component active, s not (this is verbose but simple)
        elif (weights[s]<k) and (weights[t]>=k):
            weights[t] += weights[s]
            weights[s] = 0                                # for debugging purposes (can check final weight)
            
            conn[s] = t                                   # update union find root


        # both components active
        elif (weights[s]>=k) and (weights[t]>=k):
            # F is defined here for both
            # redefine vertices wrt function value F
            w = s if F[s]<=F[t] else t
            v = t if F[s]<=F[t] else s
            
            weights[w] += weights[v]
            weights[v] = 0               # for debugging purposes
            
            conn[v] = w                  # update union find root 
            
            PD.append((F[v],f))          # add point to persistence diagram
        else:
            raise Exception("No case matches - bug")
    #------------------------------------------

    inf_bars = set([source(conn,i) for i in verts])   # find infinite components
    PD+=[(F[j],np.inf) for j in inf_bars]
    return np.array(PD)

############################################################
# Compute persistence diagram and filtration from MST
############################################################
def computeFiltrationFromMST(edges:List[Edge],verts:List[int],k:int)->Tuple[List[Edge],Dict[int,float],np.array]:
    """
     This is one of two internal functions. This takes an MST (list of (int,int,float))
     and returns the persistence diagram and filtation. 

     - input edges: list of tuples (int,int,float) which gives the MST
             verts: a list with vertex ids (should match edge ids)
             k: parameter for filtration

     - returns:  E: MST with updated weights, 
                 F: Dictionary for vertex function values
                 PD: numpy array with 1st column birth second column death

      Notes: 1. this does not return points
                on the diagonal
    """
     # TODO: Add checks and errorchecking
    # Initialization 
    conn = dict(zip(verts,verts))               # simplified union find 
                                                # connected components look up table

    weights = dict.fromkeys(verts,1)            # weight function
    F = dict.fromkeys(verts,0)                  # vertex filtration function
    PD = []                                     # initialize empty persistence diagram

    edges.sort(key = lambda x: x[2])            # make sure edges are sorted
   
    non_active = {v:set([v,]) for v in verts}   # to keep track of non-active 
                                                # components to update filtration

    #------------------------------------------
    # Main loop - iterate over edges in order
    # - this is massively simplified since 
    #   all edges are negative 
    #------------------------------------------
    for x,y,f in edges:  # edges is (x,y) with edge weight f
       
        #  find representative vertices
        x_source = source(conn,x)
        y_source = source(conn,y)
        
        # apply lexicographical ordering (to prevent further tests)
        # NOTE: since these are representative vertices, we cannot 
        #       control the ordering a priori
        s = x_source if x_source<y_source else y_source
        t = y_source if x_source<y_source else x_source

        total_weight = weights[s]+weights[t]     # total weight of new component

        #---------------------------------------
        # Cases: 
        #---------------------------------------
        
        # neither component active
        if (weights[s]<k) and (weights[t]<k):
            # update weights to new root
            weights[s] = total_weight
            weights[t] = 0                               # for debugging purposes

            conn[t] = s                                  # update union find root
            
            non_active[s].update(non_active[t])          # the pop empties the list and copies
                                                         # it into the root list for the component

            # if it has become active then record 
            # function of root vertex 
            if (weights[s]>=k):
                for v in non_active[s]:
                    F[v] = f                             # update weights
               
        # s component active, tnot
        elif (weights[s]>=k) and (weights[t]<k):
            weights[s] += weights[t]
            weights[t] = 0                               # for debugging purposes
            
            conn[t] = s                                  # update union find root
            
            for v in non_active[t]:
                F[v] = f                                 # update weights
              
        # t component active, s not (this is verbose but simple)
        elif (weights[s]<k) and (weights[t]>=k):
            weights[t] += weights[s]
            weights[s] = 0                               # for debugging purposes
            
            conn[s] = t                                  # update union find root
            for v in non_active[s]:
                F[v] = f                                 # update weights

        # both components active
        elif (weights[s]>=k) and (weights[t]>=k):
            # F is defined here for both
            # redefine vertices wrt function value F
            w = s if F[s]<=F[t] else t
            v = t if F[s]<=F[t] else s
            
            weights[w] += weights[v]
            weights[v] = 0               # for debugging purposes
            
            conn[v] = w                  # update union find root 
            
            PD.append((F[v],f))          # add point to persistence diagram
        else:
            raise Exception("No case matches - bug")
    #------------------------------------------

    inf_bars = set([source(conn,i) for i in verts])   # find infinite components
    PD+=[(F[j],np.inf) for j in inf_bars]

   
    E = [(e[0],e[1], max([e[2],F[e[0]],F[e[1]]])) for e in edges]
    E.sort(key = lambda x: x[2])
    return E,F,np.array(PD)
    
############################################################
# Wrapper function which first computes MST depending
# on the input and returns the persistence diagram
############################################################
def computeDiagram(X: Union[npt.ArrayLike, List[Edge]], k: int, distance_matrix=True) -> np.array:
    """
     This takes either a weighted graph or a distance matrix (Check if it
     can be sparse?) and returns the persistence diagram for the k filtration
     - input X: either a distance matrix or list of tuples (int,int,float) 
                representing the weighted graph
                We assume the graph is connected.
             k: parameter for filtration
             distance_matrix: bool for whether the input is a distance matrix
     - returns: PD: numpy array with 1st column birth second column death

      Notes: 1. this does not return points
                on the diagonal
             2. only returns the diagram 
    """
    if distance_matrix:
        mst = minimum_spanning_tree(X)    # compute MST
     
        # create ordered list of edges
        rows,cols = mst.nonzero()
        edges = [(i,j,mst[i,j]) for i,j in zip(rows,cols)]
        edges.sort(key = lambda x: x[2])
    
         # Initialization of vertices
        verts = list(range(X.shape[0]))        # make list of vertices    
        return computeDiagramFromMST(edges,verts,k)
    else:
        mst = computeMST(X)
        # Initialization of vertices (assume connected)
        verts = list(set([e[0] for e in mst]+ [e[1] for e in mst]))     # make list of vertices
        return computeDiagramFromMST(mst,verts,k)



############################################################
# Wrapper function which first computes MST depending
# on the input and returns the persistence diagram
############################################################
def computeFiltration(X: Union[npt.ArrayLike, List[Edge]], k: int, distance_matrix=True) -> Tuple[List[Edge], Dict[int, float], np.array]:
    """
     This takes either a weighted graph or a distance matrix (Check if it
     can be sparse?) and returns the filtration and persistence diagram 
     for the k filtration
     - input 
             X: either a distance matrix or list of tuples (int,int,float) 
                representing the weighted graph
                We assume the graph is connected.
             k: parameter for filtration
             distance_matrix: bool for whether the input is a distance matrix
     - returns:
             MST: list of edges with weights
             F: dictionary of vertex filtration values. 
             PD: numpy array with 1st column birth second column death

      Notes: 1. this does not return points
                on the diagonal
    """
    if distance_matrix:
        mst = minimum_spanning_tree(X)    # compute MST

        # create ordered list of edges
        rows,cols = mst.nonzero()
        edges = [(i,j,mst[i,j]) for i,j in zip(rows,cols)]
        edges.sort(key = lambda x: x[2])
    
         # Initialization of vertices
        verts = list(range(X.shape[0]))        # make list of vertices    
        return computeFiltrationFromMST(edges,verts,k)
    else:
        mst = computeMST(X)
        # Initialization of vertices (assume connected)
        verts = list(set([e[0] for e in mst]+ [e[1] for e in mst]))     # make list of vertices
        return computeFiltrationFromMST(mst,verts,k)


############################################################
# Compute minimum spanning tree from graph         
############################################################
def computeMST(E:List[Edge])->List[Edge]:
    """
    Simple implementation for computing the MST from a graph
    - input
        E: list of tuples representing edges (int,int,float)
    - output
        MST: list of tuples representing edges (int,int,float) 
             in MST
    Ignores isolated vertices
    """
    E.sort(key = lambda x: x[2])                                # the edges must be sorted
    
    # Initialization 
    verts = list(set([e[0] for e in E]+ [e[1] for e in E]))     # make list of vertices
    conn = dict(zip(verts,verts))                               # simplified union find 
                                                                # connected components look up table
    MST = []                                                    # initialize MST
    
    for x,y,f in E:
        #  find representative vertices
        x_source = source(conn,x)
        y_source = source(conn,y)
        
        # if they are the same then move on, otherwise
        if(x_source!=y_source):
             # apply lexicographical ordering (to prevent further tests)
             # NOTE: since these are representative vertices, we cannot 
             #       control the ordering a priori

            s = x_source if x_source<y_source else y_source     # We know they are difffernet here
            t = y_source if x_source<y_source else x_source 

            conn[t] = s # perform merge
            MST.append((x,y,f))
                        
    return MST

############################################################
# Get threshold for a desired number of clusters from 
# persistence diagram
############################################################
def getThreshold(PD:np.array, m:int,multiplicative:bool=True)->float:
    """
    Return the threshold required from a persistence diagram to get
    m clusters - either multiplicatively (default) or additively
    - input
        PD: persistence diagram [nx2] numpy array (for some n)
        m: number of clusters
        multiplicative: bool, if true used death/birth
    - output
        threshold: float value to set (should be positive)
    """
    # test that no vertices have value 0
    if multiplicative and (np.count_nonzero(PD[:,0])<PD.shape[0]):
        raise Exception("Cannot use multiplicative if vertex can take value 0")
    
    # compute persistence values

    if multiplicative:     
        pers_values = PD[:,1]/PD[:,0]
    else:
        pers_values = PD[:,1] - PD[:,0]

    pers_values = np.sort(pers_values)[::-1]        # sort in decreasing order

    return (pers_values[m]+pers_values[m-1])/2


############################################################
# Custom comparison function - should never 
# be called directly
############################################################
def comp(a: Tuple[int,float],b: Tuple[int,float])->bool:
    if a[1]!=b[1]:
        return a[1]<b[1]
    else:
        return a[0]<b[0]
    

############################################################
# Compute clusters - assumes persistence diagram computed
# using persistence-based clustering
############################################################
def computeClusters(F:Dict[int,float], E:List[Edge], alpha:float, multiplicative:bool=True)->Dict[int,int]:
    """
    Return clusters for a given threshold using the  persistence-based clustering algorithm
    
    input:
       - F: filtration values on vertices in the form of a dictionary[int,float]
       = E: MST with filtration values on edges as a list of tuples (int,int,float) 
       - alpha: threshold
       - multiplicative: bool, true if death/birth used, false means death-birth used
    output:
       - clusters:  a lookup for each vertexreturning the cluster representative (minimum)
                    dictionary [int,int]
    """
    
    verts = list(F.keys())                   # make list of vertices
    conn = dict(zip(verts,verts))            # simplified union find 

    E.sort(key = lambda x: x[2])             # re-sort edges just in case

    # test that no vertices have value 0
    vals = np.array([F[i] for i in F.keys()])
    if multiplicative and (np.count_nonzero(vals)<len(verts)):
        raise Exception("Cannot use multiplicative if vertex can take value 0")
 
    # main loop
    for x,y,f in E:
         #  find representative vertices
        x_source = source(conn,x)
        y_source = source(conn,y)
        
        # apply function and in case of equality lexicographical ordering
        # (to prevent further tests)
        # NOTE: since these are representative vertices, we cannot 
        #       control the ordering a priori
        s = x_source if comp((x_source,F[x_source]),(y_source,F[y_source])) else y_source
        t = y_source if comp((x_source,F[x_source]),(y_source,F[y_source])) else x_source

        # all edges are negative so only need to check threshold
        # whether to merge
        if multiplicative:
            if f/F[t]<alpha:
                conn[t]=s
        else:
            if f-F[t]<alpha:
                conn[t]=s

    # do one more compression just in case
    for v in conn.keys():
        conn[v] = source(conn,v)
    

    return conn


############################################################
# Given persistence diagram return number of significant 
# clusters and threshold for getting  
# Note: This is always multiplicative weights
############################################################
def numSignificantClusters(PD:np.array, pvalue:float)->Tuple[int,float]:
    """
     This figures out how many significant clusters and the appropriate
     threshold based on universality 
     - input 
            PD: persistence diagram - an (n x 2) numpy array 
            pvalue: measure of significance
             
     - returns: 
            num_clusters: number of clusters
            threshold: for getting appropriate number of clusters                
    """ 
    # compute ell values
    ell = np.log(np.log( PD[:,1]/PD[:,0]))
    ell = np.sort(ell)[::-1]
    ell = ell[1:]    

    avg_ell = np.mean(ell)
    ell = (ell-avg_ell) 

    # statistically test
    pvals = 1-st.gumbel_l().cdf(ell)
    pvals.sort()

    res, pv, _, _  = mt.multipletests(pvals, alpha=pvalue, method='bonferroni')
    num_clusters = 1+np.sum(res == True)
    threshold = getThreshold(PD,num_clusters)

    return num_clusters,threshold



############################################################
# Compute clusters - returning only significant 
# clusters according to the p-value
############################################################
def kClustering(X: Union[npt.ArrayLike, List[Edge]], k: int, pvalue: float = 0.05, distance_matrix=True) -> Dict[int, int]:
    """
     This takes either a weighted graph or a distance matrix (Check if it
     can be sparse?) and returns statistically significant clusters 
     based on universality 
     - input X: either a distance matrix or list of tuples (int,int,float) 
                representing the weighted graph
                We assume the graph is connected.
             k: parameter for filtration
             pvalue: measure of significance
             distance_matrix: bool for whether the input is a distance matrix
     - returns: clusters: a dictionary for each vertex to the cluster head
                
      Notes: to obtain number of clusters you can run len(clusters.keys())
    """
    mst,F,PD = computeFiltration(X,k,distance_matrix)
    num,alpha = numSignificantClusters(PD,pvalue)
    return computeClusters(F,mst,alpha)