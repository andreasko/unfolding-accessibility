'''
Created on Mar 25, 2014

@author: koher
'''
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from networkx.generators.random_graphs import fast_gnp_random_graph 
from networkx.convert import to_scipy_sparse_matrix

def makedata(time,size,prob):
    graphs = [to_scipy_sparse_matrix(fast_gnp_random_graph(size,prob),dtype=int) for ii in range(time)]
    return graphs

def group_edges_by_times(edges, maxtime, mintime=0):
    """ returns list of tupels: [(d,[(u,v),...]),...] """
    dct = defaultdict(list)
    for u, v, d in edges:
        dct[d].append((u, v))
    dct_s = dict.fromkeys(range(0, maxtime-mintime), [])
    for d in dct:
        dct_s[d - mintime] = dct[d]
    return dct_s.items()

def readdata(filename):
    """ Reading data from a textfile formatted as follows: 
        first_node | second_note | timestep.
        Returns a Dictionary (key=timestep value=adjacency-matrix)
    """

    # Auslesen der Daten als Array
    netarray = np.loadtxt(filename, dtype = int)
    
    # Entpacktes Auslesen, um schneller Knoten und Zeiten zu bekommen
    u, v, times = np.loadtxt(filename, dtype = int, unpack=True)
    
    # Anzahl der Knoten: groesster aller Indizes
    number_of_nodes = max(max(u)+1, max(v)+1)
    #print number_of_nodes
    
    # das dict selbst
    # edges in bessere Form bringen
    edges = group_edges_by_times(netarray, max(times), min(times))

    # das dict selbst
    point = {}
    for d, es in edges:
        us = [u for u,v in es]
        vs = [v for u,v in es]
        bs = [True for i in range(len(es))]
        
        m = sp.csr_matrix((bs,(us,vs)),
                shape=(number_of_nodes, number_of_nodes), dtype=np.int8)
        
        point[d] = m+m.transpose()

    return point


if __name__ == "__main__": # (Als Testumgebung)
    A = readdata('sexual_contacts.dat')
    print A[2232]

