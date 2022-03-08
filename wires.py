# -*- coding: utf-8 -*-

'''
The module "wires" used to create the network structure was 
imported and adapted from the model reported in the work by Loeffler, Alon, et al.,
"Topological properties of neuromorphic nanowire networks." 
Frontiers in Neuroscience 14 (2020): 184.
https://github.com/aloe8475/CODE/blob/master/Analysis/Generate%20Networks/wires.py
'''

from itertools import *
from scipy.spatial.distance import cdist

import numpy as np
import networkx as nx

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

from matplotlib.lines import Line2D




###############################################################################

def generate_wires_distribution(number_of_wires=1500, 
                                wire_av_length=14.0, 
                                wire_dispersion=5.0,
                                Lx=3e3, Ly=3e3, 
                                this_seed=42):
    '''
    Drops nanowires on the device of sides Lx, Ly. 
    
    Parameters
    ----------
    number_of_wires : int 
        Total number of wires to be sampled
    wire_av_length : float 
        Average wire length in mum
    wire_dispersion : float 
        Dispersion/scale of length distribution in mum
    wire_length : float 
        Length of the nanowire in mum (default = 14)
    centroid_dispersion : float 
        Scale parameter for the general normal distribution from 
        which centroids of wires are drawn in mum
    gennorm_shape : float 
        Shape parameter of the general normal distribution from 
        which centroids of wires are drawn. As this number increases, 
        the distribution approximates a uniform distribution.
    Lx : float 
        Horizontal legth of the device in mum
    Ly : float 
        Vertical length of the device in mum
    seed : int
        Seed of the random number generator to always generate the same distribution
    
    Returns
    -------
    dict
        A dictionary with the centre coordinates, the end point coordinates, and
        orientations. The `outside` key in the dictionary is 1 when
        the wire intersects an edge of the device and is 0 otherwise.
    '''

    np.random.seed(this_seed)
    
    # wire lengths
    wire_lengths     = generate_dist_lengths(number_of_wires, wire_av_length, wire_dispersion)
    xc, yc = np.random.rand(number_of_wires) * Lx, np.random.rand(number_of_wires) * Ly
    theta  = generate_dist_orientations(number_of_wires)
    
    xa, ya  = xc - wire_lengths/2.0 * np.cos(theta), yc - wire_lengths/2.0 * np.sin(theta) # coordinates for one end 
    xb, yb  = xc + wire_lengths/2.0 * np.cos(theta), yc + wire_lengths/2.0 * np.sin(theta) # coordinates for the other end

    # Compute the Euclidean distance between pair of region centres
    wire_distances = cdist(np.array([xc, yc]).T, np.array([xc, yc]).T, metric='euclidean')    

    # Find values outside the domain
    a = np.where(np.vstack([xa, xb, ya, yb]) < 0.0, True, False).sum(axis=0)
    b = np.where(np.vstack([xa, xb]) > Lx, True, False).sum(axis=0)
    c = np.where(np.vstack([ya, yb]) > Ly, True, False).sum(axis=0)
    
    outside = a + b + c

    return dict(xa=xa,ya=ya,
                xc=xc,yc=yc,
                xb=xb,yb=yb,
                theta=theta,
                avg_length = wire_av_length,
                wire_lengths = wire_lengths,
                this_seed = this_seed,
                outside = outside,
                length_x = Lx,
                length_y = Ly,
                number_of_wires = number_of_wires,
                wire_distances = wire_distances)


###############################################################################

def generate_dist_lengths(number_of_wires, wire_av_length, wire_dispersion):
    '''
    Generates the distribution of wire lengths
    '''

    mu = wire_av_length
    sigma = wire_dispersion

    wire_lengths = np.random.normal(mu, sigma, int(number_of_wires))
    
    
    i=0

    for i in range (0,len(wire_lengths)):
        while wire_lengths[i] < 0:
            wire_lengths[i]= np.random.normal(mu, sigma, 1)
    

    return wire_lengths


###############################################################################

def generate_dist_orientations(number_of_wires):

     return np.random.rand(int(number_of_wires))*np.pi # uniform random angle in [0,pi)


###############################################################################
     
def find_segment_intersection(p0, p1, p2, p3):
    """
    Find *line segments* intersection using line equations and 
    some boundary conditions.
    First segment is defined between p0, p1 and 
    second segment is defined between p2, p3
          p2
          |  
    p0 ------- p1
          |
          p3
    Parameters
    ----------
    p0 : array
        x, y coordinates of first wire's start point 
    p1 : array
        x, y coordinates of first wire's end point
    p2 : array
        x, y coordinates of second wire's start point 
    p3 : array
        x, y coordinates of second wire's end point
    Returns
    -------
    xi, yi: float 
       x, y coordinates of the intersection
    TODO: + change input to a list instead of individual points; or,
          + make point a class with x, y coordinates so we avoid using 
          indexing (x: pX[0]; y:pX[1])
          + polish these docstring with standard input/ouput definitions
    """
    
    # Check that points are not the same
    if np.array_equal(p0, p1) or np.array_equal(p2, p3):
        return False 

     # Check that an overlapping interval exists
    if max(p0[0], p1[0]) < min(p2[0], p3[0]) or max(p2[0], p3[0]) < min(p0[0], p1[0]):
        return False 
    else:
        # xi, yi have to be included in
        interval_xi = [max(min(p0[0],p1[0]), min(p2[0],p3[0])), min(max(p0[0],p1[0]), max(p2[0],p3[0]))]
        interval_yi = [max(min(p0[1],p1[1]), min(p2[1],p3[1])), min(max(p0[1],p1[1]), max(p2[1],p3[1]))]

    # Find the intersection point between nanowires
    A1 = (p0[1]-p1[1])/(p0[0]-p1[0]) # will fail if division by zero
    A2 = (p2[1]-p3[1])/(p2[0]-p3[0]) 
    b1 = p0[1] - A1*p0[0] 
    b2 = p2[1] - A2*p2[0] 

    xi = (b2 - b1) / (A1 - A2)
    yi = A1 * xi + b1

    #The last thing to do is check that xi, yi are included in interval_i:
    if xi  < min(interval_xi) or xi > max(interval_xi):
        return False 
    elif yi < min(interval_yi) or yi > max(interval_yi):
        return False
    else:
        return xi, yi



###############################################################################
        
def detect_junctions(wires_dict):
    """
    Find all the pairwise intersections of the wires contained in wires_dict.
    Adds four keys to the dictionary: junction coordinates, edge list, and
    number of junctions.
    Parameters
    ----------
    wires_dict: dict
    Returns
    -------
    wires_dict: dict 
        with added keys
    """
    logging.info('Detecting junctions')
    xi, yi, edge_list = [], [], []
    for this_wire, that_wire in combinations(range(wires_dict['number_of_wires']), 2):

        xa, ya = wires_dict['xa'][this_wire], wires_dict['ya'][this_wire]
        xb, yb = wires_dict['xb'][this_wire], wires_dict['yb'][this_wire]

        p0 = np.array([xa, ya])
        p1 = np.array([xb, yb])

        xa, ya = wires_dict['xa'][that_wire], wires_dict['ya'][that_wire]
        xb, yb = wires_dict['xb'][that_wire], wires_dict['yb'][that_wire]

        p2 = np.array([xa, ya])
        p3 = np.array([xb, yb])

        # Find junctions
        J = find_segment_intersection(p0, p1, p2, p3)

        if J is not False:
            # Save coordinates
            xi.append(J[0])
            yi.append(J[1])
            # Save node indices for every edge
            edge_list.append([this_wire, that_wire])

    # Save centres coordinates and edge list to dict
    # if there are junctions
    if len(edge_list) is not 0:
        wires_dict['number_of_junctions'] = len(edge_list)
        wires_dict['xi'] = np.asarray(xi)
        wires_dict['yi'] = np.asarray(yi)
        wires_dict['edge_list'] = np.asarray(edge_list)
        logging.info('Finished detecting junctions')
        return wires_dict
    
    raise Exception('There are no junctions in this network')


###############################################################################
    
def generate_adj_matrix(wires_dict):
    """
    This function will produce adjaceny matrix of 
    the physical network
    Parameters
    ----------
    wires_dict: dict
        a dictionary with all the wires position and junctions/intersection 
        positions.
    Returns
    ------- 
    wires_dict: dict
        The same dictionary with added key:value pairs adjacency matrix 
    """

    # Create array -- maybe use sparse matrix?
    adj_matrix_shape = (wires_dict['number_of_wires'], wires_dict['number_of_wires'])
    adj_matrix = np.zeros(adj_matrix_shape, dtype=np.float32)
    adj_matrix[wires_dict['edge_list'].astype(np.int32)[:, 0], wires_dict['edge_list'].astype(np.int32)[:, 1]] = 1.0
    
    # Make the matrix symmetric
    adj_matrix = adj_matrix + adj_matrix.T

    wires_dict['adj_matrix'] = adj_matrix

    return wires_dict


###############################################################################
    
def generate_graph(wires_dict):
    """
    This function will produce a networkx graph.
    Parameters
    ----------
    wires_dict: dict
        a dictionary with all the wires position and junctions/intersection 
        positions.
    Returns
    ------- 
    wires_dict: dict
        The same dictionary with added key:value pairs networkx graph object.
    """


    # Create graph - this is going to be a memory pig for large matrices
    wires_dict = generate_adj_matrix(wires_dict)
    G = nx.from_numpy_matrix(np.matrix(wires_dict['adj_matrix']))

    wires_dict['G'] = G

    return wires_dict


###############################################################################

def draw_wires(ax, wires_dict):
    """
    Draw wires on a given set of axes.
    
    Wires outside the domain are light gray dashed lines. 
    Wires inside the domain are light gray solid lines. 
    The centre of the wires is marked with a red 'o' marker. 
    
    ax -- matplotlib axes to draw needle symbol
    wires_dict  -- dictionary output from generate_distribution
    """    
    # Make local variables
    xa, ya = wires_dict['xa'], wires_dict['ya']
    xb, yb = wires_dict['xb'], wires_dict['yb']
    xc, yc = wires_dict['xc'], wires_dict['yc']

    for this_wire in range(wires_dict['number_of_wires']):
        line = [Line2D([xa[this_wire],xb[this_wire]],[ya[this_wire],yb[this_wire]], color=(0.42, 0.42, 0.42)),
                Line2D([xc[this_wire]],[yc[this_wire]], color='r', marker='o', ms=2, alpha=0.77)] 
        
        '''
        if wires_dict['outside'][this_wire]:
            line = [Line2D([xa[this_wire],xb[this_wire]],[ya[this_wire],yb[this_wire]], color='k', ls='--', alpha=0.2)] 
                   #Line2D([xc[this_wire]],[yc[this_wire]],  color='k', marker='o', ms=4, alpha=0.1)] 
        else:   
            line = [Line2D([xa[this_wire],xb[this_wire]],[ya[this_wire],yb[this_wire]], color=(0.42, 0.42, 0.42)),
                    Line2D([xc[this_wire]],[yc[this_wire]], color='r', marker='o', ms=2, alpha=0.77)] 
        '''
        for l in line: 
            ax.add_line(l)

    return ax


###############################################################################

def draw_junctions(ax, wires_dict):
    """
    Draw the circles at the junctions
    """

    xi, yi = wires_dict['xi'], wires_dict['yi']

    for this_junction in range(wires_dict['number_of_junctions']):
        line = [Line2D([xi[this_junction]],[yi[this_junction]], color='b', marker='o', ms=1.5, alpha=0.77)]
        for l in line: 
            ax.add_line(l)
    return ax

