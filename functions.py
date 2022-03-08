# -*- coding: utf-8 -*-


import math
import random
from networkx import grid_graph
import numpy as np

import networkx as nx

##################    -    GRID GRAPH DEFINITION     -    #####################

def define_grid_graph(xdim, ydim):

    Ggrid = grid_graph(dim=[xdim, ydim])
    G = nx.convert_node_labels_to_integers(Ggrid, first_label=0, ordering='default', label_attribute='pos')

    return G

#########  -    GRID GRAPH DEFINITION_2  (with random diagonals) -    #########

def define_grid_graph_2(xdim,ydim,groundnode_pos,sourcenode_pos):

    ##define a grid graph

    Ggrid = grid_graph(dim=[xdim, ydim])
    
    ##define random diagonals
    for x in range (xdim-1):
        for y in range(ydim-1):
            k = random.randint(0, 1)
            if k == 0:
                Ggrid.add_edge((x, y), (x+1, y+1))
            else:
                Ggrid.add_edge((x+1, y), (x, y+1))
            

    ##define a graph with integer nodes and positions of a grid graph
    G=nx.convert_node_labels_to_integers(Ggrid, first_label=0, ordering='default', label_attribute='pos')

    return G

########################  - GRAPH INITIALIZATION     -    #####################
    
def initialize_graph_attributes(G,Yin):
    #add the initial conductance
    for u,v in G.edges():
        G[u][v]['Y']=Yin            #assign initial ammittance to all edges                                                         #assign initial high resistance state in all junctions
        G[u][v]['R']=1/G[u][v]['Y']
        G[u][v]['deltaV']=0
        G[u][v]['g']=0
        
        
    ##initialize
    for n in G.nodes():
        G.nodes[n]['pad']=False
        G.nodes[n]['source_node']= False
        G.nodes[n]['ground_node']= False
        
    return G

#################  - MODIFIED VOLTAGE NODE ANALYSIS     -    ##################

def mod_voltage_node_analysis(G, Vin, sourcenode, groundnode):
    ## MODIFIED VOlTAGE NODE ANALYSIS

    # definition of matrices
    matZ = np.zeros(shape=(G.number_of_nodes(), 1))
    matG = np.zeros(shape=(G.number_of_nodes()-1, G.number_of_nodes()-1))
    matB = np.zeros(shape=(G.number_of_nodes()-1, 1))
    matD = np.zeros(shape=(1, 1))

    # filling Z matrix
    matZ[-1] = Vin

    # filling Y matrix as a combination of G B D in the form [(G B) ; (B' D)]

    # elements of G
    for k in range(0, G.number_of_nodes()):
        if k < groundnode:
            nodeneighbors = list(G.neighbors(k))  # list of neighbors nodes
            for m in range(0, len(nodeneighbors)):
                matG[k][k] = matG[k][k] + G[k][nodeneighbors[m]]['Y']  # divided by 1
                if nodeneighbors[m] is not groundnode and nodeneighbors[m]<groundnode:
                    matG[k][nodeneighbors[m]] = -G[k][nodeneighbors[m]]['Y']
                if nodeneighbors[m] is not groundnode and nodeneighbors[m]>groundnode:
                    matG[k][nodeneighbors[m]-1] = -G[k][nodeneighbors[m]]['Y']
        if k > groundnode:
            nodeneighbors = list(G.neighbors(k))  # list of neighbors nodes
            for m in range(0, len(nodeneighbors)):
                matG[k-1][k-1] = matG[k-1][k-1] + G[k][nodeneighbors[m]]['Y']  # divided by 1
                if nodeneighbors[m] is not groundnode and nodeneighbors[m]<groundnode:
                    matG[k - 1][nodeneighbors[m]] = -G[k][nodeneighbors[m]]['Y']
                if nodeneighbors[m] is not groundnode and nodeneighbors[m]>groundnode:
                    matG[k - 1][nodeneighbors[m]-1] = -G[k][nodeneighbors[m]]['Y']
    # matB
    if sourcenode < groundnode:
        matB[sourcenode] = 1
    elif sourcenode > groundnode:
        matB[sourcenode-1] = 1

    # matY
    submat1 = np.hstack((matG, matB))
    submat2 = np.hstack((np.transpose(matB), matD))
    matY = np.vstack((submat1, submat2))

    # solve X matrix from Yx = z
    invmatY = np.linalg.inv(matY)  # inverse of matY
    
    matX = np.matmul(invmatY, matZ)  # Ohm law

    # add voltage as a node attribute
    for n in G.nodes():
        if n == groundnode:
            G.nodes[n]['V'] = 0
        elif n < groundnode:
            G.nodes[n]['V'] = matX[n][0]
        elif n > groundnode:
            G.nodes[n]['V'] = matX[n - 1][0]
            
    ###DEFINE CURRENT DIRECTION

    # transform G to a direct graph H

    H = G.to_directed()  # transform G to a direct graph

    # add current as a node attribute

    for u, v in H.edges():
        H[u][v]['I'] = (H.nodes[u]['V'] - H.nodes[v]['V']) * H[u][v]['Y']
        H[u][v]['Irounded'] = np.round(H[u][v]['I'], 2)

    # set current direction
    for u in H.nodes():  # select current direction
        for v in H.nodes():
            if H.has_edge(u, v) and H.has_edge(v, u):
                if H[u][v]['I'] < 0:
                    H.remove_edge(u, v)
                else:
                    H.remove_edge(v, u)

    return H


#################  - CALCULATE  NETWORK RESISTANCE     -    ####################
    

def calculate_network_resistance(H, sourcenode):
    
    I_fromsource = 0
    for u,v in H.edges(sourcenode):
        a= H[u][v]['I']
        I_fromsource=I_fromsource+a

    
    Rnetwork=H.nodes[sourcenode]['V']/I_fromsource
    
    return Rnetwork


#######################  - CALCULATE V source     -    ########################
    
    
def calculate_Vsource(H, sourcenode):

    Vsource=H.nodes[sourcenode]['V']
    
    return Vsource


#######################  - CALCULATE I source     -    ########################

def calculate_Isource(H, sourcenode):
    
    I_from_source=0
    for u,v in H.edges(sourcenode):
        a= H[u][v]['I']
        I_from_source=I_from_source+a
    
    return I_from_source


#################  - UPDATE EDGE WEIGHT (Miranda's model)   -    ##############
    
def update_edge_weigths(G,delta_t,Y_min, Y_max,kp0,eta_p,kd0,eta_d):
    

    for u,v in G.edges():
        
        G[u][v]['deltaV']=abs(G.nodes[u]['V']-G.nodes[v]['V'])
    
        G[u][v]['kp']= kp0*math.exp(eta_p*G[u][v]['deltaV'])
        
        G[u][v]['kd']= kd0*math.exp(-eta_d*G[u][v]['deltaV'])
        
        G[u][v]['g']= (G[u][v]['kp']/(G[u][v]['kp']+G[u][v]['kd']))*(1-(1-(1+(G[u][v]['kd']/G[u][v]['kp'])*G[u][v]['g']))*math.exp(-(G[u][v]['kp']+G[u][v]['kd'])*delta_t))
    
        G[u][v]['Y']= Y_min*(1-G[u][v]['g'])+Y_max*G[u][v]['g']
        
        G[u][v]['R']=1/G[u][v]['Y']
    
    return G

###############################################################################

