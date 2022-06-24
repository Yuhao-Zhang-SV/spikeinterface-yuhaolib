# -*- coding: utf-8 -*-
"""
visualization of connectivity between neurons

@author: Xiang LI
"""

import xianglib.glmcc as glmcc
import matplotlib.pyplot as plt
import subprocess as proc
import sys
import numpy as np
import math
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import importlib
from google.colab import output

from estherlib.uniform_60_filtering import uniform_60_filt
from estherlib.returnDataInfo import *
from estherlib.datasetObjectModule import *
from estherlib.cachingModule import *
from estherlib.runRoutines import *

import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display
from ipyfilechooser import FileChooser
from tqdm import tqdm

output.enable_custom_widget_manager()

def __generate_spikeTrain(dataObj, threshold = 100):
    sorting = dataObj.sorting_MS4
    spike_train = []
    unit_list = sorting.get_unit_ids()
    for i in range(len(unit_list)):
        line = sorting.get_unit_spike_train(unit_id = unit_list[i]).tolist()
        if len(line) >= threshold:
            for i in range(len(line)):
                line[i]= line[i]/(dataObj.samplingFrequency/1000.0)
            spike_train.append(line)
    return spike_train

def __generate_crossCorrelogram(cell1, cell2, T):
    '''
    Modified from https://github.com/NII-Kobayashi/GLMCC
    make Cross correlogram.

    Parameters
    ----------
    cell1 : TYPE
        DESCRIPTION.
    cell2 : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    for i in range(len(cell1)):
        if cell1[i]>T*1000.0 or cell1[i]<0:
            del cell1[i]
    for i in range(len(cell2)):
        if cell2[i]>T*1000.0 or cell2[i]<0:
            del cell2[i]
    # make c_ij(spike time)
    w = int(glmcc.WIN)
    c = []
    min_index = -1
    max_index = -1

    for i in range(len(cell2)):
        min = cell2[i] - w
        max = cell2[i] + w

        min_j = glmcc.index_linear_search(cell1, min, min_index)
        min_index = min_j
        max_j = glmcc.index_linear_search(cell1, max, max_index)
        max_index = max_j

        c_i = []
        for j in range(max_j - min_j):
            if (cell1[min_j + j] - cell2[i]) < glmcc.WIN:
                c_i.append(cell1[min_j + j] - cell2[i])

        c.extend(c_i)

    
    # make histogram
    bin_width = glmcc.DELTA # bin width
    bin_num = int(2 * w / bin_width) # the number of bin
    
    hist_array = np.histogram(np.array(c), bins=bin_num, range=(-1*w, w))
    result = [0, 0, 0, 0]
    result[0] = c
    result[1] = hist_array[0].tolist()
    result[2] = len(cell1)
    result[3] = len(cell2)

    return result
    
def __generate_J(cell_list):
    DataNum = len(cell_list)
    J_f_list = []
    pbar = tqdm(total=DataNum*(DataNum-1)/2, desc='fit cross-correlogram')
    count = 0
    for i in range(DataNum):
        for j in range(i):
            T = 5400  # why use this constant.Authors refused to explain. The original paper used these, and authors didn't give a complete reasnos
            cc_list = __generate_crossCorrelogram(cell_list[i], cell_list[j], T)
            #set tau
            tau = [4, 4]
            beta = 4000  #why this constant is so large.
            
            #find maximum log_likelihood and par, delay_synapse
            delay_synapse = 1
            par, log_pos, log_likelihood = glmcc.GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse)
        
            for m in range(2, 5):
                tmp_par, tmp_log_pos, tmp_log_likelihood = glmcc.GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], m)
                if tmp_log_likelihood > log_likelihood:
                    log_pos = tmp_log_pos
                    log_likelihood = tmp_log_likelihood
                    par = tmp_par
                    delay_synapse = m
            
        #Connection parameters
            nb = int(glmcc.WIN/glmcc.DELTA)
            cc_0 = [0 for l in range(2)]
            max = [0 for l in range(2)]
            for l in range(2):
                cc_0[l] = 0
                max[l] = int(tau[l] + 0.1)
            
                if l == 0:
                    for m in range(max[l]):
                        cc_0[l] += np.exp(par[nb+int(delay_synapse)+m])
                if l == 1:
                    for m in range(max[l]):
                        cc_0[l] += np.exp(par[nb-int(delay_synapse)-m])

                cc_0[l] = cc_0[l]/max[l]
                    
                n12 = tau[l]*cc_0[l]
                if n12 <= 10:
                    par[glmcc.NPAR-2+l] = 0
            D1 = 0
            D2 = 0
            tmp_par, tmp_log_pos, log_likelihood_p = glmcc.GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse, cond = 1)
            tmp_par, tmp_log_pos, log_likelihood_n = glmcc.GLMCC(cc_list[1], cc_list[0], tau, beta, cc_list[2], cc_list[3], delay_synapse, cond = 2)
            D1 = log_likelihood - log_likelihood_p
            D2 = log_likelihood - log_likelihood_n
        
            #construct J file
            J_f_list_line = [i, j, round(par[glmcc.NPAR-1], 6), round(par[glmcc.NPAR-2], 6), round(D2, 6)
                         ,round(D1, 6)]
            J_f_list.append(J_f_list_line)
            count += 1
            if count == 10:
                pbar.update(10)
                count = 0
    pbar.close()
    return J_f_list
    
def __generate_W(J_list, DataNum):
    n = DataNum
    scale = 1.277
    z_a = 15.14
    W = [[0 for i in range(n)] for j in range(n)]
    #calculate W
    for i in range(0, len(J_list)):    
        W[J_list[i][0]][J_list[i][1]] = round(glmcc.calc_PSP_LR(J_list[i][2], J_list[i][4], z_a), 6)
        W[J_list[i][1]][J_list[i][0]] = round(glmcc.calc_PSP_LR(J_list[i][3], J_list[i][5], z_a), 6)
    return np.array(W)

def __plot_3D_W(W):
    G = nx.from_numpy_array(W, parallel_edges=False, create_using = nx.MultiDiGraph)
    spring_3D = nx.spring_layout(G,dim=3, seed=18) #maybe modified to use different layout
    Num_nodes = W.shape[0]
    x_nodes = [spring_3D[i][0] for i in range(Num_nodes)]# x-coordinates of nodes
    y_nodes = [spring_3D[i][1] for i in range(Num_nodes)]# y-coordinates
    z_nodes = [spring_3D[i][2] for i in range(Num_nodes)]# z-coordinates
    edge_list = G.edges()
    arrow_tip_ratio = 0.1
    arrow_starting_ratio = 0.9

    x_edges=[]
    y_edges=[]
    z_edges=[]
    x_cones=[]
    y_cones=[]
    z_cones=[]
    u_cones=[]
    v_cones=[]
    w_cones=[]
    edges_widget = []
    for edge in G.edges.data("weight"):
        edges_widget.append(edge[2])
    #need to fill these with all of the coordiates
    for edge in edge_list:
        #format: [beginning,ending,None]
        x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
        x_edges += x_coords

        y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
        y_edges += y_coords
        
        z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
        z_edges += z_coords

        x_cone = [spring_3D[edge[0]][0]+arrow_starting_ratio*(spring_3D[edge[1]][0]-spring_3D[edge[0]][0])]
        x_cones += x_cone

        y_cone = [spring_3D[edge[0]][1]+arrow_starting_ratio*(spring_3D[edge[1]][1]-spring_3D[edge[0]][1])]
        y_cones += y_cone

        z_cone = [spring_3D[edge[0]][2]+arrow_starting_ratio*(spring_3D[edge[1]][2]-spring_3D[edge[0]][2])]
        z_cones += z_cone
        #make the same length
        norm = np.sqrt((spring_3D[edge[1]][0]-spring_3D[edge[0]][0])**2+(spring_3D[edge[1]][1]-spring_3D[edge[0]][1])**2+(spring_3D[edge[1]][2]-spring_3D[edge[0]][2])**2)
        u_cones += [arrow_tip_ratio*(spring_3D[edge[1]][0]-spring_3D[edge[0]][0])/norm]
        v_cones += [arrow_tip_ratio*(spring_3D[edge[1]][1]-spring_3D[edge[0]][1])/norm]
        w_cones += [arrow_tip_ratio*(spring_3D[edge[1]][2]-spring_3D[edge[0]][2])/norm]
    edge_color = []
    #negative red, positive blue
    for i in range(len(edges_widget)):
        if edges_widget[i]<0:
            edge_color += ['red','red','red']
        else:
            edge_color += ['blue','blue','blue']
    trace_edges = go.Scatter3d(x=x_edges,
                        y=y_edges,
                        z=z_edges,
                        mode='lines',
                        line=dict(color=edge_color, width=2),
                        hoverinfo='none')
    #create a trace for the nodes
    trace_nodes = go.Scatter3d(x=x_nodes,
                         y=y_nodes,
                        z=z_nodes,
                        mode='markers',
                        marker=dict(symbol='circle',
                                    size=15,
                                    color=np.zeros(Num_nodes), #color the nodes according to their community
                                    colorscale=['lightgreen','magenta'], #either green or mageneta
                                    line=dict(color='black', width=0.5)),
                        text=np.arange(1,Num_nodes+1),
                        hoverinfo='text')
    trace_cones = go.Cone(
    x=x_cones,
    y=y_cones,
    z=z_cones,
    u=u_cones,
    v=v_cones,
    w=w_cones,
    showlegend=False,
    showscale=False,
    hoverinfo='none',
    colorscale=[[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']]
)
    #we need to set the axis for the plot 
    axis = dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')
    #also need to create the layout for our plot
    layout = go.Layout(title="neuron connectivity",
                width=650,
                height=625,
                showlegend=False,
                scene=dict(xaxis=dict(axis),
                        yaxis=dict(axis),
                        zaxis=dict(axis),
                        ),
                margin=dict(t=100),
                hovermode='closest')
    #Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes, trace_cones]
    fig = go.Figure(data=data, layout=layout)

    fig.show()

def __matrix_plot_W(W):
    plt.imshow(W)
    plt.colorbar()
    plt.show()


def Connection_analysis(dataObj = None):
    class weight_class():
        def __init__(self):
            self.W = None
            self.dataNum = None
    style = {'description_width': 'initial'}
    #generate W
    threshold_widget = widgets.IntText(
            value = 100,
            description = 'minimum spike number',
            style = style,
            disabled = False
            )
    save_path_widget = FileChooser('/content/drive/MyDrive/EPhys Recordings/')
    save_path_widget.title = 'save weight matrix'
    save_path_widget.default_filename = 'output.csv'
    save_button = widgets.Button(description="save")
    generate_button = widgets.Button(description="generate")
    weight = weight_class()
    #plot W
    W_path_widget = FileChooser('/content/drive/MyDrive/EPhys Recordings/')
    W_path_widget.title = 'load weight matrix'
    W_path_widget.filter_pattern = '*.csv'
    load_button = widgets.Button(description="load")
    #matrix plot
    matrix_plot_button = widgets.Button(description="matrix plot")
    #3D plot
    plot_button = widgets.Button(description="3D plot")
    def split_path(path):
        path_split = path.split('/')
        result_path = '/'+path_split[1]+'/'+path_split[2]+'/My Drive'
        for i in range(5,len(path_split)):
          result_path = result_path + '/'+path_split[i]
        return result_path
    def on_button_clicked_load_W(b):
        path = split_path(W_path_widget.selected)
        try:
            W = np.genfromtxt(path, delimiter=',')
        except (RuntimeError, TypeError, NameError, IOError, ValueError, AttributeError, OSError) as e: #Why this part raises so many errors?
            print(e)
            return
        weight.W = W
    load_button.on_click(on_button_clicked_load_W)
    def on_button_clicked_save_W(b):
        path = split_path(save_path_widget.selected)
        if not path.endswith('.csv'):
            print('must be csv file')
            return
        if not weight.W is None:
            np.savetxt(path, weight.W, delimiter=",")
    save_button.on_click(on_button_clicked_save_W)
    def on_button_clicked_generate_W(b):
        if dataObj is None:
            print('No data Object detected')
            return
        spike_train = __generate_spikeTrain(dataObj, threshold_widget.value)
        J_list = __generate_J(spike_train)
        weight.dataNum = len(spike_train)
        del spike_train
        weight.W = __generate_W(J_list, weight.dataNum)
        del J_list
    generate_button.on_click(on_button_clicked_generate_W)
    def on_button_clicked_matrix_plot(b):
        if weight.W is None:
            print('No weight matrix detected')
            return
        __matrix_plot_W(weight.W)
    matrix_plot_button.on_click(on_button_clicked_matrix_plot)
    def on_button_clicked_3d_plot(b):
        if weight.W is None:
            print('No weight matrix detected')
            return
        __plot_3D_W(weight.W)
    plot_button.on_click(on_button_clicked_3d_plot)
    path_box = widgets.HBox([save_path_widget, W_path_widget, threshold_widget])
    button_box = widgets.HBox([save_button, load_button, generate_button])
    plot_box = widgets.HBox([matrix_plot_button, plot_button])
    dimension_box = widgets.VBox([path_box, button_box, plot_box])
    display(dimension_box)
    