# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 14:19:02 2022

@author: Xiang LI
"""

import numpy as np
import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.toolkit as st

import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.offsetbox import AnchoredOffsetbox
from google.colab import output
from mpl_toolkits.mplot3d import Axes3D

import importlib

from estherlib.uniform_60_filtering import uniform_60_filt
from estherlib.returnDataInfo import *
from estherlib.datasetObjectModule import *
from estherlib.cachingModule import *
from estherlib.runRoutines import *

from intan.load_intan_rhd_format import read_data
import pandas as pd
import statistics
from scipy.fft import fft
import os
from multiprocessing import Pool
from functools import partial
from IPython.core.debugger import set_trace
from ipyfilechooser import FileChooser

output.enable_custom_widget_manager()

def pca_analysis(dataObj, recording = None, sorting = None, pca_score = None, max_chan = None, units = None):
    """
    Description
    ----------
    Perform pca analysis in a widget.
    
    Parameters
    ----------
    recording : RecordingExtractor, optional
        recording. The default is None.
    sorting : SortingExtractor, optional
        sorting. The default is None.
    pca_score : int list, optional
        If not provided, and recording or sorting is not complete, raise error. The default is None.
    max_chan : int list, optional
        If not provided, and recording or sorting is not complete, raise error. The default is None.
    units : int list, optional
        unit ids. Default get unit ids from sorting. If Both units and sorting are missed, get unit ids as range of len(pca_score)-1

    Returns
    -------
    None.
    
    """
    if recording == None:
        recording = dataObj.recording_cmr
    if sorting == None:
        sorting = dataObj.sorting_MS4
    if pca_score != None and max_chan != None:
        if len(pca_score) != len(max_chan):
            raise ValueError("unit numbers not match")
    elif recording != None and sorting != None: # compute pca_score and max_chan
        loaded_recording_cmr = si.load_extractor(dataObj.cacheFilePath)
        loaded_recording_cmr.annotate(is_filtered=True)
        
        dataObj.waveformCachePath = dataObj.cacheFilePath + "_waveforms"
        # load the waveform object
        we = si.extract_waveforms(loaded_recording_cmr, sorting, dataObj.waveformCachePath, ms_before = 2, ms_after = 2, max_spikes_per_unit = 10000, overwrite=True)
        if pca_score == None:
            pca = st.postprocessing.compute_principal_components(we, load_if_exists=True, n_components=3, mode='by_channel_local')
            pca_score = [pca.get_projections(unit_id = unit) for unit in sorting.get_unit_ids()]  # a list of (units, times, PCA_Dims)
        if max_chan == None:
            max_chan = [1] * sorting.get_unit_ids()
    for i in range(len(max_chan)):
        if max_chan[i] == None:
            raise ValueError("No enough data to perform PCA")
    for i in range(pca_score[0].shape[0]):
        for j in range(pca_score[0].shape[1]):
            for k in range(pca_score[0].shape[2]):
                if pca_score[0][i, j, k] == None:
                    raise ValueError("No enough data to perform PCA")
    max_channel =len(recording.get_channel_ids()) -1
    if units == None:
        if sorting != None:
            units = sorting.get_unit_ids()
        else:
            units = range(len(pca_score)-1)
    # direct changing valuein the readout of slider has not effect. Add an additional inttext for each input to do bidirectional change
    channel_widget_1 = widgets.IntSlider(min=0,max = max_channel, value = 0, readout = False)
    channel_1 = widgets.IntText(min=0, max = max_channel, value = 0, layout = widgets.Layout(width='50px'), disabled=False)
    def update_channel_widget_1(*args):
      channel_widget_1.value = channel_1.value
    def update_channel_1(*args):
      channel_1.value = channel_widget_1.value
    channel_widget_1.observe(update_channel_1, 'value')
    channel_1.observe(update_channel_widget_1, 'value')
    # package channel_widget_1 and channel_1
    channel_box_1 = widgets.HBox([channel_widget_1, channel_1])
    pc_widget_1 = widgets.Dropdown(options = [1,2,3], value = 1)
    # direct changing valuein the readout of slider has not effect. Add an additional inttext for each input to do bidirectional change
    channel_widget_2 = widgets.IntSlider(min=0,max = max_channel, value = 0, readout = False)
    channel_2 = widgets.IntText(min=0, max = max_channel, value = 0, layout = widgets.Layout(width='50px'), disabled=False)
    def update_channel_widget_2(*args):
      channel_widget_2.value = channel_2.value
    def update_channel_2(*args):
      channel_2.value = channel_widget_2.value
    channel_widget_2.observe(update_channel_2, 'value')
    channel_2.observe(update_channel_widget_2, 'value')
    # package channel_widget_2 and channel_2
    channel_box_2 = widgets.HBox([channel_widget_2, channel_2])
    pc_widget_2 = widgets.Dropdown(options = [1,2,3], value = 2)
    # direct changing valuein the readout of slider has not effect. Add an additional inttext for each input to do bidirectional change
    channel_widget_3 = widgets.IntSlider(min=0,max = max_channel, value = 0, readout = False)
    channel_3 = widgets.IntText(min=0, max = max_channel, value = 0, layout = widgets.Layout(width='50px'), disabled=False)
    def update_channel_widget_3(*args):
      channel_widget_3.value = channel_3.value
    def update_channel_3(*args):
      channel_3.value = channel_widget_3.value
    channel_widget_3.observe(update_channel_3, 'value')
    channel_3.observe(update_channel_widget_3, 'value')
    # package channel_widget_3 and channel_3
    channel_box_3 = widgets.HBox([channel_widget_3, channel_3])
    pc_widget_3 = widgets.Dropdown(options = [1,2,3], value = 3)
    dimension_widget = widgets.IntSlider(min=1,max = 3, value = 2, description = "dimension: ")
    legend_widget = widgets.Checkbox(value = False, description = 'legend')
    axis_widget = widgets.Checkbox(value = True, description = 'axis')
    # combine pca score and max channel
    unit_list = []
    for i in range(len(units)):
        unit_list.append("unit {:d} with maximum channel {:d}".format(units[i], max_chan[i]))
        
    unit_select = widgets.SelectMultiple(
    options=unit_list,
    value=[unit_list[0]],
    description='units',
    disabled=False
    )
    # dimension customization layout
    label_box = widgets.VBox([widgets.Label(value='      '), widgets.Label(value='dimension 1: '), widgets.Label(value='dimension 2: '), widgets.Label(value='dimension 3: ')])
    left_box = widgets.VBox([widgets.Label(value='channel'), channel_box_1, channel_box_2, channel_box_3])
    right_box = widgets.VBox([widgets.Label(value='principal component'), pc_widget_1, pc_widget_2, pc_widget_3])
    dimension_box = widgets.HBox([label_box, left_box, right_box])
    display(unit_select)
    display(dimension_box)
    display(dimension_widget)
    display(legend_widget)
    display(axis_widget)
    
    def plot_pca(unit_list, channel_1, pc_1, channel_2, pc_2, channel_3, pc_3, dimension, legend, axis):
        pc_1 = pc_1 - 1
        pc_2 = pc_2 - 1
        pc_3 = pc_3 - 1
        if dimension == 1:  #1D
            fig = plt.figure()
            for unit in unit_list:
                unit = int(unit.split()[1]) - 1
                plt.plot(pca_score[unit][:, channel_1, pc_1],np.zeros((len(pca_score[unit][:, channel_1, pc_1]),1)), '*', label="unit {:d}".format(unit + 1))
        elif dimension == 2: #2D
            fig = plt.figure()
            for unit in unit_list:
                unit = int(unit.split()[1]) - 1
                plt.plot(pca_score[unit][:, channel_1, pc_1], pca_score[unit][:, channel_2, pc_2], '*', label="unit {:d}".format(unit + 1))
        elif dimension == 3: #3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for unit in unit_list:
                unit = int(unit.split()[1]) - 1
                ax.scatter3D(pca_score[unit][:, channel_1, pc_1], pca_score[unit][:, channel_2, pc_2], pca_score[unit][:, channel_3, pc_3], marker = '*', label="unit {:d}".format(unit + 1))
        if legend == True:
            plt.legend(bbox_to_anchor=(1.23, 1.0), loc ="upper right")
        if axis == False:
            plt.gca().set_axis_off()
        plt.savefig('pca_score.jpg', dpi = 1200)
        
    output = widgets.interactive_output(plot_pca, {'unit_list':unit_select, 'channel_1':channel_widget_1, 'pc_1':pc_widget_1, 'channel_2':
         channel_widget_2, 'pc_2':pc_widget_2, 'channel_3':channel_widget_3, 'pc_3':pc_widget_3, 
         'dimension':dimension_widget, 'legend':legend_widget, 'axis':axis_widget})
    display(output)
    