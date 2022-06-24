# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 18:14:21 2022

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


def curation_widget(dataObj):
    if dataObj is None:
        print("dataObj is empty")
        return
    recording = dataObj.recording_cmr
    sorting = dataObj.sorting_MS4
    select_widget = widgets.Dropdown(
    options=['SNR', 'ISI', 'silhouette'],
    value='ISI',
    description='curation metric:',
    disabled=False,
    )
    threshold = widgets.widgets.FloatText(
    value=0.05,
    description='threshold:',
    disabled=False
    )
    run = widgets.Button(description="run")
    save = widgets.Button(description="save")
    output = widgets.Button(description="output")
    class sorting_holder():
        def __init__(self):
            self.soting = None
            self.type = None
            self.threshold = None
    curation_holder = sorting_holder()
    def curation(options, threshold):
        if options is "SNR":
            sorting_snr = st.curation.threshold_snrs(sorting, recording, threshold=threshold, threshold_sign='less')
            print("units pass SNR test: ",sorting_KL_snr.get_unit_ids())
            print("ratio of units pass SNR test {:.2f}% ".format(100*len(sorting_snr.get_unit_ids())/len(sorting.get_unit_ids())))
            return sorting_snr
        elif options is "ISI":
            sorting_isi = st.curation.threshold_isi_violations(sorting, threshold, 'greater',recording.get_num_frames())
            print("units pass ISI test: ",sorting_isi.get_unit_ids())
            print("ratio of units pass ISI test {:.2f}% ".format(100*len(sorting_isi.get_unit_ids())/len(sorting.get_unit_ids())))
            return sorting_isi
        elif options is "silhouette":
            sorting_silhouette = st.curation.threshold_silhouette_scores(sorting, recording, threshold, 'less')
            print("units pass silhouette test: ",sorting_silhouette.get_unit_ids())
            print("ratio of units pass silhouette test {:.2f}% ".format(100*len(sorting_silhouette.get_unit_ids())/len(sorting.get_unit_ids())))
            return sorting_silhouette
    def on_button_clicked_run(b):
        curation_sorting = f(options.value, threshold.value)
        curation_holder.sorting = curation_sorting
        curation_holder.type = options.value
        curation_holder.threshold = threshold.value
    run.on_click(on_button_clicked_run)
    select_box = widgets.VBox([select_widget, threshold])
    button_box = widgets.VBox([run, save, output])
    dimension_box = widgets.HBox([select_box, button_box])
    display(dimension_box)
        