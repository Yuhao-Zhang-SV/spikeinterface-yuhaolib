# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:21:51 2022

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


#correlation
def correlation_widget(dataObj):
  '''
    visualize the autocorrelation and cross-correlation

    Parameters
    ----------
    dataObj : TYPE
        Input data.

    Returns
    -------
    None.

  '''
  recording = dataObj.recording_cmr
  sorting = dataObj.sorting_MS4
  unit_widget = widgets.SelectMultiple(
  options=dataObj.units,
  value=[dataObj.units[0]],
  description='units',
  disabled=False
  )
  bin_widget = widgets.IntText(
  value = 2,
  description='Bin size:',
  disabled=False
  )
  window_widget = widgets.IntText(
  value = 50,
  description='Window size:',
  disabled=False
  )
  select_widget = widgets.Dropdown(
    options=['auto', 'cross'],
    value='auto',
    description='figure model:',
    disabled=False,
  )
  def f(units, bin, window, select):
    if select == 'auto':
      sw.plot_autocorrelograms(sorting, bin_size=bin, window=window, unit_ids=units)
    elif select == 'cross':
      sw.plot_crosscorrelograms(sorting, bin_size=bin, window=window, unit_ids=units)
  run = widgets.Button(description="plot")
  def on_button_clicked(b):
    f(unit_widget.value, bin_widget.value, window_widget.value, select_widget.value)
  run.on_click(on_button_clicked)
  para_box = widgets.VBox([bin_widget, window_widget])
  dimension_box = widgets.HBox([unit_widget, para_box, select_widget])
  display(dimension_box)
  display(run)


#some bug, DONT USE!
def statistics_widgets(dataObj):
  recording = dataObj.recording_cmr
  sorting = dataObj.sorting_MS4
  unit_widget = widgets.SelectMultiple(
  options=dataObj.units,
  value=[dataObj.units[0]],
  description='units',
  disabled=False
  )
  channel_list = dataObj.recording_cmr.get_channel_ids()
  channel_widget = widgets.SelectMultiple(
    options=channel_list,
    value = [channel_list[0]],
    description='channels',
    disabled=False
  )
  select_widget = widgets.Dropdown(
    options=['spectrum', 'spectrogram', 'unit waveform'],
    value='spectrogram',
    description='figure model:',
    disabled=False
  )
  def f(units, channels, select):
    if select=='spectrum':
      sw.plot_spectrum(recording)
    elif select=='spectrogram':
      sw.plot_spectrogram(recording, channel=0, nfft=2048)
    elif select=='unit waveform':
      sw.plot_unit_waveforms(recording, sorting, channel_ids=channels, unit_ids=units)
  run = widgets.Button(description="plot")
  def on_button_clicked(b):
    f(unit_widget.value, channel_widget.value, select_widget.value)
  run.on_click(on_button_clicked)
  dimension_box = widgets.HBox([unit_widget, channel_widget, select_widget])
  display(dimension_box)
  display(run)


def histogram_widget(dataObj):
  recording = dataObj.recording_cmr
  sorting = dataObj.sorting_MS4
  sample_rate = recording.get_sampling_frequency()
  num_frame = recording.get_num_frames()
  time_limit = num_frame/sample_rate #time length in s
  time_limit_ms = time_limit*1000 #time length in ms
  unit_widget = widgets.SelectMultiple(
  options=dataObj.units,
  value=[dataObj.units[0]],
  description='units',
  disabled=False
  )
  select_widget = widgets.Dropdown(
    options=['raster', 'isi', 'histogram'],
    value='raster',
    description='figure model:',
    disabled=False
  )
  time_select = widgets.Dropdown(
    options=['milliseconds', 'seconds'],
    value='seconds',
    description='time unit:',
    disabled=False
  )
  start_widget = widgets.BoundedFloatText(
      min = 0,
      max = time_limit,
      value = 0,
      step = 0.0001,
      description='start time',
      disable=False
  )
  end_widget = widgets.BoundedFloatText(
      min = 0,
      max = time_limit,
      value = 0,
      step = 0.0001,
      description='end time',
      disable=False
  )
  window_widget = widgets.BoundedFloatText(
      min = 0,
      max = time_limit,
      value = 1,
      step = 0.0001,
      description='window',
  )
  overlap_widget = widgets.BoundedFloatText(
      min = 0,
      max = time_limit,
      value = 0,
      step = 0.0001,
      description='overlap'
  )
  def update_time_unit(*args):
    if time_select.value == 'milliseconds':
      start_widget.max = time_limit_ms
      end_widget.max = time_limit_ms
      window_widget.max = time_limit_ms
      overlap_widget.max = time_limit_ms
    elif time_select.value == 'seconds':
      start_widget.max = time_limit
      end_widget.max = time_limit
      window_widget.max = time_limit
      overlap_widget.max = time_limit
    start_widget.value = 0
    end_widget.value = 0
    window_widget.value = 0
    overlap_widget.value = 0
  time_select.observe(update_time_unit, 'value')
  def histogram(units, start, end, window, overlap = 0):
    num_frame = int((end-start)/(window-overlap))
    spike_count = np.zeros([len(units), num_frame])
    for i in range(num_frame):
        if len(units) == 1:
            spike_count[0][i] = len(sorting.get_unit_spike_train(unit_id=units[0], start_frame=start+i*(window-overlap), end_frame=start+(i+1)*(window-overlap)))
        for j, id in enumerate(units):
            spike_count[j][i] = len(sorting.get_unit_spike_train(unit_id=id, start_frame=start+i*(window-overlap), end_frame=start+(i+1)*(window-overlap)))
    return spike_count
  def f(units, select, time_model, start, end, window, overlap):
    if select =='raster':
      if time_model == 'milliseconds':
        sw.plot_rasters(sorting, unit_ids=units, time_range = [start/1000, end/1000])
      elif time_model == 'seconds':
        sw.plot_rasters(sorting, unit_ids=units, time_range = [start, end])
    elif select == 'isi':
        sw.plot_isi_distribution(sorting, unit_ids=units)
    elif select == 'histogram':
      if time_model == 'milliseconds':
        spike_count = histogram(units, start*sample_rate/1000, end*sample_rate/1000, window*sample_rate/1000, overlap*sample_rate/1000)
      elif time_model == 'seconds':
        spike_count = histogram(units, start*sample_rate, end*sample_rate, window*sample_rate, overlap*sample_rate)
      if (end - start) % (window - overlap) != 0:
          end_real = end - ((end - start) % (window - overlap))
      else:
          end_real = end
      if len(units) == 1:
        axs = plt.subplot(1, 1, 1)
        plt.title('spike count')
        axs.step(np.arange(start, end_real, window-overlap), spike_count[0])
        axs.label_outer()
      else:
        fig, axs = plt.subplots(len(units), sharex=True, sharey=True, gridspec_kw={'hspace': 0})
        fig.suptitle('spike count')
        for i in range(len(units)):
            axs[i].step(np.arange(start, end_real, window-overlap), spike_count[i])
        for ax in axs:
            ax.label_outer()
  select_box = widgets.VBox([select_widget, time_select])
  time_box = widgets.VBox([start_widget, end_widget, window_widget, overlap_widget])
  dimension_box = widgets.HBox([unit_widget, select_box, time_box])
  run = widgets.Button(description="plot")
  def on_button_clicked(b):
    f(unit_widget.value, select_widget.value, time_select.value, start_widget.value, end_widget.value, window_widget.value, overlap_widget.value)
  run.on_click(on_button_clicked)
  display(dimension_box)
  display(run)
