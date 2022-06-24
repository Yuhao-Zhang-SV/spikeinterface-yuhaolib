# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:24:22 2022

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

def spike_viewer(spikes2 = None, units = None, samplingFrequency = None, recording = None, dataObj = None):
    '''
    
    @author: Jay Reddy

    Parameters
    ----------
    spikes2 : TYPE, optional
        DESCRIPTION. The default is None.
    units : TYPE, optional
        DESCRIPTION. The default is None.
    samplingFrequency : TYPE, optional
        DESCRIPTION. The default is None.
    recording : TYPE, optional
        DESCRIPTION. The default is None.
    dataObj : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    class AnchoredScaleBar(AnchoredOffsetbox):
        def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, barcolor="black", barwidth=None, 
                 **kwargs):
            """
            Draw a horizontal and/or vertical  bar with the size in data coordinate
            of the give axes. A label will be drawn underneath (center-aligned).
            - transform : the coordinate frame (typically axes.transData)
            - sizex,sizey : width of x,y bar, in data units. 0 to omit
            - labelx,labely : labels for x,y bars; None to omit
            - loc : position in containing axes
            - pad, borderpad : padding, in fraction of the legend font size (or prop)
            - sep : separation between labels and bars in points.
            - **kwargs : additional arguments passed to base class constructor
            """
            from matplotlib.patches import Rectangle
            from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
            bars = AuxTransformBox(transform)
            if sizex:
                bars.add_artist(Rectangle((0,0), sizex, 0, ec=barcolor, lw=barwidth, fc="none"))
            if sizey:
                bars.add_artist(Rectangle((0,0), 0, sizey, ec=barcolor, lw=barwidth, fc="none"))

            if sizex and labelx:
                self.xlabel = TextArea(labelx, minimumdescent=False)
                bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
            if sizey and labely:
                self.ylabel = TextArea(labely)
                bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)

            AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=True, **kwargs)
    def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
        """ Add scalebars to axes
        Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
        and optionally hiding the x and y axes
        - ax : the axis to attach ticks to
        - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
        - hidex,hidey : if True, hide x-axis and y-axis of parent
        - **kwargs : additional arguments passed to AnchoredScaleBars
        Returns created scalebar object
        """
        def f(axis):
            l = axis.get_majorticklocs()
            return len(l)>1 and (l[1] - l[0])
    
        if matchx:
            kwargs['sizex'] = f(ax.xaxis)
            kwargs['labelx'] = str(kwargs['sizex'])
        if matchy:
            kwargs['sizey'] = f(ax.yaxis)
            kwargs['labely'] = str(kwargs['sizey'])
        
        sb = AnchoredScaleBar(ax.transData, **kwargs)
        ax.add_artist(sb)

        if hidex : ax.xaxis.set_visible(False)
        if hidey : ax.yaxis.set_visible(False)
        if hidex and hidey: ax.set_frame_on(False)

        return sb
    
    def f(unit,channels, event, ms_buffer):
        t0 = spikes2[unit-1][event]
        t0 = t0/samplingFrequency
        offset = ms_buffer/1000
        fig = sw.plot_timeseries(recording, time_range= [t0-offset, t0+2*offset], color = 'black',channel_ids=channels )
        ax = fig.ax
        add_scalebar(ax,
               False, 
               True, 
               hidex=True, 
               hidey=True, 
               sizex= 0.050,
               sizey=50, 
               labelx='50 ms', 
               labely='50 $\mu$V')
        ax.set_xticks([t0])
        ax.ticklabel_format(style = 'plain', useOffset = False)
        plt.savefig('test.jpg', dpi = 1200)
    if dataObj != None:
        if spikes2 == None:
            spikes2 = dataObj.spikes2
        if units == None:
            units = dataObj.units
        if samplingFrequency == None:
            samplingFrequency = dataObj.samplingFrequency
        if recording == None:
            recording = dataObj.recording_cmr
    
    if spikes2 == None or units == None or samplingFrequency == None or recording == None:
        raise ValueError("No enough data of spike") 
    unit_widget = widgets.Dropdown(options = units, value = 2)

    max_events =len(spikes2[unit_widget.value -1]) -1
    event_widget = widgets.IntSlider(min=1,max = max_events, value = 20)
    buffer_widget = widgets.IntSlider(min=1,max = 2000, value = 40)
    
    channel_list = dataObj.recording_cmr.get_channel_ids()
    channel_widget = widgets.SelectMultiple(
    options=channel_list,
    value = [channel_list[0]],
    description='channels',
    disabled=False
    )

    def update_event_range(*args):
        event_widget.max = len(spikes2[unit_widget.value -1]) -1

    unit_widget.observe(update_event_range, 'value')

    interact(f, unit = unit_widget, channels = channel_widget, event =event_widget, ms_buffer = buffer_widget)



def plot_peristimulus(dataObj):
    '''
    @author: Jay Reddy
    generate the widget to plot the peristimulus histogram
    '''
    def peristimulus_histogram(filedir, stop = None):
        # stimdata is a readout of the trigger input. Whenever stimdata == 1, 
        # the optical stimulus is turned on
        rawdata=np.fromfile(filedir)
        stimdata = list(rawdata)
        plt.plot(stimdata)
        if stop <= 1e-6:
            stop = None
        # Find all the times the stimulus turns on and when it turns off
        thresh = (max(stimdata)-min(stimdata))/2
        on_times = (np.diff(stimdata,prepend=False)>thresh).nonzero()[0]
        print(f'Averaging across {len(on_times)} trials')
        off_times = (np.diff(stimdata,prepend=False)< -1*thresh).nonzero()[0]
        unittimes = []
        # for spikes in good units, re-index the spike time to the most recent stimulus
        # i.e. if the stimulus is at 30s, and the spike happens at 30.2s, then the
        # new spike time is 0.2s following the stimulus. Since we have many trials,
        # this aggregates the "peri-stimulus" times across trials. 
        goodunits = range(100)
        for unitspikes in [unitspikes for ind, unitspikes in enumerate(dataObj.spikes2) if ind+1 in goodunits]:
            sweeps= []
            i = 0
            max_i =len(on_times)-1 
            sweep_times =[]
            sweep_start = on_times[i]
            sweep_end = on_times[i+1]
            for t in unitspikes:
                if t < sweep_start:
                    # spike is before time period we care about
                    continue
                if t > sweep_start and t < sweep_end:
                    sweep_times.append((t - sweep_start)/30000)
                # keep moving until 
                while t > sweep_end:
                    i = i + 1
                    if i < max_i:
                        sweep_start = on_times[i]
                        sweep_end = on_times[i+1]
                        if len(sweep_times)>=1:
                            sweeps.append(sweep_times)
                        sweep_times = []
                        if t < sweep_start and t < sweep_end:
                            sweep_times.append((t - sweep_start)/30000)
                    if i >= max_i:
                        break
            if len(sweeps)>0:
                unittimes.append(np.concatenate(sweeps))
        num_units =len(unittimes)
        import seaborn as sns
        fig, axes = plt.subplots(num_units,1)
        for i in range(num_units):
            ax = axes[i]
            sns.histplot(unittimes[i], bins = 1000, ax=ax)
            xvals = np.linspace(0, 0.04)
            yvals = 0*xvals + 10
            ax.fill_between(xvals,yvals,color="green", alpha=0.3)
            if stop:
                ax.set_xlim(0,stop)
        plt.tight_layout()
        fig.set_figheight(32)
    filedir_widget = widgets.Textarea(
        value='/content/drive/My Drive/EPhys Recordings/Chamanzar Lab Recording/Jay Rodent in vivo/08-19-2021/M1 ReaChr/continuous_40ms_410mV_0.25hzbatchcontinuous_40mstrigger.RAW',
        placeholder='path to trigger',
        disabled=False,
        height = 'auto',
        descrption='file path'
        )
    stop_widget = widgets.FloatText(
    value=0.0,
    description='Stop:',
    disabled=False
    )
    run = widgets.Button(description="generate histogram")
    def on_button_clicked(b):
        peristimulus_histogram(filedir_widget.value, stop_widget.value)
    input_box = widgets.HBox([filedir_widget, stop_widget])
    run.on_click(on_button_clicked)
    display(input_box)
    display(run)

def waveform_widget(dataObj):
    '''
    compare waveform and template in a single figure. Dont choose too many units and channels, or the figure will be indistinguishable

    Parameters
    ----------
    dataObj : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    recording = si.load_extractor(dataObj.cacheFilePath)
    recording.annotate(is_filtered=True)
    sorting = dataObj.sorting_MS4
    unit_list = []
    dataObj.waveformCachePath = dataObj.cacheFilePath + "_waveforms"
    we = si.extract_waveforms(recording, sorting, dataObj.waveformCachePath, ms_before = 1, ms_after = 2, max_spikes_per_unit = None, overwrite=True)

    unit_widget = widgets.SelectMultiple(
    options=dataObj.units,
    value=[dataObj.units[0]],
    description='units',
    disabled=False
    )

    radius_widget = widgets.IntSlider(value=100, min=0, max=1000, step=5, description='radius_um:', disabled=False, orientation='horizontal', readout=True, readout_format='d')
    
    select_widget = widgets.Dropdown(
    options=['waveform', 'template', 'both'],
    value='template',
    description='figure model:',
    disabled=False,
    )

    def f(units, radius, selection):
        if selection == 'waveform':
            sw.plot_unit_waveforms(we, unit_ids=units, radius_um=radius, lw=0.3)
        elif selection == 'template':
            sw.plot_unit_templates(we, unit_ids=units, radius_um=radius, lw=0.3)
        elif selection == 'both':
            sw.plot_unit_waveforms(we, unit_ids=units, radius_um=radius, lw=0.3)
            sw.plot_unit_templates(we, unit_ids=units, radius_um=radius, lw=0.3)
    run = widgets.Button(description="plot")
    def on_button_clicked(b):
        f(list(unit_widget.value), radius_widget.value, select_widget.value)
    run.on_click(on_button_clicked)
    dimension_box = widgets.HBox([unit_widget, radius_widget, select_widget])
    display(dimension_box)
    display(run)
