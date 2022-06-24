import numpy as np
import matplotlib.pyplot as plt

import spikeinterface.core as si
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
from estherlib.cachingModule import *
from estherlib.runRoutines import *
from yuhaolib.datasetObjectModule import *

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

def initial_import():
    global spikeinterface, si, se, st, ss, sc, sw
    spikeinterface = __import__('spikeinterface', globals(), locals())
    si = spikeinterface.core
    se = spikeinterface.extractors
    st = spikeinterface.toolkit
    ss = spikeinterface.sorters
    sc = spikeinterface.comparison
    sw = spikeinterface.widgets

    global np, plt, mpimg, pd, scipy, cm, os, sys, copy, gc, importlib
    np = __import__('numpy', globals(), locals())
    _temp = __import__('matplotlib', globals(), locals())
    plt = _temp.pyplot
    maimg = _temp.image
    pd = __import__('pandas', globals(), locals())
    scipy = __import__('scipy', globals(), locals())
    cm = _temp.cm
    os = __import__('os', globals(), locals())
    sys = __import__('sys', globals(), locals())
    copy = __import__('copy', globals(), locals())
    gc = __import__('gc', globals(), locals())
    importlib = __import__('importlib', globals(), locals())

def dataObj_constructor():
    class dataObj_holder():
        def __init__(self):
            self.dataObj = None
    dataInd_widget = widgets.IntText(
    value=50,
    disabled=False
    )
    spikeProminence_widget = widgets.IntText(
        value=100,
        disabled=False
    )
    sortingThreshold_widget = widgets.IntText(
        value=5,
        disabled=False
    )
    bandpassLow_widget = widgets.IntText(
        value=500,
        disabled=False
    )
    bandpassHigh_widget = widgets.IntText(
        value=5000,
        disabled=False
    )
    '''
    filedir_widget = widgets.Textarea(
        value='/content/drive/My Drive/EPhys Recordings/Chamanzar Lab Recording/chamLabRecordingVariables.xlsx',
        placeholder='type in dataInfoDir',
        disabled=False,
    )
    '''
    filedir_widget = FileChooser('/content/drive/My Drive/EPhys Recordings/Chamanzar Lab Recording/')
    filedir_widget.title = 'data infor sheet'
    left_lab_box = widgets.VBox([widgets.Label(value='dataInd: '), widgets.Label(value='spikeProminence: '), widgets.Label(value='sortingThreshold: ')])
    left_box = widgets.VBox([dataInd_widget, spikeProminence_widget, sortingThreshold_widget])
    right_lab_box = widgets.VBox([widgets.Label(value='bandpassLow: '), widgets.Label(value='bandpassHigh: '), widgets.Label(value='dataInfoDir: ')])
    right_box = widgets.VBox([bandpassLow_widget, bandpassHigh_widget, filedir_widget])
    display_widget = widgets.Checkbox(value = False, description = 'display')
    mountainsort_widget = widgets.Checkbox(value = False, description = 'overide mountainsort')
    loadvariables_widget = widgets.Checkbox(value = False, description = 'override loadVariables')
    check_box = widgets.VBox([display_widget, mountainsort_widget, loadvariables_widget])
    dimension_box = widgets.HBox([left_lab_box, left_box, right_lab_box, right_box, check_box])
    run = widgets.Button(description="create dataObj")
    holder = dataObj_holder()
    def split_path(path):
        path_split = path.split('/')
        result_path = '/'+path_split[1]+'/'+path_split[2]+'/My Drive'
        for i in range(5,len(path_split)):
          result_path = result_path + '/'+path_split[i]
        return result_path
    def on_button_clicked(b):
      dataInfoDir = split_path(filedir_widget.selected)
    #   holder.dataObj = Dataset(dataInfoDir = dataInfoDir, dataInd = dataInd_widget.value, spikeProminence = spikeProminence_widget.value, sortingThreshold = sortingThreshold_widget.value, bandpassLow = bandpassLow_widget.value, bandpassHigh = bandpassHigh_widget.value ,detect_sign=-1, display = display_widget.value)
      holder.dataObj = Dataset(dataInfoDir = dataInfoDir, dataInd = dataInd_widget.value, spikeProminence = spikeProminence_widget.value, sortingThreshold = sortingThreshold_widget.value, bandpassLow = bandpassLow_widget.value, bandpassHigh = bandpassHigh_widget.value ,detect_sign=0, display = display_widget.value)
      holder.dataObj.loadMountainSort(overide = mountainsort_widget.value)
      holder.dataObj.loadVariables(overide = loadvariables_widget.value)
      print('dataObj initialized')
    run.on_click(on_button_clicked)
    display(dimension_box)
    display(run)
    return holder

'''
Since Pool needs to pickle (serialize) everything it sends to its worker-processes (IPC). 
Pickling actually only saves the name of a function and unpickling requires re-importing the function by name. 
For that to work, the function needs to be defined at the top-level, 
nested functions won't be importable by the child and already trying to pickle them raises an exception (more).
Which means all the auxiliary functions used in Concatenate_widget() should be defined
outside the function Concatenate_widget() 

'''
def listFiles(path, recursive = True): #adapted from 112 website
        if (os.path.isdir(path) == False):
            # base case:  not a folder, but a file, so return singleton list with its path
            if (path.endswith('.rhd')):
                return [path]
            else:
                return []
        if recursive == True:
                  # recursive case: it's a folder, return list of all paths
            files = [ ]
            for filename in os.listdir(path):
                files += listFiles(path + os.sep + filename)
            
            return files
    
# returns list of base file names (truncated path) for an inputted list of files with full path names
def fileBaseOnly(fileList):
        result = []
        for fileName in fileList:
            result.append(os.path.basename(fileName))
        return result
    
# below is general version
def getTime(fileName):
        fileName1 = fileName.split(".")[-2]
        fileInfo = fileName1.split('_')
        print("fileInfo", fileInfo)
        #print(seconds)
        seconds = int((fileInfo[-1])[4:])
        minutes = int((fileInfo[-1])[2:4])
        hours = int((fileInfo[-1])[0:2])
        time = hours*3600 + minutes*60 + seconds
        return time
    
# Get the name in format: {name}_date_time.rhd
def getName(fileName):
        fileName1 = fileName.split(".")[0]
        # get everything before date,time
        name = '_'.join(fileName1.split('_')[:-2])
        return name
    
def removePath(fileName):
      fn = fileName.split("/")[-1]
      return fn
    
# function that checks size of 2D array - used for assert in sortFiles
def sizeOf(list2D):
        result = 0
        for batch in list2D:
            for file in batch:
                result +=1
        return result            
            
# function that takes a .rhd file list and outputs 2-d array, groups lists of rhd files that belong in same experiment batch
def sortFiles(fileList):
        # this may need to be modified to account for files that are out of order
        print("fileList", fileList)
        assert(len(fileList) >= 1)
        result = []
        batch = []
        batch.append(fileList[0])
        for i in range(len(fileList)-1):
            file1 = fileList[i]
            file1Time = getTime(file1)
            file2 = fileList[i+1]
            file2Time = getTime(file2)
            timeDiff = file2Time - file1Time
            ## Filenames for different recording sessions may also be different
            file1Name = getName(file1)
            file2Name = getName(file2)
            if ((58 <= timeDiff <= 62) or (298 <= timeDiff <= 302))and (file1Name == file2Name): #300 is for 5 min files
                batch.append(file2)
            else:
                result.append(batch)
                batch = []
                batch.append(file2)        
        result.append(batch)
        # checks to make sure all input .rhd files are grouped into a batch
        assert(sizeOf(result) == len(fileList))
        return result
    
# Helper function for parallelized processing
def processFile(filename, totalChanNum= 32, includeADC= False, includeDIO= False):
          converted = read_data(filename) # converted should be the numpy array         
          #we can get the number of channels from the size of currFileArray
          currFileArray = converted['amplifier_data']       
          channel_info = converted['amplifier_channels']         
          sample_rate = converted['sample_rate']       
          if converted['record_time'] != None:
              record_time = converted['record_time']
          else:
              record_time = -1
          impedances = [(c['electrode_impedance_magnitude'],
                        c['electrode_impedance_phase']) for c in channel_info]
          channel_numbers = [c['chip_channel'] for c in channel_info]
          channel_offset = np.array([int(c['port_number']) -2 for c in channel_info]) *32
          channel_numbers = list(np.array(channel_numbers)+ channel_offset)
          if includeADC:
              assert 'board_adc_data' in converted, "includeADC set to True, but file contains no adc data"
              currFileArray =  np.append(currFileArray, converted['board_adc_data'], axis = 0)
          if includeDIO:
              assert 'board_dig_in_data' in converted, "includeDIO set to True, but file contains no dio data"
              print(converted.keys())
              digdata = converted['board_dig_in_data']
              currFileArray =  np.append(currFileArray, digdata, axis = 0) 
          print("currFileArray size", currFileArray.shape)
          return (currFileArray, channel_numbers, impedances)
    
# function takes in a 2d list of file names, allData[batch][all files in batch]
# where batch represents uninterrupted experiment, and files represents each rhd data file in that experiment run
def parallelProcessData(allData, fileName, totalChanNum = 32, includeADC= False, includeDIO =False):
        print("allData", allData)
        for batch in allData:
            print(batch[0])
            with Pool(processes=8) as pool:
              f = partial(processFile, totalChanNum = totalChanNum, includeADC=includeADC, includeDIO=includeDIO)
              npArrays = pool.map(f, batch)
              channel_numbers = npArrays[0][1]
              impedances = npArrays[0][2]
              sample_rate = npArrays[0][3]
              record_time = npArrays[0][4]
              npArrays = [arr[0] for arr in npArrays]
              batchResult = np.concatenate(npArrays, axis = 1)
              # slice out DIO segment
              batchName = removePath(getName(batch[0]))
              if includeDIO:
                  print('chop out DIO')
                  diodata = batchResult[-1,:]
                  plt.plot(diodata)
                  diodata.tofile(fileName+batchName+'trigger.RAW')
                  batchResult = batchResult[:-1, :]
              assert(len(batchResult) == totalChanNum)
              batchResult.tofile(fileName + batchName + ".RAW")
              print(impedances)
              print(channel_numbers)
              np.savetxt(fileName+batchName+'_impedances.csv', impedances, delimiter = ',')
              np.savetxt(fileName+batchName+'_chananels.csv', channel_numbers, delimiter = ',')
              print("CREATED FILE", fileName + batchName + ".RAW")
        return (fileName + batchName + ".RAW", sample_rate, record_time)
# function takes in a 2d list of file names, allData[batch][all files in batch]
# where batch represents uninterrupted experiment, and files represents each rhd data file in that experiment run
def sequentialProcessData(allData, fileName, totalChanNum = 32, includeADC= False, includeDIO =False, dtype = np.float64):
    print("allData", allData)
    for batch in allData:
        print(batch[0])
        f = partial(processFile, totalChanNum = totalChanNum, includeADC=includeADC, includeDIO=includeDIO)

        batchResult, channel_numbers, impedances = None,None,None

        batchResult_index = 0
        for element in batch:
          data, channel_numbers,impedances = f(element)

          # do conversion here to reduce footprint: 
          data = data.astype(dtype)

          if batchResult is None:
            # batchResult doesn't exist yet, so this is the first element
            # let's initialize it

            # We can use the number of items in the batch and the size of the
            # individual batch element to allocate the proper array size
            num_elems = len(batch)
            arr_ax1, arr_ax2 = data.shape
            batch_shape = (arr_ax1, arr_ax2*num_elems)
            
            batchResult = np.empty(batch_shape, dtype= dtype)

          # fill batchResult with the current data chunk
          # we know the start and stop indices based on the chunk size
          _, data_len = data.shape
          start_index,stop_index = batchResult_index, batchResult_index + data_len

          # do the assignment
          batchResult[:, start_index:stop_index] = data

          # update the index for the next run
          batchResult_index += data_len
          
        # trim the ndArray to the right size, since the last chunk can be a
        # partial chunk

        ax1_len, ax2_len = batchResult.shape
        trimmed_shape = (ax1_len, batchResult_index)
        batchResult.resize(trimmed_shape)

        # slice out DIO segment
        batchName = removePath(getName(batch[0]))
        if includeDIO:
            print('chop out DIO')
            diodata = batchResult[-1,:]
            plt.plot(diodata)
            diodata.tofile(fileName+batchName+'trigger.RAW')
            batchResult = batchResult[:-1, :]
        assert(len(batchResult) == totalChanNum)
        
        batchResult.tofile(fileName + batchName + dtype.__name__ + ".RAW")
        print(impedances)
        print(channel_numbers)
        np.savetxt(fileName+batchName+'_impedances.csv', impedances, delimiter = ',')
        np.savetxt(fileName+batchName+'_chananels.csv', channel_numbers, delimiter = ',')
        print("CREATED FILE", fileName + batchName +  dtype.__name__+ ".RAW")
def Concatenate_widget():
    #This is modified from ConcatenateRHDFiles25GB.ipynb
    # Concatenating channels
    def process_concatenate(base_dir, fileName, totalChanNum, includeADC=False, includeDIO=False, dtype=np.float64):
        print(dtype)
        # define base_dir as the folder in which the relevant data files are held
        RHDfileListRaw = listFiles(base_dir) #return list of full file path for files that end in ".rhd" in current directory
        sortedRHDfileListRaw = sorted(RHDfileListRaw) #alphabetize .rhd file list
        print("alphabetized RHDfileListRaw", sortedRHDfileListRaw) #sorts list of all RHD files in directory alphabetically
        # note, if you're specifying directory, don't use fileBaseOnly, because you need full path name
        allDataSorted = sortFiles(sortedRHDfileListRaw) #returns 2-D list of lists of RHDfile path names grouped together in experiment batches 
        print("grouped .rhd files", allDataSorted)
        print("num groups", len(allDataSorted))
        # for compressed data, say dtype=np.float32, if this optional parameter is removed, it will default to float64
        sequentialProcessData(allDataSorted, fileName, totalChanNum, includeADC=includeADC, includeDIO=includeDIO, dtype = dtype)
    
    def generate_excel_path():
        excel_split = excel_widget.selected.split('/')
        excel_path = '/'+excel_split[1]+'/'+excel_split[2]+'/My Drive'
        for i in range(5,len(excel_split)):
          excel_path = excel_path + '/' + excel_split[i]
        return excel_path
    def catch_xlsl(sheet_name):
        excel_path = generate_excel_path()
        df = pd.read_excel(excel_path, sheet_name= sheet_name)
        return df
    style = {'description_width': 'initial'}
    path_widget = FileChooser('/content/drive/MyDrive/EPhys Recordings/')
    path_widget.title = 'recording folder'
    path_widget.show_only_dirs = True
    excel_widget = FileChooser('/content/drive/My Drive/EPhys Recordings/Chamanzar Lab Recording/')
    excel_widget.title = 'excel file'
    path_box = widgets.VBox([path_widget, excel_widget])
    totalChanNum_widget = widgets.IntText(
            value = 16,
            placeholder = 'total number of channels',
            disabled=False,
            height = 'auto',
            description='channelNum'
            )
    includeADC_widget = widgets.Checkbox(
            value = False,
            description='includeADC'
            )
    includeDIO_widget = widgets.Checkbox(
            value = False,
            description='includeDIO'
            )
    datasheet_widget = widgets.Textarea(
        value='datasetInfo',
        disabled = False,
        style=style,
        description = 'datasheet'
        )
    datatype_widget = widgets.RadioButtons(
        options = ['np.float32', 'np.float64'],
        description = 'Recording DataType',
        value = 'np.float32',
        disabled = False
        )
    index_widget = widgets.IntText(
        value = 0,
        disabled = False,
        description = 'index'
    )
    name_widget = widgets.Textarea(
        value='',
        disabled = False,
        style=style,
        description = 'file name shown in xlsx'
    )
    cache_widget = widgets.Textarea(
        value='',
        disabled = False,
        style=style,
        description = 'path of cache'
    )
    Raw_widget = widgets.Textarea(
        value='',
        disabled = False,
        style=style,
        description = 'path of RAW file'
    )
    tissue_widget = widgets.Textarea(
        value='',
        disabled = False,
        style=style,
        description = 'tissue'
    )
    dataChannel_widget = widgets.SelectMultiple(
        options=[0],
        value=[0],
        description='data Channel',
        disabled=False
    )
    sampling_widget = widgets.IntText(
        value=30000,
        min = 0,
        description = 'sampling rate',
        disabled = False
    )
    recording_widget = widgets.IntText(
        value=-1,
        description = 'recording length',
        style=style,
        disabled = False
    )
    unitToRem_widget = widgets.SelectMultiple(
        options=[0],
        value=[0],
        description='units to Rem',
        disabled=False
    )
    originalChan_widget = widgets.IntText(
        value = 0,
        disabled = False,
        style=style,
        description = 'original channel number'
    )
    probe_widget = widgets.Textarea(
        value='',
        disabled = False,
        style=style,
        description = 'path of probe map'
    )
    channelPath_widget = widgets.Textarea(
        value='',
        disabled = False,
        style=style,
        description = 'path of channel map'
    )
    str_box_1 = widgets.HBox([index_widget, name_widget])
    str_box_2 = widgets.HBox([cache_widget, Raw_widget, tissue_widget])
    str_box_3 = widgets.HBox([dataChannel_widget, sampling_widget, recording_widget])
    str_box_4 = widgets.HBox([unitToRem_widget, originalChan_widget])
    str_box_5 = widgets.HBox([probe_widget, channelPath_widget])
    xlsx_box = widgets.VBox([str_box_1, str_box_2, str_box_3, str_box_4, str_box_5])
    run_xlsx = widgets.Button(description="write excel",
                              disabled = False)

    def activate_write(raw_name, sample_rate, record_time):
        df = catch_xlsl(datasheet_widget.value)
        index_widget.value = df['index-int'][len(df['index-int'])-1]+1
        cache_widget.value= path_widget.value+'cache'
        Raw_widget.value = raw_name
        sampling_widget.value = sample_rate
        recording_widget.value = record_time
        dataChannel_widget.options = list(range(totalChanNum_widget.value))
        unitToRem_widget.options = list(range(totalChanNum_widget.value))
        originalChan_widget.value = totalChanNum_widget.value

    def generate_xlsx(df):
        '''
        Index(['index-int', 'fileName-str', 'cacheDirPath-str', 'RAWfilePath-str',
       'tissue-str', 'dataChannels-intList', 'samplingRate-int',
       'recordingLength-float', 'unitsToRem-intList', 'unitsToMerge-intList',
       'activeTime-floatList', 'originalChanNum-int', 'probeMapPath-str',
       'channelMapPath-str']
        '''
        df_append = {'index-int':index_widget.value,
                     'fileName-str':name_widget.value,
                     'cacheDirPath-str':cache_widget.value,
                     'RAWfilePath-str':Raw_widget.value,
                     'tissue-str':tissue_widget.value,
                     'dataChannels-intList':list(dataChannel_widget.value),
                     'samplingRate-int':sampling_widget.value,
                     'recordingLength-float':recording_widget.value,
                     'unitsToRem-intList':list(unitToRem_widget.value),
                     'unitsToMerge-intList':[],
                     'activeTime-floatList':[],
                     'originalChanNum-int':originalChan_widget.value,
                     'probeMapPath-str':probe_widget.value,
                     'channelMapPath-str':channelPath_widget.value
                     }
        df = df.append(df_append, ignore_index = True)
        print(df)
        excel_path = generate_excel_path()
        df.to_excel(excel_path, datasheet_widget.value, index=False)
        
    run_concatenate = widgets.Button(description="Concatenate")
    def on_button_clicked_main(b):
        path_split = path_widget.selected_path.split('/')
        rhd_path = '/'+path_split[1]+'/'+path_split[2]+'/My Drive'
        for i in range(5,len(path_split)):
          rhd_path = rhd_path + '/'+path_split[i]
        process_concatenate(rhd_path, rhd_path+'/batch', totalChanNum_widget.value, includeADC_widget.value, includeDIO_widget.value, datatype_widget.value)
        # activate_write(raw_name, sample_rate, record_time)
    def on_button_clicked_write(b):
        df = catch_xlsl(datasheet_widget.value)
        generate_xlsx(df)
    run_xlsx.on_click(on_button_clicked_write)
    run_concatenate.on_click(on_button_clicked_main)
    select_box = widgets.VBox([totalChanNum_widget, includeADC_widget, includeDIO_widget, datatype_widget])
    dimension_box = widgets.HBox([path_box, select_box, datasheet_widget])
    display(dimension_box)
    display(run_concatenate)
    display(xlsx_box)
    display(run_xlsx)