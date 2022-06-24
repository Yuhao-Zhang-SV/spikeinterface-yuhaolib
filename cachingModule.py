import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import pandas as pd
import scipy.io
from matplotlib import cm
import os
import copy
import sys
import gc
from multiprocessing import Pool
from functools import partial

from jaylib.histogram import plot_firing_rate
from jaylib.accumulate import plot_cumulative_spikes
from estherlib.adaptiveFiltering import *
from estherlib.returnDataInfo import *

import unicodedata


from stellalib.custom_raster import custom_raster
#from madilib.waveform_viewer import waveform_subplot

def saveFilteredDataToRAW(dataset, spike_prominence, fullFileName) :   
    dataset.tofile(fullFileName)
    print("CREATED FILE : {0}".format(fullFileName))

def parallelAdapativeFilteringHelper(tup, 
                                    sampling_frequency, 
                                    bandpassLow,
                                    bandpassHigh,
                                    filePath, 
                                    spikeProminence, 
                                    printFFT):
    print(f'paralleltuple: {tup}')
    rawdata, figFileName = tup
    return adaptiveFiltering(rawdata, 
                            figFileName, 
                            sampling_frequency, 
                            bandpassLow, 
                            bandpassHigh, 
                            filePath, 
                            spikeProminence, 
                            printFFT)

def returnRAWFiltered(spike_prominence, fileName, RAWfilePath, cacheFilePath, chanList, sampling_frequency,  bandpassLow, bandpassHigh, totalChanNum, display = True):
    print(f"Loaded Dataset: {fileName}")
    prominenceExt = "prominence{0}_new/".format(spike_prominence)
    fullFilePath = cacheFilePath + prominenceExt
    fullFilePath = unicodedata.normalize("NFKD", fullFilePath)
    RAWfilePath = unicodedata.normalize("NFKD", RAWfilePath)
    print("searching for " + fullFilePath)
    # check if file with filtering prominence exists, if not, create it
    if os.path.exists(fullFilePath) == False:
        os.mkdir(fullFilePath)
    
    # create folder to store images if folder doesn't already exist
    imagesExt = "images_new/"
    imagesFolderFilePath = fullFilePath + imagesExt
    if os.path.exists(imagesFolderFilePath) == False:
        os.mkdir(imagesFolderFilePath)

    fullFileName = fullFilePath + 'filtered {dataset_name} data with {prominence_value} spike prominence.RAW'.format(dataset_name = fileName, prominence_value = spike_prominence)
    num_channels = len(chanList)

    if os.path.exists(fullFileName) == True:
        # filtered file exists
        fullFileName = unicodedata.normalize("NFKD", fullFileName)
        if 'float32' in RAWfilePath:
            rawdata = np.fromfile(fullFileName, dtype = np.float32)
        else:
            rawdata = np.fromfile(fullFileName)
        print("Filtered data FOUND IN CACHE: " + fullFileName)
        # only the selected data channels were filtered and saved
        data = rawdata.reshape((num_channels, int(rawdata.size/num_channels)))
    else:
        # filtered file does not exist, filter raw data

        ### Jay Experimental change: load the data as float32 if that name is in the file. This should reduce the
        ### memory overhead by 2.

        print ("Filtered data NOT IN CACHE: " + fullFileName)
        if 'float32' in RAWfilePath:
            rawdata = np.fromfile(RAWfilePath, dtype = np.float32)
        else:
            rawdata = np.fromfile(RAWfilePath)
        print("RAWfilePath: ", RAWfilePath)
        rawdata = rawdata.reshape((int(totalChanNum),int(rawdata.size/totalChanNum))) #totalChanNum used to be 32
        print("Raw data shape", rawdata.shape)

        #geom = np.zeros((num_channels, 2)) #removed on 1-28
        #geom[:, 0] = range(num_channels) #removed on 1-28
        ####
        # Jay's parallelized version
        # ####
        # rawdataByChannel = list(rawdata)
        # figFileNames = [fileName + "Chan{0}".format(i) for i in range(totalChanNum)]
        # tuplemap = list(zip(rawdataByChannel,figFileNames))
        # f = partial(parallelAdapativeFilteringHelper,
        #             sampling_frequency=sampling_frequency, 
        #             bandpassLow=bandpassLow, 
        #             bandpassHigh=bandpassHigh, 
        #             filePath = imagesFolderFilePath, 
        #             spikeProminence = spike_prominence, 
        #             printFFT = display)

        # with Pool(processes = 4) as pool:
        #   data =pool.map(f, tuplemap)
        #   data =np.ndarray(data)
        ## original version
  
        data = rawdata[tuple(chanList),:]
        for ii, chan in enumerate(chanList):
            figFileName = fileName + "Chan{0}".format(ii)
            #line below removed on 1-28, replaced uniform_60_filter with adaptiveFiltering
            #data[ii] = uniform_60_filt(rawdata[chan], figFileName, sampling_frequency, filePath = imagesFolderFilePath, spikeProminence = spike_prominence, printFFT = display)
            data[ii] = adaptiveFiltering(rawdata[chan], figFileName, sampling_frequency, bandpassLow, bandpassHigh, filePath = imagesFolderFilePath, spikeProminence = spike_prominence, printFFT = display)

        print(data.size)
        # save filtered data for later reload
        saveFilteredDataToRAW(data, spike_prominence, fullFileName)
    
    print("Filtered data array shape", data.shape)
    return data

def exportSpikes2(dataArray, unit_ids, fileName):
    '''
    Specifically saves spikes2 data as excel for computing correlation matrices
    
        Parameters:
            dataArray (array) : relevant data array to save
            unit_ids (list of ints) : list of spikes units
            fileName (string) : full filename to save excel data
        
        Returns:
            None
    '''
    fileName = unicodedata.normalize("NFKD", fileName)
    empty_data = {}
    # This assumes dataArray is in spike order
    for ii, row in enumerate(dataArray):
        empty_data[str(unit_ids[ii])] = row

    max_len = max(map(len, empty_data.values()))

    for item in empty_data.items():
        key = item[0]
        value = item[1]
        newValue = value.copy()
        newValue.resize(max_len,)
        empty_data[key] = newValue

    empty_df = pd.DataFrame.from_dict(empty_data)
    empty_df_replaced = empty_df.replace(0, np.nan)

    if os.path.exists(fileName + ".xlsx") == True:
        print("FILE EXISTS, OVERIDING: " + fileName + ".xlsx")
    else:
        print("FILE DOES NOT EXIST, CREATING: " + fileName + ".xlsx")

    empty_df_replaced.to_excel(fileName + ".xlsx")
    print("CREATED FILE: " + fileName + ".xlsx")
    return None

def saveMatfile(data_dict, fileName):
    fileName = unicodedata.normalize("NFKD", fileName)
    if os.path.exists(fileName + ".mat") == True:
        print("FILE EXISTS, OVERIDING: " + fileName + ".mat")
    else:
        print("FILE DOES NOT EXIST, CREATING: " + fileName + ".mat")
    scipy.io.savemat(fileName + '.mat', data_dict)
    print("CREATED FILE: " + fileName + ".mat")
    return None

def savePickleFile(dataArray, fileName):
    fileName = unicodedata.normalize("NFKD", fileName)
    if os.path.exists(fileName + ".npy") == True:
        print("FILE EXISTS, OVERIDING: " + fileName + ".npy")
    else:
        print("NO FILE FOUND, CREATING: " + fileName + ".npy")
    np.save(fileName + '.npy', dataArray, allow_pickle = True)
    print("CREATED FILE: " + fileName + ".npy" )
    return None


def saveProcessedInfo(exportDataDict, exportFileNameDict, cacheFilePath, spike_prominence, sortingThreshold, fullFilePath):
    '''
    saveProcessedInfo saves pca_Scores, amplitudes, waveforms, spikes2, and spikes info of the relevant dataset if it exists

        Parameters: 
            exportDataDict (dict of arrays) : dictionary of numpy arrays of the relevant data
            exportFileNameDict (dict of strings) : dictionary of filenames of relevant data
            cacheFilePath (string) : file path of the cache for the correct dataset
            spike_prominence (float) : spike prominence of the desired dataset
            sortingThreshold (float) : sorting threshold of the desired dataset
        
        Returns:
            None
    '''
    prominenceExt = "prominence{0}_new/".format(spike_prominence)
    thresholdExt = "threshold{0}_new/".format(sortingThreshold)
    cacheFilePath = unicodedata.normalize("NFKD", cacheFilePath)
    if os.path.exists(cacheFilePath + prominenceExt) == False:
        os.mkdir(cacheFilePath + prominenceExt)

    if os.path.exists(cacheFilePath + prominenceExt + thresholdExt) == False:
        os.mkdir(cacheFilePath + prominenceExt + thresholdExt)

    # this case is for edited variable set
    if os.path.exists(fullFilePath) == False:
        os.mkdir(fullFilePath)

    # saving variable data from sorting
    variableList = ['pca_scores', 'amplitudes', 'waveforms', 'spikes2', 'spikes']
    for variable in variableList:
        savePickleFile(exportDataDict[variable], fullFilePath + exportFileNameDict[variable])
    
    # saving variable data as numpy
    exportMatlabDataDict = {}
    exportMatlabDataDict['PCA'] = exportDataDict['pca_scores']
    exportMatlabDataDict['amplitudes'] = exportDataDict['amplitudes']
    exportMatlabDataDict['waveforms'] = exportDataDict['waveforms']
    exportMatlabDataDict['spikes'] = exportDataDict['spikes2']
    saveMatfile(exportMatlabDataDict, fullFilePath + exportFileNameDict['exportMatlabData'])
    
    # saving spikes2 as excel
    print("waveforms", exportDataDict['waveforms'])
    exportSpikes2(exportDataDict['spikes2'], list(range(1, len(exportDataDict['waveforms'])+1)), fullFilePath + exportFileNameDict['waveforms'] + "EXCEL")
    return None

def loadPickleFile(fileName):
    if os.path.exists(fileName + ".npy") == False:
        print("FILE DOES NOT EXIST: " + fileName + ".npy")
        return None
    else:
        loadedData = np.load(fileName + '.npy', allow_pickle = True)
        print("FILE FOUND AND LOADED: " + fileName + '.npy')
        return loadedData

def loadProcessedInfo(fileNameDict, fullFilePath):
    '''
    loadProcessedInfo loads pca_Scores, amplitudes, waveforms, spikes2, and spikes info of the relevant dataset if it exists

        Parameters: 
            fileNameDict (dict of strings) : dictionary of filenames of the relevant data
            fullFilePath (string) : file path to store files
        
        Returns:
            returndDict (dict of arrays) : dictionary of numpy array of data categories
    '''

    returnDict = {}

    fullBool = True # determines if all components of valuable information exist (all or nothing)

    temp = loadPickleFile(fullFilePath + fileNameDict['pca_scores'])
    if (temp is not None):
        returnDict['pca_scores'] = temp
    else:
        fullBool = False
    temp = loadPickleFile(fullFilePath + fileNameDict['amplitudes'])
    if (temp is not None):
        returnDict['amplitudes'] = temp
    else:
        fullBool = False
    temp = loadPickleFile(fullFilePath + fileNameDict['waveforms'])
    if (temp is not None):
        returnDict['waveforms'] = temp
    else:
        fullBool = False
    temp = loadPickleFile(fullFilePath + fileNameDict['spikes2'])
    if (temp is not None):
        returnDict['spikes2'] = temp
    else:
        fullBool = False
    temp = loadPickleFile(fullFilePath + fileNameDict['spikes'])
    if (temp is not None):
        returnDict['spikes'] = temp
    else:
        fullBool = False

    # the following two data files are never re-imported into an active sorting script
    if os.path.exists(fullFilePath + fileNameDict['exportMatlabData'] + ".mat") == False:
        fullBool = False
    if os.path.exists(fullFilePath + fileNameDict['waveforms'] + "EXCEL.xlsx") == False:
        fullBool = False
    
    # evaluate returns
    if fullBool == False:
        return {}
    else:
        return returnDict

