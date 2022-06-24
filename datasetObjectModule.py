import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw

from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe, plot_probe_group
from probeinterface import generate_dummy_probe, generate_linear_probe
from probeinterface import write_probeinterface, read_probeinterface
from probeinterface import write_prb, read_prb

import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import pandas as pd
import scipy.io
from matplotlib import cm
import os
import shutil
import copy
import sys
import gc

from jaylib.histogram import plot_firing_rate
from jaylib.accumulate import plot_cumulative_spikes
from estherlib.uniform_60_filtering import uniform_60_filt
from estherlib.returnDataInfo import *
from yuhaolib.cachingModule import *
from stellalib.custom_raster import custom_raster

class Dataset(object):
    def __init__(self, dataInfoDir, dataInd, spikeProminence, sortingThreshold, bandpassLow = 300, bandpassHigh = 7000, display = True, detect_sign = -1):
        print("initializing dataset...")
        self.bandpassLow = bandpassLow
        self.bandpassHigh = bandpassHigh
        self.detect_sign = detect_sign
        self.ind = dataInd
        self.dataInfoDir = dataInfoDir
        self.dataset = returnDataInfo(self.ind, self.dataInfoDir)
        self.fileName = self.dataset['fileName']
        self.filePath = self.dataset['RAWfilePath']
        self.samplingFrequency = self.dataset['samplingRate']
        self.dataChanList = self.dataset['dataChannels']
        # self.cacheFilePath = self.dataset['cacheDirPath'] + '_new'
        self.cacheFilePath = self.dataset['cacheDirPath'][:-1] +'_new' + '/'
        self.recordingLength = self.dataset['recordingLength']
        try:
            self.originalChanNum = int(self.dataset['originalChanNum'])
        except:
            self.originalChanNum = 32
        print("self.originalChanNum", self.originalChanNum)
        #self.glutTimes = self.dataset['glutTimes'] # removed 1-27
        #self.fluorOffsetTime = self.dataset['fluorOffsetTime'] # removed 1-27
        self.display = display

        self.spikeProminence = spikeProminence
        self.sortingThreshold = sortingThreshold
        self.data = None
        self.recording_cmr = None
        self.sorting_MS4 = None
        self.edited_MS4 = None
        self.spikes = None
        self.edited_spikes = None
        self.renamed_edited_units = None
        self.defineFileNames()

    def defineFileNames(self):
        '''
        Initializes filename and extension variables

            Return:
                None
        '''
        # Import Variables from Excel Sheet 
        imagesDict = returnFileNames(self.dataInfoDir, "imageCacheFileNames")
        dataDict = returnFileNames(self.dataInfoDir, "dataCacheFileNames")
        newImagesDict, newDataDict = fullyFormat(imagesDict, dataDict, self.fileName, self.spikeProminence, self.sortingThreshold)
        self.exportFileNames = {**newImagesDict, **newDataDict}

        # MountainSort file names
        self.ms4JSONfileName = self.exportFileNames['mountainSortExport']
        self.ms4JSONfileName = self.ms4JSONfileName + '.json'
        self.recording_cmr_filename = self.exportFileNames['recording_cmr_filename']

        # Define file names for images
        self.unitWaveforms = self.exportFileNames['unitWaveforms']
        self.firingRateHisto = self.exportFileNames['firingRateHisto']
        self.rasterPlot = self.exportFileNames['rasterPlot']
        try:
            self.isiPlot = self.exportFileNames['isiPlot']
        except:
            pass
        self.INDVfiringRateHisto = "INDVIDUAL UNITS " + self.firingRateHisto

        # Define file extensions
        self.prominenceExt = "prominence{0}_new/".format(self.spikeProminence)
        self.thresholdExt = "threshold{0}_new/".format(self.sortingThreshold)
        self.fullFilePath = self.cacheFilePath + self.prominenceExt + self.thresholdExt

        self.ms4FilePath = self.fullFilePath + 'mountainsort4_output_new/'
        self.editedFullFilePath = self.fullFilePath + "edited/"
        return None

    def returnFilteredData(self):
        '''
        Applies uniform_60_filt to dataset or reloads filtered RAW file from cache

            Return:
                self.data : electrophysiology data, number of channels specified in returnDataInfo
        '''
        # Load Stored Filtered RAW Data or Apply Notch Filter and Store
        print(self.cacheFilePath)
        if os.path.exists(self.cacheFilePath) == False:
            try:
                os.mkdir(self.cacheFilePath)
            except FileNotFoundError:
                os.makedirs(self.cacheFilePath)
            else:
                print("Fail to create cache directory...")
        #needs totalChaNum, which is set to 32 as default
        self.data = returnRAWFiltered(self.spikeProminence, self.fileName, self.filePath, self.cacheFilePath, self.dataChanList, self.samplingFrequency, self.bandpassLow, self.bandpassHigh, totalChanNum = self.originalChanNum, display = self.display)

        return self.data

    def deriveRecordingData(self):
        '''
        Recalculates self.recording_cmr

            Return:
                self.recording_cmr
        '''
        print("Calculating recording_cmr...")
        if self.data is None:
            self.returnFilteredData()

        # instantiate recording object
        self.recording = si.NumpyRecording(traces_list = self.data.T, sampling_frequency = float(self.samplingFrequency))

        # set channel map
        
        # ProbeMapPath defines the physical location of each channel along the probe shank
        probeMapPath = str(self.dataset['probeMapPath']) 
        
        # channelMapPath defines which physical channel is represented by each index of our data matrix
        recordingChannelPath = str(self.dataset['channelMapPath'])
        map_available = os.path.exists(probeMapPath) and os.path.exists(recordingChannelPath)
        # load probe info if available in the sheet
        if map_available:
            print("Channel map available. Loading probe geometry")
            print(f"Probe File: {probeMapPath}")
            recording_channel_ids = np.loadtxt(recordingChannelPath)

            # limit by ChanList from sheet if we are exluding channels:
            subset_channel_ids = recording_channel_ids[self.dataChanList]
            recording_channel_ids = [int(id) for id in subset_channel_ids]
            print(f"Recording Channels: {recording_channel_ids}")

            # probeMapPath points to .prb file
            probe_group_data = read_prb(probeMapPath)
            probe_data = ProbeGroup()   # probe data is first stored as a ProbeGroup object in the memory
            for probe in probe_group_data.probes:
                probe_data.add_probe(probe.get_slice(recording_channel_ids))
                probe_data.set_global_device_channel_indices(list(range(len(recording_channel_ids))))   # match global ids to each channel
            
            # set probe info to recording object
            self.recording = self.recording.set_probegroup(probe_data, group_mode='by_probe')
        # assume linear arrangement for channel locations if probe info is unavailable in the sheet
        else:
            print('Channel and probe map not found. Assuming linear arrangement')
            probe_data = ProbeGroup()
            probe = Probe()
            probe = generate_linear_probe(num_elec=self.recording.get_num_channels(), ypitch=50)
            probe_data.add_probe(probe)
            probe_data.set_global_device_channel_indices(list(range(self.recording.get_num_channels())))
            self.recording = self.recording.set_probegroup(probe_data, group_mode='by_probe')
            

        # mountainsort bandpass removed on 1-28, bandpassing is done in adaptiveFiltering.py
        self.recording_cmr = st.preprocessing.common_reference(self.recording, reference='global', operator='median')

        # print relevant info the user
        print('Num. channels = {}'.format(len(self.recording.get_channel_ids())))
        print('Sampling frequency = {} Hz'.format(self.recording.get_sampling_frequency()))
        print('Num. timepoints = {}'.format(self.recording.get_num_frames()))
        print('Stdev. on channel = {}'.format(np.std(self.recording.get_traces(segment_index=0, channel_ids = [0]))))
        
        self.saveRecordingCMR()
        
        return self.recording_cmr
    
    def saveRecordingCMR(self):
        '''
        Saves calculated self.recording_cmr (automatically calls self.deriveRecordingData() if necessary)

            Return:
                None
        '''
        print("Computing recording_cmr...")
        if self.recording_cmr is None:
            self.deriveRecordingData()
        
        recording_cmr_file = self.cacheFilePath + self.prominenceExt + self.exportFileNames['recording_cmr_filename']

        # this file needs to be saved as both .npy and .mat
        # or (os.path.exists(recording_cmr_file + ".mat") == False)
        if (os.path.exists(recording_cmr_file + ".npy") == False):
            raw_array = self.recording_cmr.get_traces()
            export_dict = {"voltages": raw_array}
            # Saves recording_cmr_arrays if not already saved
            savePickleFile(raw_array, recording_cmr_file) #save as numpy
            # saveMatfile(export_dict, recording_cmr_file) #save as matlab file
        return None

    def loadMountainSort(self, saveOnly = False, overide = False):
        '''
        Runs MountainSort or loads saved MountainSort run from cache

            Return:
                None
        '''
        MS4fileExists = os.path.exists(self.ms4FilePath + self.ms4JSONfileName)

        if (MS4fileExists == False) or (overide == True):
            # Run MountainSort
            ms4_params = ss.get_default_params('mountainsort4')
            ms4_params['detect_threshold'] = self.sortingThreshold
            ms4_params['detect_sign'] = self.detect_sign
            if self.recording_cmr is None:
                self.deriveRecordingData()
            
            # deal with "overide" flag
            
            # if overide is True or it is the first time running MountainSort, save the recording object to self.cacheFilePath (note that the directory must not exist for save() method, so we should remove directory if cacheFilePath already exists.)
            if os.path.exists(self.cacheFilePath) == True and (overide == True or os.path.isfile("provenance.json") == False):
                shutil.rmtree(self.cacheFilePath)
                recording_saved = self.recording_cmr.save(folder=self.cacheFilePath, progress_bar=True)
            if os.path.exists(self.cacheFilePath) == False:
                recording_saved = self.recording_cmr.save(folder=self.cacheFilePath, progress_bar=True)
            # load the recording object from cache
            print('cache file path is: ', self.cacheFilePath)
            recording_loaded = si.load_extractor(self.cacheFilePath)

            print("Running MountainSort...")
            self.sorting_MS4 = ss.run_mountainsort4(recording=recording_loaded, output_folder = self.ms4FilePath, verbose=True, **ms4_params)

            print("Saving MountainSort Results as Json")
            self.sorting_MS4.dump_to_json(file_path= (self.ms4FilePath + self.ms4JSONfileName))
                        
        else:
            if self.recording_cmr is None:
                self.deriveRecordingData()
            if saveOnly == True:
                return None
            # Load Previously Saved MountainSort
            print("Loading MountainSort Results from JSON...")
            self.sorting_MS4 = si.load_extractor(self.ms4FilePath + self.ms4JSONfileName)
        
        print('Units found by Mountainsort4:', self.sorting_MS4.get_unit_ids())
        self.units = self.sorting_MS4.get_unit_ids() 

        # Extract Spike timings
        
        self.times = []
        self.labels = []
        self.spikes = []
        for i in self.units:
            for j in self.sorting_MS4.get_unit_spike_train(i):
                self.times.append(j)
                self.labels.append(i)
                self.spikes.append((j, i))
        
        return None
    
    def loadVariables(self, saveOnly = False, overide = False):
        '''
        Extracts variables/information from MountainSort result, either recalculates and saves or reloads
        - automatically calls self.loadMountainSort() if necessary

            Return:
                None
        '''
        loadedVariables = loadProcessedInfo(self.exportFileNames, self.fullFilePath) #full file path includes cache/prominence/threshold
        # Exports other data if not saved already
        if (loadedVariables == {}) or (overide == True):
            if self.sorting_MS4 is None:
                self.loadMountainSort()
                # this will automatically load sorting_MS4 and recording_cmr if not already loaded
            ## Get other data:
            export_data = {}

            self.loaded_recording_cmr = si.load_extractor(self.cacheFilePath)
            self.loaded_recording_cmr.annotate(is_filtered=True)
            
            self.waveformCachePath = self.cacheFilePath + "_waveforms"

            # instantiate waveform object
            self.we = si.extract_waveforms(self.loaded_recording_cmr, self.sorting_MS4, self.waveformCachePath, ms_before = 2, ms_after = 2, max_spikes_per_unit = 10000, overwrite=True)

            # calculate pca scores
            self.pca = st.postprocessing.compute_principal_components(self.we, load_if_exists=True, n_components=3, mode='by_channel_local')
            scores = [self.pca.get_projections(unit_id = unit) for unit in self.sorting_MS4.get_unit_ids() ]
            self.pca_scores = scores
            export_data['pca_scores'] = scores #(units, times, PCA_Dims)

            # calculate amplitydes
            self.amplitudes = st.postprocessing.get_template_amplitudes(self.we)
            export_data['amplitudes']= self.amplitudes # Amplitudes is list (uV) for each unit

            # store and export waveforms
            self.waveforms = []
            for unit_id in self.sorting_MS4.get_unit_ids():
                waveform_matrix = self.we.get_waveforms(unit_id)
                
                # flip indices
                waveform_matrix_T = waveform_matrix.transpose((0, 2, 1))

                self.waveforms.append(waveform_matrix_T)

            export_data['waveforms']= self.waveforms

            self.spikes2 = np.array([self.sorting_MS4.get_unit_spike_train(unit) for unit in self.sorting_MS4.get_unit_ids()], dtype=object)
            export_data['spikes2'] = self.spikes2 #list of timestamps for each unit in samples. Should be divided by sampling frequency

            export_data['spikes'] = self.spikes
            # Save computed data for next time
            print("Saving Variables for Spike Prominence: ", self.spikeProminence)
            saveProcessedInfo(export_data, self.exportFileNames, self.cacheFilePath, self.spikeProminence, self.sortingThreshold, self.fullFilePath)
        else:
            # Load previously computed data
            print("Loading Variables for Spike Prominence: ", self.spikeProminence)
            self.pca_scores = loadedVariables['pca_scores']
            self.amplitudes = loadedVariables['amplitudes']
            self.waveforms = loadedVariables['waveforms']
            self.spikes2 = loadedVariables['spikes2']
            self.spikes = loadedVariables['spikes']
            self.units = list(range(1, len(self.waveforms) + 1))
            print("Asserting number of waveforms match")
            assert(len(self.waveforms) == len(self.units))
        return None