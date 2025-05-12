import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy.fft import fft,fftshift,fftfreq
from scipy.interpolate import CubicSpline
import os
import json
from galvani import BioLogic
import pandas as pd


# Auxiliary functions
def compute_crest_factor(signal):
    return max(abs(signal)) / np.sqrt(np.mean(signal**2)) 


def load_eclab_impedance(file_name):
    mpr_file = BioLogic.MPRfile(file_name)
    df = pd.DataFrame(mpr_file.data)
    return df

def amplitude_extraction_from_experiment(impedance_experiment, 
                                         frequencies_experiment, 
                                         frequencies_desired):
    amplitude_experiment = np.absolute(impedance_experiment)
    cubic_spline = CubicSpline(np.flip(frequencies_experiment), 
                               np.flip(amplitude_experiment))
    return cubic_spline(frequencies_desired)

def show_extracted_amplitude_from_experiment(impedance_experiment, 
                                         frequencies_experiment, 
                                         frequencies_desired):
    amplitude_extracted = amplitude_extraction_from_experiment(impedance_experiment, 
                                                               frequencies_experiment,
                                                               frequencies_desired)
    fig, ax = plt.subplots()
    ax.plot(frequencies_experiment, np.absolute(impedance_experiment), '-v', label = 'experiment')
    ax.scatter(frequencies_desired, amplitude_extracted, label = 'interpolated', color = 'orange', zorder=2)
    ax.set_xscale('log')
    fig.legend()
    ax.set_xlabel('Frequency / Hz')
    ax.set_ylabel('Module of Z')
    

class Multisine:
    
    def __init__(self, sampling_frequency, frequencies, amplitudes, phases = None, number_points=None):
        self.sampling_frequency = sampling_frequency
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        if phases is None:
            self.phases = np.zeros(frequencies.size)
        else:
            self.phases = phases
        if number_points is None:
            self.number_points = self.sampling_frequency/self.frequencies[0]
        else:
            self.number_points = number_points
        self.waveform = self.compute_multisine(self.phases)
        self.cf = compute_crest_factor(self.waveform)
        self.is_normalized = False
        print(f"Multisine generated. Crest factor: {self.cf:.4}")
    

    def random_phases(self):
        return np.random.random(self.frequencies.size) * 2 * np.pi

    
    def compute_multisine(self, phases, time_step = None):
        # Time array
        
        self.time = np.arange(self.number_points) / self.sampling_frequency
        # Compute multisine
        multisine = np.zeros(self.time.size)
        for i in range(0, self.frequencies.size):
            multisine += self.amplitudes[i] * np.sin(
                2 * np.pi * self.frequencies[i] * self.time + phases[i]
                )
        self.time_step = multisine.size/self.sampling_frequency
        return multisine
    
    
    def normalize_waveform(self, factor =1):
        self.waveform = np.divide(self.waveform, max(abs(self.waveform)))
        self.waveform = np.multiply(self.waveform, factor)
        self.is_normalized = True
        self.normalization_factor = factor
            

    def best_random_phases(self, iteration):
        print('Randomizing phase for minimu crest factor...')
        for i in range(iteration):
            new_phases = self.random_phases()
            new_waveform = self.compute_multisine(new_phases)
            new_cf = compute_crest_factor(new_waveform)
            print(f"Calculated from random phases: {new_cf:.4}")
            if new_cf < self.cf:
                self.phases = new_phases
                self.waveform = new_waveform
                self.cf = new_cf
                print(f"Current minimum: {self.cf:.4}")
        print(f"Current minimum crest factore: {self.cf:.4}")
                
                

    def plot(self, signal_type):
        if signal_type == 'current':
            signal_label = 'Current/A'
        elif signal_type == 'voltage':
            signal_label = 'Voltage/V'
        plt.figure()
        plt.plot(self.time, self.waveform)
        plt.xlabel('Time /s')
        plt.ylabel(f'{signal_label}')
        
    def fourier_analysis(self, repetitions = 1):
        extendend_waveform = np.repeat(self.waveform, repetitions , axis = None)
        self.fft_waveform = fftshift(fft(self.waveform)/extendend_waveform.size)
        self.freq_axis = fftshift(fftfreq(self.fft_waveform.size, 1/self.sampling_frequency))
        
    
    def plot_dft(self, freq_range):
        index_f0 = np.where(self.freq_axis == 0)[0][0]
        df = self.sampling_frequency / self.fft_waveform.size
        index_start = int(freq_range[0] / df) + index_f0
        index_end = int(freq_range[1] / df) + index_f0
        plt.figure()
        plt.vlines(self.frequencies, 0, max(abs(self.fft_waveform)), colors= 'green', label='Ideal frequencies')
        plt.plot(self.freq_axis[index_start:index_end], np.abs(self.fft_waveform[index_start:index_end]), 'o', label = 'Multisine')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Frequency / Hz')
        plt.ylabel('Magnitude / V')
    
    def plot_phase(self, freq_range, tollerance = 1e-6):
        # Clean the matematical error
        fft_clean = np.copy(self.fft_waveform)
        fft_clean[np.where(fft_clean < tollerance)] = 0
        #
        index_f0 = np.where(self.freq_axis == 0)[0][0]
        df = self.sampling_frequency / self.fft_waveform.size
        index_start = int(freq_range[0] / df) + index_f0
        index_end = int(freq_range[1] / df) + index_f0
        plt.figure()
        plt.vlines(self.frequencies, -1, 1, colors= 'green', label='Ideal frequencies')
        plt.plot(self.freq_axis[index_start:index_end], np.angle(fft_clean[index_start:index_end])/np.pi, 'o', label = 'Multisine')
        plt.legend()
        # plt.yscale('log')
        plt.xscale('log')
        plt.ylim((-1,1))
        plt.xlabel('Frequency / Hz')
        plt.ylabel('Phase / $\pi$')
    
    def plot_phase_full(self, tollerance =1e-6):
        # Clean the matematical error
        fft_clean = np.copy(self.fft_waveform)
        fft_clean[np.where(fft_clean < tollerance)] = 0
        plt.figure()
        plt.vlines(self.frequencies,-1, 1, colors= 'green', label='Ideal frequencies')
        plt.plot(self.freq_axis, np.angle(fft_clean)/np.pi, 'o', label = 'Multisine')
        plt.legend()
        # plt.yscale('log')
        plt.ylim((-1,1))
        plt.xlabel('Frequency / Hz')
        plt.ylabel('Phase / $\pi$')
        
    def plot_phase2(self):
        plt.figure()
        plt.phase_spectrum(self.waveform)
    
    def scale_waveform(self, oscillation_pp, offset=0):
        '''Scale the normalzide oscilattion to a spacific value and optionally
        apply an offset. Specify values in V or A!'''
        self.waveform = self.waveform * oscillation_pp
        self.waveform = self.waveform + offset
        

    def save(self, name):
        '''
        Save waveform and respective metadata in a dedicated folder.
        '''
        # Create a new folder if needed
        if not os.path.exists(name):
            os.makedirs(name)
        # Save
        self.save_waveform(name)
        self.save_metadata(name)
        

    def save_waveform(self, name):
        with open(name+'/waveform.txt', 'w') as file:
            np.savetxt(file, self.waveform)
    

    def save_BL_UP(self, waveform_type, name):
        def save(time, waveform, name, waveform_type):
            if waveform_type == 'current':
                waveform_header = 'Current/A'
            elif waveform_type == 'voltage':
                waveform_header = 'Voltage/V'
            np.savetxt(name, 
                       np.stack((time, waveform), axis=1), 
                       header = 'Time/s\t'+waveform_header,
                       comments='',
                       fmt= '%.6f')
        # Split the waveform in chanks of 4000 samples and save each of them
        max_size = 4000
        num_files = ceil(self.waveform.size/max_size)
        start_index = 0
        for i in range(0, num_files):
            start_index = max_size * i
            end_index = max_size * (i+1)
            if i == num_files-1:
                end_index = self.waveform.size             
            save(self.time[start_index:end_index]-self.time[start_index], 
                 self.waveform[start_index:end_index],
                 f'{name}_part{i+1}.txt',
                 waveform_type)
            
            
    def save_BL_PI(self, waveform_type, name):
        if waveform_type == 'current':
            waveform_header = 'Current/A'
        elif waveform_type == 'voltage':
            waveform_header = 'Voltage/V'
        np.savetxt(name, 
                   np.stack((self.time, self.waveform), axis=1), 
                   header = 'Time/s\t'+waveform_header,
                   comments='',
                   fmt= '%.6f')

    
    def save_metadata(self, name):
        '''
        Save metadata as a dictionary in a json file.
        '''
        # Create a dictionary with all metadata
        metadata = {'Multisine period / s' : 1/self.frequencies[0],
                    'Sample frequency / Hz': self.sampling_frequency,
                    'Lower frequency / Hz': self.frequencies[0],
                    'Higher frequency / Hz': self.frequencies[-1],
                    'Number of frequencies' : self.frequencies.size,
                    'Crest factor': self.cf,
                    'Frequencies / Hz' : self.frequencies.tolist(),
                    'Amplitudes / mV' : self.amplitudes.tolist(),
                    'Phases / rad' : self.phases.tolist()}
        if self.is_normalized: 
            metadata['Normalized'] = 'Yes' 
            metadata['Normalization factor'] = self.normalization_factor
        else: 
            metadata['Normalized'] = 'No' 
        # Save to json file
        with open(name + '/waveform_metadata.json', 'w') as jfile:
            json.dump(metadata, jfile)
        
    