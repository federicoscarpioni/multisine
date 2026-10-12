import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy.fft import fft,fftshift,fftfreq
from scipy.interpolate import CubicSpline, UnivariateSpline
import os
import json
from galvani import BioLogic
import pandas as pd

from .phase_optim import optimize_phases_crest_continuation
from .harmonics_gen import (
    odd_quasilog, odd_random_grid, unexcited_odd_harmonics, describe_grid,
)




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
    cubic_spline = CubicSpline(frequencies_experiment, 
                               amplitude_experiment)
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


def load_multisine(path: str):
    with open(path+'/waveform_metadata.json') as file:
        data = json.load(file)
    ms = Multisine(
        data['Generation frequency / Hz'],
        np.array(data['Frequencies / Hz']),
        np.array(data['Amplitudes / mV']),
        np.array(data['Phases / rad']),
    )
    ms.is_normalized = True
    ms.phase_policy = data.get('Phase policy', 'unknown')
    if 'Harmonics' in data:
        ms.harmonics = np.array(data['Harmonics'], dtype=int)
    ms.orp = data.get('ORP design')
    return ms

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
        self.period = int(self.number_points/self.sampling_frequency)
        self.harmonics = None
        self.phase_policy = 'given' if phases is not None else 'zero'
        self.orp = None
        print(f"Multisine generated. Crest factor: {self.cf:.4}")


    @classmethod
    def generate_orp(cls, sampling_frequency, f0, h_max, amplitude=1.0,
                     L=5, harmonics=None, seed=None, number_points=None):
        '''
        Build an odd random phase multisine, ready for BLTVA analysis.

        Bundles the three things the method needs together, because they are
        only correct together:

        1. an odd-only harmonic grid, so even-order distortion lands on empty
           even harmonics;
        2. odd harmonics deliberately left unexcited, so odd-order distortion
           has somewhere to appear;
        3. phases drawn uniformly at random, which is what makes the best
           linear approximation well defined and the distortion estimate
           meaningful.

        Parameters
        ----------
        sampling_frequency : float
            Generation rate in Hz.
        f0 : float
            Fundamental, i.e. 1/period. Harmonic h sits at ``h * f0``.
        h_max : int
            Highest harmonic to excite.
        amplitude : float or array
            Per-tone amplitude. A scalar is applied to every tone.
        L : int
            Group size for the dense construction: one odd harmonic in every
            L is dropped (Hallemans TIM 2021, Definition 1). Ignored when
            ``harmonics`` is given.
        harmonics : array of int, optional
            Use this grid instead of building a dense one -- for a sparse
            quasi-logarithmic grid from :func:`odd_quasilog`, say. On a sparse
            grid no omission is needed: almost every odd harmonic is already
            unexcited, which serves the same purpose as the L-grouping.
        seed : int, optional
            Seeds both the harmonic omission and the phases, so the whole
            excitation is reproducible from this one number.

        Warning
        -------
        Do **not** follow this with :meth:`optimize_crest_factor` or
        :meth:`best_random_phases` if you intend to estimate nonlinear
        distortion. Both replace the phases with a set chosen for low crest
        factor, which is no longer a uniform random draw, and the theory
        behind the distortion estimate no longer holds. Nothing downstream can
        detect this from the recorded data -- the distortion variances would
        simply be wrong rather than absent. Choose one: a low crest factor, or
        a trustworthy distortion estimate.
        '''
        rng = np.random.default_rng(seed)
        dense = harmonics is None
        if dense:
            harmonics = odd_random_grid(h_max, L=L, seed=seed)
        else:
            harmonics = np.asarray(harmonics, dtype=int)
            harmonics = harmonics[harmonics <= h_max]

        report = describe_grid(harmonics)
        if not report['supports_distortion_estimate']:
            raise ValueError(
                f'this grid cannot support a distortion estimate: {report}. '
                'Excite odd harmonics only, and leave some of them unexcited.')

        frequencies = harmonics * float(f0)
        if np.ndim(amplitude) == 0:
            amplitudes = np.full(frequencies.size, float(amplitude))
        else:
            amplitudes = np.asarray(amplitude, dtype=float)
        phases = rng.uniform(0, 2 * np.pi, frequencies.size)

        obj = cls(sampling_frequency, frequencies, amplitudes, phases,
                  number_points)
        obj.harmonics = harmonics
        obj.phase_policy = 'uniform_random'
        obj.orp = {'grid': 'dense_L_grouped' if dense else 'supplied',
                   'L': L if dense else None, 'seed': seed, **report}
        return obj

    def compute_crest_factor(self):
        self.cf = compute_crest_factor(self.waveform)

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
        print('Randomizing phase for minimum crest factor...')
        for i in range(iteration):
            new_phases = self.random_phases()
            new_waveform = self.compute_multisine(new_phases)
            new_cf = compute_crest_factor(new_waveform)
            print(f"Calculated from random phases: {new_cf:.4}")
            if new_cf < self.cf:
                self.phases = new_phases
                self.waveform = new_waveform
                self.cf = new_cf
                self._phases_no_longer_random('best_of_random')
                print(f"Current minimum: {self.cf:.4}")
        print(f"Current minimum crest factore: {self.cf:.4}")


    def optimize_crest_factor(self, p_schedule=(2, 4, 8, 16, 32), **minimize_kwargs):
        '''
        Minimize crest factor by optimizing phases, starting from a
        random phase guess. Uses a smooth Lp-norm surrogate for the peak
        (see phase_optim.py), with p ramped up over p_schedule.
        '''
        print('Optimizing phases for minimum crest factor...')
        phase0 = self.random_phases()
        result = optimize_phases_crest_continuation(
            self.frequencies, self.amplitudes, phase0,
            self.sampling_frequency, self.number_points,
            p_schedule=p_schedule, **minimize_kwargs,
        )
        new_phases = result.x
        new_waveform = self.compute_multisine(new_phases)
        new_cf = compute_crest_factor(new_waveform)
        print(f"Calculated from optimized phases: {new_cf:.4}")
        if new_cf < self.cf:
            self.phases = new_phases
            self.waveform = new_waveform
            self.cf = new_cf
            self._phases_no_longer_random('crest_optimized')
        print(f"Current minimum crest factor: {self.cf:.4}")


    def _phases_no_longer_random(self, policy):
        '''
        Record that the phases are no longer a uniform random draw.

        The distortion estimate of an ORP measurement rests on the phases
        being drawn independently and uniformly, so that E{exp(j*phi)} = 0.
        Choosing phases for a low crest factor breaks that, and no downstream
        tool can detect it from the recorded waveform. Saying so in the
        metadata is the only way the information survives to analysis time.
        '''
        was_orp = self.phase_policy == 'uniform_random'
        self.phase_policy = policy
        if was_orp:
            print('Note: phases are no longer a uniform random draw, so this '
                  'is no longer an ORP excitation. Nonlinear distortion '
                  'estimates from it will not be trustworthy.')


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
        plt.ylabel('Phase / \pi')
    
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
        plt.ylabel('Phase / \pi')
        
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
        metadata = {'Period / s' : self.period,
                    'Generation frequency / Hz': self.sampling_frequency,
                    'Lowest frequency / Hz': self.frequencies[0],
                    'Highest frequency / Hz': self.frequencies[-1],
                    'Number of frequencies' : self.frequencies.size,
                    'Crest factor': self.cf,
                    'Number of points': self.waveform.size,
                    'Frequencies / Hz' : self.frequencies.tolist(),
                    'Amplitudes / mV' : self.amplitudes.tolist(),
                    'Phases / rad' : self.phases.tolist()}
        if self.is_normalized:
            metadata['Normalized'] = 'Yes'
            metadata['Normalization factor'] = self.normalization_factor
        else:
            metadata['Normalized'] = 'No'
        # Provenance of the excitation design. The harmonic numbers and the
        # phase policy are what an analysis tool needs in order to know
        # whether a nonlinear distortion estimate is defensible, and neither
        # can be recovered by inspecting the waveform after the fact.
        metadata['Phase policy'] = self.phase_policy
        if self.harmonics is not None:
            metadata['Harmonics'] = np.asarray(self.harmonics).tolist()
        if self.orp is not None:
            metadata['ORP design'] = self.orp
        # Save to json file
        with open(name + '/waveform_metadata.json', 'w') as jfile:
            json.dump(metadata, jfile)
        
    