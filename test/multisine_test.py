from multisine.multisine import Multisine, compute_crest_factor
import numpy as np

harmonics = np.loadtxt('data/harmonics_8dec_quasi-log-8pts_no_intermod_second.txt')
base_frequency = 1
max_frequency = 1
frequencies = harmonics * base_frequency

# Remove higher frequencies
frequencies = frequencies[0:np.where(frequencies>max_frequency)[0][0]]
frequencies[-1] = max_frequency

amplitudes = np.ones(frequencies.size)*1  # V
sampling_frequency = 10000
ms1 = Multisine(sampling_frequency, frequencies, amplitudes,)
ms1.best_random_phases(500)
ms1.normalize_waveform(int(2**16 /2 - 1))
ms1.plot('voltage')
ms1.fourier_analysis(10)
ms1.plot_dft((0.001,sampling_frequency//2))
ms1.fourier_analysis(10)
ms1.plot_dft((0.001,sampling_frequency//2))
# ms1.plot_phase((0.1,250))
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
ms1.save('C:/multisine_collection/2412161033_signle_sine_1Hz')

#%% Split the multisine in two
splitting_index = 28
phases = ms1.phases

msfirst = Multisine(1000,frequencies[0:splitting_index], amplitudes[0:splitting_index], phases[0:splitting_index])
msfirst.plot('voltage')
msfirst.fourier_analysis(6)
msfirst.plot_dft((0.001,sampling_frequency//2))
msfirst.normalize_waveform()


base_frequency = 100
max_frequency = 100000
frequencies_high = harmonics * base_frequency 
frequencies_high = frequencies_high[0:np.where(frequencies_high>max_frequency)[0][0]]
frequencies_high[-1] = max_frequency
mssecond = Multisine(1000000,frequencies_high, amplitudes[splitting_index:splitting_index+20], phases[splitting_index:splitting_index+20],10000)
mssecond.plot('voltage')
mssecond.fourier_analysis(6)
mssecond.plot_dft((0.001,1000000//2))
mssecond.normalize_waveform()
msfirst.save('C:/multisine_collection/2412111607multisine_splitted_100kHz-10mHz_8ptd_fgen1MHz_flat_norm_random_phases/high_freqs/')
mssecond.save('C:/multisine_collection/2412111607multisine_splitted_100kHz-10mHz_8ptd_fgen1MHz_flat_norm_random_phases/low_freqs/')

#%% Recompute the full multisine to check the properties

perdiodfirst = 100
periodsecond = 1

mstotal = np.tile(mssecond.waveform, int(perdiodfirst/periodsecond))
for i in range(0,msfirst.waveform.size):
    mstotal[i*10000] = mstotal[i*10000] + msfirst.waveform[i]

cf_mstotal = compute_crest_factor(mstotal)
cf_mstotal
