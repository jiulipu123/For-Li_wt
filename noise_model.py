import h5py
import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
'''
actual noise
'''
file_path=r"C:\Users\86130\Desktop\LDC2_sangria_training_v2.h5"
def extract_obs_data(file_path):
    obs_data={}
    with h5py.File(file_path,'r') as f:
        obs_tdi=f['obs/tdi'][:]
        if obs_tdi.ndim == 2 and obs_tdi.shape[1] == 1:
            obs_tdi=obs_tdi.flatten()
        obs_data['obs_time']=obs_tdi['t']
        obs_data['obs_X']=obs_tdi['X']
        obs_data['obs_Y']=obs_tdi['Y']
        obs_data['obs_Z']=obs_tdi['Z']
        obs_data['obs_A']=(obs_tdi['Z']-obs_tdi['X'])/np.sqrt(2)
        obs_data['obs_E']=(obs_tdi['X']-2*obs_tdi['Y']+obs_tdi['Z'])/np.sqrt(6)
        obs_data['obs_dt']=obs_tdi['t'][1]-obs_tdi['t'][0]
        N=len(obs_data['obs_time'])
        obs_data['obs_frequency']=np.fft.fftfreq(N,obs_data['obs_dt'])
        obs_data['obs_df']=obs_data['obs_frequency'][1]-obs_data['obs_frequency'][0]
        obs_data['obs_X_f']=np.fft.fft(obs_data['obs_X'])
        obs_data['obs_Y_f']=np.fft.fft(obs_data['obs_Y'])
        obs_data['obs_Z_f']=np.fft.fft(obs_data['obs_Z'])
        obs_data['obs_A_f']=np.fft.fft(obs_data['obs_A'])
        obs_data['obs_E_f']=np.fft.fft(obs_data['obs_E'])
    return obs_data

def extract_all_sky_signals(file_path):
    sky_signals={}
    sources=['dgb','igb','mbhb','vgb']
    with h5py.File(file_path,'r') as f:
        for source in sources:
            tdi_path=f'sky/{source}/tdi'
            if tdi_path in f:
                tdi_data=f[tdi_path][:]
                if tdi_data.ndim==2 and tdi_data.shape[1]==1:
                    tdi_data=tdi_data.flatten()
                sky_signals[source]=tdi_data['X']
    return sky_signals

def calculate_data_noise(obs_data,sky_signals):
    total_sky_signal_X=np.zeros_like(obs_data['obs_X'])
    for source_name,signal_data in sky_signals.items():
        total_sky_signal_X+=signal_data
    instrument_noise_t=obs_data['obs_X']-total_sky_signal_X
    return instrument_noise_t

def calculate_galactic_confusion_noise(file_path):
    with h5py.File(file_path,'r') as f:
        dgb_tdi=f['sky/dgb/tdi'][:]
        if dgb_tdi.ndim==2 and dgb_tdi.shape[1]==1:
            dgb_tdi.flatten()
        igb_tdi=f['sky/igb/tdi'][:]
        if igb_tdi.ndim and igb_tdi.shape[1]==1:
            igb_tdi=igb_tdi.flatten()
        galactic_confusion=dgb_tdi['X']+igb_tdi['X']
    return galactic_confusion

def calculate_noise_psd(obs_data,instrument_noise,galactic_confusion=None):
    if galactic_confusion is not None:
        noise=instrument_noise+galactic_confusion
    else:
        noise=instrument_noise

    fs=1/obs_data['obs_dt']
    nperseg=8192
    noverlap=nperseg//2
    freqs,S_n=signal.welch(
        noise,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant',
        scaling='density',
        average='median'
    )
    return freqs,S_n

obs_data=extract_obs_data(file_path)
sky_signals=extract_all_sky_signals(file_path)
instrument_noise_t=calculate_data_noise(obs_data,sky_signals)
#galactic_confusion=calculate_galactic_confusion_noise(file_path)
freqs,S_n=calculate_noise_psd(obs_data,instrument_noise_t)
plt.figure(figsize=(12,8))
plt.loglog(freqs,S_n,'r')
plt.show()