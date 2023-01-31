# -*- coding: utf-8 -*-
# ECG AFib Detection Functions

#initializing
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from matplotlib import pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from skimage.measure import shannon_entropy
from sklearn.decomposition import PCA
import pickle
import pandas as pd


def my_metric(y_true, y_pred):
    """
    Metric for training phase of model that multiply precision by recall value

    Parameters:
    y_true (tensor): ground truth labels
    y_pred (tensor): predicted labels

    Returns: 
    multiplication of precision and recall
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())

    return precision_keras*recall_keras
#end my_metric

def autocorr(x):
    '''
        Computes Auto Correlation of signal x

        Parameters:
        x (array): Signal.

        Returns: 
        acorr (array): Autocorrelation of input signal.
    '''
    
    #calculating correlation
    norm = x - np.mean(x)
    result = np.correlate(norm, norm, mode='full')
    acorr = result[int(result.size/2)-30:-30]
    # acorr /= ( x.var() * np.arange(int(x.size), 0, -1) )
    acorr /= max(acorr)

    return acorr
#end autocorr

def autocorr_diff_peak_periodicity_detetction(sig_filtered, fc, plot_status=False):
    '''
        Computes periodicity of signal.
        Return an periodicity percentage [0-1].
        * 0 = non-periodic.
        * 1 = periodic.

        Parameters:
        sig_filtered (array): Signal.
        fc (float): Sampling frequency of signal.
        plot_status (boolean): Show Autocorrelation plot

        Returns: 
        periodicity_percentage (float): periodicity percentage of signal.
        peaks (array): Autocorrelation peaks location.
        sig_autocor (array): Autocorrelation signal.
    '''

    #constants
    height_thres = 0  #threshold
    max_heartrate = 180
    

    #variables
    periodicity_status = 0
    periodicity_found = []

    threshold = fc * 0.13  # based on normal heart rate variabality < 130 ms
    min_distance =  int(fc/(max_heartrate/60))
    min_period_length = 3
    num_of_sections = 1

    period_length = len(sig_filtered)/fc

    if period_length > min_period_length:
        num_of_sections = int(period_length/3)
    #end if

    for s in range(num_of_sections):
        sig_autocor = autocorr(sig_filtered[s*int((period_length/num_of_sections)*fc):(s+1)*int((period_length/num_of_sections)*fc)])
        min_height = np.percentile(sig_autocor,75)
        peaks, properties = find_peaks(sig_autocor, height=min_height ,prominence=0.15, distance = min_distance)
        if len(peaks)>=3:
            maxdiff = max(abs(np.diff(np.diff(peaks)))) # ~ max HRV
            # print(maxdiff)
            if maxdiff<threshold:
                periodicity_found.append(1)
            #end if
        else:
            periodicity_found.append(0)
        #end if

        if plot_status:
            #plotting
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(25, 8))
            title = 'Periodicity Detection using autocorrelation, Samples ' + str(s*int((period_length/num_of_sections)*fc)) + '-' + str((s+1)*int((period_length/num_of_sections)*fc))
            plt.title(title)
            plt.plot(sig_autocor, label='Autocorrelation')
            plt.scatter(peaks, sig_autocor[peaks], marker = 'x', s=80, c='red', Label='Peaks')
            plt.legend()
            
        #end if

    #end for

    periodicity_percentage = sum(periodicity_found)/num_of_sections

    # print(periodicity_status)

    return periodicity_percentage, peaks, sig_autocor
#end autocorr_diff_peak_periodicity_detetction

def statistical_feature(sig, fs):
    '''
        Computes statistical features of signal.
        features:
        * periodicity percentage
        * skewness
        * kurtosis
        * shannon entropy
        * mean
        * variance

        Parameters:
        sig (array): Signal.
        fs (float): Sampling frequency of signal.

        Returns: 
        (list): calculated features from the original and the autocorrelation signals.
    '''

    #cleaning the ECG signal
    ecg_cleaned = nk.ecg_clean(sig, sampling_rate=fs)
    
    #calculating features
    periodicity_percentage, peaks, sig_autocor = autocorr_diff_peak_periodicity_detetction(ecg_cleaned, fs)
    skewness = stats.skew(sig_autocor)
    kurt = stats.kurtosis(sig_autocor)
    shannon_ecg = shannon_entropy(ecg_cleaned)
    shannon_auto = shannon_entropy(sig_autocor)
    mean_auto = np.mean(sig_autocor)
    var_auto = np.var(sig_autocor)

    return [periodicity_percentage, skewness, kurt, shannon_ecg, shannon_auto, mean_auto, var_auto]
#end statistical_feature

def HRV_calc(sig, fs, plot=False):
    '''
        Calculating HRV features for each segment of signal.

        Parameters:
        sig (array): ECG Signal.
        fs (float): Sampling frequency of signal.

        Returns: 
        hrv_indices (list): calculated HRV features from the signals.
    '''

    #Cleaning the ECG signal and Finding peaks
    ecg_cleaned = nk.ecg_clean(sig, sampling_rate=fs)
    ecg_cleaned = nk.signal_filter(ecg_cleaned, highcut=70, method="butterworth",order=5)

    # R-R peaks detection
    _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, correct_artifacts=True, method='neurokit')
    rr1 = info['ECG_R_Peaks']
    diff_mean_peaks1 = abs(np.mean(ecg_cleaned)-np.mean(ecg_cleaned[rr1]))

    peaks, info = nk.ecg_peaks(-ecg_cleaned, sampling_rate=fs, correct_artifacts=True, method='neurokit')
    rr2 = info['ECG_R_Peaks']
    diff_mean_peaks2 = abs(np.mean(-ecg_cleaned)-np.mean(ecg_cleaned[rr2]))

    if diff_mean_peaks1 < diff_mean_peaks2:
        sig = -sig
    #end if

    # Calculating HRV Data for each segment
    # Clean signal and Find peaks
    ecg_cleaned = nk.ecg_clean(sig, sampling_rate=fs)
    ecg_cleaned = nk.signal_filter(ecg_cleaned, highcut=70, method="butterworth",order=5)
    peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, correct_artifacts=True, method='neurokit')
    # peaks = find_peaks(ecg_cleaned, height=100, threshold=None, distance=rate/4, prominence=50, wlen=None, plateau_size=None)[0]
    # Compute HRV indices 
    peaks = info['ECG_R_Peaks']
    hrv_indices = nk.hrv(peaks, sampling_rate=fs, show=False)

    if plot:
        #plotting
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(25, 8))
        plt.plot(ecg_cleaned, label='ECG Filtered')
        plt.scatter(peaks, ecg_cleaned[peaks], marker = 'x', s=80, c='red', Label='Peaks')
        plt.legend()
    #end if

    return hrv_indices
#end HRV_calc

def AFib_Model_Input_Features(sig, fs, verbose=False, plot=False):
    '''
        Calculating AFib Model Input features.

        Parameters:
        sig (array): ECG Signal.
        fs (float): Sampling frequency of signal.
        verbose (boolean): print PCA details

        Returns: 
        principalComponents_standard (list): PCA Components calculated using HRV and statistical features.
    '''

    hrv_keys = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD',
       'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN',
       'HRV_IQRNN', 'HRV_Prc20NN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20',
       'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN', 'HRV_HF', 'HRV_VHF', 'HRV_HFn',
       'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S', 'HRV_CSI',
       'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS',
       'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a',
       'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a',
       'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry',
       'HRV_MFDFA_alpha1_Fluctuation', 'HRV_MFDFA_alpha1_Increment',
       'HRV_ApEn', 'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn',
       'HRV_CMSEn', 'HRV_RCMSEn', 'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    
    #checking sig dimension
    if np.array(sig).ndim == 1:
        sig = np.expand_dims(np.array(sig),axis=1)
    elif np.array(sig).ndim == 2:
        pass
    else:
        print('signal has too much dimensions')
    #end if

    #calculating features
    hrv_temp = HRV_calc(sig, fs, plot=plot)
    hrv_temp = hrv_temp[hrv_keys]
    hrv_temp = hrv_temp.fillna(0)
    hrv_temp = np.array(hrv_temp)[0]

    stat_feature = statistical_feature(sig,fs)

    features = np.concatenate([hrv_temp, stat_feature])

    # #applying PCA on features
    # pca_std = load_PCA_model()
    # principalComponents_standard = pca_std.transform(np.expand_dims(np.array(features),axis=0))

    # if verbose:
    #     print('PCA components variances:', pca_std.explained_variance_ratio_)
    #     print('Sum of PCA components variances:', sum(pca_std.explained_variance_ratio_))
    # #end if

    return features
#end AFib_Model_Input_Features

def load_PCA_model():
    # Getting back the objects:
    with open('PCA_model.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        PCA_model = pickle.load(f)

    return PCA_model
#end load_PCA_model