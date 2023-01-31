# -*- coding: utf-8 -*-
# PreProcessing Functions

#initializing
import pywt
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd

def filter_bank(index_list, wavefunc='db6', lv=7, m=3, n=6, plot=False):
    
    '''
    Calculating wavelet decomposition of given signal and return selected levels coefficients

    Parameters:
    index_list (array): Input Sequence
    wavefunc (str): Function of Wavelet, 'db6' default
    lv (int): Decomposing Level  
    m, n (int): Level of Threshold Processing
    plot (boolean): status of ploting wavelet signals

    Returns: 
    coeff (list): coefficients of the selected levels
    '''

    # Decomposing 
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   #  Decomposing by levels，cD is the details coefficient
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn function

    # Denoising
    # Soft Threshold Processing Method
    for i in range(m,n+1):   #  Select m~n Levels of the wavelet coefficients，and no need to dispose the cA coefficients(approximation coefficients)
        cD = coeff[i]
        Tr = np.sqrt(2*np.log2(len(cD)))  # Compute Threshold
        for j in range(len(cD)):
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) * (np.abs(cD[j]) -  Tr)  # Shrink to zero
            else:
                coeff[i][j] = 0   # Set to zero if smaller than threshold
                
    # Reconstructing
    coeffs = {}
    for i in range(len(coeff)):
        coeffs[i] = copy.deepcopy(coeff)
        for j in range(len(coeff)):
            if j != i:
                coeffs[i][j] = np.zeros_like(coeff[j])
    
    for i in range(len(coeff)):
        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        if len(coeff[i]) > len(index_list):
            coeff[i] = coeff[i][:-1]
        
    if plot:     
        denoised_index = np.sum(coeff, axis=0)   
        data = pd.DataFrame({'Raw': index_list, 'Denoised': denoised_index})
        data.plot(figsize=(10,10),subplots=(2,1))
        data.plot(figsize=(10,5))
   
    return coeff
#end filter_bank

def rolling_rms(sig, N=200):
    '''
        Applying RMS filter on signal x with window size of N

        Parameters:
        sig (array): Signal.
        N (int): window size

        Returns: 
        RMS filtered signal
    '''
 
    xc = np.cumsum(abs(sig)**2);
    return np.sqrt((xc[N:] - xc[:-N]) / N)
    
def wavelet_denoising(sig, wavelet_funcs, verbose=True):
    from src.functions.preprocessing import filter_bank

    wavelet_signal = []
    temp = []

    for wl in wavelet_funcs:
        if verbose:
            print('[INFO] using',wl,'Wavelet packet')
        #end if verbose

        if len(wavelet_signal)==0:
            for i in range(len(sig)):
                if i%5000==0 and verbose:
                      print('   [INFO]',i,'segments were denoised')
                #end if verbose
                coeff=filter_bank(sig[i], wavefunc=wl, plot=False)
                temp.append(np.sum(coeff, axis=0))
            #end for
            wavelet_signal = temp
            temp = []

            if verbose:
                print('   [INFO]',i,'segments were denoised')
            #end if verbose

        else:
            for i in range(len(sig)):
                if i%5000==0 and verbose:
                    print('   [INFO]',i,'segments were denoised')
                #end ifverbose
                coeff=filter_bank(sig[i], wavefunc=wl, plot=False)
                wavelet_signal[i] = np.mean([wavelet_signal[i], np.sum(coeff, axis=0)], axis=0)
            #end for

            if verbose:
                print('   [INFO]',i,'segments were denoised')
            #end if verbose
    #end for

    return wavelet_signal
#end wavelet_denoising