# -*- coding: utf-8 -*-
# BCG signal Processing Functions

#initializing
import numpy as np

def finding_start_points(bcg):
    """
    Finds the starting point(s) of BCG signal in a long signal

    Parameters:
    bcg (array): long BCG signal with missing parts.

    Returns: 
    start_points_index (list): start point(s) of BCG signal
    """
    
    start_points_index = []
    last_point = []
    counter = 0
    no_change_after_last_start = True

    for i in range(len(bcg)):
        if last_point==bcg[i]:
            counter += 1
            if counter==10:
              no_change_after_last_start = True
            #end if
        elif last_point!=bcg[i] and no_change_after_last_start==True:
            no_change_after_last_start = False
            counter = 0
            last_point = bcg[i]
            start_points_index.append(i)
        #end if
    #end for

    # if len(start_points_index)!=0:
    #     start_points_index.pop(0)
    # #end if

    return start_points_index
#end finding_start_points


def finding_end_points(bcg):
    """
    Finds the ending point(s) of BCG signal in a long signal

    Parameters:
    bcg (array): long BCG signal with missing parts.

    Returns: 
    end_points_index (list): end point(s) of BCG signal
    """

    end_points_index = []
    counter = 0
    last_point = []
    for i in range(len(bcg)):
        if last_point==bcg[i]:
            counter += 1
        else:
            last_point = bcg[i]
            counter = 0
        #end if
        
        if counter == 30:
            end_points_index.append(i-30)
        #end if
    #end for
    
    # deleting fake end points at the start of signal
    if len(end_points_index)!=0 and end_points_index[0]==0:
        end_points_index.pop(0)
    #end if

    return end_points_index
#end finding_end_points

def clean_signal(phase1_dict):
    """
    Deleting parts of signal that have a constant value or not showing any significant change. 

    Parameters:
    phase1_dict (dict): dictionary of subjects, each with three synced signals ('ECG', 'BCG_hi', 'BCG_low')

    Returns: 
    phase1_clean_dict (dict): dictionary of subjects, each with three clean (just useful parts) synced signals ('ECG', 'BCG_hi', 'BCG_low') 
    """

    clean_sig_ecg = []
    clean_sig_bcg_low = []
    clean_sig_bcg_hi = []
    phase1_clean_dict = dict()

    subjs = phase1_dict.keys()

    for subject in subjs:
        print('[INFO] Cleaning subject',subject,'signals')

        for i in range(len(phase1_dict[subject]['BCG_low'])):
            try:
                if i%10 == 0 :
                    print('   [INFO]',i,'segments were cleaned')
                #end if

                start_point = []
                end_point = []
                ecg = np.array(phase1_dict[subject]['ECG'][i][0])
                bcg_hi = np.array(phase1_dict[subject]['BCG_hi'][i][0])
                bcg_low = np.array(phase1_dict[subject]['BCG_low'][i][0])

                #finding start and end points in signals
                sig = bcg_low
                # print(len(sig))
                if len(sig)!=0:

                    start_point = finding_start_points(sig[:,1])
                    end_point = finding_end_points(sig[:,1])

                    if len(start_point)!=0:
                        pass
                    else:
                        ecg_bcg_start_diff = ecg[0,0] - sig[0,0]
                        # print(ecg_bcg_start_diff)
                        if ecg_bcg_start_diff<-0.032:
                            start_point.append(0)
                        elif np.std(sig[:,1])!=0:
                            start_point.append(0)
                        #end if
                    #end if

                    if len(end_point)!=0:
                        pass
                    else:
                        ecg_bcg_start_diff = ecg[-1,0] - sig[-1,0]
                        # print(ecg_bcg_start_diff)
                        if ecg_bcg_start_diff>0.032:
                            end_point.append(-1)
                        elif np.std(sig[:,1])!=0:
                            end_point.append(-1)
                        #end if
                    #end if
                #end if
                # print('len(start_point)',len(start_point))
                # print('len(end_point)',len(end_point))
                
                #selecting signals based on start and end points
                for j in range(len(start_point)):
                    diff = np.array(end_point)-start_point[j]
                    # print(diff)
                    end = np.argmin(diff)
                    # print(start_point[j])
                    # print(end_point[end])
                    clean_sig_bcg_low.append(bcg_low[start_point[j]:end_point[end],:])
                    clean_sig_bcg_hi.append(bcg_hi[start_point[j]:end_point[end],:])

                    #finding ecg signal sync to bcg signal
                    start_time = bcg_low[start_point[j],0]
                    end_time = bcg_low[end_point[end],0]

                    # print(start_time)
                    # print(end_time)

                    ecg_start_time_index = np.argmin(ecg[:,0] - start_time)
                    ecg_end_time_index = np.argmin(abs(ecg[:,0] - end_time))
                    # print(ecg_start_time_index)
                    # print(ecg_end_time_index)

                    clean_sig_ecg.append(ecg[ecg_start_time_index:ecg_end_time_index,:])
                #end for
            except:
                print('something went wrong on segment',i) 
        #end for segments
        print('   [INFO]',i,'segments were cleaned')
        phase1_clean_dict[subject] = {'ECG': clean_sig_ecg, 'BCG_hi': clean_sig_bcg_hi, 'BCG_low': clean_sig_bcg_low}

    #end for subject

    return phase1_clean_dict
#end clean_merge_signal

def removing_timestamps(sig, signal_coloumn=1):
    """
    Removing time stamps from the signal

    Parameters:
    sig (list): the signal to clean
    signal_coloumn (int): coloumn containing signal values

    Return:
    sig_clean (list): signal without timestamps
    """

    sig_clean = []
    for i in range(len(sig)):
        sig_clean.append(np.array(sig[i])[:,signal_coloumn])
    #end for

    return sig_clean
#end removing_timestamps