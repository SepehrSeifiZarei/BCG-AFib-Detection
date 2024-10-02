# -*- coding: utf-8 -*-
# Basic Functions

#initializing
import os
import pickle

def finding_files_path(path=os.getcwd(), suffix='csv', containing_word='', find_subject=False, verbose=False):
    """
    finding file paths end with suffix in the given address

    Parameters:
    path (str): path to look for files
    suffix (str): suffix of the target files
    containing_word (str): word should be included in path
    find_subject (bool): if 'True', subject ids also extracted from the path
    verbose (bool): if 'True', number of paths and subject ids will be printed

    Returns: 
    csv_paths (list): list of path to found files
    subjects_id (list): list of subjects
    """

    #initializing
    csv_paths = []
    subjects_id = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(suffix) and (containing_word in root.lower() or containing_word in file.lower()):
                # print(root)
                # print(file)
                if find_subject:
                    subjects_id.append(file.split('_')[0])
                #end if
                csv_paths.append(os.path.join(root, file))
            #end if
        #end for
    #end for
    
    csv_paths.sort()
    subjects_id.sort()
    
    if verbose:
        print('[INFO]',len(csv_paths),suffix, 'Paths')
        if find_subject:
            print('subjects: ', subjects_id)
        #end if
    #end if

    if find_subject:
        return csv_paths, subjects_id
    else:
        return csv_paths
#end finding_csv_path

def save_variables(variables_list, file_name='', path='',verbose=True):
    """
    Saveing Variables in a File

    Parameters:
    variables_list (list): list of variables to save
    file_name (str): name of the saved file
    path (str): path for saving the file
    verbose (bool): if 'True', saving pass will be shown
    """

    # changing current directory to path
    if len(path)>0:
        last_directory = os.getcwd()
        os.chdir(path)
    #end if

    # Saving the objects:
    if str(file_name).endswith('.pkl'):
        save_name = str(file_name)
    else:
        save_name = str(file_name)+'.pkl'
    #end if
    
    with open(save_name, 'wb',) as f:  # Python 3: open(..., 'wb')
        pickle.dump(variables_list, f,) 

    # changing current directory to last one
    if len(path)>0:
        os.chdir(last_directory)
    #end if

    if verbose:
        print('[INFO] variables saved to the path successfully')
        print('   [INFO] in', path, save_name)
    #end if

#end save_variables

def load_variables(file_name='', path='', ):
    """
    Loading Variables from a File

    Parameters:
    file_name (str): name of the saved file
    path (str): path of the save file
    """

    # Loading the objects:
    save_name = str(path) + str(file_name) + '.pkl'
    if str(str(path) + str(file_name)).endswith('.pkl'):
        save_name = str(path) + str(file_name)
    #end if
    
    with open(save_name, 'rb') as f:  # Python 3: open(..., 'rb')
        return pickle.load(f)

#end load_variables