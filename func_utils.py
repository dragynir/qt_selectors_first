import pandas as pd
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
import os
from tqdm import tqdm

# CONSTANT dictionary that defines extensions of files that can be read by Pandas functions
EXT = {'txt': pd.read_table, 'csv': pd.read_csv, 
       'xls': pd.read_excel, 'xlsx': pd.read_excel}


def read_all_data(dir='data'):
    """
    Reads ALL data from 'dir' with extension .txt, .csv, .xls(x).
    It drops all columns that does not have a name!
    
    Arguments:
        dir: str
            Name of directory, wherein data is stored
    
    Return:
        DataFrames: dict
            Dictionary wherein keys are names of files that were read, 
            and values are pandas.DataFrame objects
    """
    
    path = dir + '/'
    allFiles = os.listdir(path)
    files, names, exts = parser(allFiles)
    cols_parser = lambda x: x.split(':')[0].casefold() != 'unnamed'
    
    DataFrames = {}
    for f, n, e in zip(files, names, exts):
        DataFrames[n] = EXT[e](path + f, usecols=cols_parser)
    return DataFrames


def parser(allNames):
    files, names, exts = [], [], []
    for n in allNames:
        s = n.split('.')
        if len(s) == 2:
            if s[1] in EXT.keys():
                files.append(n)
                names.append(s[0])
                exts.append(s[1])
    return files, names, exts

INTERPS = {'linear' : LinearNDInterpolator, 
           'cubic' : CloughTocher2DInterpolator}


def fill_interp(df, cols, interpolation='linear', k_dims=['X', 'Y'], points=None):
    """
    Does the 2D interpolation of the data in 'df'. 
    
    Arguments:
        df: pandas.DataFrame
            Table with data to be interpolated
        
        cols: list of str
            Names of columns to be used for interpolation.
            If None, then all columns except fisrt THREE that must be ('FFID', 'X', 'Y')
        
        interpolation: str ('cubic' or 'linear')
            'cubic' - scipy.interpolate.CloughTocher2DInterpolator
            'linear' - scipy.interpolate.LinearNDInterpolator
            By default 'linear'
        
        k_dims: list of TWO str
            Specifies the name of columns with 2D coordinates. By default ['X', 'Y']
            'df' must contain these columns. 
            
        points: None or numpy array (n, 2)
            Specifies points that are used to interpolate the data onto them.
            If None, interpolation works to fill in the gaps with NaN values.
            If numpy array (n, 2) , interpolation works to interpolate the data onto 'points',
            The first columns must correspond to k_dims[0], the second is for k_dims[1] 
            
    """
    assert interpolation in INTERPS.keys(), "Interpolation must be one of {}".format(INTERPS.keys())
    
    df_new = pd.DataFrame()

#     for c in tqdm(cols, desc='Interpolation', leave=None):
    for c in cols:
        df_ = df.loc[df[c].notna()].copy()
        xy_ = df_[k_dims].values
        v_ = df_[c].values
        interp = INTERPS[interpolation](xy_, v_, fill_value=v_.mean())
        df_new[c] = interp(points)

    return df_new
    
    
def del_noise(df, factor):
    if isinstance(factor, type(None)):
        factor = np.inf
    df_sc = ((df - df.mean()) / df.std()).copy()
    inds = (df_sc <= factor).prod(axis=1) == 1
    return inds
    
    
def merge_all(dataframes, xy=['X', 'Y'], interpolation='linear', sigmaNoise=4, suffixes=None):
    dfs = dataframes.values()
    if isinstance(suffixes, type(None)):
        suffixes = ['_' + str(i) for i in range(len(dfs))]
        

    df_min = min(dfs, key=lambda x: len(x))
    xyid = ['FFID'] + xy 
    df_all = df_min[xyid].copy()
    XY = df_all[xy].values
    
    for i, df in tqdm(enumerate(dfs), total=len(dfs), desc='Interpolating'):
#     for i, df in enumerate(dfs):
        cols = list(df.columns)
        for k in xyid:
            try: cols.remove(k)
            except: continue
        new_cols = [c + suffixes[i] for c in cols]
        df_new = df[cols].copy()
        df_new.columns = new_cols
        noise_inds = del_noise(df_new, sigmaNoise)
        df_new[xy] = df.loc[:, xy].copy()
        df_all[new_cols] = fill_interp(df_new[noise_inds], new_cols, interpolation=interpolation, 
                                       k_dims=xy, points=XY)

    return df_all
    
