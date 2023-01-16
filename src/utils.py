import pickle
import h5py
import numpy as np
import pandas as pd
import os
import cv2

from multiprocessing import Pool, cpu_count
from contextlib import closing
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from scipy.special import kl_div

import matplotlib.pyplot as plt

# INITIALIZATION

try:
    NUM_PROCESSES = cpu_count()
except:
    NUM_PROCESSES = 16
print(f'NUM_PROCESSES = {NUM_PROCESSES}\n', end='')
print(f'pid: {os.getpid()}\n', end='')



# VITERBI

# the following parameters are transitions probabilities for the Viterbi algorithm,
# they have been hand-tuned to get visibly good transformations
kk = 1/32
trp = 1/2*kk + 2/3*(1-kk)
transitions = np.array([trp/2, 1-trp, trp/2])
logtr = np.log(transitions)

def spectrogram_normalization(df:pd.DataFrame):
    """
    Makes computations on input dataframe (magnitude, pesudo-standardization,
        Doppler effect mitigation and time-grouping, solving H1-L1 time inconsitencies)

    df: input dataframe

    Returns two numpy.ndarray in a list, one per detector. These can be direclty used by viterbi_2w
    """
    c = []
    for im in ['H1', 'L1']:
        sdf = df.loc[im]
        val = sdf.values
        var = val.var(axis=1) # spectrogram is transposed
        val = np.square(np.abs(val)) / var.reshape((var.size,-1)) / val.shape[1]
        np.maximum(val[:,1:], val[:,:-1], val[:,1:])
        np.maximum(val[:,:-1], val[:,1:], val[:,:-1])
        sdf = pd.DataFrame(val, index=sdf.index, columns=[f'{im}_{i}' for i in range(360)])
        sdf = sdf.groupby(lambda ts:int(ts)//(1800)).sum()
        sdf.sort_index(inplace=True)
        c.append(sdf)
    dfs = pd.concat(c, axis=1)
    dfs.fillna(0, inplace=True)
    c = [dfs.filter(regex=(f"{im}.*")).values for im in ['H1','L1']]
    return c


def viterbi_2w(spec:list, plot=False):
    """
    Computes Viterbi transformations of input dataframe, bot forward and backward

    spec: list of numpy.ndarray, one per detector

    Returns [fw, bw] transformations, each one is a numpy.ndarray
    """
    if plot:
        print('.\b', end='')
        fig, ax = plt.subplots(2, 2, figsize=(9,6))
        ax[0,0].pcolormesh(spec[0].T)
        ax[1,0].pcolormesh(spec[1].T)
    val = sum(spec) # sum of numpy matrices
    r = [None, None]
    for ver in range(2):
        v = val.copy()
        if ver:
            v = v[::-1,:]
        l = np.zeros((v.shape[1],))
        for j in range(1, v.shape[0]):
            l[0] = v[j-1,0] + logtr[1]
            np.maximum(v[j-1,1:]+logtr[1], v[j-1,:-1]+logtr[2], l[1:])
            np.maximum(l[:-1], v[j-1,1:]+logtr[0], l[:-1])
            v[j,:] += l
        vstd = (v.std(axis=1)[...,np.newaxis])
        vstd[vstd==0] = 1
        v = (v-(v.mean(axis=1)[..., np.newaxis]))/vstd
        if ver:
            v = v[::-1,:]
        r[ver] = v.T
    if plot:
        ax[0,1].pcolormesh(r[0])
        ax[1,1].pcolormesh(r[1])
        plt.show()
    return r



# FEATURE UTILITIES

def peaks(somma, th=5, base=0):
    """
    Compute ratio between second highest peak and highest peak.
    The input is split in contiguous regions with values all above threshold.
    In a region, the peak is the leftmost highest value.
    The ratio is computed among different regions' peaks.

    somma: 1d array
    th: threshold = somma.max()/th
    base: subtract threshold from peaks if 1
    """
    a = 0
    b = 0
    t = 0
    f = False
    thresh = (somma.max())/th
    d = thresh*base
    for x in np.concatenate((somma, np.array([0]))):
        if x <= thresh:
            if f:
                f = False
                if t>a:
                    b = a
                    a = t
                elif t>b:
                    b = t
                t = 0
        else:
            f = True
            t = max(t,x-d)
    if t>a:
        b = a
        a = t
    elif t>b:
        b = t
    t = 0
    if a == 0:
        return 1
    return b/a

def areas(somma, th=5, base=0):
    """
    Compute ratio between highest area above threshold and sum of the areas above threshold.
    The input is split in contiguous regions with values all above threshold.
    For each region, the area is computed.

    somma: 1d array
    th: threshold = somma.max()/th
    base: subtract threshold from points over threshold if 1
    """
    a = 0
    b = 0
    t = 0
    f = False
    thresh = somma.max()/th
    d = thresh*base
    for x in np.concatenate((somma, np.array([0]))):
        if x <= thresh:
            if f:
                f = False
                if t>a:
                    b += a
                    a = t
                elif t>b:
                    b += t
                t = 0
        else:
            f = True
            t += x-d
    if t>a:
        b += a
        a = t
    elif t>b:
        b += t
    if a == 0:
        return 1
    return a/(a+b)



# HOUGH

import cv2
def normalize(v):
    """
    min-max normalization over columns
    """
    vdelta = (v.max(axis=0)-v.min(axis=0))
    vdelta[vdelta==0] = 1
    return (v-v.min(axis=0))/vdelta

def viterbi2hough(v, thickness='auto'):
    """
    Compute Hough from Viterbi

    v:          Viterbi values
    thickness:  plot the thickness of the line according to its importance

    th:         thresholded Viterbi plot
    hough_v:    result of line extraction on Viterbi plot
    hough_th:   result of line extraction on the thresholded Viterbi plot
    l:          most important line (rho, theta, n_votes)
    """

    v = normalize(v)

    gray = np.uint8(v*255) # scale in range 0-255
    t = np.quantile(gray, 0.995) 
    _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY) # threshold
    lines = cv2.HoughLinesWithAccumulator(th, 1, np.pi/180, 0)

    try:
        l = lines[0] # line with the highest vote
    except:
        return th, None, None, [(0, np.pi/2, 0)]
    else:
        hough_th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        hough_v = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        rho = l[0][0]
        theta = l[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
        pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
        
        thick = int(max(l[0][2]/200, 1)) if thickness=='auto' else 1
        cv2.line(hough_th, pt1, pt2, (255,0,0), thick, cv2.LINE_AA)
        cv2.line(hough_v, pt1, pt2, (255,0,0), thick, cv2.LINE_AA)

        return th, hough_v, hough_th, l

def extract_lines(v):
    """
    Extract the most important line from H1 and L1 
    
    df: dataFrame containing both H1 and L1 measurements
    """
    _, _, _, hline = viterbi2hough(v)
    rhoH, thetaH, votesH = hline[0]
    return pd.Series([rhoH, thetaH, votesH], index=["rho", "theta", "votes"]).T

# LBP 

def viterbi2lbp(v):
    """
    Compute local binary pattern from Viterbi
    """
    x = local_binary_pattern(v, 8, 1, 'default')
    hist, _ = np.histogram(x, 256)
    return pd.Series(hist, index=[i for i in range(256)]).T

def read_wrapper_lbp(x):
    return read_record(x, extract_lbp)

def extract_lbp(df):
    v = viterbi(df)
    return viterbi2lbp(v)

def symmetric_kldiv(x, y):
    return np.sum(kl_div(x, y)/2 + kl_div(y, x)/2)



# READ UTILS

labels = None
labels_test = None

def read_labels(train:bool=True):
    """
    Read record ids and labels (or dummy predictions for test)
    """
    global labels
    global labels_test
    if train:
        labels = pd.read_csv('../../data/train/_labels.csv', index_col='id')
        return labels
    else:
        labels_test = pd.read_csv('../../data/test/sample_submission.csv', index_col='id')
        return labels_test

read_labels(True)
read_labels(False)

def readcgw1(record:str, labels, preprocess=(lambda x:x), train:bool=True) -> tuple:
    """
    Read and preprocess a single record

    record: record id
    labels: labels, numpy.ndarray
    preprocess: function to apply
    train: whether to read from train or test set

    Returns a tuple with (record id, preprocess output, target (int))
    """
    if train:
        label = labels.loc[record].values[0]
        if label not in [0,1]:
            return None, None
        target = label
    else:
        target = None
    if train:
        filename = f'../../data/train/{record}.hdf5'
    else:
        os.system(f'unzip -q ../../data/test/{record}.hdf5.zip')
        filename = f'{record}.hdf5'
    with h5py.File(filename,'r') as g:
        f = g[record]
        dfl = pd.DataFrame(data=np.array(f['L1']['SFTs']).T, columns=f['frequency_Hz'])
        dfl['timestamp'] = f['L1']['timestamps_GPS']
        dfh = pd.DataFrame(data=np.array(f['H1']['SFTs']).T, columns=f['frequency_Hz'])
        dfh['timestamp'] = f['H1']['timestamps_GPS']
    dfl['interferometer'] = 'L1'
    dfh['interferometer'] = 'H1'
    df = pd.concat([dfh, dfl])
    df = df.set_index(['interferometer', 'timestamp'])
    df = df.sort_index()
    if not train:
        os.remove(filename)
    r = preprocess(record, df)
    return (record, r, target)

# global variables accessed by readcgw1_wrap parallelized function
preprocess_global = lambda x,y:y
train_global = True

def readcgw1_wrap(record:str):
    return readcgw1(record, labels, preprocess_global, train_global,)

def readcgws(records:list=None, preprocess=(lambda x,y:y), train=True):
    """
    Reads records in parallel applying preprocess to each.

    records: list of record ids
    preprocess: function to apply
    train: whether to read from train or test set

    Returns a list of objects returned by readcgw1, sorted by record id
    """
    global preprocess_global
    global train_global
    preprocess_global = preprocess
    train_global = train
    if records is None:
        records = read_labels(train=train)
        if train:
            records = records[records['target']!=-1]
        records = records.index.values
    with closing(Pool(processes=NUM_PROCESSES, maxtasksperchild=1)) as pool:
        print(f'pid: {os.getpid()}\n', end='')
        kxys = list(tqdm(pool.imap(readcgw1_wrap, records), total=len(records)))
    return sorted(kxys)

def read_data(preprocess, train:bool, cache_read:bool=False, cache_write:bool=False):
    """
    Read and preprocess data from disk

    preprocess(recordid, dataframe): function to use to preprocess data
    train: reads train set if true, test set if false
    cache_read: try to read stored preprocessed data from disk, NOTE: does not handle different preprocess functions
    cache_write: try to store preprocessed data on disk for future use, NOTE: does not handle different preprocess functions
    """
    data = None
    if cache_read:
        try:
            with open('data_2w_' + ('train' if train else 'test') + '.pickle', 'rb') as infile:
                data = pickle.load(file=infile)
        except:
            cache_read = False
    if not cache_read:
        data = readcgws(preprocess=preprocess, train=train)
        if cache_write:
            try:
                with open('data_2w_' + ('train' if train else 'test') + '.pickle', 'wb') as outf:
                    pickle.dump(obj=data, file=outf)
            except Exception as e:
                print(f'Could not store preprocessed data, reason:\n{str(e)}\n', end='')
    return data

def read_cache_dfs(infiles:list=None, index_col:list=[], read_function=None, outfiles:list=None):
    dfs = []
    if infiles is not None:
        for infile in infiles:
            try:
                df = pd.read_csv(infile, index_col=index_col)
                dfs.append(df)
            except Exception as e:
                print(e)
                infiles = None
    if infiles is None:
        dfs = read_function()
        if outfiles is not None:
            for outfile, df in zip(outfiles, dfs):
                try:
                    df.to_csv(outfile)
                except Exception as e:
                    print(e)
    return tuple(dfs)
