import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from numpy.fft import ifft
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from multiprocessing import Pool, cpu_count
import multiprocessing
from contextlib import closing
from tqdm import tqdm
import cv2
from skimage.feature import local_binary_pattern
from scipy.special import kl_div
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

NUM_PROCESSES = cpu_count()
print(f'NUM_PROCESSES = {NUM_PROCESSES}\n', end='')
print(f'pid: {os.getpid()}\n', end='')

labels = None
labels_test = None

def readcgw_old(record_id, phase='train'):
    with h5py.File(f'../../data/{phase}/{record_id}.hdf5','r') as g:
        f = g[record_id]
        dfh = pd.DataFrame(data=np.array(f['H1']['SFTs']).T, columns=f['frequency_Hz'], index=f['H1']['timestamps_GPS'])
        dfl = pd.DataFrame(data=np.array(f['L1']['SFTs']).T, columns=f['frequency_Hz'], index=f['L1']['timestamps_GPS'])
    return dfh, dfl

def read_labels(train:bool=True):
    global labels
    global labels_test
    if train:
        labels = pd.read_csv('../../data/train/_labels.csv', index_col='id')
        return labels
    else:
        labels_test = pd.read_csv('../../data/test/sample_submission.csv', index_col='id')
        return labels_test

def readcgw1(record:str, labels, preprocess=(lambda x:x), train:bool=True, viterbi:bool=True) -> tuple:
    if train:
        phase = 'train'
        label = labels.loc[record].values[0]
        if label not in [0,1]:
            return None, None
        target = label
    else:
        phase = 'test'
        target = None
    if viterbi:
        with np.load(f'../../data/viterbi/{phase}/{record}.npz') as infile:
            r = infile['arr_0']
        r = preprocess(record, r)
    else:
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

preprocess_global = lambda x,y:y
train_global = True
viterbi_global = True

def readcgw1_wrap(record:str):
    r = readcgw1(record, labels, preprocess_global, train_global, viterbi_global)
    # with open(f'kxys/{record}.pickle', 'wb') as outf:
    #     pickle.dump(obj=[r], file=outf)
    return r

def readcgws(records:list=None, preprocess=(lambda x,y:y), train=True, viterbi=False, maxtasksperchild=1):
    global preprocess_global
    global train_global
    global viterbi_global
    preprocess_global = preprocess
    train_global = train
    viterbi_global = viterbi
    if records is None:
        records = read_labels(train=train)
        if train:
            records = records[records['target']!=-1]
        records = records.index.values
    with closing(Pool(processes=NUM_PROCESSES, maxtasksperchild=maxtasksperchild)) as pool:
        print(f'pid: {os.getpid()}\n', end='')
        kxys = list(tqdm(pool.imap(readcgw1_wrap, records), total=len(records)))
    return sorted(kxys)

def plot_roc(y_true, y_score):
    RocCurveDisplay.from_predictions(y_true, y_score)

def ifft_bp(a, f0, df):
    pad = round(f0 / df)
    b = np.concatenate((np.zeros(pad), a, np.flip(np.conjugate(a)), np.zeros(pad-1)))
    return ifft(b).real

def autocorrelate(record:str, df:pd.DataFrame) -> tuple:
    ans = []
    for interferometer in ['H1','L1']:
        x = df.loc[interferometer]
        xd = {}
        for i in x.index:
            md = i%86400
            l = xd.get(md)
            if l is None:
                xd[md] = [i]
            else:
                l.append(i)
        bf = x.columns[0]
        deltaf = np.average(np.diff(x.columns))
        somma = 0
        count = 0
        for j in xd.values():
            if len(j) == 1:
                continue
            a = ifft_bp(x.loc[j[0]].values, bf, deltaf)
            for i in j[1:]:
                b = ifft_bp(x.loc[i].values, bf, deltaf)
                somma += np.dot(a,b)
                count += 1
                a = b
        ans.append(somma/count)
    return tuple(ans)

def spectrogram(df):
    plt.pcolormesh(df.index, np.float64(df.columns), np.abs(df.values).T)


def spectrogram_hl(record, df, preprocess=lambda x:np.abs(x), dirname='figs', plot=False):
    dfh = df.loc['H1']
    dfl = df.loc['L1']
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].pcolormesh(dfh.index, np.float64(dfh.columns), preprocess(dfh.values.T))
    ax[1].pcolormesh(dfl.index, np.float64(dfl.columns), preprocess(dfl.values.T))
    fig.suptitle(record)
    if plot:
        print('.\b', end='')
        plt.show()
    else:
        fig.savefig(f'{dirname}/{record}.png')

def viterbi(m, alpha=0.05):
    m = np.float64(np.abs(m))
    for j in range(1, m.shape[1]):
        for i in range(1,m.shape[0]-1):
            m[i,j] += alpha*(m[i-1:i+2,j-1].max()) + (1-alpha)*m[i,j-1]
        m[0,j] += alpha*(m[:2,j-1].max()) + (1-alpha)*m[0,j-1]
        m[-1,j] += alpha*(m[-2:,j-1].max()) + (1-alpha)*m[-1,j-1]
    for j in range(m.shape[1]):
        m[:,j] -= m[:,j].mean()
        m[:,j] /= m[:,j].std()
    m -= m.min()
    m /= m.max()
    return m

def projected_stats(df):
    h = df.loc['H1'].apply(lambda x: np.abs(x)).sum(axis=0).describe()[1:].rename('h_{}'.format)
    l = df.loc['L1'].apply(lambda x: np.abs(x)).sum(axis=0).describe()[1:].rename('l_{}'.format)
    hl = pd.concat([h, l]).T
    return hl

def projected_stats_from_values(values):
    features = pd.DataFrame(values).apply(lambda x: np.abs(x)).sum(axis=0).describe()[1:]
    return features.T

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import StandardScaler

def baseline(df, labels):
    X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=.8, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y1 = [x[1] for x in y_pred]
    roc = RocCurveDisplay.from_predictions(y_test, y1)

def softmax(x):
    y = np.exp(x)
    y -= y.min()
    y /= y.max()
    return y


def merge(vh,vl,d):
    w = min(vh.shape[1], vl.shape[1])
    vh = vh[:,:w]
    vl = vl[:,:w]
    for i in range(d):
        vh = softmax(vh)
        vl = softmax(vl)
    return vh*vl

def viterbi_spectrogram(record, df, d) -> np.ndarray:
    dfh = df.loc['H1']
    dfl = df.loc['L1']
    vh = viterbi(dfh.values.T)
    vl = viterbi(dfl.values.T)
    return merge(vh, vl, d)

def k_fold_validation(X, y):
    kf = KFold(random_state=42, shuffle=True)
    r = []
    k = []
    for train, test in kf.split(X):
        clf = LogisticRegression(max_iter=int(1e6))
        clf.fit(X.iloc[train], y[train])
        y_score = clf.predict_proba(X.iloc[test])
        y_true = y[test]
        y_score = [x[1] for x in y_score]
        r.append(roc_auc_score(y_true, y_score))
        k.append(clf.coef_)
    return r,k

# VITERBI SOAP
kk = 1/32
trp = 1/2*kk + 2/3*(1-kk)
kkdp = 1/16
dpp = 1/2*kkdp + 1/3*(1-kkdp)
transitions = np.array([trp/2, 1-trp, trp/2])
doppler = np.array([1/3,1/3,1/3])
logtr = np.log(transitions)
logdp = np.log(doppler)
assert logtr.size & 1 == 1 and logdp.size & 1 == 1

def viterbi_soap(record, df:pd.DataFrame):
    h1df = df.loc['H1']
    c = []
    for im in ['H1', 'L1']:
        sdf = df.loc[im]
        val = sdf.values
        var = val.var(axis=1) # spectrogram is transposed
        val = np.square(np.abs(val)) / var.reshape((var.size,-1)) / val.shape[1]
        valm = np.zeros(val.shape)
        for j in range(val.shape[0]):
            for k in range(1, val.shape[1]-1):
                valm[j,k] = val[j,k-1:k+2].max()
            valm[j,0] = val[j,:2].max()
            valm[j,-1] = val[j,-2:].max()
        c.append(valm)
    v = np.zeros((max(x.shape[0] for x in c), max(x.shape[1] for x in c)))
    for x in c:
        v[:x.shape[0], :x.shape[1]] += x
    d = logtr.size//2
    for j in range(1, v.shape[0]):
        for k in range(v.shape[1]):
            r = range(max(0,k-d), min(v.shape[1],k+d+1))
            v[j,k] += max(v[j-1,i] + logtr[i-k+d] for i in r) # spectrogram is transposed
    for j in range(1,v.shape[0]):
        v[j,:] -= v[j,:].mean()
        v[j,:] /= v[j,:].std()
    return v.T

def viterbi_2w(record, df:pd.DataFrame):
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
    c = [dfs[[f'{im}_{i}' for i in range(360)]].values for im in ['H1','L1']]
    val = sum(c)
    r = [None, None]
    for ver in range(2):
        v = val.copy()
        if ver:
            v = v[::-1,:]
        d = logtr.size//2
        l = np.zeros((v.shape[1],))
        for j in range(1, v.shape[0]):
            l[0] = v[j-1,0] + logtr[1]
            np.maximum(v[j-1,1:]+logtr[1], v[j-1,:-1]+logtr[2], l[1:])
            np.maximum(l[:-1], v[j-1,1:]+logtr[0], l[:-1])
            v[j,:] += l
        v = (v-(v.mean(axis=1)[..., np.newaxis]))/(v.std(axis=1)[...,np.newaxis])
        if ver:
            v = v[::-1,:]
        r[ver] = v.T
    return r


def peak_1(somma, th=5, base=0):
    a = 0
    b = 0
    t = 0
    f = False
    thresh = (somma.max())/th
    d = thresh*base
    for x in np.concatenate((somma,np.array([0]))):
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
    return b/a

def peak_2(somma, th=5, base=0):
    a = 0
    b = 0
    t = 0
    f = False
    thresh = (somma.max())/th
    d = thresh*base
    for x in np.concatenate((somma,np.array([0]))):
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
    return b/a

def peaks(df):
    return pd.DataFrame([[peak_1(somma,5), peak_1(somma,5,1), peak_2(somma,5), peak_2(somma,5,1)] for somma in df.values], index=df.index, columns=['p10', 'p11', 'p20', 'p21'])

# HOUGH

import cv2
def normalize(v):
    """
    min-max normalization over columns
    """
    return (v-v.min(axis=0))/(v.max(axis=0)-v.min(axis=0))

def viterbi2hough(v, thickness='auto'):
    """
    Compute hough from viterbi

    v:          viterbi values
    thickness:  plot the thickness of the line according to its importance

    th:         thresholded viterbi plot
    hough_v:    result of line extraction on viterbi plot
    hough_th:   result of line extraction on the thresholded viterbi plot
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
        return th, None, None
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

def extract_lines(record, vit):
    """
    Extract the most important line from H1 and L1 
    
    df: dataFrame containing both H1 and L1 measurements
    """
    _, _, _, hline = viterbi2hough(vit)
    
    rhoH, thetaH, votesH = hline[0]

    return pd.Series([rhoH, thetaH, votesH], index=["rho", "theta", "votes"]).T

# LBP 

def viterbi2lbp(v):
    """
    Compute local binary pattern from viterbi
    """
    x = local_binary_pattern(v, 8, 1, 'default')
    hist, _ = np.histogram(x, 256)
    return pd.Series(hist, index=[i for i in range(256)]).T


def read_wrapper_lbp(x):
    return read_record(x, extract_lbp)


def extract_lbp(df): # TO CHECK
    v = viterbi(df)
    return viterbi2lbp(v)

def symmetric_kldiv(x, y):
    return np.sum(kl_div(x, y)/2 + kl_div(y, x)/2)

def symmetric_kldiv_pole(x, y):
    if np.isclose(x,y).all():
        return np.inf
    return symmetric_kldiv(x, y)

# TRAIN THE FOLLOWING CLASSIFIER
# clf = KNeighborsClassifier(metric=symmetric_kldiv, weights='distance')

# GAB

def hshow(hough):
    """
    Show the result of hough line extraction
    """
    plt.imshow(hough[::-1, :, :], aspect='auto') # flip to match the other plots


def vshow(r, v):
    """
    Show viterbi plot
    r:  record
    v:  viterbi values between 0 and 1 (min-max normalized)
    """
    plt.pcolormesh(r.index, np.float64(r.columns), v)



def read_record(record: str, preprocess=lambda x:x, phase: str="train") -> tuple:
    """
    Read a record and return a df containing H1 and L1 measurements

    str:        record name
    preprocess: pd.DataFrame -> pd.DataFrame or pd.Series
                function to be applied to df for feature extraction 
    """

    with h5py.File(f'./{phase}/{record}.hdf5','r') as g:
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
    return preprocess(df)


def read_wrapper_lines(x):
    """
    Wrapper function for line extraction
    """
    return read_record(x, extract_lines) # NOTE: extract_lines interface has changed


def read_extract(id_target_train, read_func, parallel=True, ):
    """
    Read and extract feature with parallel processes

    id_target_train:    df containing /train/_labels.csv
    read_record:        wrapper function of read_record(x, **kwargs)
    """
    if parallel:
        try:
            n = multiprocessing.cpu_count()
        except NotImplementedError:
            n = NUM_PROCESSES # default
        pool = multiprocessing.Pool(n)
        df = pd.DataFrame(pool.map(read_func, id_target_train.index))
    else:
        df = pd.DataFrame([read_func(x) for x in id_target_train.index])
    return df