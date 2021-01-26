#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from load_data import *
sys.path.append(os.environ['pyutil'])
from pylib import *

try:
    from sklearn.model_selection import train_test_split as tts, \
        StratifiedShuffleSplit, StratifiedKFold, GroupKFold
except:
    from sklearn.cross_validation import train_test_split as tts, \
        StratifiedShuffleSplit, StratifiedKFold, GroupKFold


verbose=True
train_size=0.75
random_state=42

def split_single(X,y,**kwargs):    
    kwargs.setdefault('stratify',y)
    kwargs.setdefault('train_size',train_size)
    kwargs.setdefault('random_state',random_state)

    idx = np.int64(np.arange(len(y)))
    tridx,teidx = tts(idx,**kwargs)
    X_train,X_test = [X[tridx]],[X[teidx]]
    y_train,y_test = [y[tridx]],[y[teidx]]
    return X_train,y_train,X_test,y_test

def split_multi(X,y,n_splits,**kwargs):
    kwargs.setdefault('train_size',train_size)
    kwargs.setdefault('random_state',random_state)
    
    sss = StratifiedShuffleSplit(n_splits=n_splits,**kwargs)
    X_train,y_train,X_test,y_test = [],[],[],[]
    for tridx,teidx in sss.split(y,y):
        X_train.append(X[tridx])
        y_train.append(y[tridx])
        X_test.append(X[teidx])
        y_test.append(y[teidx])

    return X_train,y_train,X_test,y_test

def split_kfold(X,y,n_folds,**kwargs):
    kwargs.setdefault('shuffle',True)
    kwargs.setdefault('random_state',random_state)
    
    skf = StratifiedKFold(n_splits=n_folds,**kwargs)
    X_train,y_train,X_test,y_test = [],[],[],[]
    for tridx,teidx in skf.split(y,y):
        X_train.append(X[tridx])
        y_train.append(y[tridx])
        X_test.append(X[teidx])
        y_test.append(y[teidx])
    return X_train,y_train,X_test,y_test

def split_paths(X,y,n_folds,**kwargs):
    '''
    Given filenames X in the following format...
        /path/to/image_tiles/filename_0/{tp,fp,tn,pos,neg}/...
        /path/to/image_tiles/filename_1/{tp,fp,tn,pos,neg}/...
                         ...
        /path/to/image_tiles/filename_n/{tp,fp,tn,pos,neg}/...

    ...the function will assume the following group structure...

        group_0 = /path/to/image_tiles/filename_0/*
        group_1 = /path/to/image_tiles/filename_1/*
                       ...
        group_n = /path/to/image_tiles/filename_n/*
    '''
    splargs = ['tp','fp','tn','pos','neg']
    bases = []
    for path in X:
        for a in splargs:
            sa = '/%s/'%a
            if sa in path:
                bases.append(path.split(sa)[0])
                break
            
    ubase,uids = np.unique(bases,return_inverse=True)
    print(len(ubase),'unique base paths:',n_folds,'splits')
    gkf = GroupKFold(n_folds)
    X_train,y_train,X_test,y_test = [],[],[],[]
    for tridx, teidx in gkf.split(X,y,uids):
        X_train.append(X[tridx])
        y_train.append(y[tridx])
        X_test.append(X[teidx])
        y_test.append(y[teidx])        

    return X_train,y_train,X_test,y_test
    
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="split_train_test")

    parser.add_argument("-n","--num_splits", help="number of splits",
                        type=int, default=1)
    parser.add_argument("-m","--mode", type=str, default='kfold',
                        help="split mode (single|multiple|kfold|path)")
    parser.add_argument("--excludefile", type=str, default=None,
                        help="File containing list of samples to exclude")
    
    
    parser.add_argument("labelfile", type=str, help="label file")
    
    args = parser.parse_args(sys.argv[1:])
    exclf = args.excludefile
    
    labf = args.labelfile
    k = int(args.num_splits)
    mode = args.mode
    if k==1 and mode=='kfold':
        mode='single'

    print('loading',labf)
    (X,y) = load_file(labf,load_func=lambda f: f,class_mode='binary')
    X = np.array(X.files)
    if exclf is not None:
        exclude = np.loadtxt(exclf,dtype=str)
        badmask = np.zeros(len(X),dtype=np.bool8)
        for i,xi in enumerate(X):
            for exci in exclude:
                if basename(exci) in xi:
                    badmask[i]=1
                    break
        n_bad = np.count_nonzero(badmask)
        if n_bad!=0:
            print('excluded',n_bad,'samples listed in',exclf)
            X = X[~badmask]
            y = y[~badmask]

    if mode=='single':
        X_train,y_train,X_test,y_test = split_single(X,y)
    elif mode=='multiple':
        X_train,y_train,X_test,y_test = split_multi(X,y,k)
    elif mode=='kfold':
        X_train,y_train,X_test,y_test = split_kfold(X,y,k)
    elif mode=='path':
        X_train,y_train,X_test,y_test = split_paths(X,y,k)

    for fold in range(k):
        if verbose:
            print('fold',fold+1,'of',k)
            print('Training classes',counts(y_train[fold]))
            print('Testing classes',counts(y_test[fold]))

        foldid = 'fold%dof%d'%(fold+1,k) if k>1 else 'split'
        
        train_out = []
        for f,l in zip(X_train[fold],y_train[fold]):
            train_out.append(' '.join([str(f),str(l)]))

        trainf = labf.replace('.txt','_%s_train.txt'%foldid)
        with open(trainf,'w') as fid:
            print('\n'.join(train_out),file=fid)
        print('saved',trainf)

        test_out = []
        for f,l in zip(X_test[fold],y_test[fold]):
            test_out.append(' '.join([str(f),str(l)]))

        testf = labf.replace('.txt','_%s_test.txt'%foldid)
        with open(testf,'w') as fid:
            print('\n'.join(test_out),file=fid)
        print('saved',testf)
        
        
