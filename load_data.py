#!/usr/bin/env python
from __future__ import division, print_function, absolute_import, unicode_literals

from os.path import exists as pathexists

import numpy as np
from skimage.io import ImageCollection, imread as skimread 

counts = lambda a: dict(zip(*np.unique(a,return_counts=True)))

def balance_classes(y,**kwargs):
    verbose = kwargs.get('verbose',False)
    ulab = np.unique(y)
    K = len(ulab)
    yc = counts(y)
    nsamp_tomatch = max(yc.values())
    balance_idx = np.array([])
    if verbose:
        print('Total (unbalanced) samples: %d\n'%len(y))

    for j in range(K):
        idxj = np.where(y==ulab[j])[0]
        nj = len(idxj)
        naddj = nsamp_tomatch-nj
        addidxj = idxj[np.random.randint(0,nj-1,naddj)]
        if verbose:
            print('Balancing class %d with %d additional samples\n'%(ulab[j],
                                                                     naddj))
        balance_idx = addidxj if j==0 else np.r_[balance_idx, addidxj]

    return balance_idx

def _load_path(labelf,nmax=np.inf,asarray=False,load_func=None,check_path=False,
               conserve_memory=True,memory_slots=1,transpose=None,
               exclude_pattern=None,mean_image=None,balance=False):

    load_func = load_func or skimread
    
    #exclude_pattern = '/tn/'
    imglabs = np.loadtxt(labelf,dtype=str)
    n = imglabs.shape[0]
    if nmax < n:
        keepidx = np.random.permutation(n)[:nmax]
        imglabs = imglabs[keepidx]
        n = nmax
    elif nmax == np.inf:
        nmax = n
        
    imgfiles,labs = [],[]
    nskip = 0
    for i,(imgf,lab) in enumerate(imglabs):
        if exclude_pattern and exclude_pattern in imgf:
            continue
        if (i+1) % (n//10)==1:
            print('loading image',i,'of',n,'(%d of %d selected)'%(len(labs),nmax))

        if i>=nmax:
            break

        if check_path and not pathexists(imgf):
            warn('image "%s" not found, skipping'%imgf)
            nskip+=1
            continue
            
        imgfiles.append(imgf)
        labs.append(lab)

    print('loaded %d of %d images (%d skipped)'%(len(labs),len(imglabs),nskip))
        
    if balance:
        print('balancing classes')
        imgfiles = np.array(imgfiles)
        labs = np.array(labs)
        balance_idx = balance_classes(labs)
        imgfiles = list(np.r_[imgfiles,imgfiles[balance_idx]])
        nulab = len(np.unique(labs))
        nbal = len(np.unique(labs[balance_idx]))
        labs = list(np.r_[labs,labs[balance_idx]])
        print('balanced %d of %d classes by sampling %d images with replacement'%(nbal,nulab,
                                                                               len(balance_idx)))

    imgs = ImageCollection(imgfiles,conserve_memory=conserve_memory,
                           load_func=load_func) #memory_slots=memory_slots
    if asarray:
        imgs = imgs.concatenate()
    labs = np.array(labs,dtype=np.int8)
    return imgs, labs

def compute_mean(X_train,X_test,meanf):
    if pathexists(meanf):
        return loadmat(meanf)['mean_image']
    mean_image = np.sum(X_train,axis=0)+np.sum(X_test,axis=0)
    mean_image /= X_train.shape[0]+X_test.shape[0]
    
    savemat({'mean_image':mean_image},meanf)
    return mean_image

def load_data(*args,**kwargs):
    balance_train = kwargs.pop('balance_train',False)
    nargs = len(args)
    X_train,y_train = [],[]
    X_test,y_test = [],[]
    if args[0]:
        X_train,y_train = _load_path(args[0],balance=balance_train,**kwargs)
    
    if nargs>1 and args[1]:
        X_test,y_test = _load_path(args[1],**kwargs)
    return X_train,y_train,X_test,y_test

if __name__ == '__main__':
    import sys
    import pylab as pl

    from tilepredictor import imread_tile
    from model_package import collect_batch, perturb_batch
    from keras.utils.np_utils import to_categorical 
    
    basepath=sys.argv[1]
    # basepath='/lustre/bbue/ch4/srcfinder/tiles/thompson_training/256/ang20150419t163741_det'#ang20150421t181252
    # basepath='/lustre/bbue/ch4/srcfinder/tiles/thompson_training/256/ang20150419t163741_det'
    # basepath='/lustre/bbue/ch4/srcfinder/tiles/thompson_training/224/ang20150419t163741'
    # basepath='/lustre/bbue/ch4/srcfinder/tiles/thorpe_training/256/ang20160910t182946_det_notn'
    # basepath='/lustre/bbue/ch4/srcfinder/tiles/thorpe_training/256/ang20160910t182946_det_notn'

    trainf = basepath+'_train.txt'
    testf = basepath+'_test.txt'
    tile_shape = [200,200]
    
    load_func = lambda imgf: imread_tile(imgf,tile_shape=tile_shape)
    loadargs = (trainf,testf)
    
    loadkwargs = {'conserve_memory':True,'load_func':load_func,
                  'exclude_pattern':None} # 'exclude_pattern':'/tn/'}

    X_train,y_train,_,_ = load_data(*loadargs,**loadkwargs)
    pos_idx,neg_idx = np.where(y_train==1)[0],np.where(y_train==0)[0]
    print(y_train.shape,len(pos_idx),len(neg_idx))
    X_train,y_train = X_train[::15],y_train[::15]
    pos_idx,neg_idx = np.where(y_train==1)[0],np.where(y_train==0)[0]
    npos,nneg = len(pos_idx),len(neg_idx)
    n = float(npos+nneg)
    naugpos,naugneg=5,1
    batch_size = 256//(naugpos+naugneg)
    
    nposb = int(batch_size*(npos/n)+0.5)
    nnegb = int(batch_size*(nneg/n)+0.5)
    randperm = np.random.permutation
    batch_idx = np.r_[randperm(pos_idx)[:nposb],
                      randperm(neg_idx)[:nnegb]]
    
    print(y_train.shape,len(pos_idx),len(neg_idx),nposb,nnegb)
    X_batch,y_batch = collect_batch(X_train,to_categorical(y_train),
                                    batch_idx=batch_idx)


    X_train_batch,y_train_batch = perturb_batch(X_batch,y_batch,
                                                naugpos=naugpos,
                                                naugneg=naugneg)
    pos_idx,neg_idx = np.where(np.argmax(y_train_batch,-1)==1)[0],\
                      np.where(np.argmax(y_train_batch,-1)==0)[0]
    print(y_train_batch.shape,len(pos_idx),len(neg_idx)),raw_input()
    
    fig,ax = pl.subplots(2,2,sharex=True,sharey=True)
    ax[0,0].imshow(X_train[pos_idx[0]].squeeze())
    ax[0,1].imshow(X_train[pos_idx[1]].squeeze())
    ax[1,0].imshow(X_train[neg_idx[0]].squeeze())
    ax[1,1].imshow(X_train[neg_idx[1]].squeeze())
    pl.show()
