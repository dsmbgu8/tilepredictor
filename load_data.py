#!/usr/bin/env python
from __future__ import division, print_function, absolute_import, unicode_literals

from os.path import exists as pathexists, expanduser

import numpy as np
from skimage.io import ImageCollection

from tilepredictor_util import *
from pylib import *

require_load_func=True


def _parse_label_file(labelf,nmax=np.inf):
    imglabs = np.loadtxt(labelf,dtype=str)
    if nmax < imglabs.shape[0]:
        keepidx = np.random.permutation(n)[:nmax]
        imglabs = imglabs[keepidx]

    return imglabs

def _load_path(labelf,load_func,nmax=np.inf,asarray=False,check_paths=False,
               conserve_memory=True,memory_slots=1,transpose=None,balance=False,
               class_mode='categorical',exclude_pattern=None,mean_image=None):

    #exclude_pattern = '/tn/'
    imglabs = _parse_label_file(labelf,nmax=nmax)
    n = imglabs.shape[0]    

    nmax = min(nmax,n)
    imgfiles,labs = [],[]
    nskip = 0
    pmsg = 'loading %d (of %d) images'%(nmax,len(imglabs))
    pbar = progressbar(pmsg,n)
    for i,(imgif,lab) in enumerate(pbar(imglabs)):
        # paths can contain environment variables and "~"
        imgf = os.path.expanduser(os.path.expandvars(imgif))
        if exclude_pattern and exclude_pattern in imgf:
            continue

        if i>=nmax:
            break

        if check_paths and not pathexists(imgf):
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
        balance_labs = labs[balance_idx]
        nlab = len(np.unique(labs))
        nballab = len(np.unique(balance_labs))
        labs = list(np.r_[labs,balance_labs])
        imgfiles = list(np.r_[imgfiles,imgfiles[balance_idx]])
        print('balanced %d of %d classes by sampling %d images'
              'with replacement'%(nballab,nlab,len(balance_idx)))

    imgs = ImageCollection(imgfiles,conserve_memory=conserve_memory,
                           load_func=load_func) #memory_slots=memory_slots
    if asarray:
        imgs = imgs.concatenate()
    labs = np.array(labs,dtype=np.int8)
    if class_mode=='categorical':
        labs = labs.squeeze()
        labs = to_categorical(labs)
        
    return imgs, labs

def compute_mean(X_train,X_test,meanf):
    if pathexists(meanf):
        return loadmat(meanf)['mean_image']
    mean_image = np.sum(X_train,axis=0)+np.sum(X_test,axis=0)
    mean_image /= X_train.shape[0]+X_test.shape[0]
    
    savemat({'mean_image':mean_image},meanf)
    return mean_image

def load_data(*args,**kwargs):
    nargs = len(args)
    assert(nargs>=1)
    if 'load_func' not in kwargs:
        if require_load_func:
            raise Exception('load_data requires load_func (e.g., skimage.io.imread) to load_data function!')
        else:
            print('load_func not specified, using skimage.io.imread as default reader')
            from skimage.io import imread as _imread
            kwargs['load_func'] = _imread

    # only balance training set
    balance_train = kwargs.pop('balance_train',False)
    train_set = [_load_path(args[0],balance=balance_train,**kwargs)]

    # test set(s)
    test_sets = [_load_path(argi,**kwargs) for argi in args[1:]]
                 
    return train_set+test_sets

def load_image_data(train_file,test_file,**kwargs):
    load_func = kwargs.pop('load_func',None)
    if load_func is None:
        if require_load_func:
            raise Exception('load_image_data requires load_func (e.g., skimage.io.imread) to load_data function!')
        else:
            print('load_func not specified, using skimage.io.imread as default reader')
            from skimage.io import imread as _imread
            load_func = _imread

    balance_train = kwargs.pop('balance_train',False)
    collect_test = kwargs.pop('collect_test',True)
    test_percent = kwargs.pop('test_percent',0.2)
    exclude_pattern = kwargs.pop('exclude_pattern',None)
    mean_image = kwargs.pop('mean_image',None)
    class_mode = kwargs.pop('class_mode','categorical')
    if train_file==test_file:
        warn('train_file==test_file, sampling test data from train_file')
        test_file = None

    (X_train,y_train),(X_test,y_test) = load_data(train_file,test_file,
                                                  load_func=load_func,
                                                  class_mode=class_mode,
                                                  balance_train=False,
                                                  exclude_pattern=exclude_pattern,
                                                  mean_image=mean_image,
                                                  **kwargs)
    train_img_files = np.array(X_train.files)
    if y_train.ndim==1 or y_train.shape[1]==1:
        y_train = to_categorical(y_train)

    train_lab = to_binary(y_train)
    nb_classes = y_train.shape[1]

    if len(y_test)==0:
        msg='No test_file provided'
        if test_percent > 0.0:
            from sklearn.model_selection import train_test_split

            msg+=', testing on %d%% of training data'%int(test_percent*100)
            train_idx, test_idx = train_test_split(np.arange(y_train.shape[0]),
                                                   train_size=1.0-test_percent,
                                                   stratify=train_lab,
                                                   random_state=random_state)

            # remove any test samples present in the training data
            test_idx = test_idx[~np.isin(train_img_files[test_idx],
                                         train_img_files[train_idx])]
            test_img_files = train_img_files[test_idx]
            train_img_files = train_img_files[train_idx]
            X_test = imgfiles2collection(test_img_files,load_func,**kwargs)
            y_test,y_train = y_train[test_idx],y_train[train_idx]
            train_lab = to_binary(y_train)
            test_lab = to_binary(y_test)
        else:
            msg+=', test_percent=0, no validation will be performed'
            X_test,y_test = [],[]
            test_lab,test_img_files = [],[]
        print(msg)
    else:
        test_img_files = np.array(X_test.files)
        X_test = imgfiles2collection(test_img_files,load_func,**kwargs)
        test_lab = to_binary(y_test)          
        
    if balance_train:
        balance_idx = balance_classes(train_lab)
        train_img_files = np.r_[train_img_files,train_img_files[balance_idx]]
        y_train = np.r_[y_train,y_train[balance_idx]]
        print('Balanced %d classes by sampling %d images with '
              'replacement'%(nb_classes,len(balance_idx)))
        train_lab = to_binary(y_train)
        
    X_train = imgfiles2collection(train_img_files,load_func,**kwargs)
    
    if len(y_test)!=0:
        if y_test.ndim==1 or y_test.shape[1]==1:
            y_test = to_categorical(y_test)
        test_lab = to_binary(y_test)
        if collect_test:
            print('Collecting %d test samples'%len(y_test))
            X_test = X_test.concatenate()
            #X_test,y_test = collect_batch(X_test,y_test,verbose=1)


    print("Training samples: {}, input shape: {}".format(len(X_train),X_train[0].shape))
    print('Training classes:')
    class_stats(train_lab,verbose=1)
    print("Test samples: {}, shape: {}".format(len(X_test),X_test[0].shape))        
    print('Test classes:')
    class_stats(test_lab,verbose=1)
    
    return (X_train,y_train,train_img_files),(X_test,y_test,test_img_files)

def imageccomp2tilegen(imagelistf,tile_dim,n_tiles,conserve_memory=True):

    from skimage.transform import integral_image
    from scipy.sparse import dok_matrix as spmat
    detlist = np.loadtxt('detf.txt',dtype=str)
    lablist = np.loadtxt('labf.txt',dtype=str)
    imglist = np.c_[detlist,lablist]

    #detfiles,ccompfiles,labfiles = imglist.T
    from sklearn.feature_extraction.image import extract_patches_2d
    tile_size = (tile_dim,tile_dim)
    tile_off = tile_dim//2
    for i,ifiles in enumerate(imglist):
        detf,labf = map(expanduser,ifiles)
        idet = np.uint8(imread(detf)[...,:3])
        labs = imlabel(imread(labf)[...,:3].any(axis=2))

        ipatch = np.dstack([idet,labs])

        for lj in range(1,labs.max()):
            (ib,il),(it,ir) = maskbbox(labs==lj,border=tile_off)
            ntj = max(1,max(it-ib,ir-il)//tile_dim)
            p = extract_patches_2d(ipatch[ib:it,il:ir], tile_size, ntj, random_state=42)        
            pj,yj = p[...,:-1],p[...,-1].any(axis=-1)
            for j in range(p.shape[0]):
                pj,yj = p[j][...,:-1],p[j][...,-1].any()
                nj+=1    
            
        p = extract_patches_2d(ipatch, tile_size, n_tiles, random_state=42)
        pj,yj = p[...,:-1],p[...,-1].any(axis=-1)
        nj=0
        for j in range(p.shape[0]):            
            pj,yj = p[j][...,:-1],p[j][...,-1].any()
            if not pj.any():
                continue
            nj += 1
            print('tile',nj,'#comp=',np.count_nonzero(pj.any(axis=2)),'lab=',yj)




if __name__ == '__main__':
    import sys    
    import pylab as pl

    tile_dim = 150
    n_tiles = 50
    imageccomp2tilegen('',tile_dim,n_tiles,conserve_memory=True)
    raw_input()
    
    from tilepredictor import imread_tile
    from model_package import collect_batch, imaugment_perturb, batch_datagen_params 
    #from keras.preprocessing.image import ImageDataGenerator
    
    basepath=sys.argv[1]
    # basepath='/Users/bbue/Research/ARIA/iceberg/iceberg'
    # basepath='/lustre/bbue/ch4/srcfinder/tiles/thompson_training/256/ang20150419t163741_det'#ang20150421t181252
    # basepath='/lustre/bbue/ch4/srcfinder/tiles/thompson_training/256/ang20150419t163741_det'
    # basepath='/lustre/bbue/ch4/srcfinder/tiles/thompson_training/224/ang20150419t163741'
    # basepath='/lustre/bbue/ch4/srcfinder/tiles/thorpe_training/256/ang20160910t182946_det_notn'
    # basepath='/lustre/bbue/ch4/srcfinder/tiles/thorpe_training/256/ang20160910t182946_det_notn'
    trainf = basepath+'_train.txt'
    testf = basepath+'_test.txt'

    tile_shape = [75,75] #[200,200]
    
    load_func = lambda imgf: imread_tile(imgf,tile_shape=tile_shape)
    loadargs = (trainf,testf)
    
    loadkwargs = {'conserve_memory':True,'load_func':load_func,
                  'exclude_pattern':None} # 'exclude_pattern':'/tn/'}

    X_train,y_train,_,_ = load_data(*loadargs,**loadkwargs)
    pos_idx,neg_idx = np.where(y_train==1)[0],np.where(y_train==0)[0]

    image_files = np.array(X_train.files)
    flights = np.array([Xi.split('/')[-3] for Xi in image_files])    
    flights_uniq = np.unique(flights)
    for i,flighti in enumerate(flights_uniq):
        fmaski = flights==flighti
        yi = y_train[fmaski]
        fpos,fneg = np.count_nonzero(yi==1),np.count_nonzero(yi==0)
        fn = fpos+fneg
        print(', '.join([flighti,fpos,fneg,fn]))
        if fn<50:
            print(image_files[fmaski]),raw_input()
    
    datagen = ImageDataGenerator(**batch_datagen_params)

    datagen.fit(X_train,seed=42)
    
    epochs=5
    batch_size = 500
    batch_idx = np.arange(batch_size,dtype=int)
    batches = 0
    n_batches = len(X_train) / batch_size
    flowkw = dict(batch_size=batch_size,shuffle=True,seed=42,
                  save_to_dir=None,save_prefix='',save_format='png')
    random_transform = datagen.random_transform
    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for X_batch, y_batch in datagen.flow(X_train, y_train, **flowkw):
            n_bi = X_batch.shape[0]
            if n_bi < batch_size:
                n_aug = batch_size-n_bi
                aug_idx = batch_idx[randperm(batch_size)] % n_bi
                bal_idx = balance_classes(y_batch[aug_idx],verbose=True)
                aug_idx = np.r_[aug_idx,aug_idx[bal_idx]]
                aug_idx = aug_idx[randperm(len(aug_idx),n_aug)]
                X_aug = map(random_transform,X_batch[aug_idx])
                y_aug = y_batch[aug_idx]
                X_batch = np.r_[X_batch,X_aug]
                y_batch = np.r_[y_batch,y_aug]

            #model.fit(x_batch, y_batch)
            print("Batch",batches,X_batch.shape)
            
            fig,ax = pl.subplots(1,3,sharex=True,sharey=True)
            ax[0].imshow(X_batch[-3])
            ax[1].imshow(X_batch[-1])
            ax[2].imshow(X_batch[-2])
            pl.show()
            batches += 1
            if batches >= n_batches:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
            
    print('Done at',batches,'batches'), raw_input()
    
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


    X_train_batch,y_train_batch = imaugment_perturb(X_batch,y_batch,
                                                naugpos=naugpos,
                                                naugneg=naugneg)
    pos_idx,neg_idx = np.where(to_binary(y_train_batch)==1)[0],\
                      np.where(to_binary(y_train_batch)==0)[0]
    print(y_train_batch.shape,len(pos_idx),len(neg_idx)),raw_input()
    
    fig,ax = pl.subplots(2,2,sharex=True,sharey=True)
    ax[0,0].imshow(X_train[pos_idx[0]].squeeze())
    ax[0,1].imshow(X_train[pos_idx[1]].squeeze())
    ax[1,0].imshow(X_train[neg_idx[0]].squeeze())
    ax[1,1].imshow(X_train[neg_idx[1]].squeeze())
    pl.show()
