#!/usr/bin/env python
from __future__ import division, print_function, absolute_import, unicode_literals

from os.path import exists as pathexists

import numpy as np
from skimage.io import ImageCollection, imread as skimread 

from tilepredictor_util import *

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
    from model_package import collect_batch, imaugment_perturb, batch_datagen_params 
    #from keras.utils.np_utils import to_categorical 
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
    pos_idx,neg_idx = np.where(np.argmax(y_train_batch,-1)==1)[0],\
                      np.where(np.argmax(y_train_batch,-1)==0)[0]
    print(y_train_batch.shape,len(pos_idx),len(neg_idx)),raw_input()
    
    fig,ax = pl.subplots(2,2,sharex=True,sharey=True)
    ax[0,0].imshow(X_train[pos_idx[0]].squeeze())
    ax[0,1].imshow(X_train[pos_idx[1]].squeeze())
    ax[1,0].imshow(X_train[neg_idx[0]].squeeze())
    ax[1,1].imshow(X_train[neg_idx[1]].squeeze())
    pl.show()
