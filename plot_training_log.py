#!/usr/bin/env python

from __future__ import absolute_import, division, print_function#, unicode_literals

import sys,os
import csv
from pylib import *
from pandas import DataFrame

if len(sys.argv)==1:
    print('usage: %s training_log.csv'%sys.argv[0])
    sys.exit(0)

csvfile = sys.argv[1]

reader = csv.reader(open(csvfile, 'rU')) #, dialect=csv.excel_tab)
trhdr  = reader.next()
trhist = np.array([map(float,l) for l in reader if len(l)==len(trhdr)])
trhist = DataFrame(trhist,columns=trhdr)
lr_col = 'clr_lr' if 'clr_lr' in trhdr else 'lr'

epoch = trhist['epoch']
loss=trhist['loss']
vloss=trhist['val_loss']
vpre=trhist['val_precision']
vrec=trhist['val_recall']
vfs=trhist['val_fscore']
lr = trhist[lr_col]
msg = 'epochs %d'%epoch.max()
np.set_printoptions(precision=2)
if 'cputime' in trhdr:
    cputime = trhist['cputime']
    msg += ', total cputime: %.2f minutes'%(cputime.sum()/60.0)
    msg += '\nextrema(cputime): "%s"'%str((np.c_[extrema(cputime)]))
    msg += ', median(cputime): "%s"'%str((np.c_[[median(cputime)]]))
    msg += ', mad(cputime): "%s"'%str((np.c_[[mad(cputime)]]))
    
i = np.argmax(vfs)
vfsi,vprei,vreci = vfs[i],vpre[i],vrec[i]
lossi,vlossi = loss[i],vloss[i]

dper = 0.2

fsmin,fsmax = extrema(np.r_[vprei,vreci,vfsi].ravel())
fsdelt = (fsmax-fsmin) * 2*dper
print('fsmin,fsmax: "%s"'%str((fsmin,fsmax)))

lossmin,lossmax = extrema(vloss)
lossdelt = (lossmax-lossmin) * dper

lrmin,lrmax = extrema(lr)
lrdelt = (lrmax-lrmin) * dper


nsub=3
sz=10
fig,ax = pl.subplots(nsub,1,sharex=True,sharey=False,figsize=(12,nsub*2+1))
ax[0].scatter(epoch,vpre,sz,label='val_precision max=%f, fsmax=%f'%(vpre.max(),vprei),color='m')
ax[0].scatter(epoch,vrec,sz,label='val_recall max=%f, fsmax=%f'%(vrec.max(),vreci),color='b')
ax[0].scatter(epoch,vfs,sz,label='val_fscore max=%f'%vfsi,color='r')
ax[0].set_ylim(fsmin-fsdelt,fsmax+fsdelt)
ax[0].legend()
ax[0].set_title(msg)

ax[1].scatter(epoch,loss,sz,label='loss min=%f, fsmin=%f'%(loss.min(),lossi),color='b')
ax[1].scatter(epoch,vloss,sz,label='val_loss min=%f, fsmin=%f'%(vloss.min(),vlossi),color='r')
ax[1].set_ylim(lossmin-lossdelt,lossmax+lossdelt)
ax[1].legend()

ax[2].scatter(epoch,lr,sz,label='learning rate',color='b')
ax[2].set_ylim(lrmin-lrdelt,lrmax+lrdelt)
              
ax[2].legend()

pl.show()
