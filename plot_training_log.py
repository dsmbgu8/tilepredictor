#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys,os
import csv
from pylib import *
from pandas import DataFrame

if __name__ == '__main__':
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
    msg = 'Total epochs %d'%epoch.max()
    np.set_printoptions(precision=2)
    if 'cputime' in trhdr:
        cputime = trhist['cputime']
        msg += ', cputime: %.2f (m)'%(cputime.sum()/60.0)
        msg += '\n(cputime/epoch (s) median: %.3f, mad: %.3f)'%(median(cputime),mad(cputime))

    i = np.argmax(vfs)
    vfsi,vprei,vreci = vfs[i],vpre[i],vrec[i]
    lossi,vlossi = loss[i],vloss[i]

    dper = 0.2

    fsmin,fsmax = extrema(vfs)
    fsdelt = (fsmax-fsmin) * 2*dper
    print('fsmin,fsmax: "%s"'%str((fsmin,fsmax)))


    recmin,recmax = extrema(vrec)
    recdelt = (recmax-recmin) * 2*dper
    print('recall min,max: "%s"'%str((recmin,recmax)))

    premin,premax = extrema(vpre)
    predelt = (premax-premin) * 2*dper
    print('precision min,max: "%s"'%str((premin,premax)))

    axmin = max(0,min([fsmin,recmin,premin]))
    axmax = min(1,max([fsmax,recmax,premax]))
    axdelt = (axmax-axmin) * dper

    lossmin,lossmax = extrema(np.hstack([loss,vloss]))
    il = np.argmin(vloss)
    vlfsi,vlprei,vlreci = vfs[il],vpre[il],vrec[il]
    lossdelt = (lossmax-lossmin) * dper
    print('loss min,max: "%s"'%str((lossmin,lossmax)))

    lrmin,lrmax = extrema(lr)
    lrdelt = (lrmax-lrmin) * dper

    nsub=5
    sz=10
    fig,ax = pl.subplots(nsub,1,sharex=True,sharey=False,figsize=(9,nsub*2))
    ax[0].scatter(epoch,loss,sz,label='loss min=%.4f'%(loss.min()),color='b')
    ax[0].scatter(epoch,vloss,sz,label='val_loss min=%.4f (f=%.4f,p=%.4f,r=%.4f)'%(vloss.min(),vlfsi,vlprei,vlreci),color='r')
    ax[0].set_ylim(lossmin-lossdelt,lossmax+lossdelt)

    ax[1].scatter(epoch,vfs,sz,label='val_f1 max=%.4f (l=%.4f,p=%.4f,r=%.4f)'%(vlfsi,vlossi,vprei,vreci),color='r')
    ax[1].set_ylim(axmin-axdelt,axmax+axdelt)
    ax[2].scatter(epoch,vpre,sz,label='val_precision max=%.4f (r=%.4f)'%(vpre.max(),vrec[vpre==vpre.max()].values[-1]),color='m')
    ax[2].set_ylim(axmin-axdelt,axmax+axdelt)
    ax[3].scatter(epoch,vrec,sz,label='val_recall max=%.4f (p=%.4f)'%(vrec.max(),vpre[vrec==vrec.max()].values[-1]),color='b')
    ax[3].set_ylim(axmin-axdelt,axmax+axdelt)

    flrmax = vfs[lr==lr.max()].mean()
    flrmed = vfs[lr==lr.median()].mean()
    flrmin = vfs[lr==lr.min()].mean()
    ax[4].scatter(epoch,lr,sz,label='learning rate (fmin=%.4f,fmed=%.4f,fmax=%.4f)'%(flrmin,flrmed,flrmax),
                  color='b')
    ax[4].set_ylim(lrmin-lrdelt,lrmax+lrdelt)

    for j in range(nsub):
        ax[j].legend()
        ax[j].axvline(i,c='r',ls=':')
        ax[j].axvline(il,c='b',ls=':')        
    pl.suptitle(msg)

    do_show=False
    if do_show:
        pl.show()
    else:
        savefig(csvfile.replace('.csv','.png'))
