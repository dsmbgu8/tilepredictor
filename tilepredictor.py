#!/usr/bin/env python
from __future__ import division, print_function, absolute_import, unicode_literals

import sys
from glob import glob

from pylib import *

tilepredictor_home = pathsplit(__file__)[0]
sys.path.append(abspath(tilepredictor_home))

from imtiler.imtiler.util import *
from tilepredictor_util import *
from model_package import *

test_epoch = 1
save_epoch = 1


# softmax probs are:
# [0.0,0.5] class 0
# [0.5,1.0] class 1
# by default, we map these to [0.0,0.5] and store the class label separately
# scale_probs maps the [0.0,0.5] probs to [0.0,1.0] per class
scale_probs = True
randomize = False

# network architecture / training parameters
nb_hidden = 1024 # nodes in last FC layer before classification
nb_classes = 2 # we currently only consider binary classification problems 
output_shape = [nb_hidden,nb_classes]

augment_size = [1,1]

def collect_tile_uls(tile_path,tile_id='det',tile_ext='.png'):
    """
    collect_tile_files(imgid,tile_dir,tile_dim,tile_id,tile_ext=tile_ext)
    
    Summary: collect tiles and uls for precomputed path tile_dir
    
    Arguments:
    - imgid: base image for precomputed tiles
    - tile_dir: base tile directory, must be of the form:
    
        tile_path = tile_dir/tile_dim/imgid/tile_prefix
    
      ...and must contain filenames of the form:
    
        tile_file = *tile_id*tile_ext

      e.g., bue_training/100/ang20160914t203630/fp/*det*.png
            bue_training/120/ang20160914t203630/det/*rgb*.png
    
    - tile_dim: tile dimension
    - tile_id: tile type identifier (e.g., det, rgb)
    
    Keyword Arguments:
    - tile_ext: tile extension (default=.png)
    
    Output:
    - output
    """
    
    tile_files = []
    if not tile_path:
        return tile_files
    elif not pathexists(tile_path):
        warn('tile_dir "%s" not found'%tile_path)
        return tile_files
        
    tile_pattern = '*'+tile_id+'*'+tile_ext    
    for prefix in tile_prefix:        
        load_pattern = pathjoin(tile_path,prefix,tile_pattern)
        tile_files.extend(glob(load_pattern))

    tile_uls = map(tilefile2ul,tile_files)
    return tile_files, tile_uls

def collect_image_uls(img_test,tile_dim,tile_stride):
    from skimage.measure import label as imlabel
    uls = set([])
    nzmask = img_test[:,:,:3].any(axis=2)
    if (~nzmask).all():
        return uls
    nzcomp = imlabel(nzmask, background=0, return_num=False, connectivity=2)    
    nzrows,nzcols = nzmask.nonzero()
    nzlabs = nzcomp[nzrows,nzcols]
    ucomp,ucounts = np.unique(nzlabs,return_counts=True)        
    
    if tile_stride >= 1:
        stride = tile_stride
    else:
        stride = max(1,tile_stride*tile_dim)
        
    stride = int(stride)
    print('Collecting salience uls for',len(ucomp),
          'nonzero components with stride=',stride,'pixels')

    # offset by -1 pixel to 
    tile_off = tile_dim-1
    for ulab,unum in zip(ucomp,ucounts):
        umask = nzlabs==ulab
        urow,ucol = nzrows[umask],nzcols[umask]
        rmin,rmax = urow.min()-tile_off,urow.max()+tile_off
        cmin,cmax = ucol.min()-tile_off,ucol.max()+tile_off

        # add center pixel for each ccomp by default
        rmu,cmu = int(round(urow.mean())),int(round(ucol.mean()))
        uls.add((rmu,cmu))
        for r in range(rmin,rmax+stride,stride):
            ri = min(r,rmax)
            for c in range(cmin,cmax+stride,stride):
                uls.add((ri,min(c,cmax)))

        cstep = (cmax-cmin)/stride
        rstep = (rmax-rmin)/stride
        if rstep>1 or cstep>1:
            print('Collected',len(uls),'tiles for component',ulab,'with',
                  unum,'pixels, rstep=',rstep,'cstep=',cstep)

    uls = np.int32(list(map(list,uls)))    
    print(uls.shape[0],'total uls collected')
    
    return uls

def generate_tiles(img_test,tile_uls,tile_dim):
    for tile_ul in tile_uls:
        tile_img = extract_tile(img_test,tile_ul,tile_dim,verbose=False)
        if tile_img.any():
            yield tile_img

def generate_image_batch(img_test,tile_uls,tile_dim,batch_size,
                         preprocess=None):
    if preprocess is None:
        preprocess = lambda Xi: Xi
    n_test = len(tile_uls)
    n_out = min(n_test,batch_size)
    X_batch = np.zeros([n_out,tile_dim,tile_dim,3])
    b_off = 0
    while b_off < n_test:
        b_end = min(b_off+batch_size,n_test)
        print('Computing predictions for samples %d through %d'%(b_off,b_end))
        b_i=0
        while b_off < n_test and b_i < batch_size:
            tile_img = extract_tile(img_test,tile_ul[b_off],tile_dim,verbose=False)
            # note: the loader script should already preprocess each test sample
            if tile_img.any():
                X_batch[b_i] = preprocess(tile_img)
                b_i += 1
            b_off += 1
        yield X_batch[:b_i]
            
def generate_test_batch(X_test,batch_size,preprocess=None):
    if preprocess is None:
        preprocess = lambda Xi: Xi

    tile_dim = X_test[0].shape[0]
    n_test = len(X_test)
    n_out = min(n_test,batch_size)
    X_batch = np.zeros([n_out,tile_dim,tile_dim,3])

    b_off = 0
    while b_off < n_test:
        b_end = min(b_off+batch_size,n_test)
        print('Computing predictions for samples %d through %d'%(b_off,b_end))
        for i in range(b_off,b_end):
            # note: the loader script should already preprocess each test sample
            X_batch[i-b_off] = preprocess(X_test[i])
        b_off += batch_size
        yield X_batch

def compute_predictions(model,X_test,n_classes,scale_probs=True):
    pred_prob = model.predict(X_test)
    pred_lab = np.int8(np.argmax(pred_prob,axis=-1))
    pred_prob = np.amax(pred_prob,axis=-1)

    if scale_probs:
        pred_prob = 2*(pred_prob-0.5)

    return dict(pred_lab=pred_lab,pred_prob=pred_prob)

def aximshow(ax,img,ylab,vmin=None,vmax=None,cmap=None):
    transpose = [1,0] if img.shape[0]>img.shape[1] else [0,1]
    if img.ndim>2:
        transpose.append(2)
    ret = ax.imshow(img.transpose(transpose),vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set_ylabel(ylab)
    return ret

def plot_pred_images(img_data,pred_out,mapinfo=None,img_lab=[],
                     output_dir=None,output_prefix='',
                     mask_zero=False,mask_prob=False,
                     mask_nodata=False,do_show=False):

    img_pred = pred_out['img_pred']
    img_prob = pred_out['img_prob']
    img_mask = pred_out['img_mask']

    def save_png_output(imagebase,img_data):
        outf = pathjoin(output_dir,'_'.join([output_prefix,imagebase]))+'.png'
        img_out = img_data.copy()
        img_out[...,3] = np.where(img_mask,0,255)
        imsave(outf,img_out)
        print('Saved',outf)

    def save_envi_output(imagebase,img_data,mapinfo,mask_nodata=True):
        mapout = mapinfo.copy()
        mapout['rotation']=-mapout['rotation']
        img_out = img_data.copy()
        if mask_nodata:
            img_out[img_mask] = -9999
            
        outf = pathjoin(output_dir,'_'.join([output_prefix,imagebase]))
        array2img(outf,img_out,mapinfostr=mapdict2str(mapout),overwrite=1)
        print('Saved',outf)        
    
    if output_dir and mapinfo:        
        save_envi_output('prob_softmax',np.float32(img_prob),mapinfo,
                         mask_nodata=mask_nodata)
        save_envi_output('pred_softmax',np.int16(img_pred),mapinfo,
                         mask_nodata=mask_nodata)
        if not mask_nodata:
            save_envi_output('nodata_mask',np.int16(img_mask),mapinfo,
                             mask_nodata=mask_nodata)

    invscalef = 1/255. if img_data.max()>1. else 1.
    img_test = img_data.copy() * invscalef
    
    nrows,ncols = img_test.shape[0],img_test.shape[1]    
    
    # pixels with no predictions
    img_counts = img_pred.copy()
    img_total = img_counts.sum(axis=2)    

    # find valid pixels with at least 1 prediction, ignore everything else
    img_invalid = ~np.isfinite(img_total) | img_mask
    img_haspred  = (img_total!=0)
    img_nopred  = (img_total==0)
    
    # class predictions (+1/-1 pos/neg, 0=ignore)
    img_pos = np.bool8(np.argmax(img_counts,axis=2)==1)
    img_neg = ~(img_pos | img_invalid | img_nopred)
    img_class = np.zeros([nrows,ncols],dtype=np.int8)
    img_class[img_pos] =  1
    img_class[img_neg] = -1

    # prediction counts
    img_vote = np.zeros_like(img_total,dtype=np.float32)
    img_vote[img_pos] =  img_counts[img_pos,1]
    img_vote[img_neg] = -img_counts[img_neg,0]
    
    # vote confidence (#classpredictions/#totalpredictions per-pixel)
    img_vcon = np.zeros_like(img_total,dtype=np.float32)
    img_vcon[img_pos] = img_vote[img_pos] / img_total[img_pos]
    img_vcon[img_neg] = img_vote[img_neg] / img_total[img_neg]

    # prob confidence (probs*#classpredictions/#totalpredictions per-pixel)
    img_pcon = np.zeros_like(img_total,dtype=np.float32)
    img_pcon[img_pos] = img_vcon[img_pos]*img_prob[img_pos,1]
    img_pcon[img_neg] = img_vcon[img_neg]*img_prob[img_neg,0]
                      
    # flatten predicted class probabilities
    img_prob = np.where(img_pos,img_prob[...,1],-img_prob[...,0])
    img_prob[img_invalid | img_nopred] = 0 # ignore pixels without predictions

    if mask_zero:
        img_zero = (img_test==0).all(axis=2)
        img_mask[img_zero] = 1
        pzeromask = img_zero.sum()/float(nrows*ncols)
        print('Zero-valued pixels masked: %5.2f%%'%(100*pzeromask))
        
    if mask_prob:
        img_thr = np.abs(img_prob)<prob_thresh
        pthrmask = img_thr.sum()/float(nrows*ncols)
        img_mask[img_thr] = 1 
        print('P<%5.2f% pixels masked: %5.2f%%'%(100*prob_thresh,100*pthrmask))

    if img_mask.any() != 0:
        pmask = np.count_nonzero(img_mask)
        print('Masked pixels: %5.2f%%'%(100*pmask/float(nrows*ncols)))
        # use mask as alpha channel if img_test == rgb
        if img_test.shape[-1]==3:
            img_test = np.dstack([img_test,float32(~img_mask)])
        img_class[img_mask] = 0
        img_pred[img_mask] = 0
        img_prob[img_mask] = 0
        img_vcon[img_mask] = 0
        img_pcon[img_mask] = 0

    if not pathexists(output_dir):
        print('Creating output directory %s'%output_dir)
        os.makedirs(output_dir)
        
    # save float32 envi images before converting to RGBA pngs 
    if output_dir is not None:
        save_envi_output('pred',np.int16(img_pred),mapinfo)
        save_envi_output('prob',np.float32(img_prob),mapinfo)
        
    max_vote = max(abs(extrema(img_vote)))    
    
    fig,ax = pl.subplots(5,1,sharex=True,sharey=True,figsize=(16,10))
    aximshow(ax[0],img_test,'test') 
    aximshow(ax[1],img_class,'class',vmin=-1.0,vmax=1.0)          
    aximshow(ax[2],img_prob,'prob',vmin=-1.0,vmax=1.0)
    aximshow(ax[3],img_vcon,'vcon',vmin=-1.0,vmax=1.0)
    aximshow(ax[4],img_pcon,'pcon',vmin=-1.0,vmax=1.0)

    if len(img_lab)!=0:
        lab_mask = (img_lab>0)
        img_over = np.zeros([nrows,ncols,4])
        img_over[lab_mask, :] = [.4,.2,.6,.5]
        aximshow(ax[0],img_over,'test+labels')
        aximshow(ax[1],img_over,'class+labels')    
        
    if output_dir is not None:
        img_vote = array2rgba(img_vote,  vmin=-max_vote,vmax=max_vote)        
        img_class = array2rgba(img_class,vmin=-1.0,vmax=1.0)
        img_prob = array2rgba(img_prob,  vmin=-1.0,vmax=1.0)
        img_vcon = array2rgba(img_vcon,  vmin=-1.0,vmax=1.0)
        img_pcon = array2rgba(img_pcon,  vmin=-1.0,vmax=1.0)        
        save_png_output('vote',img_vote)
        save_png_output('pred',img_class)
        save_png_output('vcon',img_vcon)
        save_png_output('prob',img_prob)
        save_png_output('pcon',img_pcon)

def write_csv(csvf,imgid,pred_list,tile_dim,prob_thresh=0.0,imgmap=None):
    from LatLongUTMconversion import UTMtoLL
    # geolocate detections with .hdr files
    #tile_out = pred_out['pred_list']
    
    keep_mask = float32(pred_list[:,-2])>=prob_thresh
    tile_keep = pred_list[keep_mask,:]
    line,samp = np.float32(tile_keep[:,[1,2]]).T
    preds,probs = tile_keep[:,-1],tile_keep[:,-2]    

    if imgmap:        
        zone,hemi = imgmap['zone'],imgmap['hemi']
        zonealpha = zone + ('N' if hemi=='North' else 'M')

    outcsv = []
    for i in range(tile_keep.shape[0]):
        entryi = [imgid,'%d'%line[i],'%d'%samp[i],preds[i],probs[i]]
        if imgmap:
            utmx,utmy = sl2xy(samp[i],line[i],mapinfo=imgmap)
            lat,lon = UTMtoLL(23,utmy,utmx,zonealpha)
            entryi.extend(['%18.4f'%utmx,'%18.4f'%utmy,zone,hemi,lat[0],lon[0]])
        outcsv.append(', '.join(map(lambda v: str(v).strip(),entryi)))
    outcsv = np.array(outcsv,dtype=str)
    outcsv = outcsv[np.argsort((line*(samp.max()+1))+samp)]
    with(open(csvf,'w')) as fid:
        fid.write('\n'.join(list(outcsv))+'\n')
    print('saved',csvf) 

def compute_salience(model, img_test, tile_dim, tile_stride,
                     output_dir, output_prefix, tile_dir=None, img_map=None,
                     scale_probs=scale_probs, randomize=randomize,
                     verbose=False, do_show=False):
    
    rows,cols = img_test.shape[:2]
    img_pred = np.zeros([rows,cols,2],dtype=np.int32)
    img_prob = np.zeros([rows,cols,2],dtype=np.float32)

    rows,cols,bands = img_test.shape
    if bands==4:        
        img_rgb = img_test[...,:3]
        img_mask = (img_test[...,-1]==0)
    else:
        img_rgb = img_test
        img_mask = np.zeros([rows,cols],dtype=np.bool8)

    if not tile_dir:
        tile_uls = collect_img_uls(img_rgb,tile_dim,tile_stride)
    else:
        tile_files,tile_uls = collect_tile_uls(tile_dir)
    n_tiles = len(tile_uls)
        
    if randomize:
        tile_uls = tile_uls[randperm(n_tiles)]
    
    tile_gen = generate_tiles(img_rgb,tile_uls,tile_dim)
    pmsg = 'Collecting model output'
    print(pmsg+' for %d tiles'%n_tiles)
    pbar = progressbar(pmsg,n_tiles)
    pred_list = np.float32([n_tiles,4])
    for i,tile_image in pbar(enumerate(tile_gen)):
        tile_ul = tile_uls[i]
        tile_rows = slice(tile_ul[0],tile_ul[0]+tile_dim,None)
        tile_cols = slice(tile_ul[1],tile_ul[1]+tile_dim,None)

        pred_input = model.preprocess(tile_image).transpose(model.transpose)
        model_out = model.predict(pred_input[np.newaxis])[0]
        tile_pred = np.int8(np.argmax(model_out,axis=-1))
        tile_prob = np.amax(model_out,axis=-1) 

        if i<3:
            print('tile ul: "%s"'%str((tile_ul)))
            for bi in range(pred_input.shape[-1]):
                bmin,bmax=extrema(pred_input[...,bi].ravel())
                print('band[%d] min=%.3f, max=%.3f'%(bi,bmin,bmax))            

            tile_prob_scaled = 2*(tile_prob-0.5)
            model_scaled = model_out.copy()
            model_scaled[tile_pred] = tile_prob_scaled
            model_scaled[1-tile_pred] = 1-tile_prob_scaled
            print('tile_pred:            "%s"'%str((tile_pred)))
            print('tile_prob (unscaled): "%s"'%str((tile_prob)))
            print('tile_prob (scaled):   "%s"'%str((tile_prob_scaled)))
            print('model_out (unscaled): "%s"'%str((model_out)))
            print('model_out (scaled):   "%s"'%str((model_scaled)))

        if scale_probs:
            # probs originally in range [0.5,1.0]
            tile_prob = 2*(tile_prob-0.5)
            model_out[tile_pred] = tile_prob
            model_out[1-tile_pred] = 1-tile_prob
            
        # increment the index of the predicted class for this tile location
        img_pred[tile_rows,tile_cols,tile_pred] += 1

        # also update the sum of probability scores 
        img_prob[tile_rows,tile_cols,:] += model_out

        # keep state for all of the predictions 
        pred_list[i] = [tile_ul[0],tile_ul[1],tile_prob,tile_pred]

    # get average probabilities for each class by normalizing by pred counts
    for i in range(2):
        predi = img_pred[...,i]
        img_prob[...,i] /= np.float32(np.where(predi!=0,predi,1))
    
    pred_list = np.array(pred_list,dtype=np.float32)
    pos_mask = pred_list[:,-1]==1
    neg_mask = pred_list[:,-1]==0
    npos = np.count_nonzero(pos_mask)
    nneg = np.count_nonzero(neg_mask)
    total = float(npos+nneg)
    pos_probs = pred_list[pos_mask,-2]
    neg_probs = pred_list[neg_mask,-2]
    if npos!=0:
        print('positive predictions:',npos,npos/total,'prob min/max/mean/std:',
              pos_probs.min(),pos_probs.max(),pos_probs.mean(),pos_probs.std())
    if nneg!=0:
        print('negative predictions:',nneg,nneg/total,'prob min/max/mean/std:',
              neg_probs.min(),neg_probs.max(),neg_probs.mean(),neg_probs.std())

    if verbose:
        print('\n'.join(map(lambda s: ' '.join(s),pred_list)))

    pred_out = dict(pred_list=pred_list,img_pred=img_pred,
                    img_prob=img_prob,img_mask=img_mask)

    if output_dir:
        if not pathexists(output_dir):
            warn('Output dir "%s" not found'%output_dir)
            return {}
        
        plot_pred_images(img_rgb,pred_out,mapinfo=img_map,img_lab=img_lab,
                         output_dir=output_dir,output_prefix=output_prefix,
                         mask_zero=False, do_show=do_show)
        
    return pred_out


def image_salience(model, img_data, tile_dim, tile_stride,
                   output_dir, output_prefix, img_map=None, img_lab=[],
                   scale_probs=scale_probs, verbose=False,
                   transpose=True, do_show=False):
    from skimage.util.shape import view_as_windows
    
    if tile_stride >= 1:
        stride = tile_stride
    else:
        stride = max(1,(tile_stride*tile_dim))
    stride = int(stride)
    
    img_test = (img_data.transpose((1,0,2)) if transpose else img_data).copy()
    rows,cols,bands = img_test.shape

    # pad to fit into (modulo stride) and (modulo tile_dim) increments
    print('image rows,cols: "%s"'%str((rows,cols)))
    cadd = stride-(cols%stride)
    radd = stride-(rows%stride)
    cadd += tile_dim-(cols+cadd)%tile_dim
    radd += tile_dim-(rows+radd)%tile_dim
    
    if cadd > 0:
        csbuf = np.zeros([rows,tile_dim,bands])
        cebuf = np.zeros([rows,cadd,bands])
        img_test = np.hstack([csbuf,img_test,cebuf])
        cols += tile_dim+cadd
        
    if radd > 0:
        rsbuf = np.zeros([tile_dim,cols,bands])
        rebuf = np.zeros([radd,cols,bands])
        img_test = np.vstack([rsbuf,img_test,rebuf])
        rows += tile_dim+radd

    print('padded rows,cols: "%s"'%str((rows,cols)))
        
    # need to copy the rgb bands so view_as_windows will work(?)
    img_rgb = img_test[...,:3].copy()
    img_mask = np.zeros([rows,cols],dtype=np.bool8)
    if bands==4:        
        img_mask[img_test[...,-1]==0] = 1
 
    #ridx,cidx = map(lambda a: a.ravel(),np.meshgrid(range(cols),range(rows)))
    img_pred = np.zeros([rows,cols,2],dtype=np.int32)
    img_prob = np.zeros([rows,cols,2],dtype=np.float32)

    input_win = (tile_dim,tile_dim,3)


    half_dim = tile_dim//2
    rrange = np.arange(0,rows-tile_dim+1,stride)
    crange = np.arange(0,cols-tile_dim+1,stride)
    n_rb,n_cb = len(rrange),len(crange)
    pmsg = 'Collecting predictions'
    print(pmsg+', window size = %d x %d tiles (tile_dim=%d, stride=%d)'%(n_rb,
                                                                         n_cb,
                                                                         tile_dim,
                                                                         stride))
    pbar = progressbar(pmsg,n_rb)
    for i,rbeg in pbar(enumerate(rrange)):
        rend = rbeg+tile_dim
        rinput = img_rgb[rbeg:rend]

        #print('rinput.shape: "%s"'%str((rinput.shape)))
        rwin = view_as_windows(rinput, input_win, step=stride).squeeze()

        if rwin.ndim == 3:
            rwin = rwin[np.newaxis]
        elif rwin.ndim != 4:
            warn('Unknown window dimensions, unable to proceed')
            return []
        
        cmask = rwin.reshape(rwin.shape[0],-1).any(axis=1)        
        if (~cmask).all():
            continue

        #print(rwin.shape,ridx.shape,np.count_nonzero(ridx))
        #cmask = view_as_windows(img_mask[rbeg:rend], input_win, step=stride).squeeze()

        # rinput = [cols/tile_dim,3,tile_shape,tile_shape]
        rinput = model.preprocess(rwin[cmask])
        if model.transpose:
            rinput = rinput.transpose([0]+[d+1 for d in model.transpose])

        batch_size = min(np.count_nonzero(cmask),32)
        #begtime = gettime()
        # probrowi = [cols/tile_dim,2] softmax class probabilities
        probrowi = model.predict(rinput,batch_size=batch_size)

        # predrowi = [cols/tile_dim,1] softmax class predictions
        #predrowi = np.int8(np.argmax(probrowi,axis=-1))
        
        # if print_state and (i%10)==0:
        #     print('prediction time: %0.3f seconds'%(gettime()-begtime))
        #     print('row block %d of %d'%(i,n_rb))
        # rimage = img_rgb[rbeg:rend]
        #     print('rimage.shape: "%s"'%str((rimage.shape)))
        #     print('rwin.shape: "%s"'%str((rwin.shape)))
        #     print('rinput.shape: "%s"'%str((rinput.shape)))            

        
        #rpred = np.int8(np.argmax(routput,axis=-1))
        #rprob = np.amax(routput,axis=-1) 
        #print('routput.shape: "%s"'%str((routput.shape)))
        #print('rpred.shape: "%s"'%str((rpred.shape)))
        #print('rprob.shape: "%s"'%str((rprob.shape)))

        # rpred = img_pred[rbeg:rend]
        # rprob = img_prob[rbeg:rend]
        # output_win = (tile_dim,tile_dim,2)
        # predwin = view_as_windows(rpred, output_win, step=stride).squeeze()
        # probwin = view_as_windows(rprob, output_win, step=stride).squeeze()

        # predwin[rpred] += 1
        # probwin[...,0] += routput[...,0]
        # probwin[...,1] += routput[...,1]
        
        #for j,cbeg in enumerate(range(0,cols-tile_dim,stride)):
        #    cend = cbeg+tile_dim
        #img_pred[ridx+rbeg,cidx,rpred] 
        #img_prob[ridx+rbeg,cidx,:] += routput

        for cbeg,probij in zip(crange[cmask],probrowi):
            cend,predij = cbeg+tile_dim, np.int8(probij[1]>probij[0])

            if scale_probs:
                probij[predij] = 2*(probij[predij]-0.5)
                probij[1-predij] = 1-probij[predij]
            
            img_prob[rbeg:rend,cbeg:cend,0] += probij[0]
            img_prob[rbeg:rend,cbeg:cend,1] += probij[1]
            img_pred[rbeg:rend,cbeg:cend,predij] += 1
            
    # get average probabilities for each class by normalizing by pred counts
    img_total = img_pred.sum(axis=2)
    # find valid pixels with at least 1 prediction
    img_haspred = img_total!=0

    img_prob[img_haspred,0] /= img_total[img_haspred]
    img_prob[img_haspred,1] /= img_total[img_haspred]

    img_amax = np.argmax(img_prob,axis=2)
    img_pos,img_neg = (img_amax==1),(img_amax==0)    
    if 1:
        print('img_prob[img_pos,0].min(): "%s"'%str((img_prob[img_pos,0].min())))
        print('img_prob[img_pos,1].min(): "%s"'%str((img_prob[img_pos,1].min())))
        print('img_prob[img_pos,0].max(): "%s"'%str((img_prob[img_pos,0].max())))
        print('img_prob[img_pos,1].max(): "%s"'%str((img_prob[img_pos,1].max())))
        print('img_prob.sum(axis=2).max(): "%s"'%str((img_prob.sum(axis=2).max())))

    # if scale_probs:
    #     img_prob[img_pos,1] = 2*(img_prob[img_pos,1]-0.5)
    #     img_prob[img_pos,0] = 1-img_prob[img_pos,1]
    #     img_prob[img_neg,0] = 2*(img_prob[img_neg,0]-0.5)
    #     img_prob[img_neg,1] = 1-img_prob[img_neg,0]
    #     img_prob[~img_valid] = 0

    if 0:
        print('after')
        print('img_prob[img_pos,0].min(): "%s"'%str((img_prob[img_pos,0].min())))
        print('img_prob[img_pos,1].min(): "%s"'%str((img_prob[img_pos,1].min())))    
        print('img_prob[img_pos,0].max(): "%s"'%str((img_prob[img_pos,0].max())))
        print('img_prob[img_pos,1].max(): "%s"'%str((img_prob[img_pos,1].max())))
        print('img_prob.sum(axis=2).max(): "%s"'%str((img_prob.sum(axis=2).max())))
        raw_input()

    
        
    # crop row,col buffers
    rbeg,rend = tile_dim,rows-radd
    cbeg,cend = tile_dim,cols-cadd

    img_rgb  = img_rgb[rbeg:rend,cbeg:cend,:]
    img_prob = img_prob[rbeg:rend,cbeg:cend,:]
    img_pred = img_pred[rbeg:rend,cbeg:cend,:]
    img_mask = img_mask[rbeg:rend,cbeg:cend]
        
    if transpose:
        # transpose back to original shape
        img_rgb  = img_rgb.transpose((1,0,2))
        img_pred = img_pred.transpose((1,0,2))
        img_prob = img_prob.transpose((1,0,2))
        img_mask = img_mask.transpose((1,0))

    pred_out = dict(img_prob=img_prob,
                    img_pred=img_pred,
                    img_mask=img_mask)

    if output_dir:
        if not pathexists(output_dir):
            warn('Output dir "%s" not found'%output_dir)
            return {}
        
        plot_pred_images(img_rgb,pred_out,mapinfo=img_map,img_lab=img_lab,
                         output_dir=output_dir,output_prefix=output_prefix,
                         do_show=do_show)
        
    return pred_out
            
if __name__ == '__main__':
    import load_data

    import argparse
    parser = argparse.ArgumentParser(description="Tile Predictor")

    # model initialization params
    parser.add_argument("-m", "--model_package", default=default_package,
                        help="Model package (%s)"%('|'.join(valid_packages)),
                        type=str)
    parser.add_argument("-f", "--model_flavor", default=default_flavor,
                        help="Model flavor (%s)"%('|'.join(valid_flavors)),
                        type=str)
    # output paths
    parser.add_argument("-s", "--state_dir", type=str, default=default_state_dir,
                        help="Path to save network output state (default=%s)"%default_state_dir)
    parser.add_argument("-o", "--output_dir", type=str,
                        help="Path to save output images/metadata (default=state_dir)")

    
    parser.add_argument("-w", "--weight_file", type=str,
                        help="Weight file to write or load")
    parser.add_argument("-d", "--tile_dim", type=int, default=tile_dim,
                        help="Dimension of input tiles")
    parser.add_argument("-b", "--batch_size", default=batch_size, type=int,
                        help="Batch size (default=%d)"%batch_size)
    parser.add_argument("--seed", default=random_state, type=int,
                        help="Random seed (default=%d)"%random_state)
    parser.add_argument("-a", "--augment", default=1.0, type=float, 
                        help="Augmentation ratio (pos/neg, default=1)"%augment_size)
    
    
    # model training params
    parser.add_argument("--train_file", help="Training data file", type=str)    
    parser.add_argument("--test_file", help="Test data file", type=str)
    parser.add_argument("--mean_file", help="Mean image data file", type=str)
    
    parser.add_argument("--test_epoch", type=int, default=test_epoch,
                        help="Epochs between testing (default=%d)"%test_epoch)
    parser.add_argument("--test_percent",type=float, default=0.2,
                        help="Percentage of test data to use during validation")
    parser.add_argument("--save_epoch", type=int, default=save_epoch,
                        help="Epochs between save cycles (default=%d)"%save_epoch)
    parser.add_argument("--save_preds", action='store_true',
                        help="Write preds to text file every save_epoch iterations'")
    parser.add_argument("--conserve_memory", action='store_true',
                        help="Conserve memory by not caching train/test tiles'")
    parser.add_argument("--balance", action='store_true',
                        help='Balance minority class in training data')
    
    # prediction threshold
    parser.add_argument("--prob_thresh", type=float, default=0.0,
                        help="Threshold on prediction probabilities [0,100]")

    # image salience params
    parser.add_argument("--image_dir", type=str,
                        help="Path to input images(s) for salience map generation")
    parser.add_argument("--image_load_pattern", type=str, default=load_pattern,
                        help="Load pattern for input/test images(s) (default=%s)"%load_pattern)
    
    parser.add_argument("--tile_stride", type=float, default=0.5,
                        help="Tile stride (# pixels or percentage of tile_dim) for salience map generation (default=0.5)")
    parser.add_argument("--tile_dir", type=str,
                        help="Path to directory containing precomputed tiles for each image in image_dir")
    parser.add_argument("-p","--plot", action='store_true',
                        help="Plot salience outputs")

    
    # misc
    parser.add_argument("--pred_best", action='store_true',
                        help="Load best model in state_dir and compute/save preds for test_file'")
    parser.add_argument("--clobber", action='store_true',
                        help="Overwrite existing files.")
    
    # hdr file params for geolocalization (optional)
    parser.add_argument("--hdr_dir", type=str,
                        help=".hdr file path for geocoding (default=image_dir)")
    parser.add_argument("--hdr_load_pattern", type=str,
                        help="Load pattern to locate .hdr file(s) for geocoded images (default=image_load_pattern*.hdr)")

    parser.add_argument("--label_dir", type=str,
                        help="Path to labeled input image(s) for salience map comparisons (default=None)")
    parser.add_argument("--label_load_pattern", type=str,
                        help="Load pattern to locate label image file(s) for salience map comparisons images (default=None)")

    
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Enable verbose output")
    
    args = parser.parse_args(sys.argv[1:])
    model_weightf = args.weight_file
    train_file    = args.train_file

    if not (train_file or model_weightf):
        print('Error: weight_file, train_file or a local load_data.py required to initialize model')
        parser.print_help()
        sys.exit(1)

    model_package = args.model_package
    model_flavor  = args.model_flavor
    tile_dim      = args.tile_dim
    batch_size    = args.batch_size
    
    test_file     = args.test_file
    save_preds    = args.save_preds

    mean_file     = args.mean_file

    test_epoch    = args.test_epoch
    test_percent  = args.test_percent
    save_epoch    = args.save_epoch
    conserve_mem  = args.conserve_memory
    balance_train = args.balance
    augment       = args.augment
    prob_thresh   = args.prob_thresh
    thresh_str    = '' if prob_thresh==0 else '%d'%prob_thresh

    # output directories
    state_dir     = args.state_dir
    output_dir    = args.output_dir or state_dir

    # salience map generation
    img_dir     = args.image_dir 
    img_pattern = args.image_load_pattern 
    tile_dir      = args.tile_dir
    tile_stride   = args.tile_stride
        
    pred_best     = args.pred_best
    if pred_best:
        save_preds = True

    hdr_dir       = args.hdr_dir or img_dir
    hdr_pattern   = args.hdr_load_pattern or splitext(img_pattern)[0]+'*.hdr'

    label_dir     = args.label_dir
    label_pattern = args.label_load_pattern
    
    verbose       = args.verbose
    clobber       = args.clobber
    do_show       = args.plot
    
    tile_shape    = [tile_dim,tile_dim]
    input_shape   = [tile_shape[0],tile_shape[1],tile_bands]

    mean_image    = mean_file
    if mean_file:
        mean_image = imread_image(mean_file)

    if pred_best:
        save_preds = True
        model_weight_dir = pathjoin(state_dir,model_package,model_flavor,'*.h5')

    print('Initializing model',model_flavor,model_package)
    model = compile_model(input_shape,output_shape,
                          model_state_dir=state_dir,
                          model_flavor=model_flavor,
                          model_package=model_package,
                          model_weightf=model_weightf)

    def preprocess_tile(img):
        return model.preprocess(img).transpose(model.transpose)
    
    def load_preprocess_tile(imgf):
        return preprocess_tile(imread_tile(imgf,tile_shape=tile_shape))
    
    memory_slots = 1 if conserve_mem else batch_size
    loadparams = {'load_func':load_preprocess_tile,
                  'conserve_memory':conserve_mem,
                  'balance_train':balance_train,
                  'memory_slots':memory_slots,
                  'exclude_pattern':None, # 'exclude_pattern':'/tn/',
                  'mean_image':mean_image}
    if train_file or test_file:
        if train_file==test_file:
            warn('train_file==test_file, sampling test data from train_file')
            test_file = None

        loadargs = (train_file,test_file)
        X_train,y_train,X_test,y_test = load_data.load_data(*loadargs,**loadparams)
        train_img_files = np.array(X_train.files)
        if test_file:
            test_img_files = np.array(X_test.files)
        else:
            test_img_files = []

        if len(y_test)==0:
            msg='No test data provided'
            if test_percent > 0.0:
                from sklearn.model_selection import train_test_split as trtesplit
                train_img_files = np.array(X_train.files)
                if y_train.ndim==1 or y_train.shape[1]==1:
                    train_lab = y_train.copy()
                else:
                    train_lab = np.argmax(y_train,axis=-1)            
                msg+=', testing on %d%% of training data'%int(test_percent*100)
                train_idx, test_idx = trtesplit(np.arange(y_train.shape[0]),
                                                train_size=1.0-test_percent,
                                                stratify=train_lab,
                                                random_state=random_state)
                test_img_files = train_img_files[test_idx]
                train_img_files = train_img_files[train_idx]

                imgf2colkw = dict(load_func=loadparams['load_func'],
                                 conserve_memory=loadparams['conserve_memory'])
                X_train = imgfiles2collection(train_img_files,**imgf2colkw)
                X_test = imgfiles2collection(test_img_files,**imgf2colkw)
                y_train, y_test = y_train[train_idx], y_train[test_idx]
            else:
                msg+=', test_percent=0, no validation will be performed'
                X_test,y_test = [],[]                
            warn(msg)

        if train_file:
            model.train(X_train,y_train,X_test,y_test,batch_size=batch_size,
                        train_ids=train_img_files,test_ids=test_img_files,
                        save_epoch=save_epoch,save_preds=save_preds,
                        test_epoch=test_epoch,test_percent=test_percent)

    if not model.initialized:
        print('Error: model not sucessfully initialized')
        sys.exit(1)
        
    if test_file:
        if y_test.ndim==1 or y_test.shape[1]==1:
            test_lab = y_test.copy()
            y_test = to_categorical(y_test, nb_classes)
        else:
            test_lab = np.argmax(y_test,axis=-1)
            
        n_pos,n_neg = np.count_nonzero(test_lab==1),np.count_nonzero(test_lab!=1)
        n_test = n_pos+n_neg
        
        msg  = 'Computing predictions for test_file "%s"'%shortpath(test_file)
        msg += '%d (#pos=%d, #neg=%d) samples'%(n_test,n_pos,n_neg)
        print(msg)

        pred_out = compute_predictions(model,X_test,nb_classes,batch_size,
                                       scale_probs=scale_probs,
                                       preprocess=False)

        pred_lab = pred_out['pred_lab']
        pred_prob = pred_out['pred_prob']

        pred_mets = compute_metrics(test_lab,pred_lab)
        pred_file = splitext(test_file)[0]+'_pred.txt'
        write_predictions(pred_file, test_img_files, test_lab, pred_lab,
                          pred_prob, pred_mets, fprfnr=True, buffered=False)
        print('Saved test predictions to "%s"'%pred_file)

    if img_dir:
        if isdir(img_dir):
            img_files = glob(pathjoin(img_dir,img_pattern))
        else:
            img_files = [img_dir]
        print('Computing salience for %d images in path "%s"'%(len(img_files),
                                                               img_dir))

        img_map=None
        img_tile_dir=None
        for imagef in img_files:
            if not pathexists(imagef):
                warn('Image file "%s" not found, skipping'%imagef)
                continue
            img_base = basename(imagef)
            img_id = filename2flightid(imagef)
            img_output_prefix = img_base
            img_csvf = pathjoin(output_dir,img_output_prefix+'%s.csv'%thresh_str)
            if not clobber and pathexists(img_csvf):
                print('Output "%s" exists, skipping'%img_csvf)
                continue

            img_data = imread_image(imagef,bands=4)
            print('img_data info: "%s"'%str((img_data.shape,img_data.dtype,
                                             extrema(img_data.ravel()))))
            
            if tile_dir and pathexists(tile_dir):
                img_tile_dir = pathjoin(tile_dir,str(tile_dim),img_id)
                
            if hdr_dir and pathexists(hdr_dir):
                img_map = get_imagemap(img_id,hdr_dir,hdr_pattern)



            print('label_dir: "%s"'%str((label_dir)))
            print('label_pattern: "%s"'%str((label_pattern)))
            img_lab = []
            if label_dir and pathexists(label_dir):
                img_lab = get_imagelabs(img_id,label_dir,label_pattern)
                print('img_lab info: "%s"'%str((img_lab.shape,img_lab.dtype,
                                                extrema(img_lab.ravel()))))

                if 0:
                    fig,ax = pl.subplots(1,1,sharex=True,sharey=True,figsize=(12,4))
                    img_over = np.zeros_like(img_data,dtype=np.float32)
                    lab_mask = bwdilate(thickboundaries(img_lab>0),selem=disk(2))
                    img_over[lab_mask, :] = [1.0,0.0,0.0,0.9]
                    aximshow(ax,img_data,img_id)                 
                    aximshow(ax,img_over,img_id+'+labels')
                    pl.show()
                
            salience_out = image_salience(model,img_data,
                                          tile_dim,tile_stride,
                                          output_dir,
                                          img_output_prefix,
                                          img_lab=img_lab,
                                          img_map=img_map,
                                          scale_probs=scale_probs,
                                          verbose=verbose,
                                          do_show=do_show)

            #tile_out = pred_out['pred_list']
            #write_csv(img_csvf,img_id,salience_out,tile_dim,prob_thresh,
            #          imgmap=img_map)
            print('Completed salience processing for imageid "%s"'%img_id)

            if do_show:
                pl.ioff();
                pl.show()
