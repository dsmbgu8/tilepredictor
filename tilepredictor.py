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
num_gpus = 1

def import_load_func(module_func):
    import importlib
    print('Importing load_func:',module_func)
    module,func = module_func.split('.')
    func_lib = importlib.import_module(module)
    load_func = getattr(func_lib, func)
    return load_func

def aximshow(ax,img,ylab,vmin=None,vmax=None,cmap=None):
    transpose = [1,0] if img.shape[0]>img.shape[1] else [0,1]
    if img.ndim>2:
        transpose.append(2)
    ret = ax.imshow(img.transpose(transpose),vmin=vmin,vmax=vmax,cmap=cmap)
    ax.set_ylabel(ylab)
    return ret

def plot_pred_images(img_data,pred_out,mapinfo=None,lab_mask=[],
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
        
    # save float32 envi images before converting to RGBA pngs 
    if output_dir is not None:
        save_envi_output('prob',np.float32(img_prob),mapinfo)
        save_envi_output('vcon',np.float32(img_vcon),mapinfo)
        save_envi_output('pcon',np.float32(img_pcon),mapinfo)
        
    max_vote = max(abs(extrema(img_vote)))    
    
    fig,ax = pl.subplots(5,1,sharex=True,sharey=True,figsize=(16,10))
    aximshow(ax[0],img_test,'test') 
    aximshow(ax[1],img_class,'class',vmin=-1.0,vmax=1.0)          
    aximshow(ax[2],img_prob,'prob',vmin=-1.0,vmax=1.0)
    aximshow(ax[3],img_vcon,'vcon',vmin=-1.0,vmax=1.0)
    aximshow(ax[4],img_pcon,'pcon',vmin=-1.0,vmax=1.0)

    if len(lab_mask)!=0:
        lab_mask = (lab_mask>0)
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
        save_png_output('pred',img_class)
        save_png_output('prob',img_prob)        
        save_png_output('vcon',img_vcon)
        save_png_output('pcon',img_pcon)

def write_csv(csvf,imgid,pred_list,tile_dim,prob_thresh=0.0,img_map=None):
    from LatLongUTMconversion import UTMtoLL
    # geolocate detections with .hdr files
    #tile_out = pred_out['pred_list']

    header = ['imgid','line','samp','prob_pos']
    assert(pred_list.shape[1]==len(header)-1)

    keep_mask = float32(pred_list[:,-1])>=prob_thresh
    if (~keep_mask).all():
        warn('no detections with prediction probability >= %f'%prob_thresh)
        return
    
    pred_keep = pred_list[keep_mask,:]
    # offset line,samp by tile center
    line,samp = (tile_dim//2)+np.float32(pred_keep[:,[1,2]]).T
    probpos = pred_keep[:,-1]       
    if img_map:        
        zone,hemi = img_map['zone'],img_map['hemi']
        zonealpha = zone + ('N' if hemi=='North' else 'M')
        header.extend(['lat','lon','utmx','utmy','zone','hemi'])

    outcsv = []
    #sortv = (line*(samp.max()+1))+samp # sort by line,sample
    sortv = -probpos # sort by descending probability
    sorti = np.argsort(sortv)
    for i in range(pred_keep.shape[0]):
        entryi = [imgid,'%d'%line[i],'%d'%samp[i],probpos[i]]
        if img_map:
            utmx,utmy = sl2xy(samp[i],line[i],mapinfo=img_map)
            lat,lon = UTMtoLL(23,utmy,utmx,zonealpha)
            entryi.extend([lat[0],lon[0],'%18.6f'%utmx,'%18.6f'%utmy,zone,hemi])
        outcsv.append(', '.join(map(lambda v: str(v).strip(),entryi)))
    outcsv = np.array(outcsv,dtype=str)
    outcsv = outcsv[sorti]
    with open(csvf,'w') as fid:
        fid.write('\n'.join(['# '+', '.join(header)]+list(outcsv))+'\n')
    print('saved',csvf) 

def image_salience(model, img_data, tile_stride, output_dir, output_prefix,
                   img_map=None, lab_mask=[], scale_probs=scale_probs,
                   verbose=False, transpose=False, do_show=False):
    from skimage.util.shape import view_as_windows
    input_shape = model.layers[0].input_shape 
    if model.backend=='tensorflow':
        # channels last
        tile_dim,tile_bands = input_shape[2],input_shape[3]
    elif model.backend=='theano':
        # channels first
        tile_dim,tile_bands = input_shape[2],input_shape[1]

    assert(tile_bands == img_data.shape[-1])
        
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
        bands = 3
        
    #ridx,cidx = map(lambda a: a.ravel(),np.meshgrid(range(cols),range(rows)))
    img_pred = np.zeros([rows,cols,2],dtype=np.int32)
    img_prob = np.zeros([rows,cols,2],dtype=np.float32)

    tdims  = [tile_dim,tile_dim]
    tshape = tdims+[bands]
    rshape = [-1]+tshape
    rtranspose = None
    if model.transpose:
        rtranspose = [0]+[d+1 for d in model.transpose]

    rrange = np.arange(0,rows-tile_dim+1,stride)
    crange = np.arange(0,cols-tile_dim+1,stride)
    n_rb,n_cb = len(rrange),len(crange)
    pmsg = 'Collecting predictions'
    print(pmsg+', window size = %d x %d tiles (tile_dim=%d, stride=%d)'%(n_rb,
                                                                         n_cb,
                                                                         tile_dim,
                                                                         stride))
    print_status = True

    pred_list = []    
    cidx = np.arange(n_cb,dtype=np.int32)
    rmask = np.ones(n_cb,dtype=np.bool8)
    inittime = gettime()
    preptime,predtime,posttime = 0.,0.,0.
    pbar = progressbar(pmsg,n_rb)
    for rbeg in pbar(rrange):
        preptime = gettime()
        rend = rbeg+tile_dim

        rwin = view_as_windows(img_rgb[rbeg:rend], tshape, step=stride)
        # only keep tiles that contain nonzero values
        #rmask[:] = rwin.reshape(rwin.shape[0],-1).any(axis=1)
        #if (rmask==0).all():
        #    continue

        # nkeep = np.count_nonzero(rmask)
        # rinput = [nkeep,tile_shape,tile_shape,3]
        # rwini = rwin[rmask]
        rinput = model.preprocess(rwin.reshape(rshape))
        if rtranspose:
            rinput = rinput.transpose(rtranspose)
        preptime += (gettime()-preptime)
        
        predtime = gettime()
        # probi = [nkeep,2] softmax class probabilities
        #probi = model.predict(rinput,batch_size=min(nkeep,n_cb))
        rprob = model.predict(rinput)
        rpred = np.int8(to_binary(rprob))
        
        #rprob = rprob.reshape([rstep,-1,2])
        #rpred = rpred.reshape([rstep,-1])
        #rwseq = WindowSequence(rinput,batch_size=rinput.shape[0]//stride)        
        #probi = model.base.predict_generator(rwseq,stride)
        predtime += (gettime()-predtime)

        posttime = gettime()
        # predi = [nkeep,1] softmax class predictions

        # if scale_probs:
        #     ckeep = cidx[:nkeep]
        #     probi[ckeep,predi] = 2*(probi[ckeep,predi]-0.5)
        #     probi[ckeep,1-predi] = 1-probi[ckeep,predi]

        # probwin = view_as_windows(img_prob[rbeg:rend], tdims+[2], step=stride).squeeze()
        # predwin = view_as_windows(img_pred[rbeg:rend], tdims+[1], step=stride).squeeze()        

        # probwin = probwin.swapaxes(1,3)
        # predwin = predwin.swapaxes(1,3)
        # probwin[cidx[rmask]] = probi

        # if print_state and (i%10)==0:
        #     print('prediction time: %0.3f seconds'%(gettime()-begtime))
        #     print('row block %d of %d'%(i,n_rb))

        # crangei = crange[rmask]
        

        for j,cbeg in enumerate(crange):
            cend = cbeg+tile_dim            
            img_prob[rbeg:rend,cbeg:cend,:] += rprob[j]
            img_pred[rbeg:rend,cbeg:cend,rpred[j]] += 1
            pred_list.append([rbeg,cbeg,rprob[j,1]])

        posttime += (gettime()-posttime)

    pred_list = np.float32(pred_list)
    print('total prediction time (%d tiles): %0.3f seconds'%(pred_list.shape[0],
                                                             gettime()-inittime))
    if print_status:
        print('preprocessing time: %0.3f seconds'%(preptime/n_rb))
        print('prediction time: %0.3f seconds'%(predtime/n_rb))
        print('postprocessing time: %0.3f seconds'%(posttime/n_rb))


    # get average probabilities for each class by normalizing by pred counts
    img_total = img_pred.sum(axis=2)
    # find valid pixels with at least 1 prediction
    img_haspred = img_total!=0
    
    #img_prob[img_haspred,0] /= img_total[img_haspred]
    #img_prob[img_haspred,1] /= img_total[img_haspred]

    img_amax = np.argmax(img_prob,axis=2)
    img_pos,img_neg = (img_amax==1),((img_amax==0) & img_haspred)
    if 1:
        print('img_prob[img_pos,0].min(): "%s"'%str((img_prob[img_pos,0].min())))
        print('img_prob[img_pos,0].max(): "%s"'%str((img_prob[img_pos,0].max())))
        print('img_prob[img_pos,1].min(): "%s"'%str((img_prob[img_pos,1].min())))
        print('img_prob[img_pos,1].max(): "%s"'%str((img_prob[img_pos,1].max())))
        print('img_prob[img_neg,0].min(): "%s"'%str((img_prob[img_neg,0].min())))
        print('img_prob[img_neg,0].max(): "%s"'%str((img_prob[img_neg,0].max())))
        print('img_prob[img_neg,1].min(): "%s"'%str((img_prob[img_neg,1].min())))
        print('img_prob[img_neg,1].max(): "%s"'%str((img_prob[img_neg,1].max())))
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
        print('img_prob[img_pos,0].max(): "%s"'%str((img_prob[img_pos,0].max())))
        print('img_prob[img_pos,1].min(): "%s"'%str((img_prob[img_pos,1].min())))    
        print('img_prob[img_pos,1].max(): "%s"'%str((img_prob[img_pos,1].max())))
        print('img_prob[img_neg,0].min(): "%s"'%str((img_prob[img_neg,0].min())))
        print('img_prob[img_neg,0].max(): "%s"'%str((img_prob[img_neg,0].max())))
        print('img_prob[img_neg,1].min(): "%s"'%str((img_prob[img_neg,1].min())))
        print('img_prob[img_neg,1].max(): "%s"'%str((img_prob[img_neg,1].max())))
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
                    img_mask=img_mask,
                    pred_list=pred_list)

    if output_dir:        
        plot_pred_images(img_rgb,pred_out,mapinfo=img_map,lab_mask=lab_mask,
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
    parser.add_argument("-s", "--state_dir", type=str, default=None,
                        help="Path to save network output state (default=%s)"%default_state_dir)
    parser.add_argument("-o", "--output_dir", type=str,
                        help="Path to save output images/metadata (default=state_dir)")

    module_func = 'tilepredictor_util.imread_image'
    parser.add_argument("--load_func", help="Image loader function (default=%s)"%module_func,
                        type=str,default=module_func)    
    
    
    parser.add_argument("-w", "--weight_file", type=str,
                        help="Weight file to write or load")
    parser.add_argument("-d", "--tile_dim", type=int, default=tile_dim,
                        help="Dimension of input tiles")
    parser.add_argument("-b", "--batch_size", default=batch_size, type=int,
                        help="Batch size (default=%d)"%batch_size)
    parser.add_argument("--seed", default=random_state, type=int,
                        help="Random seed (default=%d)"%random_state)
    
    
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
    #parser.add_argument("--save_preds", action='store_true',
    #                    help="Write preds to text file every save_epoch iterations'")
    #parser.add_argument("--conserve_memory", action='store_true',
    #                    help="Conserve memory by not caching train/test tiles'")
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
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Number of GPUs to use (default=%d)"%num_gpus)
        
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

    mean_file     = args.mean_file

    test_epoch    = args.test_epoch
    test_percent  = args.test_percent
    save_epoch    = args.save_epoch
    #conserve_mem  = args.conserve_memory
    balance_train = args.balance
    prob_thresh   = args.prob_thresh
    thresh_str    = '' if prob_thresh==0 else '%d'%prob_thresh

    # output directories
    state_dir     = args.state_dir
    output_dir    = args.output_dir or state_dir

    # salience map generation
    img_dir       = args.image_dir 
    img_pattern   = args.image_load_pattern 
    tile_dir      = args.tile_dir
    tile_stride   = args.tile_stride
        
    pred_best     = args.pred_best
    module_func   = args.load_func
    
    save_model    = True
    save_preds    = True

    hdr_dir       = args.hdr_dir or img_dir
    hdr_pattern   = args.hdr_load_pattern or splitext(img_pattern)[0]+'*.hdr'

    label_dir     = args.label_dir
    label_pattern = args.label_load_pattern
    
    verbose       = args.verbose
    clobber       = args.clobber
    num_gpus      = args.num_gpus
    do_show       = args.plot
    
    tile_shape    = [tile_dim,tile_dim]
    input_shape   = [tile_shape[0],tile_shape[1],tile_bands]

    mean_image    = mean_file
    if mean_file:
        mean_image = imread_image(mean_file)

    if pred_best:
        save_preds = True
        model_weight_dir = pathjoin(state_dir,model_package,model_flavor,'*.h5')

    model = compile_model(input_shape,output_shape,
                          model_state_dir=state_dir,
                          model_flavor=model_flavor,
                          model_package=model_package,
                          model_weightf=model_weightf,
                          num_gpus=num_gpus)

    # convert the mudule_func string to an actual python function 
    load_func = import_load_func(module_func)
    
    def preprocess_tile(img,model=model,doplot=False,verbose=0):
        pre = model.preprocess(img,verbose=verbose)
        if doplot:
            imgtitle='Before preprocessing: '+str(img.shape,extrema(img))
            pretitle='After preprocessing: '+str(pre.shape,extrema(pre))
            pl.ioff()
            fig,ax = pl.subplots(1,2,sharex=True,sharey=True)
            ax[0].imshow(img)
            ax[0].set_title(imgtitle)
            ax[1].imshow(pre.transpose(1,2,0))
            ax[1].set_title(pretitle)
            pl.show()
        return pre

    debug=0
    def load_tile(tilef,tile_shape=tile_shape,verbose=debug):
        tile = load_func(tilef,verbose=verbose)
        if tile.shape[:2] != tile_shape:
            tile = resize_tile(tile,tile_shape=tile_shape)
        return preprocess_tile(tile,verbose=verbose)

    if train_file or test_file:
        collect_test=True
        loadparams = dict(load_func=load_tile,
                          conserve_memory=True,
                          balance_train=False, # balance below, not here
                          exclude_pattern=None, # 'exclude_pattern='/tn/',
                          mean_image=mean_image,
                          class_mode='categorical',
                          collect_test=collect_test)
                
        train_data,test_data = load_data.load_image_data(train_file,test_file,
                                                         **loadparams)
        (X_train,y_train,train_images) = train_data
        (X_test,y_test,test_images) = test_data
        
        assert((y_train.ndim == 2) and (y_train.shape[1]==2))
        assert((y_test.ndim == 2) and (y_test.shape[1]==2))

        validation_data = (X_test,y_test)
                  
        if train_file:
            #train_gen = self.imaugment_batches(X_train,y_train,n_batches)
            train_gen = datagen_arrays(X_train,y_train,batch_size,
                                       datagen_params=train_datagen_params,
                                       fill_partial=True,shuffle=True,
                                       verbose=2)

            callback_params=dict(nb_epochs=nb_epochs,
                                 monitor='val_loss',
                                 save_epoch=save_epoch,
                                 save_preds=save_preds,
                                 save_model=save_model,
                                 test_epoch=test_epoch)
            model.init_callbacks(**callback_params)
            model.update_validation_callback(validation_data[0],
                                             validation_data[1],
                                             test_images)
            n_batches = len(X_train)//batch_size
            print(', '.join(['Training network for %d epochs'%nb_epochs,
                             'batch size=%d'%batch_size,
                             'batches/epoch=%d'%n_batches]))            
            model.train(train_gen,nb_epochs,n_batches,
                        validation_data=validation_data,
                        **callback_params)

    if not model.initialized:
        print('Error: model not sucessfully initialized')
        sys.exit(1)
        
    if test_file:
        test_lab = to_binary(y_test)
        n_pos,n_neg = class_stats(test_lab)
        n_test = n_pos+n_neg
        
        msg  = 'Computing predictions for test_file "%s"'%shortpath(test_file)
        msg += '%d (#pos=%d, #neg=%d) samples'%(n_test,n_pos,n_neg)
        print(msg)

        pred_out = compute_predictions(model,X_test,nb_classes,batch_size,
                                       scale_probs=scale_probs,
                                       preprocess=False)

        pred_lab = pred_out['pred_lab']
        pred_prob = pred_out['pred_prob']

        pred_mets = compute_metrics(to_binary(y_test),pred_lab)
        pred_file = splitext(test_file)[0]+'_pred.txt'
        write_predictions(pred_file, test_img_files, to_binary(y_test),
                          pred_lab, pred_prob, pred_mets, fnratfpr=0.01,
                          buffered=False)
        print('Saved test predictions to "%s"'%pred_file)

    if img_dir:
        if isdir(img_dir):
            img_files = glob(pathjoin(img_dir,img_pattern))
        else:
            img_files = [img_dir]
        print('Computing salience for %d images in path "%s"'%(len(img_files),
                                                               img_dir))
        
        if not pathexists(output_dir):
            print('Creating output directory %s'%output_dir)
            os.makedirs(output_dir)

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
            lab_mask = []
            if label_dir and pathexists(label_dir):
                lab_mask = get_lab_mask(img_id,label_dir,label_pattern)
                if len(lab_mask) != 0:
                    print('lab_mask info: "%s"'%str((lab_mask.shape,lab_mask.dtype,
                                                    extrema(lab_mask.ravel()))))

                plot_labs=False
                if plot_labs and len(lab_mask) != 0:
                    fig,ax = pl.subplots(1,1,sharex=True,sharey=True,figsize=(12,4))
                    img_over = np.zeros_like(img_data,dtype=np.float32)
                    lab_mask = bwdilate(thickboundaries(lab_mask>0),selem=disk(2))
                    img_over[lab_mask, :] = [1.0,0.0,0.0,0.9]
                    aximshow(ax,img_data,img_id)                 
                    aximshow(ax,img_over,img_id+'+labels')
                    pl.show()

            salience_out = image_salience(model,img_data,tile_stride,
                                          output_dir,
                                          img_output_prefix,
                                          lab_mask=lab_mask,
                                          img_map=img_map,
                                          scale_probs=scale_probs,
                                          verbose=verbose,
                                          do_show=do_show)

            pred_list = salience_out['pred_list']
            write_csv(img_csvf,img_id,pred_list,tile_dim,prob_thresh,
                      img_map=img_map)
            print('Completed salience processing for imageid "%s"'%img_id)

            if do_show:
                pl.ioff();
                pl.show()


                
