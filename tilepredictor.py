#!/usr/bin/env python
from __future__ import division, print_function, absolute_import, unicode_literals

import sys
from glob import glob

from util.aliases import *

tilepredictor_home = pathsplit(__file__)[0]
sys.path.append(abspath(tilepredictor_home))

from imtiler.util import *
from tilepredictor_util import *
from model_package import *

batch_size = 256
augment_size = [1,1]

test_epoch = 1
save_epoch = 1

plot_predictions = True
plot_salience = True

scale_probs = False
randomize = False

def collect_file_uls(tile_path,tile_id='det',tile_ext='.png'):
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

def collect_salience_uls(image_test,tile_dim,tile_stride):
    from skimage.measure import label as imlabel
    nzmask = image_test[:,:,:3].any(axis=2)
    nzrows,nzcols = nzmask.nonzero()
    nzcomp = imlabel(nzmask, background=0, return_num=False, connectivity=2)
    nzlabs = nzcomp[nzrows,nzcols]
    ucomp,ucounts = np.unique(nzlabs,return_counts=True)
    stride = int(max(1,tile_stride*tile_dim))
    print('Collecting salience uls for',len(ucomp),
          'nonzero components with stride',stride)
    
    uls = []
    half_dim = tile_dim//2
    for ulab,unum in zip(ucomp,ucounts):
        umask = nzlabs==ulab
        urow,ucol = nzrows[umask],nzcols[umask]
        ul = []
        rmin,cmin = urow.min(),ucol.min()
        rmax,cmax = max(rmin+stride,urow.max()),max(cmin+stride,ucol.max())
        for r in range(rmin,rmax,stride):
            for c in range(cmin,cmax,stride):
                ul.append([r-stride,c-stride])

        cstep = (cmax-cmin)/stride
        rstep = (rmax-rmin)/stride
        if rstep>1 or cstep>1:
            print('Collected',len(ul),'tiles for component',ulab,'with',unum,'pixels',
                  'rstep',rstep,'cstep',cstep)
        ul = np.int32(ul).reshape([-1,2])
        uls = r_[uls,ul] if len(uls)!=0 else ul

    print(uls.shape[0],'uls collected')
    return uls

def generate_tiles(image_test,tile_uls,tile_dim):
    for tile_ul in tile_uls:
        tile_img = extract_tile(image_test,tile_ul,tile_dim,verbose=False)
        if tile_img.any():
            yield tile_img

def compute_predictions(model,X_test,n_classes,batch_size,scale_probs=True,
                        preprocess=False):
    n_test = len(X_test)
    tile_dim = X_test[0].shape[0]

    n_out = min(n_test,batch_size)
    X_batch = np.zeros([n_out,tile_dim,tile_dim,3])
    pred_prob = np.zeros([n_test,n_classes])
    b_off = 0
    while b_off < n_test:
        b_end = min(b_off+batch_size,n_test)
        print('Computing predictions for samples %d through %d'%(b_off,b_end))
        for i in range(b_off,b_end):
            # note: the loader script should already preprocess each test sample
            Xi = X_test[i]
            if preprocess:
                Xi = model.preprocess(Xi).transpose(model.transpose)
            X_batch[i-b_off] = Xi
        pred_prob[b_off:b_end] = model.predict(X_batch[:b_end-b_off])
        b_off += batch_size
    pred_lab = np.int8(np.argmax(pred_prob,axis=-1))
    pred_prob = np.amax(pred_prob,axis=-1)

    if scale_probs:
        pred_prob = 2*(pred_prob-0.5)

    return dict(pred_lab=pred_lab,pred_prob=pred_prob)
            
def plot_pred_images(pred_out,mapinfo=None,output_dir=None,output_prefix='',
                     mask_zero=True,mask_prob=False):
    
    def save_png_output(imagebase,image_data):
        outf = pathjoin(output_dir,'_'.join([output_prefix,imagebase]))+'.png'
        imsave(outf,image_data)
        print('Saved',outf)

    def save_envi_output(imagebase,image_data,mapinfo):
        mapinfostr=None
        mapout = mapinfo.copy()
        mapout['rotation']=-mapout['rotation']

        outf = pathjoin(output_dir,'_'.join([output_prefix,imagebase]))
        array2img(outf,image_data,mapinfostr=mapdict2str(mapout),overwrite=1)
        print('Saved',outf)

    image_test = float32(pred_out['image_test'])
    image_pred = pred_out['image_pred']
    image_prob = pred_out['image_prob']

    scalef = 255.0 if image_test.max()>1.0 else 1.0
    image_test = image_test/scalef
    
    nrows,ncols = image_test.shape[0],image_test.shape[1]    
    
    # pixels with no predictions
    image_nopreds = ~(image_pred.any(axis=2) & np.isfinite(image_pred).all(axis=2))

    # class predictions
    image_class = np.where(np.argmax(image_pred,axis=2)==1,1,-1)
    image_class[image_nopreds] = 0
    pos_mask = image_class==1
    
    # prediction counts
    image_vote = np.where(pos_mask,image_pred[:,:,1],-image_pred[:,:,0])
    image_total = image_pred.sum(axis=2)
    # hit confidence (#classpredictions/#totalpredictions per-pixel)
    image_vcon = image_vote/np.where(image_total!=0,image_total,1)

    # prob confidence (probs*#classpredictions/#totalpredictions per-pixel)
    image_pcon = image_prob*image_vcon[:,:,np.newaxis]
    image_pcon = np.where(pos_mask,image_pcon[:,:,1],image_pcon[:,:,0])
                      
    # predicted class probabilities
    image_prob = np.where(pos_mask,image_prob[:,:,1],-image_prob[:,:,0])
    image_prob[image_nopreds] = 0 # ignore pixels without predictions

    image_mask = np.zeros([nrows,ncols],dtype=bool8)
    if mask_zero:
        image_zero = (image_test==0).all(axis=2)
        image_mask[image_zero] = 1
        pzeromask = image_zero.sum()/float(nrows*ncols)
        print('Zero-valued pixels masked: %5.2f%'%(100*pzeromask))
        
    if mask_prob:
        image_thr = np.abs(image_prob)<prob_thresh
        pthrmask = image_thr.sum()/float(nrows*ncols)
        image_mask[image_thr] = 1 
        print('P<%5.2f% pixels masked: %5.2f%'%(100*prob_thresh,100*pthrmask))

    pmask = image_mask.sum()/float(nrows*ncols)
    print('Total pixels masked: %5.2f%%'%(100*pmask))
    if pmask != 0:
        image_test = d_[image_test,float32(~image_mask)] # alpha channel
        image_pred[image_mask] = 0
        image_prob[image_mask] = 0
        image_vcon[image_mask] = 0
        image_pcon[image_mask] = 0

    if not pathexists(output_dir):
        print('Creating output directory %s'%output_dir)
        os.makedirs(output_dir)
        
    # save float32 envi images before converting to RGBA pngs 
    if output_dir is not None:
        save_envi_output('pred',np.int16(image_pred),mapinfo)
        save_envi_output('prob',np.float32(image_prob),mapinfo)
        
    max_vote = max(abs(extrema(image_vote)))

    def aximshow(ax,img,ylab,vmin=None,vmax=None):
        transpose = [1,0] if img.shape[0]>img.shape[1] else [0,1]
        if img.ndim>2:
            transpose.append(2)
        ret = ax.imshow(img.transpose(transpose),vmin=vmin,vmax=vmax)
        ax.set_ylabel(ylab)
        return ret
    
    fig,ax = pl.subplots(5,1,sharex=True,sharey=True,figsize=(16,10))
    aximshow(ax[0],image_test,'test') 
    aximshow(ax[1],image_class,'pred',vmin=-1.0,vmax=1.0)          
    aximshow(ax[2],image_prob,'prob',vmin=-1.0,vmax=1.0)          
    aximshow(ax[3],image_vcon,'vcon',vmin=-1.0,vmax=1.0)          
    aximshow(ax[4],image_pcon,'pcon',vmin=-1.0,vmax=1.0)          
    
    image_vote = array2rgba(image_vote,  vmin=-max_vote,vmax=max_vote)        
    image_class = array2rgba(image_class,vmin=-1.0,vmax=1.0)
    image_prob = array2rgba(image_prob,  vmin=-1.0,vmax=1.0)
    image_vcon = array2rgba(image_vcon,  vmin=-1.0,vmax=1.0)
    image_pcon = array2rgba(image_pcon,  vmin=-1.0,vmax=1.0)
        
    if output_dir is not None:
        save_png_output('vote',image_vote)
        save_png_output('pred',image_class)
        save_png_output('vcon',image_vcon)
        save_png_output('prob',image_prob)
        save_png_output('pcon',image_pcon)

    pl.ioff(); pl.show()

def write_csv(csvf,imgid,pred_out,tile_dim,prob_thresh=0.0,imgmap=None):
    from LatLongUTMconversion import UTMtoLL
    # geolocate detections with .hdr files
    tile_out = pred_out['tile_preds']
    keep_mask = float32(tile_out[:,-2])>=prob_thresh
    tile_keep = tile_out[keep_mask,:]
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

def compute_salience(model, image_test, tile_dim, tile_stride,
                     output_dir, output_prefix, tile_dir=None, img_map=None,
                     scale_probs=scale_probs, randomize=randomize,
                     verbose=False):
    if not tile_dir:
        tile_uls = collect_salience_uls(image_test,tile_dim,tile_stride)
    else:
        tile_files,tile_uls = collect_file_uls(tile_dir)
    n_tiles = len(tile_uls)
    
    rows,cols = image_test.shape[:2]
    image_pred = np.zeros([rows,cols,2],dtype=np.int32)
    image_prob = np.zeros([rows,cols,2],dtype=np.float32)

    if randomize:
        tile_uls = tile_uls[randperm(n_tiles)]
    
    tile_preds = []
    tile_gen = generate_tiles(image_test,tile_uls,tile_dim)
    pbar = progressbar('Collecting model output for %d tiles'%n_tiles,n_tiles)    
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

            print('model_out:          "%s"'%str((model_out)))
            print('tile_pred:          "%s"'%str((tile_pred)))
            print('tile_prob_unscaled: "%s"'%str((tile_prob)))
            print('tile_prob_scaled:   "%s"'%str((2*(tile_prob-0.5))))

        if scale_probs:
            tile_prob = 2*(tile_prob-0.5)
                        
        image_pred[tile_rows,tile_cols,tile_pred] += 1
        image_prob[tile_rows,tile_cols,:] += model_out

        tile_probstr = '%6.3f'%(100*tile_prob)
        tile_preds.append([i]+list(tile_ul)+[tile_probstr,tile_pred])

    # get average probabilities by normalizing by counts
    image_prob /= np.where(image_pred,image_pred,1)

    tile_preds = np.array(tile_preds,dtype=str)
    pos_mask = tile_preds[:,-1]=='1'
    neg_mask = tile_preds[:,-1]=='0'
    npos = np.count_nonzero(pos_mask)
    nneg = np.count_nonzero(neg_mask)
    total = float(npos+nneg)
    pos_probs = float32(tile_preds[pos_mask,-2])
    neg_probs = float32(tile_preds[neg_mask,-2])
    if npos!=0:
        print('positive predictions:',npos,npos/total,'prob min/max/mean/std:',
              pos_probs.min(),pos_probs.max(),pos_probs.mean(),pos_probs.std())
    if nneg!=0:
        print('negative predictions:',nneg,nneg/total,'prob min/max/mean/std:',
              neg_probs.min(),neg_probs.max(),neg_probs.mean(),neg_probs.std())

    if verbose:
        print('\n'.join(map(lambda s: ' '.join(s),tile_preds)))

    pred_out = dict(tile_preds=tile_preds,image_test=image_test,
                    image_pred=image_pred,image_prob=image_prob)

    if output_dir:
        if not pathexists(output_dir):
            warn('Output dir "%s" not found'%output_dir)
            return {}
        
        plot_pred_images(pred_out,mapinfo=img_map,output_dir=output_dir,
                         output_prefix=output_prefix,mask_zero=False)
        
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
    parser.add_argument("-p", "--prob_thresh", type=float, default=0.0,
                        help="Threshold on prediction probabilities [0,100]")

    # image salience params
    parser.add_argument("--image_dir", type=str,
                        help="Path to input images(s) for salience map generation")
    parser.add_argument("--image_load_pattern", type=str, default=load_pattern,
                        help="Load pattern for input/test images(s) (default=%s)"%load_pattern)
    parser.add_argument("--tile_stride", type=float, default=0.5,
                        help="Tile stride (percentage of tile_dim) for salience map generation (default=0.5)")
    parser.add_argument("--tile_dir", type=str,
                        help="Path to directory containing precomputed tiles for each image in image_dir")
    
    # misc
    parser.add_argument("--pred_best", action='store_true',
                        help="Load best model in state_dir and compute/save preds for test_file'")
    
    # hdr file params for geolocalization (optional)
    parser.add_argument("--hdr_dir", type=str,
                        help=".hdr file path for geocoding")
    parser.add_argument("--hdr_load_pattern", type=str, default='*.hdr',
                        help="Load pattern to locate .hdr file(s) for geocoded images (default='*.hdr')")

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
    image_dir    = args.image_dir 
    image_pattern = args.image_load_pattern 
    tile_dir      = args.tile_dir
    tile_stride   = args.tile_stride
        
    pred_best     = args.pred_best
    if pred_best:
        save_preds = True

    hdr_dir       = args.hdr_dir # (optional) path to (geocoded) .hdr files 
    hdr_pattern   = args.hdr_load_pattern # (optional) suffix of img with .hdr

    verbose       = args.verbose
    
    tile_shape    = [tile_dim,tile_dim]
    input_shape   = [tile_shape[0],tile_shape[1],tile_bands]

    mean_image    = mean_file
    if mean_file:
        mean_image = imread_image(mean_file)

    if pred_best:
        save_preds = True
        model_weight_dir = pathjoin(state_dir,model_package,model_flavor,'*.h5')
    print('Initializing model',model_flavor,model_package)
    model = compile_model(input_shape,nb_classes,
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
        if test_file:
            test_image_files = np.array(X_test.files)
        else:
            test_image_files = []

        if len(y_test)==0:
            msg='No test data provided'
            if test_percent > 0.0:
                from sklearn.model_selection import train_test_split as trtesplit
                train_image_files = np.array(X_train.files)
                if y_train.ndim==1 or y_train.shape[1]==1:
                    train_lab = y_train.copy()
                else:
                    train_lab = np.argmax(y_train,axis=-1)            
                msg+=', testing on %d%% of training data'%int(test_percent*100)
                train_idx, test_idx = trtesplit(np.arange(y_train.shape[0]),
                                                train_size=1.0-test_percent,
                                                stratify=train_lab,
                                                random_state=random_state)
                test_image_files = train_image_files[test_idx]
                train_image_files = train_image_files[train_idx]            
                X_train = imgfiles2collection(train_image_files,
                                             load_func=loadparams['load_func'],
                                             conserve_memory=loadparams['conserve_memory'])
                X_test = imgfiles2collection(test_image_files,
                                            load_func=loadparams['load_func'],
                                            conserve_memory=loadparams['conserve_memory'])
                y_train, y_test = y_train[train_idx],y_train[test_idx]


            else:
                msg+=', test_percent=0, no validation will be performed'
                X_test,y_test = [],[]                
            warn(msg)

        if train_file:
            train_image_files = np.array(X_train.files)
            model.train(X_train,y_train,X_test,y_test,batch_size=batch_size,
                        train_ids=train_image_files,test_ids=test_image_files,
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
        
        msg = 'Computing predictions for test_file "%s"'%shortpath(test_file)
        msg += '%d (#pos=%d, #neg=%d) samples'%(n_test,n_pos,n_neg)
        print(msg)

        pred_out = compute_predictions(model,X_test,nb_classes,batch_size,
                                       scale_probs=scale_probs,
                                       preprocess=False)

        pred_lab = pred_out['pred_lab']
        pred_prob = pred_out['pred_prob']

        pred_mets = compute_metrics(test_lab,pred_lab)
        pred_file = splitext(test_file)[0]+'_pred.txt'
        write_predictions(pred_file, test_image_files, test_lab, pred_lab,
                          pred_prob, pred_mets, fprfnr=True, buffered=False)
        print('Saved test predictions to "%s"'%pred_file)

    if image_dir:
        if isdir(image_dir):
            image_files = glob(pathjoin(image_dir,image_pattern))
        else:
            image_files = [image_dir]
        print('Computing salience for %d images in path "%s"'%(len(image_files),
                                                               image_dir))

        img_map=None
        img_tile_dir=None
        for imagef in image_files:
            if not pathexists(imagef):
                warn('Image file "%s" not found, skipping'%imagef)
                continue
            img_base = basename(imagef)
            img_id = filename2flightid(imagef)
            img_data = imread_image(imagef)
            if tile_dir and pathexists(tile_dir):
                img_tile_dir = pathjoin(tile_dir,str(tile_dim),img_id)
            if hdr_dir and pathexists(hdr_dir):
                img_map = get_imagemap(img_id,hdr_dir,hdr_pattern)

            img_output_prefix=img_base
            salience_out = compute_salience(model,img_data,
                                            tile_dim,tile_stride,
                                            output_dir,img_output_prefix,
                                            tile_dir=img_tile_dir,
                                            img_map=img_map,
                                            scale_probs=scale_probs,
                                            randomize=randomize,
                                            verbose=verbose)
            csvf = pathjoin(output_dir,img_output_prefix+'%s.csv'%thresh_str)
            write_csv(csvf,img_id,salience_out,tile_dim,prob_thresh,imgmap=img_map)
            print('Completed salience processing for imageid "%s"'%img_id)
