#!/usr/bin/env python
from __future__ import division, print_function, absolute_import, unicode_literals

import sys
from glob import glob

from util.aliases import *

tilepredictor_home = pathsplit(__file__)[0]
sys.path.append(abspath(tilepredictor_home))

from imtiler.util import *
from model_package import *

random_state = 42

image_ext = '.png'
image_bands = 3

tile_ext = image_ext
tile_bands = image_bands

# tile_dim must match number of network input units
#tile_dim = 200 
tile_dim = 256 

batch_size = 256

test_epoch = 1
save_epoch = 10

# list of all imaginable tile prefixes
tile_prefix = ['tp','tn','fp','fn','pos','neg','pos_det','neg_det']

tile_transpose = (0,1,2) # (2,0,1) -> (rows,cols,bands) to (bands,rows,cols)

plot_predictions = True

from skimage.io import imread
from skimage.transform import resize as imresize

def imcrop(img,crop_shape):
    croprows,cropcols = crop_shape[0],crop_shape[1]
    nrows,ncols = img.shape[0],img.shape[1]
    croph = max(0,(nrows-croprows)//2)
    cropw = max(0,(ncols-cropcols)//2)
    return img[croph:(nrows-croph),cropw:(ncols-cropw)]

def imread_tile(f,tile_shape,resize='resize',quantize=None):
    assert(len(tile_shape)==2)
    tile = imread(f)
    nr,nc = tile.shape[0],tile.shape[1]
    scalef = 255 if tile.dtype==float else 1
    nbands = 1 if tile.ndim == 2 else tile.shape[2]
    if nbands==1:
        tile = tile.squeeze()
        tile = d_[tile,tile,tile]
    elif nbands == 4:
        tile = tile[:,:,:-1]
        
    if resize=='extract':
        roff = nr-tile_shape[0]
        coff = nc-tile_shape[1]
        r = 0 if roff<0 else randint(roff)
        c = 0 if coff<0 else randint(coff)
        print('before',tile.shape)
        tile = extract_tile(tile,(r,c),tile_shape[0])
        print(tile.shape)
    elif resize=='crop':
        tile = imcrop(tile,tile_shape)

    nr,nc = tile.shape[0],tile.shape[1]
    if resize=='resize' or (nr,nc) != tile_shape:
        tile = imresize(tile,tile_shape,preserve_range=True)

    tile = scalef*tile
    if quantize=='u8' and nbands!=1:
        tile = 255*(tile.sum(axis=2)/(np.float64(256**nbands)-1))
        
    return np.uint8(tile)

def imread_image(f,image_bands=3):
    print('Loading image',f)
    image = imread(f)
    assert(image.shape[2]==image_bands)
    scalef=1 if image.dtype==np.uint8 else 255
    return np.uint8(scalef*image)

def imfiles2collection(imgfiles,load_func,**kwargs):
    from skimage.io import ImageCollection
    kwargs.setdefault('conserve_memory',True)
    imgs = ImageCollection(imgfiles,load_func=load_func,**kwargs)
    return imgs

def imfiles2array(imgfiles,load_func,**kwargs):
    imgs = imfiles2collection(imgfiles,load_func,**kwargs)
    return imgs.concatenate()

def tilefile2ul(tile_imagef):
    tile_base = basename(tile_imagef)
    for prefix in tile_prefix:
        tile_base = tile_base.replace(prefix,'')
    return map(int,tile_base.split('_'))

def collect_tile_files(imgid,tile_dir):
    tile_files = []
    if tile_dir is None:
        return tile_files
    for prefix in tile_prefix:
        load_pattern = pathjoin(tile_dir,imgid,'*','*'+prefix+'*'+image_ext)
        tile_files.extend(glob(load_pattern))
    return tile_files

def gen_file_tiles(tile_files):
    for tilef in tile_files:
        tileimg = imread_tile(tilef)
        tileul = tilefile2ul(tilef)
        yield (tileimg, tileul)


def collect_salience_uls(image_test,tile_dim,tile_stride=None):
    image_test = atleast_3d(image_test)
    nrows,ncols,nbands = image_test.shape
    imglab = imlabel((image_test[:,:,:min(nbands,3)].sum(axis=2))!=0)
    stride = tile_stride or min(max(5,int(tile_dim/20.0)),nrows,ncols)
    uls = []
    nc = imglab.max()
    pbar = progressbar('Collecting tile locations for %d detections'%nc,nc)
    for l in pbar(range(1,nc+1)):
        lrow,lcol = where(imglab==l)
        minrow,maxrow = extrema(lrow)
        mincol,maxcol = extrema(lcol)
        rows = arange(minrow-stride,maxrow+stride-1,stride)
        cols = arange(mincol-stride,maxcol+stride-1,stride)
        luls = meshpositions(cols,rows).T
        uls = r_[uls,luls] if len(uls)!=0 else luls
    print('done, collected %d tile locations'%uls.shape[0])
    return uls

def gen_salience_tiles(image_test,tile_uls,tile_dim):
    for tileul in tile_uls:
        tileimg = extract_tile(image_test,tileul,tile_dim,verbose=False)
        if (tileimg==0).all():
            continue
        yield (tileimg, tileul)

def predict_tiles(image_test,image_tiles_uls,ntiles,verbose=False,doplot=False):
    rows,cols = image_test.shape[:2]

    image_pred = np.zeros([rows,cols,2],dtype=np.int32)
    image_prob = np.zeros([rows,cols,2],dtype=np.float32)

    tile_preds = []
    pbar = progressbar('Collecting model output for %d tiles'%ntiles,ntiles)
    for i,tile_image_ul in pbar(enumerate(image_tiles_uls)):
        tile_image,tile_ul = tile_image_ul
        tile_shape = tile_image.shape
        tile_rows = slice(tile_ul[0],tile_ul[0]+tile_shape[0],None)
        tile_cols = slice(tile_ul[1],tile_ul[1]+tile_shape[1],None)

        pred_image = tile_image.copy()
        if model.transpose:
            pred_image = tile_image.transpose(model.transpose)
        model_out = model.predict(pred_image[np.newaxis]).squeeze()
        tile_pred = np.argmax(model_out)
        tile_prob = 2*(model_out[tile_pred]-0.5)
        
        image_pred[tile_rows,tile_cols,tile_pred] += 1
        image_prob[tile_rows,tile_cols,tile_pred] += tile_prob

        tile_probstr = '%5.2f'%(100*tile_prob)
        tile_preds.append([i]+list(tile_ul)+[tile_probstr,tile_pred])

    # get average probabilities by normalizing by counts
    image_nz = image_pred!=0
    image_prob[image_nz] /= image_pred[image_nz]

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
        
    return dict(tile_preds=tile_preds,image_test=image_test,
                image_pred=image_pred,image_prob=image_prob)

def plot_pred_images(pred_out,mapinfo=None,output_dir=None,output_prefix='',
                     mask_zero=True,mask_prob=False):
    
    def save_png_output(imagebase,image_data):
        outf = pathjoin(output_dir,'_'.join([output_prefix,imagebase]))
        imsave(outf+'.png',image_data)
        print('saved',outf)

    def save_envi_output(imagebase,image_data,mapinfo):
        mapinfostr=None
        if mapinfo:
            mapout = mapinfo.copy()
            mapout['rotation']=-mapout['rotation']
            mapinfostr = mapinfo2str(mapout)

        outf = pathjoin(output_dir,'_'.join([output_prefix,imagebase]))
        array2img(outf,image_data,mapinfostr=mapinfostr)
        print('saved',outf)

    image_test = float32(pred_out['image_test'])
    image_pred = pred_out['image_pred']
    image_prob = pred_out['image_prob']

    scalef = 255.0 if image_test.max()>1.0 else 1.0
    image_test = image_test/scalef
    
    nrows,ncols = image_test.shape[0],image_test.shape[1]    
    image_transpose = (1,0,2) if nrows>ncols else (0,1,2)
    
    # pixels with no predictions
    image_nopreds = (image_pred==0).all(axis=2)

    # class predictions
    image_class = np.where(np.argmax(image_pred,axis=2),1,-1)
    image_class[image_nopreds] = 0
    pos_mask = image_class==1
    
    # prediction counts
    image_vote = np.where(pos_mask,image_pred[:,:,1],-image_pred[:,:,0])

    # hit confidence (#classpredictions/#totalpredictions per-pixel)
    image_vden = np.abs(image_pred).sum(axis=2)
    image_vcon = image_vote/np.where(image_vden!=0,image_vden,1)

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
    image_vote = array2rgba(image_vote,vmin=-max_vote,vmax=max_vote)        
    image_pred = array2rgba(image_class,vmin=-1.0,vmax=1.0)
    image_prob = array2rgba(image_prob,vmin=-1.0,vmax=1.0)
    image_vcon = array2rgba(image_vcon,vmin=-1.0,vmax=1.0)
    image_pcon = array2rgba(image_pcon,vmin=-1.0,vmax=1.0)
        
    if output_dir is not None:
        save_png_output('vote',image_vote)
        save_png_output('pred',image_pred)
        save_png_output('vcon',image_vcon)
        save_png_output('prob',image_prob)
        save_png_output('pcon',image_pcon)

    def aximshow(ax,img,ylab):
        ret = ax.imshow(img.transpose(image_transpose))
        ax.set_ylabel(ylab)
        return ret
        
    fig,ax = pl.subplots(5,1,sharex=True,sharey=True,figsize=(16,10))
    aximshow(ax[0],image_test,'test')
    aximshow(ax[1],image_pred,'pred')
    aximshow(ax[2],image_prob,'prob')
    aximshow(ax[3],image_vcon,'vcon')
    aximshow(ax[4],image_pcon,'pcon')
    pl.show()
    
def apply_model(model,tile_dim,image_path,output_path,**kwargs):
    tile_path     = kwargs.pop('tile_path',None)
    prob_thresh   = kwargs.pop('prob_thresh',0.0)
    calc_salience = kwargs.pop('calc_salience',True)
    tile_stride   = kwargs.pop('tile_stride',None)
    hdr_path      = kwargs.pop('hdr_path',None)
    hdr_suffix    = kwargs.pop('hdr_suffix',None)
    verbose       = kwargs.pop('verbose',False)
    mask_zero     = kwargs.pop('mask_zero',True)
    
    if isdir(image_path):
        image_files = glob(pathjoin(image_path,'*'+image_ext))
    else:
        image_files = [image_path]

    hdr_files = {}
    if hdr_path:
        for imagef in image_files:
            imgid = filename2flightid(imagef)
            hdrfiles = glob(pathjoin(hdr_path,imgid+'*'+hdr_suffix+'*.hdr'))
            msgtup=(imgid,hdr_path,hdr_suffix)
            if len(hdrfiles)==0:
                warn('no matching hdr for %s in %s with suffix %s'%msgtup)
                return            
            hdrf = hdrfiles[0]
            if len(hdrfiles)>1:
                msg = 'multiple .hdr files for %s in %s with suffix %s'%msgtup
                msg += '(using %s)'%hdrf
                warn(msg) 
            imgmap = mapinfo(openimg(hdrf.replace('.hdr',''),hdrf=hdrf),
                             astype=dict)
            imgmap['rotation'] = -imgmap['rotation']               
            hdr_files[imgid] = imgmap
                        
    for imagef in image_files:
        imgid = filename2flightid(imagef)
        imgmap = hdr_files.get(imgid,None)
        image_test = imread_image(imagef)
        if calc_salience:
            print('Computing salience map for image id {}'.format(imgid))
            salience_uls = collect_salience_uls(image_test,tile_dim,
                                                tile_stride=tile_stride)
            salience_tile_uls = gen_salience_tiles(image_test,salience_uls,
                                                   tile_dim)
            salience_out = predict_tiles(image_test,salience_tile_uls,
                                         len(salience_uls),verbose=verbose)
            
            output_prefix = '_'.join([imgid,'salience'])
            plot_pred_images(salience_out,mapinfo=imgmap,output_dir=output_dir,
                             output_prefix=output_prefix,mask_zero=mask_zero)
            
        if tile_path is None:
            continue
        
        tile_files = collect_tile_files(imgid,tile_path)
        if len(tile_files)!=0:
            file_tile_uls = gen_file_tiles(tile_files)
            pred_out = predict_tiles(image_test,file_tile_uls,len(tile_files),
                                     verbose=verbose)

            if plot_predictions:
                plot_pred_images(pred_out,mapinfo=imgmap,output_dir=output_dir,
                                 output_prefix=imgid,mask_zero=True)
            
            # geolocate detections with .hdr files
            if imgmap:
                from LatLongUTMconversion import UTMtoLL
                zone,hemi = imgmap['zone'],imgmap['hemi']
                zonealpha = zone + ('N' if hemi=='North' else 'M')
                tile_out = pred_out['tile_preds']
                keep_mask = float32(tile_out[:,-2])>=prob_thresh
                tile_keep = tile_out[keep_mask,:]
                tile_center = np.float32(tile_keep[:,[1,2]])+(tile_shape[0]//2)
                line,samp = tile_center.T
                utmx,utmy = sl2xy(samp,line,mapinfo=imgmap)
                lats,lons = UTMtoLL(23,utmy,utmx,zonealpha)
                preds,probs = tile_keep[:,-1],tile_keep[:,-2]
                csvf = pathjoin(output_dir,imgid+'_preds%d.csv'%int(prob_thresh))
                outcsv = []
                for i in range(tile_keep.shape[0]):
                    entryi = [imgid,'%d'%samp[i],'%d'%line[i],
                              '%18.4f'%utmx[i],'%18.4f'%utmy[i],zone,hemi,
                              lats[i],lons[i],preds[i],probs[i]]
                    outcsv.append(', '.join(map(lambda v: str(v).strip(),entryi)))
                with(open(csvf,'w')) as fid:
                    fid.write('\n'.join(outcsv)+'\n')
                print('saved',csvf)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Tile Predictor")

    # model initialization params
    parser.add_argument("-m", "--model_package", default=default_model_package,
                        help="Package (%s)"%('|'.join(valid_model_packages)),
                        type=str)
    parser.add_argument("-f", "--model_flavor", default=default_model_flavor,
                        help="Model type (%s)"%('|'.join(valid_model_flavors)),
                        type=str)
    parser.add_argument("-w", "--weight_file", type=str,
                        help="Weight file to write or load")
    parser.add_argument("-d", "--tile_dim", type=int, default=tile_dim,
                        help="Dimension of input tiles")
    parser.add_argument("-b", "--batch_size", default=batch_size, type=int,
                        help="Batch size (default=%d)"%batch_size)
    parser.add_argument("-q", "--quantize", default=None, type=str,
                        help="Quantize input images to specified dtype")
    parser.add_argument("--seed", default=random_state, type=int,
                        help="Random seed (default=%d)"%random_state)
    
    
    # model training params
    parser.add_argument("--train_file", help="Training data file (.txt)",
                        type=str)    
    parser.add_argument("--test_file", help="Test data file (.txt)",type=str)
    parser.add_argument("--test_epoch", help="Epochs between test cycles (default=%d)"%test_epoch,
                        type=int, default=test_epoch)
    parser.add_argument("--test_percent", help="Percentage of test data to use during validation",
                        type=float,default=0.25)
    parser.add_argument("--save_epoch", help="Epochs between save cycles (default=%d)"%save_epoch,
                        type=int, default=save_epoch)
    parser.add_argument("--conserve_memory", action='store_true',
                        help="Conserve memory by not caching train/test tiles'")
    
    
    # prediction threshold
    parser.add_argument("-p", "--prob_thresh", type=float, default=0.0,
                        help="Threshold on prediction probabilities [0,100]")

    # image prediction/salience params
    parser.add_argument("-i", "--image_path", type=str,
                        help="Path to input/test images(s)")
    parser.add_argument("--stride", type=int, 
                        help="Pixel stride for salience map generation (default=tile_dim/20)")
    # test inputs
    parser.add_argument("-t", "--test_dir", type=str,
                        help="Path to tile images(s) (default=None)")

    # output paths
    parser.add_argument("--state_dir", type=str, default='./model_state',
                        help="Path to save network output state (default=./model_state)")
    parser.add_argument("-o", "--output_dir", type=str,
                        help="Path to save output images/metadata")

    # hdr file params for geolocalization (optional)
    parser.add_argument("--hdr_path", help=".hdr file path for geocoding",
                        type=str)
    parser.add_argument("--hdr_suffix", type=str, default='_img',
                        help="Suffix of .hdr file(s) for geocoded images")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Enable verbose output")
    
    args = parser.parse_args(sys.argv[1:])
    model_weightf = args.weight_file
    train_file    = args.train_file

    if not (train_file or model_weightf or pathexists('./load_data.py')):
        print('Error: weight_file, train_file or a local load_data.py required to initialize model')
        parser.print_help()
        sys.exit(1)

    model_package = args.model_package
    model_flavor  = args.model_flavor
    tile_dim      = args.tile_dim
    batch_size    = args.batch_size
            
    test_file     = args.test_file
    save_mispreds = True

    test_epoch    = args.test_epoch
    test_percent  = args.test_percent
    save_epoch    = args.save_epoch
    conserve_mem  = args.conserve_memory
    
    prob_thresh   = args.prob_thresh

    # test directories
    image_path    = args.image_path # path to image to display tile predictions
    tile_dir      = args.test_dir # contains tiles from images in image_dir
    output_dir    = args.output_dir or tile_dir
    state_dir     = args.state_dir

    #calc_salience = args.salience # compute salience map
    if image_path and pathexists(image_path):
        calc_salience = True

    tile_stride   = args.stride

    hdr_path      = args.hdr_path # (optional) path to (geocoded) .hdr files 
    hdr_suffix    = args.hdr_suffix # (optional) suffix of img with .hdr

    verbose       = args.verbose
    
    tile_shape    = [tile_dim,tile_dim]
    input_shape   = [tile_shape[0],tile_shape[1],tile_bands]

    if train_file:
        if test_file==train_file:
            msg = 'Train/test files equal, sampling test data from train_file'
            warn(msg)
            test_file = None
            
        if test_file:
            print('Training and test files provided, (re)training neural network')
        else:
            print('Training file provided, (re)training neural network')

        try:
            loadargs = ()
            loadkw = {}
            if pathexists('./load_data.py'):
                import load_data
            else:
                import load_data_caffe as load_data
                load_func = lambda imgf: imread_tile(imgf,tile_shape=tile_shape)
                loadargs = (train_file,test_file)
                memory_slots = 1 if conserve_mem else batch_size
                loadkw = {'conserve_memory':conserve_mem,'load_func':load_func,
                          'exclude_pattern':None,'memory_slots':memory_slots} # 'exclude_pattern':'/tn/'}
            print("load_data function imported from:",load_data.__file__)
        except Exception as e:
            print('Error importing load_data module, cannot train network!')
            print(e)
            sys.exit(1)
            
        X_train,y_train,X_test,y_test = load_data.load_data(*loadargs,**loadkw)
        train_ids = np.array(X_train.files)
        test_ids = [] if len(X_test)==0 else np.array(X_test.files)
        
    print('Compiling model',model_flavor,model_package)
    model = compile_model(input_shape,nb_classes,model_flavor=model_flavor,
                          model_package=model_package)

    if model_weightf:
        print('Restoring weights from %s'%model_weightf)
        model.load_weights(model_weightf)

    if train_file:
        model.train_test(X_train,y_train,X_test,y_test,
                         train_ids=train_ids,test_ids=test_ids,
                         state_dir=state_dir,save_mispreds=save_mispreds,
                         test_epoch=test_epoch,save_epoch=save_epoch,
                         test_percent=test_percent,batch_size=batch_size)

    if not model.initialized:
        print('Error: model not sucessfully initialized')
        sys.exit(1)

    if image_path:
        apply_model(model,tile_dim,image_path,output_dir,tile_path=tile_dir,
                    verbose=verbose,hdr_path=hdr_path,hdr_suffix=hdr_suffix,
                    prob_thresh=prob_thresh,calc_salience=calc_salience,
                    tile_stride=tile_stride)
