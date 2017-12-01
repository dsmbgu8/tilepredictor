from __future__ import absolute_import, division, print_function

import sys,os
import threading

from warnings import warn

import numpy as np

random_state = 42
image_ext = '.png'
load_pattern = "*"+image_ext
image_bands = 3

tile_dir = 'tiles'
tile_ext = image_ext
tile_bands = image_bands
tile_id = 'det'


# tile_dim must match number of network input units
#tile_dim = 200 
tile_dim = 256

tile_transpose = (0,1,2) # (2,0,1) -> (rows,cols,bands) to (bands,rows,cols)

# list of all imaginable tile prefixes
tile_ids = ['det','rgb']
tile_prefix = ['tp','tn','fp','fn','pos','neg']

def is_collection(X):
    from skimage.io import ImageCollection
    return type(X)==ImageCollection

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

def shortpath(path,width=2):
    spl = path.split('/')
    nspl = len(spl)
    prefix='.' if nspl<=width else '...'
    return prefix+'/'.join(spl[-min(width,nspl-1):])

def extrema(a,**kwargs):
    p = kwargs.pop('p',1.0)
    if p==1.0:
        return np.amin(a,**kwargs),np.amax(a,**kwargs)
    elif p==0.0:
        return np.amax(a,**kwargs),np.amin(a,**kwargs)
    assert(p>0.0 and p<1.0)
    axis = kwargs.pop('axis',None)
    apercent = lambda q: np.percentile(a[a==a],axis=axis,q=q,
                                       interpolation='nearest')
    return apercent((1-p)*100),apercent(p*100)

def progressbar(caption,maxval=None):
    """
    progress(title,maxval=None)
    
    Summary: progress bar wrapper
    
    Arguments:
    - caption: progress bar caption
    - maxval: maximum value
    
    Keyword Arguments:
    None 
    
    Output:
    - progress bar instance (pbar.update(i) to step, pbar.finish() to close)
    """
    
    from progressbar import ProgressBar, Bar, UnknownLength
    from progressbar import Percentage, Counter, ETA

    capstr = caption + ': ' if caption else ''
    if maxval is not None:    
        widgets = [capstr, Percentage(), ' ', Bar('='), ' ', ETA()]
    else:
        maxval = UnknownLength
        widgets = [capstr, Counter(), ' ', Bar('=')]

    return ProgressBar(widgets=widgets, maxval=maxval)    

def preprocess_img_u8(img):
    img = np.float32(img)
    img /= 255.
    img -= 0.5
    img *= 2.
    return img

def class_stats(labs,verbose=0):
    _labs = labs.squeeze()
    assert(_labs.ndim==1)
    npos = np.count_nonzero(_labs==1)
    nneg = np.count_nonzero(_labs!=1)
    if verbose:
        print('npos=',npos,'nneg=',nneg)
    return npos,nneg

def band_stats(X,n_bands=3,verbose=1):
    assert(X.ndim==4)
    band_index = -1 # 'channels_last'
    if X.shape[band_index]!=n_bands:
        band_index = 1 # 'channels_first'
        assert(X.shape[band_index]==n_bands)

    bmin = np.zeros(n_bands)
    bmax = np.zeros(n_bands)
    bmean = np.zeros(n_bands)
    for bi in range(n_bands):
        X_bi = X[...,bi] if band_index==-1 else X[:,bi]
        bmin[bi],bmax[bi]=extrema(X_bi.ravel())
        bmean[bi] = X_bi.mean()
        if verbose:
            print('band %d (shape=%s):'%(bi,str(X_bi.shape)),
                  'min=%.3f, max=%.3f, mean=%.3f'%(bmin[bi],bmax[bi],bmean[bi]))
    return bmin,bmax,bmean

def to_binary(labs):
    labssq = labs.squeeze()
    assert(labssq.ndim==2 and labssq.shape[1]==2)
    return np.int8(np.argmax(labssq,axis=-1))
    
def to_categorical(labs):
    # replacement for keras.utils.np_utils.to_categorical
    labssq = labs.squeeze()
    assert(labssq.ndim==1)
    ulabs = np.unique(labssq)
    assert(len(ulabs)==2)
    return np.int8(np.c_[labssq==ulabs[0],labssq==ulabs[1]])

def imcrop(img,crop_shape):
    croprows,cropcols = crop_shape[0],crop_shape[1]
    nrows,ncols = img.shape[0],img.shape[1]
    croph = max(0,(nrows-croprows)//2)
    cropw = max(0,(ncols-cropcols)//2)
    return img[croph:(nrows-croph),cropw:(ncols-cropw)]

def imresize(img,output_shape,**kwargs):
    from skimage.transform import resize as _imresize
    kwargs.setdefault('order',0) 
    kwargs.setdefault('clip',False)
    kwargs.setdefault('preserve_range',True)
    return _imresize(img,output_shape,**kwargs)

def imcrop(img,crop_shape):
    croprows,cropcols = crop_shape[0],crop_shape[1]
    nrows,ncols = img.shape[0],img.shape[1]
    croph = max(0,(nrows-croprows)//2)
    cropw = max(0,(ncols-cropcols)//2)
    return img[croph:(nrows-croph),cropw:(ncols-cropw)]

def imread_image(f,bands=3,dtype=np.uint8,plugin=None,verbose=0):
    from skimage.io import imread
    # plugin='imageio'
    img = imread(f,plugin=plugin)
    if verbose:
        print('Loading image',f)
        imin,imax,_ = band_stats(img[np.newaxis])

    itype = img.dtype
    ishape = img.shape
    if img.ndim==3 and img.shape[2]==4 and bands!=4:
        img = img[:,:,:3]
    assert(img.shape[2]==bands)
    scalef=255 if itype==float else 1
    imgout = dtype(scalef*img)
    if verbose:
        oshape = imgout.shape
        omin,omax,_ = band_stats(imgout[np.newaxis])
        otype = imgout.dtype
        print('Image input:  ',
              'type=%s, shape=%s, '%(str(itype),str(ishape)),
              'range = %s'%str(map(list,np.c_[imin,imax])))
        print('Image output:  '
              'type=%s, shape=%s, '%(str(otype),str(oshape)),
              'range = %s'%str(map(list,np.c_[omin,omax])))
    return imgout

def resize_tile(tile,tile_shape=[],resize='resize',dtype=np.uint8,verbose=0):
    assert(len(tile_shape)==2)
    itype = tile.dtype
    ishape = tile.shape
    if verbose:
        imin,imax = extrema(tile.ravel())

    if itype==float and (tile.ndim == 2 or tile.shape[2]==1):
        nr,nc = tile.shape[:2]
        tile = np.uint32(((2**24)-1)*tile.squeeze()).view(dtype=np.uint8)
        tile = tile.reshape([nr,nc,4])[...,:-1]     
    elif tile.shape[2] == 4:
        tile = tile[:,:,:-1]

    if resize=='extract':
        roff = tile.shape[0]-tile_shape[0]
        coff = tile.shape[1]-tile_shape[1]
        r = 0 if roff<0 else randint(roff)
        c = 0 if coff<0 else randint(coff)
        print('before extract',tile.shape)
        tile = extract_tile(tile,(r,c),tile_shape[0])
        print('after extract',tile.shape)
    elif resize=='crop':
        tile = imcrop(tile,tile_shape)

    if resize=='resize':
        tile = imresize(tile,tile_shape,preserve_range=True)
        
    scalef = 255 if itype==float else 1
    tile = dtype(scalef*tile)
    if verbose:
        omin,omax = extrema(tile.ravel())
        otype = tile.dtype
        oshape = tile.shape
        print('Before resize: '
              'type=%s, shape=%s, '
              'range = [%.3f, %.3f]'%(str(itype),str(ishape),imin,imax))
        print('After resize:  '
              'type=%s, shape=%s, '
              'range = [%.3f, %.3f]'%(str(otype),str(oshape),omin,omax))    
    return tile

def imgfiles2collection(imgfiles,load_func,**kwargs):
    from skimage.io import ImageCollection
    kwargs.setdefault('conserve_memory',True)
    imgs = ImageCollection(imgfiles,load_func=load_func,**kwargs)
    return imgs

def imgfiles2array(imgfiles,load_func,**kwargs):
    imgs = imgfiles2collection(imgfiles,load_func,**kwargs)
    return imgs.concatenate()

def tilefile2ul(tile_imagef):
    tile_base = basename(tile_imagef)
    for tid in tile_ids:
        tile_base = tile_base.replace(tid,'')
    for tpre in tile_prefix:
        tile_base = tile_base.replace(tpre,'')
    return map(int,tile_base.split('_')[-2:])

def envitypecode(np_dtype):
    from spectral.io.envi import dtype_to_envi
    _dtype = np.dtype(np_dtype).char
    return dtype_to_envi[_dtype]

def mapdict2str(mapinfo):
    mapmeta = mapinfo.pop('metadata',[])
    mapkeys,mapvals = mapinfo.keys(),mapinfo.values()
    nargs = 10 if mapinfo['proj']=='UTM' else 7
    maplist = map(str,mapvals[:nargs])
    mapkw = zip(mapkeys[nargs:],mapvals[nargs:])
    mapkw = [str(k)+'='+str(v) for k,v in mapkw]
    mapstr = '{ '+(', '.join(maplist+mapkw+mapmeta))+' }'
    return mapstr

def mapinfo(img,astype=dict):
    from collections import OrderedDict
    maplist = img.metadata.get('map info',None)
    if maplist is None:
        warn('missing "map info" string in .hdr')
        return None
    elif astype==list:
        return maplist
    
    if astype==str:
        mapstr = '{ %s }'%(', '.join(maplist))    
        return mapstr 
    elif astype==dict:
        if maplist is None:
            return {}
        
        mapinfo = OrderedDict()
        mapinfo['proj'] = maplist[0]
        mapinfo['xtie'] = float(maplist[1])
        mapinfo['ytie'] = float(maplist[2])
        mapinfo['ulx']  = float(maplist[3])
        mapinfo['uly']  = float(maplist[4])
        mapinfo['xps']  = float(maplist[5])
        mapinfo['yps']  = float(maplist[6])

        if mapinfo['proj'] == 'UTM':
            mapinfo['zone']  = maplist[7]
            mapinfo['hemi']  = maplist[8]
            mapinfo['datum'] = maplist[9]

        mapmeta = []
        for mapitem in maplist[len(mapinfo):]:
            if '=' in mapitem:
                key,val = map(lambda s: s.strip(),mapitem.split('='))
                mapinfo[key] = val
            else:
                mapmeta.append(mapitem)

        mapinfo['rotation'] = float(mapinfo.get('rotation','0'))
        if len(mapmeta)!=0:
            print('unparsed metadata:',mapmeta)
            mapinfo['metadata'] = mapmeta

    return mapinfo

def findhdr(img_file):
    from os import path
    dirname = path.dirname(img_file)
    filename,filext = path.splitext(img_file)
    if filext == '.hdr' and path.isfile(img_file): # img_file is a .hdr
        return img_file
    
    hdr_file = img_file+'.hdr' # [img_file.img].hdr or [img_file].hdr
    if path.isfile(hdr_file):
        return path.abspath(hdr_file)
    hdr_file = filename+'.hdr' # [img_file.img] -> [img_file].hdr 
    if path.isfile(hdr_file):
        return hdr_file
    return None

def openimg(imgf,hdrf=None,**kwargs):
    from spectral.io.envi import open as _open
    hdrf = hdrf or findhdr(imgf)
    return _open(hdrf,imgf,**kwargs)

def get_lab_mask(imgid,lab_path,lab_pattern,verbose=False):
    import os
    from glob import glob
    from os.path import join as pathjoin, exists as pathexists, splitext

    lab_pattern = lab_pattern or '*'
    
    lab_files  = glob(pathjoin(lab_path,lab_pattern))
    msgtup=(imgid,lab_path,lab_pattern)
    if len(lab_files)==0:
        warn('No label image for "%s" in "%s" matching pattern "%s"'%msgtup)
        return []
    labf = lab_files[0]
    if len(lab_files)>1:
        labf = None
        for lab_file in lab_files:
            if imgid in lab_file:
                labf = lab_file
                break
        if not labf:
            warn('No lab for "%s" in "%s" matching pattern "%s"'%msgtup)
            return []
        
        msg = 'Multiple label files for "%s" in "%s" matching pattern "%s"'%msgtup
        msg += ', using file "%s"'%labf
        warn(msg)

    try:
        if labf.endswith('.png'):
            labimg = imread_image(labf,bands=4,dtype=np.uint8,verbose=0)
        else:    
            labimg = openimg(labf).load().squeeze()
    except:
        warn('Unable to read label image "%s"'%labf)
        labimg = []

    return labimg
            
def get_imagemap(imgid,hdr_path,hdr_pattern,verbose=False):
    import os
    from glob import glob
    from os.path import join as pathjoin, exists as pathexists, splitext
    
    if not pathexists(hdr_path):
        warn('hdr_path "%s" not found'%hdr_path)            
        return None
    
    # remove .hdr from suffix if it's there
    hdr_files  = glob(pathjoin(hdr_path,hdr_pattern))
    msgtup=(imgid,hdr_path,hdr_pattern)
    if len(hdr_files)==0:
        warn('No hdr for "%s" in "%s" matching pattern "%s"'%msgtup)
        return None
    hdrf = hdr_files[0]
    if len(hdr_files)>1:
        hdrf = None
        for hdr_file in hdr_files:
            if imgid in hdr_file:
                hdrf = hdr_file
                break
        if not hdrf:
            warn('No hdr for "%s" in "%s" matching pattern "%s"'%msgtup)
            return None
        
        msg = 'Multiple .hdr files for "%s" in "%s" matching pattern "%s"'%msgtup
        msg += ', using file "%s"'%hdrf
        warn(msg)
    imgf = hdrf.replace('.hdr','')
    imgmap = mapinfo(openimg(imgf,hdrf=hdrf),astype=dict)
    imgmap['rotation'] = -imgmap['rotation']               

    return imgmap

def array2rgba(a,**kwargs):
    '''
    converts a 1d array into a set of rgba values according to
    the default or user-provided colormap
    '''
    from pylab import get_cmap,rcParams
    from numpy import isnan,clip,uint8,where
    cm = get_cmap(kwargs.get('cmap',rcParams['image.cmap']))
    aflat = np.float32(a.copy().ravel())
    nanmask = isnan(aflat)
    avals = aflat[~nanmask]
    vmin = float(kwargs.pop('vmin',avals.min()))
    vmax = float(kwargs.pop('vmax',avals.max()))
    if len(avals)>0 and vmax>vmin:                
        aflat[nanmask] = vmin # use vmin to map to zero, below
        aflat = clip(((aflat-vmin)/(vmax-vmin)),0.,1.)
        rgba = uint8(cm(aflat)*255)
        if nanmask.any():
            nanr = np.where(nanmask)
            rgba[nanr[0],:] = 0
        rgba = rgba.reshape(list(a.shape)+[4])
    else:
        rgba = np.zeros(list(a.shape)+[4],dtype=uint8)
    return rgba

counts = lambda a: dict(zip(*np.unique(a,return_counts=True)))
def randperm(*args):
    from numpy.random import permutation
    n = args[0]
    k = n if len(args) < 2 else args[1] 
    return permutation(n)[:k]

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

def fill_batch(X_batch,y_batch,batch_size,balance=True):
    """
    fill_batch(X_batch,y_batch,batch_size,transform=None)
    
    Summary: resamples X_batch,y_batch with replacement to generate a new
    set with the same number of samples as batch_idx
    
    Arguments:
    - X_batch: n input data samples
    - y_batch: n input labels for X_batch
    - batch_size: samples per batch
    
    Keyword Arguments:
    - transform: function to transform each sample in X_batch
    
    Output:
    - X_aug: batch_size-n augmentation samples
    - y_aug: batch_size-n augmentation labels for X_aug
    
    """

    batch_lab = to_binary(y_batch)
    n_cur = len(y_batch)
    n_aug = batch_size-n_cur
    if n_aug <= 0:
        warn('len(X_batch) <= batch_size, nothing to fill')
        return [],[]
    aug_idx = randperm(batch_size) % n_cur
    if balance:
        bal_idx = balance_classes(batch_lab[aug_idx],verbose=False)
        aug_idx = np.r_[aug_idx[bal_idx],aug_idx]
    aug_idx = aug_idx[randperm(len(aug_idx),n_aug)]
    return X_batch[aug_idx], y_batch[aug_idx]

class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    code via parag2489: https://github.com/fchollet/keras/issues/1638 
    """
    def __init__(self, it):
        
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        return next(self.it)

    def next(self):     # Py2
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

#@timeit
def collect_batch(imgs,labs,batch_idx=[],imgs_out=[],labs_out=[],verbose=0):
    # collect img_out,lab_out from collection imgs
    imgshape = imgs[0].shape
    nbatch = len(batch_idx)
    if nbatch==0:
        nbatch = len(labs)
        batch_idx = range(nbatch)
    if len(imgs_out)!=nbatch:
        imgs_out = np.zeros([nbatch]+list(imgshape),dtype=imgs[0].dtype)
        labs_out = np.zeros([nbatch,labs.shape[1]],dtype=labs[0].dtype)
    batch_iter = enumerate(batch_idx)
    if verbose:
        pmsg = 'Loading %d images into memory'%nbatch
        pbar = progressbar(pmsg,nbatch)
        batch_iter = pbar(batch_iter)
    for i,idx in batch_iter:
        imgs_out[i] = imgs[idx]
        labs_out[i] = labs[idx]
    return imgs_out,labs_out

#@timeit
def imaugment_perturb(*args,**kwargs):
    from imaugment import perturb_batch as _pb
    kwargs.setdefault(train_params=train_imaugment_params)
    return _pb(*args,**kwargs)

@threadsafe_generator
def array2gen(a,nb):
    assert(a.ndim==4)
    outshape = [-1]+list(a.shape[1:])
    for i in range(0,a.shape[0]+1,nb):
        yield a[i*nb:min((i+1)*nb,a.shape[0])].reshape(outshape)

from keras.utils.data_utils import Sequence
class WindowSequence(Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x
        
if __name__ == '__main__':
    # Binary values:      [0,1,1]
    # Categorical values: 0 -> [ 1.  0.] (argmax=0)
    #                     1 -> [ 0.  1.] (argmax=1)
    #                     1 -> [ 0.  1.] (argmax=1)
    binlabs = np.array([0,1,1]).reshape([-1,1])
    cats = to_categorical(binlabs)
    pos_cat,neg_cat = cats[1],cats[0]
    cat2bin = to_binary(cats).reshape([-1,1])
    
    print('binlabs:',binlabs)
    print('cats:\n',cats)
    print('cat2bin:',cat2bin)
    assert((cat2bin==binlabs).all())
    print("Binary 0 -> categorical %s (argmax=%d)"%(neg_cat,np.argmax(neg_cat)))
    print("       1 -> categorical %s (argmax=%d)"%(pos_cat,np.argmax(pos_cat)))
   
