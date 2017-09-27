from __future__ import absolute_import, division, print_function

import sys,os
from warnings import warn

import numpy as np

from skimage.io import imread

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

def preprocess_img_u8(img):
    img = np.float32(img)
    img /= 255.
    img -= 0.5
    img *= 2.
    return img

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

def imread_tile(f,tile_shape,resize='resize',dtype=np.uint8,verbose=0):
    assert(len(tile_shape)==2)
    tile = imread(f)
    itype = tile.dtype
    if verbose:
        imin,imax = extrema(tile.ravel())

    if tile.ndim == 2 or tile.shape[2]==1:
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
        print('before',tile.shape)
        tile = extract_tile(tile,(r,c),tile_shape[0])
        print(tile.shape)
    elif resize=='crop':
        tile = imcrop(tile,tile_shape)

    if resize=='resize' or tile.shape != tile_shape:
        tile = imresize(tile,tile_shape,preserve_range=True)
        
    scalef = 255 if itype==float else 1
    tile = dtype(scalef*tile)
    if verbose:
        omin,omax = extrema(tile.ravel())
        otype = tile.dtype
        print('Input  type=%s, range = [%.3f, %.3f]'%(str(itype),imin,imax))
        print('Output type=%s, range = [%.3f, %.3f]'%(str(otype),omin,omax))    
    return tile

def imread_image(f,image_bands=3,dtype=np.uint8,verbose=0):
    print('Loading image',f)
    image = imread(f)
    itype = image.dtype
    if verbose:
        imin,imax = extrema(image.ravel())

    if image.ndim==3 and image.shape[2]==4:
        image = image[:,:,:3]
    assert(image.shape[2]==image_bands)
    scalef=255 if itype==float else 1
    imgout = dtype(scalef*image)
    if verbose:
        omin,omax = extrema(imgout.ravel())
        otype = imgout.dtype
        print('Input  type=%s, range = [%.3f, %.3f]'%(str(itype),imin,imax))
        print('Output type=%s, range = [%.3f, %.3f]'%(str(otype),omin,omax))
    return imgout

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

def openimg(imgf,hdrf=None,**kwargs):
    from spectral.io.envi import open as _open
    hdrf = hdrf or findhdr(imgf)
    return _open(hdrf,imgf,**kwargs)

def get_imagemap(imgid,hdr_path,hdr_pattern,verbose=False):
    import os
    from glob import glob
    from os.path import join as pathjoin, exists as pathexists, splitext
    
    if not pathexists(hdr_path):
        warn('hdr_path "%s" not found'%hdr_path)            
        return None
    
    # remove .hdr from suffix if it's there
    hdr_paths  = glob(pathjoin(hdr_path,hdr_pattern))
    msgtup=(imgid,hdr_path,hdr_pattern)
    if len(hdr_paths)==0:
        warn('no hdr for "%s" in "%s" matching pattern "%s"'%msgtup)
        return None
    hdrf = hdr_paths[0]
    if len(hdr_paths)>1:
        msg = 'multiple .hdr files for "%s" in "%s" matching pattern "%s"'%msgtup
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
