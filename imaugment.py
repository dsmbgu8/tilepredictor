# many of the functions included below derived from code by Sander Dieleman
# available at the following url:
# https://github.com/benanne/kaggle-galaxies/blob/master/realtime_augmentation.py

from __future__ import absolute_import, print_function, division
import numpy as np


import skimage.transform as skimgxform
SimilarityTransform = skimgxform.SimilarityTransform
AffineTransform = skimgxform.AffineTransform
warpfast = skimgxform._warps_cy._warp_fast

### perturbation and preprocessing ###
warp_mode='wrap' # 'reflect'
img_type=np.float32
do_flip=True
do_shift=True
subpixel_shift = False

augment_train_params = {
    'zoom_range': (1.0, 1.1),
    'rotation_range': (0, 180),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
}

# do not perturb for test data
augment_test_params = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
}

def imlcn(img, sigma_mean, sigma_std):
    """
    based on matlab code by Guanglei Xiong, see http://www.mathworks.com/matlabcentral/fileexchange/8303-local-normalization
    """
    from scipy.ndimage import gaussian_filter
    means = gaussian_filter(img, sigma_mean)
    img_centered = img - means
    stds = np.sqrt(gaussian_filter(img_centered**2, sigma_std))
    return img_centered / stds

def imnormhist(img, num_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    # this function only makes sense for grayscale images.
    assert(img.ndim==2 or img.shape[2]==1)
    img_flat = img.flatten()
    imhist, bins = np.histogram(img_flat, num_bins, normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(img_flat, bins[:-1], cdf)

    return im2.reshape(img.shape)

def imfastwarp(img, tf, output_shape=[], mode=warp_mode, img_wf=[]):
    """
    This wrapper function is about five times faster than
    skimage.transform.warp.
    """
    m = tf.params

    nb = img.shape[2]
    nr,nc = output_shape if len(output_shape)==2 else img.shape[:2]
    if len(img_wf)==0:
        img_wf = np.zeros((nr,nc,nb), dtype=img_type)

    for k in xrange(nb):
        img_wf[..., k] = warpfast(img[..., k], m, mode=mode,
                                  output_shape=output_shape)
    return img_wf

## TRANSFORMATIONS ##
def build_augmentation_transform(center_shift, zoom=1.0, rotation=0,
                                 shear=0, translation=(0, 0)):
    tform_center = SimilarityTransform(translation=-center_shift)
    tform_uncenter = SimilarityTransform(translation=center_shift)

    tform_augment = AffineTransform(scale=(1/zoom, 1/zoom),
                                    rotation=np.deg2rad(rotation),
                                    shear=np.deg2rad(shear),
                                    translation=translation)
    # shift to center, augment, shift back (for the rotation/shearing)
    return tform_center + tform_augment + tform_uncenter 

def build_ds_transform(ds_factor, orig_size, target_size, do_shift=do_shift,
                       subpixel_shift=subpixel_shift):
    """
    This version is a bit more 'correct',
    it mimics the skimage.transform.resize function.
    
    if subpixel_shift is true, we add an additional 'arbitrary' subpixel shift, which 'aligns'
    the grid of the target image with the original image in such a way that the interpolation
    is 'cleaner', i.e. groups of <ds_factor> pixels in the original image will map to
    individual pixels in the resulting image.
    
    without this additional shift, and when the downsampling factor does not divide the image
    size (like in the case of 424 and 3.0 for example), the grids will not be aligned, resulting
    in 'smoother' looking images that lose more high frequency information.
    
    technically this additional shift is not 'correct' (we're not looking at the very center
    of the image anymore), but it's always less than a pixel so it's not a big deal.
    
    in practice, we implement the subpixel shift by rounding down the orig_size to the
    nearest multiple of the ds_factor. Of course, this only makes sense if the ds_factor
    is an integer.    
    """
    
    rows, cols = orig_size
    trows, tcols = target_size
    col_scale = row_scale = ds_factor
    src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
    dst_corners = np.zeros(src_corners.shape, dtype=np.float64)
    # take into account that 0th pixel is at position (0.5, 0.5)
    dst_corners[:, 0] = col_scale * (src_corners[:, 0] + 0.5) - 0.5
    dst_corners[:, 1] = row_scale * (src_corners[:, 1] + 0.5) - 0.5

    tform_ds = AffineTransform()
    tform_ds.estimate(src_corners, dst_corners)

    if do_shift:
        if subpixel_shift: 
            cols = (cols // int(ds_factor)) * int(ds_factor)
            rows = (rows // int(ds_factor)) * int(ds_factor)
            # print "NEW ROWS, COLS: (%d,%d)" % (rows, cols)

        shift_x = cols / (2 * ds_factor) - tcols / 2.0
        shift_y = rows / (2 * ds_factor) - trows / 2.0
        tform_shift_ds = SimilarityTransform(translation=(shift_x, shift_y))
        return tform_shift_ds + tform_ds
    else:
        return tform_ds

uniform = np.random.uniform
randint = np.random.randint

def random_perturbation_transform(center_shift, zoom_range, rotation_range,
                                  shear_range, translation_range,
                                  do_flip=do_flip):
    # random shift [-4, 4] - shift no longer needs to be integer!
    shift_x = uniform(*translation_range)
    shift_y = uniform(*translation_range)
    translation = (shift_x, shift_y)

    # random rotation [0, 360]
    rotation = uniform(*rotation_range) # there is no post-augmentation, so full rotations here!

    # random shear [0, 5]
    shear = uniform(*shear_range)

    # flip
    if do_flip and (randint(2) > 0): # flip half of the time
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    # random zoom [0.9, 1.1]
    # zoom = uniform(*zoom_range)
    log_zoom_range = np.log(zoom_range) 
    zoom = np.exp(uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    return build_augmentation_transform(center_shift, zoom, rotation, shear,
                                        translation)

def perturb_and_dscrop(img, tforms_ds, augment_params, target_sizes,
                       mode=warp_mode, img_buf=[]):

    tform_augment = random_perturbation_transform(**augment_params)
    # return [skimage.transform.warp(img, tform_ds + tform_augment,
    #                                output_shape=target_size,
    #                                mode='reflect').astype('float32')
    #         for tform_ds in ds_transforms]

    ntransform = len(tforms_ds)
    if len(img_buf)==0:
        img_buf = np.zeros([ntransform]+list(img.shape))
    
    for i in range(ntransform):
        yield imfastwarp(img, tforms_ds[i] + tform_augment,
                         output_shape=target_sizes[i], mode=mode)
                         #img_wf=img_buf[i])



## REALTIME AUGMENTATION ##
def perturb_gen(imgdata, imglabs, ds_factor=1.0, npos=1, nneg=1, test=False,
                transpose=None, **kwargs):
    """
    perturb_batch_gen(imgdata,imglabs,npos=1,nneg=1,test=False)
    
    Summary: generator of randomized transformations to a set of input images,
    stratified and replicated by class label
    
    Arguments:
    - imgdata: image data with dimensions [nimg, nrows, ncols, nbands] 
    - imglabs: labels for each img in imgdata [nimg x nclass] \in {0,1}
    
    Keyword Arguments:
    - ds_factor: downsampling factor in (0,1] range
    - npos: positive class replication factor (default=1)
    - nneg: negative class replication factor (default=1)
    - test: imgdata/imglabs are test examples, do not permute (default=False)
    - transpose: transpose if imgdata[i] not in (nrows,ncols,nbands) order
    
    Output:
    - augmented batch + labels (generator)
    """

    train_params = kwargs.get('train_params',augment_train_params)
    test_params = kwargs.get('test_params',augment_test_params)
    add_rot45 = kwargs.get('add_rot45',False)

    nimg = len(imgdata)
    imgi = imgdata[0]

    if not transpose:
        nrows,ncols,nbands = imgi.shape
    else:
        nbands,nrows,ncols = imgi.shape

    assert((nbands>=1) and (nbands<=4))
        
    img_size = [nrows,ncols]
    input_sizes = [img_size,img_size]
    center_shift = np.array(img_size)/2.-0.5
            
    ds_transforms = [
        build_ds_transform(ds_factor, img_size, input_sizes[0])
    ]
    
    if not test and add_rot45:
        # generate two perturbed, downsampled outputs: original+rot45
        ds_transforms.append(
            build_ds_transform(ds_factor, img_size, input_sizes[1]) + \
            build_augmentation_transform(center_shift, rotation=45))
        augment_params = train_params.copy()
    else:
        # just apply downsample operation, don't augment/replicate
        augment_params = test_params.copy()
    
    augment_params['center_shift'] = center_shift
    img_buf = np.zeros([len(ds_transforms)]+img_size+[nbands])            
    
    for i in range(nimg):
        # (bands,rows,cols) <-> (rows,cols,bands)
        #           (2,0,1) <-> (1,2,0)
        imgi = imgdata[i] if not transpose else imgdata[i].transpose((1,2,0))
        labi = imglabs[i]

        # stratified replication
        nrep = (nneg if labi[1]==0 else npos) if not test else 1
        for _ in range(nrep):
            img_a = perturb_and_dscrop(imgi, ds_transforms, augment_params,
                                       input_sizes, img_buf=img_buf)
            for img_ai in img_a:
                # swap band order to fit into [nimg,nband,nrow,ncol] output
                img_ao = img_ai if not transpose else img_ai.transpose(transpose)
                yield img_type(img_ao), labi

def perturb_batch(imgdata, imglabs, ds_factor=1.0, naugpos=1, naugneg=1, 
                  test=False, transpose=None, **kwargs):
    """
    perturb_batch(imgdata,imglabs,npos=1,nneg=1,test=False)
    
    Summary: wrapper to collect output of perturb_batch_gen as static np arrays
    
    Arguments: see perturb_batch_gen

    Keyword Arguments: see perturb_batch_gen

    Output:
    - [naug imgdata.shape[1:]] image array
    - [naug x 1] label array  
    """
    
    naug = len(imglabs)*(naugpos+naugneg)
    imgs_out = kwargs.pop('imgs_out',[])
    if len(imgs_out)==0:
        imgs_out = np.zeros([naug]+list(imgdata[0].shape),dtype=imgdata.dtype)
    labs_out = kwargs.pop('labs_out',[])
    if len(labs_out)==0:
        labs_out = np.zeros([naug,imglabs.shape[1]],dtype=imglabs.dtype)

    i = 0
    for imgi,labi in perturb_gen(imgdata,imglabs,ds_factor,
                                 naugpos,naugneg,test,
                                 transpose=transpose,**kwargs):
        imgs_out[i] = imgi
        labs_out[i] = labi
        i+=1

    return imgs_out, labs_out
    
if __name__ == '__main__':
    import pylab as pl
    from scipy.io import loadmat
    datf = '/Users/bbue/Research/IMBUE/PTF/asteroids/convnets/data/streaksv2_imgdat_cen.mat'
    mat = loadmat(datf)
    imgdata = mat['imgdat'] 
    subdata = mat['subdat']
    imgdata_cen = mat['imgdat_cen'] 
    subdata_cen = mat['subdat_cen']
    imglabs=mat['imglabs'].squeeze()
    maskdata = mat['maskdat']
    print(mat.keys())
    print(imglabs.shape,imgdata.shape)
    imgdata = imgdata[imglabs==1][:,:,:,np.newaxis]
    imglabs = imglabs[imglabs==1]
    imglabs = np.c_[imglabs,np.zeros_like(imglabs)]
    print(imgdata.shape)

    imgp,labp = perturb_batch(imgdata,imglabs)
    fig,ax = pl.subplots(3,2,sharex=True,sharey=True)
    ax[0,0].imshow(imgdata[0,...].squeeze())
    ax[0,1].imshow(imgp[0,...].squeeze())
    ax[1,0].imshow(imgdata[2,...].squeeze())
    ax[1,1].imshow(imgp[4,...].squeeze())
    ax[2,0].imshow(imgdata[-1,...].squeeze())
    ax[2,1].imshow(imgp[-1,...].squeeze())
    pl.show()
