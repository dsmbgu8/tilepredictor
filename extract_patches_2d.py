import numbers
import numpy as np
from numpy.lib.stride_tricks import as_strided

def extract_patches(arr, patch_shape=8, extraction_step=1, writeable=False):
    """Modified version of sklearn.feature_extraction.image.extract_patches

    Extracts patches of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.

    writeable : boolean
        If False, returned patches are read-only.

    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple([slice(None, None, st) for st in extraction_step])
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides, writeable=writeable)
    return patches

def extract_patches_2d(image, patch_size, max_patches=None, random_state=None,
                       writeable=False, return_index=False, randomize=True):
    """
    Modified version of sklearn.feature_extraction.image.extract_patches_2d
    
    Reshape a 2D image into a collection of patches

    The resulting patches are allocated in a dedicated array.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    image : array, shape = (image_height, image_width) or
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of ints (patch_height, patch_width)
        the dimensions of one patch

    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    random_state : int, RandomState instance or None, optional (default=None)
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.  If int, random_state is the seed used by
        the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is
        the RandomState instance used by `np.random`.

    return_index : bool (default=False)
        Optionally return (n_patches,2) array of (imin,jmin) patch coordinates

    randomize : bool (default=True)
       Randomize order of returned patches
    
    Returns
    -------
    patches : array, shape = (n_patches, patch_height, patch_width) or
         (n_patches, patch_height, patch_width, n_channels)
         The collection of patches extracted from the image, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.
    indices : array, shape = (n_patches, 2)
         Integer indices of each extracted patch

    Examples
    --------

    >>> from extract_patches_2d import *
    >>> one_image = np.arange(16).reshape((4, 4))
    >>> one_image
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> patches = extract_patches_2d(one_image, (2, 2))
    >>> print(patches.shape)
    (9, 2, 2)
    >>> patches[0]
    array([[0, 1],
           [4, 5]])
    >>> patches[1]
    array([[1, 2],
           [5, 6]])
    >>> patches[8]
    array([[10, 11],
           [14, 15]])
    """
    from sklearn.utils import check_array, check_random_state
    from sklearn.feature_extraction.image import _compute_n_patches
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]
    patch_shape = (p_h,p_w,n_colors)
    extraction_step = 1
    
    # always use extraction step = 1 here, subsample later with extraction_step
    extracted_patches = extract_patches(image, patch_shape=patch_shape,
                                        extraction_step=1,
                                        writeable=False)

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    
    max_patches = n_patches if max_patches is None else \
                  min(max_patches,n_patches)
    h_max = i_h - p_h + 1
    w_max = i_w - p_w + 1
    i_s = np.arange(0, h_max, extraction_step, dtype=np.int)
    j_s = np.arange(0, w_max, extraction_step, dtype=np.int)
    
    if randomize:
        rng = check_random_state(random_state)
        i_s = i_s[rng.randint(len(i_s), size=max_patches)]
        j_s = j_s[rng.randint(len(j_s), size=max_patches)]
    else:
        i_s,j_s = map(np.ravel,np.meshgrid(i_s,j_s))
        if max_patches != n_patches:
            # return the first max_patches patches without randomization
            i_s,j_s = i_s[:max_patches],j_s[:max_patches]
            
    patches = extracted_patches[i_s, j_s, 0]
    
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        patches = patches.reshape(n_patches, p_h, p_w)
    else:
        patches = patches.reshape(n_patches, p_h, p_w, n_colors)

    if return_index:
        return patches, np.c_[i_s,j_s]
    
    return outpatches

def patch_counts(patch_index,patch_size):
    p_h,p_w = patch_size
    max_row,max_col = patch_index.max(axis=0)+patch_size

    mask = np.zeros([max_row,max_col])
    for c,(i,j) in enumerate(patch_index):
        mask[i:i+p_h,j:j+p_w] += 1

    return mask

if __name__ == '__main__':
    # from extract_patches_2d import *
    # fix seed for reproducible output patches
    np.random.seed(42) 
    input_image = np.arange(36).reshape((6, 6))
    print('input_image')
    print(input_image)
    # array([[ 0,  1,  2,  3],
    #        [ 4,  5,  6,  7],
    #        [ 8,  9, 10, 11],
    #        [12, 13, 14, 15]])
    patch_size=(2, 2)
    patches,indices = extract_patches_2d(input_image, patch_size,
                                         randomize=False,
                                         return_index=True)
    print(patches.shape)
    # (9, 2, 2)
    print(indices.shape)
    # (9,2)
    print('patches[0]')
    print(patches[0])
    print('indices[0]')
    print(indices[0])
    
    print('patches[1]')
    print(patches[1])
    print('indices[1]')
    print(indices[1])

    print('patches[-2]')
    print(patches[-2])
    print('indices[-2]')
    print(indices[-2])

    print('patches[-1]')
    print(patches[-1])
    print('indices[-1]')
    print(indices[-1])

    mask = patch_mask(indices,patch_size)

    print('mask: "%s"'%str((mask)))
