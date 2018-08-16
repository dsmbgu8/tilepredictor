import sys,os
sys.path.insert(0,os.path.expanduser('~/Research/srcfinder'))
from srcfinder_train_util import convert_cmf_image
def cmf2rgb_load_func_250_4000(cmff):
    ppmm_min,ppmm_max = 250,4000
    return convert_cmf_image(cmff,ppmm_min,ppmm_max)
