import sys,os
SOURCEFINDER_ROOT = os.path.expanduser('~/Research/srcfinder')
SOURCEFINDER_ROOT = os.getenv('SOURCEFINDER_ROOT') or SOURCEFINDER_ROOT
sys.path.insert(0,SOURCEFINDER_ROOT)
from srcfinder_train_util import convert_cmf_image
def cmf2rgb_load_func_250_4000(cmff):
    ppmm_min,ppmm_max = 250,4000
    return convert_cmf_image(cmff,ppmm_min,ppmm_max)

def cmf2rgb_load_func_100_6000(cmff):
    ppmm_min,ppmm_max = 100,6000
    return convert_cmf_image(cmff,ppmm_min,ppmm_max)
