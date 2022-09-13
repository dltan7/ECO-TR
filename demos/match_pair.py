import os
import sys
import cv2
import matplotlib.pyplot as plt
from dotmap import DotMap
SCRIPT_DIR = sys.path.append(
    os.path.join((os.path.dirname(os.path.abspath(__file__))),'..'))
os.sys.path.append(SCRIPT_DIR)
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
from src.models.ecotr_engines import *
from src.models.utils.utils import *

if __name__ == '__main__':
    img0=cv2.imread('../assets/NIR01.jpg')
    img1=cv2.imread('../assets/photo.jpg')
    
    dict = DotMap(lower_config(get_cfg_defaults()))
    fix_randomness(42)
    engine=ECOTR_Engine(dict.ecotr)
    engine.use_cuda=True
    engine.load_weight('cuda')
    engine.MAX_KPTS_NUM=4000
    engine.ASPECT_RATIOS=[1.0,1.5]

    matches=engine.forward(img0,img1,cycle=False,level='fine')
    # xx=engine.forward_2stage(img0,img1,cycle=False)
    matches=matches[matches[:,-1]<1e-2]
    canvas = draw_matches(img0,img1,matches.copy(),'dot')
    height,width = canvas.shape[:2]

    cv2.imwrite('output.jpg',canvas)
    dpi=100
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize)
    plt.imshow(canvas)
    plt.show()
