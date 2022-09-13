import numpy as np

def two_images_side_by_side_batch(img_a, img_b):
    assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
    assert img_a.dtype == img_b.dtype
    n, h, w, c = img_a.shape
    canvas = np.zeros((n, h, 2 * w, c), dtype=img_a.dtype)
    canvas[:, :, 0 * w:1 * w, :] = img_a
    canvas[:, :, 1 * w:2 * w, :] = img_b
    return canvas

def two_images_side_by_side(img_a, img_b):
    assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
    assert img_a.dtype == img_b.dtype
    h, w, c = img_a.shape
    canvas = np.zeros((h, 2 * w, c), dtype=img_a.dtype)
    canvas[:, 0 * w:1 * w, :] = img_a
    canvas[:, 1 * w:2 * w, :] = img_b
    return canvas

def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def get_corner(kpts):
    return kpts.max(0).tolist()+kpts.min(0).tolist()   

def crop_image(img, corners, margin=0.15):
    '''get cropped images with margin'''
    img = img.copy()
    h, w = img.shape[:2]
    # right, down, left, up
    right, down, left, up = np.array(corners).astype(int)
    right = min(right + int(margin * w), w)
    down = min(down + int(margin * h), h)
    left = max(left - int(margin * w), 0)
    up = max(up - int(margin * h), 0)
    return img[up:down, left:right], (left, up)

def clip_points(kpts, height, width):
    kpts[kpts<0]=0
    kpts[:,0][kpts[:,0]>width-1]=width-1
    kpts[:,1][kpts[:,1]>height-1]=height-1
    return kpts

def get_scales(MAX_LEN, aspect_ratios, hw_a):
    len_h = make_divisible(MAX_LEN*aspect_ratios[0], 32)
    len_w = make_divisible(MAX_LEN*aspect_ratios[1], 32)
    hw_new = np.array((len_h,len_w))
    ori_scale_a = hw_new/hw_a
    scale_a = 1.0/hw_new
    return hw_new, ori_scale_a, scale_a
    
def value_2_jet(value):
    if value == 0:
        return 0, 0, 0
    elif value <= 51:
        return 255, value*5, 0
    elif value < 102:
        value -= 51
        return 255-value*5, 255, 0
    elif value <= 153:
        value -= 102
        return 0, 255, value*5
    elif value < 204:
        value -= 153
        return 0, 255-(128*value/51.0+0.5), 255
    else:
        value -= 204
        return 0, 127-(127*value/51+0.5), 255

def time_counter(fn):
    def warpper(*args, **kwargs):
        import time
        start = time.time()
        res = fn(*args, **kwargs)
        end = time.time()
        print(f'{fn.__name__} cost %.2f s ' % (end-start))
        return res
    return warpper