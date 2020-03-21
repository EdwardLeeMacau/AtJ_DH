import os
import sys
from PIL import Image
import random
import numpy as np
import argparse
from natsort import natsorted, ns
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# # training preprocess
# python3.6 Preprocess3.py --inputHazy_dir /mnt/sdb/sychuang0909/sample/IHAZE_train_unprocessed --inputGT_dir /mnt/sdb/sychuang0909/sample/GT_train_unprocessed --outputHazy_dir /mnt/sdb/sychuang0909/sample/trainData_small/Hazy --outputGT_dir /mnt/sdb/sychuang0909/sample/trainData_small/GT --index_start 1 --operation Crop --trainORval 0 --regularORrandom 1
# python3.6 Preprocess3.py --inputHazy_dir /mnt/sdb/sychuang0909/sample/trainData_small/Hazy --inputGT_dir /mnt/sdb/sychuang0909/sample/trainData_small/GT --outputHazy_dir /mnt/sdb/sychuang0909/sample/trainData_hue --outputGT_dir /mnt/sdb/sychuang0909/sample/trainData_hue --index_start 1 --operation HSVshift --trainORval 0 --hshift 10 --hshift_step 2
# python3.6 Preprocess3.py --inputHazy_dir /mnt/sdb/sychuang0909/sample/trainData_small/Hazy --inputGT_dir /mnt/sdb/sychuang0909/sample/trainData_small/GT --outputHazy_dir /mnt/sdb/sychuang0909/sample/trainData_sat --outputGT_dir /mnt/sdb/sychuang0909/sample/trainData_sat --index_start 1 --operation HSVshift --trainORval 0 --sshift 10 --sshift_step 5

# CURR_DIR = os.path.abspath(os.path.dirname(__file__)) #C:/.../Indoor
parser = argparse.ArgumentParser()
parser.add_argument('--inputHazy_dir', required=False, 
  default='./sample/trainData4_hsv/Hazy',  help='')
parser.add_argument('--inputGT_dir', required=False, 
  default='./sample/trainData4_hsv/GT',  help='')
parser.add_argument('--outputHazy_dir', required=False, 
  default='./sample/trainData4_hsv/Hazy',  help='')
parser.add_argument('--outputGT_dir', required=False, 
  default='./sample/trainData4_hsv/GT',  help='')
parser.add_argument('--output_w', required=False, type=int,
  default=640,  help='')
parser.add_argument('--output_h', required=False, type=int,
  default=640,  help='')
parser.add_argument('--index_start', required=False, type=int,
  default=1,  help='')
parser.add_argument('--operation', required=False, 
  default='Crop',  help='') # Crop, HSVshift....only one operation at a time
parser.add_argument('--trainORval', required=False, type=int,
  default=0,  help='') # train=0 val=1

# For Cropping
parser.add_argument('--regularORrandom', required=False, type=int,
  default=1,  help='') # regular=0 random=1
# Regularcrop
parser.add_argument('--regular_view_ratio', required=False, type=float,
  default=0.5,  help='')
parser.add_argument('--regular_shift_ratio', required=False, type=float,
  default=0.25,  help='') # shift (right,down) the ratio of the (regular_view_ratio*w, regular_view_ratio*h)
# Randomcrop
parser.add_argument('--small_dataset', required=False, type=int,
  default=0,  help='')

# For HSVshift
parser.add_argument('--hshift', required=False, type=int,
  default=0,  help='') # shift_hue
parser.add_argument('--hshift_step', required=False, type=int,
  default=0,  help='') # hshift*hshift_step
parser.add_argument('--hshift_start', required=False, type=int,
  default=0,  help='')
parser.add_argument('--sshift', required=False, type=int,
  default=0,  help='') # shift_sat
parser.add_argument('--sshift_step', required=False, type=int,
  default=0,  help='') # sshift*sshift_step
parser.add_argument('--vshift', required=False, type=int,
  default=0,  help='') # shift_val
parser.add_argument('--vshift_step', required=False, type=int,
  default=0,  help='') # vshift*vshift_step

opt = parser.parse_args()

def rgb_to_hsv(rgb): # does this divide 255 for me?
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def shift_hue(arr,hshift,hshift_start):
    hsv=rgb_to_hsv(arr) # does this divide 255 for me? >>>> it works fine...so i think does...
    h0=hsv[...,0] 
    hsv[...,0]=(h0 + hshift + hshift_start) % 1.0 # !!! need to % 1.0
    # if((h0 + hshift + hshift_start) <= 1).all and ((h0 + hshift + hshift_start) >= 0).all: # question....is it ok to give everbody the same hue, and leave the rest to just saturation and value?
    #     hsv[...,0]=(h0 + hshift + hshift_start) % 1.0 # !!! need to % 1.0
    rgb=hsv_to_rgb(hsv)
    return rgb

def shift_sat(arr,sshift):
    hsv=rgb_to_hsv(arr)
    s0=hsv[...,1]
    s1=(s0*(1 + sshift))
    s1[s1 > 1.] = 1.
    hsv[...,1]=s1
    # s_ones=np.ones(s0)
    # np.greater_equal()
    # if(np.greater_equal(s_ones,s0*(1 + sshift))).all : 
    #     hsv[...,1]=(s0*(1 + sshift))
    #     print('YA')
    
    # if((s0 + sshift) <= 1).all and ((s0 + sshift) >= 0).all:
    #     hsv[...,1]=(s0 + sshift) % 1.0
    #     print('(s0 + sshift)= '+str(s0 + sshift))
    rgb=hsv_to_rgb(hsv)
    return rgb

def shift_val(arr,vshift):
    hsv=rgb_to_hsv(arr)
    v0=hsv[...,2]
    if(v0 + vshift <= 100).all and (v0 + vshift >= 0).all:
        hsv[...,2]=(v0 + vshift) % 1.0
    rgb=hsv_to_rgb(hsv)
    return rgb

# For Regularcrop
opt.regular_view_ratio = 0.5
opt.regular_shift_ratio = 0.25

# For Randomcrop
if not(opt.small_dataset):
    view_ratio_list = [0.25, 0.5, 0.75, 1] # view_ratio=0.25, means that we see the view of 1/4 of the original picture
    location_count_list = [225, 15, 9, 1] # 250 * () = 
else:
    view_ratio_list = [0.5, 0.75, 1]
    location_count_list = [10, 9, 1]
if(opt.trainORval==1):
    view_ratio_list = [0.5, 0.75, 1]
    location_count_list = [30, 18, 2] # 50 * 5 = 250

inputHazy_dir = opt.inputHazy_dir
if not os.path.exists(inputHazy_dir):
    os.makedirs(inputHazy_dir)

inputGT_dir = opt.inputGT_dir 
if not os.path.exists(inputGT_dir):
    os.makedirs(inputGT_dir)

outputHazy_dir = opt.outputHazy_dir 
if not os.path.exists(outputHazy_dir):
    os.makedirs(outputHazy_dir)

outputGT_dir = opt.outputGT_dir 
if not os.path.exists(outputGT_dir):
    os.makedirs(outputGT_dir)

fnames1 = []
fnames2 = []
print("!!!!!")
for root1, _, fnames in (os.walk(inputHazy_dir)):
    for i, fname1 in enumerate( natsorted(fnames, key=lambda y: y.lower()) ):
        img1 = Image.open(os.path.join(inputHazy_dir, fname1)).convert('RGB') # img = Image.open(os.path.join(root, fname)).convert('RGB')
        fnames1.append(img1)
        print(i)
        print(inputHazy_dir)
for root2, _, fnames in (os.walk(inputGT_dir)):
    for i, fname2 in enumerate( natsorted(fnames, key=lambda y: y.lower()) ):
        img2 = Image.open(os.path.join(inputGT_dir, fname2)).convert('RGB') # img = Image.open(os.path.join(root, fname)).convert('RGB')
        fnames2.append(img2)
        print(i)
        print(inputGT_dir)
print("!!!!!^^")

output_w = opt.output_w
output_h = opt.output_h
index = opt.index_start
t0 = time.time()

if (opt.operation=='Crop'):
    # Regularcrop
    if not (opt.regularORrandom):
        for i, (imgA, imgB) in enumerate( zip(fnames1, fnames2) ):
            w, h = imgA.size # imgAsize==imgBsize
            r = opt.regular_view_ratio
            s = opt.regular_shift_ratio
            tw = int(w*r)
            th = int(h*r)
            s_right = int(tw*s)
            s_down = int(th*s)
            x1,y1,count_x1,count_y1=0,0,0,0
            while y1 < h-th: # first shift right then down
                x1,count_x1=0,0
                while x1 < w-tw:
                    imga = imgA.crop((x1, y1, x1+tw, y1+th))
                    imgb = imgB.crop((x1, y1, x1+tw, y1+th))
                    imga = imga.resize((output_w, output_h), Image.BILINEAR)
                    imgb = imgb.resize((output_w, output_h), Image.BILINEAR)
                    imga.save(os.path.join(outputHazy_dir, str(index) + '_hazy.png'))
                    imgb.save(os.path.join(outputGT_dir, str(index) + '_gt.png'))
                    index = index+1
                    count_x1 = count_x1+1
                    x1 = x1+s_right*count_x1
                if x1 >= w-tw: # last to crop, located at right edge
                    imga = imgA.crop((w-tw, y1, w, y1+th))
                    imgb = imgB.crop((w-tw, y1, w, y1+th))
                    imga = imga.resize((output_w, output_h), Image.BILINEAR)
                    imgb = imgb.resize((output_w, output_h), Image.BILINEAR)
                    imga.save(os.path.join(outputHazy_dir, str(index) + '_hazy.png'))
                    imgb.save(os.path.join(outputGT_dir, str(index) + '_gt.png'))
                    index = index+1
                count_y1 = count_y1+1
                y1 = y1+s_down*count_y1
            if y1 >= h-th: # last to crop, located at down edge
                x1,count_x1=0,0
                while x1 < w-tw:
                    imga = imgA.crop((x1, h-th, x1+tw, h))
                    imgb = imgB.crop((x1, h-th, x1+tw, h))
                    imga = imga.resize((output_w, output_h), Image.BILINEAR)
                    imgb = imgb.resize((output_w, output_h), Image.BILINEAR)
                    imga.save(os.path.join(outputHazy_dir, str(index) + '_hazy.png'))
                    imgb.save(os.path.join(outputGT_dir, str(index) + '_gt.png'))
                    index = index+1
                    count_x1 = count_x1+1
                    x1 = x1+s_right*count_x1
                if x1 >= w-tw: # last to crop, located at down,right edge
                    imga = imgA.crop((w-tw, h-th, w, h))
                    imgb = imgB.crop((w-tw, h-th, w, h))
                    imga = imga.resize((output_w, output_h), Image.BILINEAR)
                    imgb = imgb.resize((output_w, output_h), Image.BILINEAR)
                    imga.save(os.path.join(outputHazy_dir, str(index) + '_hazy.png'))
                    imgb.save(os.path.join(outputGT_dir, str(index) + '_gt.png'))
                    index = index+1
        print('after regular crop index_start: ' + str(index))

    else:
        # Randomcrop
        for v, (view_ratio, location_count)in enumerate( zip(view_ratio_list, location_count_list) ):
            for i, (imgA, imgB) in enumerate( zip(fnames1, fnames2) ):
                for j in range(0, location_count): # 100 different locations >> gives total outputs of 20*100=2000 pairs of (hazy, gt) imgs
                    w, h = imgA.size # imgAsize==imgBsize
                    r = view_ratio
                    # RandomCrop(imageSize),
                    tw = int(w*r)
                    th = int(h*r)
                    imga = imgA
                    imgb = imgB
                    if not(w == tw and h == th):
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                        imga = imgA.crop((x1, y1, x1 + tw, y1 + th))
                        imgb = imgB.crop((x1, y1, x1 + tw, y1 + th))
                        # RandomHorizontalFlip(),
                        flag = random.random() < 0.5
                        if flag:
                            imga = imga.transpose(Image.FLIP_LEFT_RIGHT)
                            imgb = imgb.transpose(Image.FLIP_LEFT_RIGHT)
                    else:
                        if(j>location_count/2):
                            imga = imga.transpose(Image.FLIP_LEFT_RIGHT)
                            imgb = imgb.transpose(Image.FLIP_LEFT_RIGHT)
                    # Resize to 640
                    imga = imga.resize((output_w, output_h), Image.BILINEAR)
                    imgb = imgb.resize((output_w, output_h), Image.BILINEAR)
                    # save all in trainData as 0_hazy.png, 0_gt.png, 1_hazy.png, 1_gt.png......
                    imga.save(os.path.join(outputHazy_dir, str(index) + '_hazy.png'))
                    imgb.save(os.path.join(outputGT_dir, str(index) + '_gt.png'))
                    index = index+1
        print('after random crop index_start: ' + str(index))

elif (opt.operation=='HSVshift'):
    for i, (imgA, imgB) in enumerate( zip(fnames1, fnames2) ):
        tt0 = time.time()
        arrA = np.array(imgA)
        arrB = np.array(imgB)
        if(opt.hshift != 0):
            # shift hue with range of (+-)hshift, and then divide by 360 before shifting
            hshift = opt.hshift
            hshift_step = opt.hshift_step
            hshift_start = opt.hshift_start/360
            for h in range(1, hshift+1): # h==1~hshift
                _h = float(h*hshift_step)/360 # shift hshift_step hues per time
                arra = shift_hue(arrA, _h, hshift_start)
                arrb = shift_hue(arrB, _h, hshift_start)
                imgA = Image.fromarray(arra, 'RGB')
                imgB = Image.fromarray(arrb, 'RGB')
                imgA.save(os.path.join(outputHazy_dir, str(index) + '_hazy.png'))
                imgB.save(os.path.join(outputGT_dir, str(index) + '_gt.png'))
                index = index + 1
                arra = shift_hue(arrA, -_h, hshift_start)
                arrb = shift_hue(arrB, -_h, hshift_start)
                imgA = Image.fromarray(arra, 'RGB')
                imgB = Image.fromarray(arrb, 'RGB')
                imgA.save(os.path.join(outputHazy_dir, str(index) + '_hazy.png'))
                imgB.save(os.path.join(outputGT_dir, str(index) + '_gt.png'))
                index = index + 1
        if(opt.sshift != 0):
            # shift saturation with range of (+-)sshift, and then divide by 100 before shifting
            sshift = opt.sshift
            sshift_step = opt.sshift_step
            for s in range(1, sshift+1): # s==1~sshift
                _s = float(s*sshift_step)/100
                # print('_s= '+str(_s))
                arra = shift_sat(arrA, _s)
                arrb = shift_sat(arrB, _s)
                imgA = Image.fromarray(arra, 'RGB')
                imgB = Image.fromarray(arrb, 'RGB')
                imgA.save(os.path.join(outputHazy_dir, str(index) + '_hazy.png'))
                imgB.save(os.path.join(outputGT_dir, str(index) + '_gt.png'))
                index = index + 1
                arra = shift_sat(arrA, -_s)
                arrb = shift_sat(arrB, -_s)
                imgA = Image.fromarray(arra, 'RGB')
                imgB = Image.fromarray(arrb, 'RGB')
                imgA.save(os.path.join(outputHazy_dir, str(index) + '_hazy.png'))
                imgB.save(os.path.join(outputGT_dir, str(index) + '_gt.png'))
                index = index + 1
        if(opt.vshift != 0):
            # shift brightness value with range of (+-)vshift, and then divide by 100 before shifting
            vshift = opt.vshift
            vshift_step = opt.vshift_step
            for v in range(1, vshift+1): # s==1~sshift
                _v = float(v*vshift_step)/100
                arra = shift_val(arrA, _v)
                arrb = shift_val(arrB, _v)
                imgA = Image.fromarray(arra, 'RGB')
                imgB = Image.fromarray(arrb, 'RGB')
                imgA.save(os.path.join(outputHazy_dir, str(index) + '_hazy.png'))
                imgB.save(os.path.join(outputGT_dir, str(index) + '_gt.png'))
                index = index + 1
                arra = shift_val(arrA, -_v)
                arrb = shift_val(arrB, -_v)
                imgA = Image.fromarray(arra, 'RGB')
                imgB = Image.fromarray(arrb, 'RGB')
                imgA.save(os.path.join(outputHazy_dir, str(index) + '_hazy.png'))
                imgB.save(os.path.join(outputGT_dir, str(index) + '_gt.png'))
                index = index + 1
        tt1 = time.time()
        print('run time per img = ' + str(tt1-tt0))
        


else:
    print('should insert operation name(ex: Crop, HSVshift)')


t1 = time.time()
print('Total running time:'+str(t1-t0))