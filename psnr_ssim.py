import numpy as np
import argparse
import os
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
import statistics

CURR_DIR = os.path.abspath(os.path.dirname(__file__)) #C:/.../Indoor/indoor

parser = argparse.ArgumentParser()
parser.add_argument('--GT_dir', required=False,
  default='./test_images/J_original',  help='')
parser.add_argument('--DH_dir', required=False,
  default='./test_images/result',  help='')
opt = parser.parse_args()

GT_dir = opt.GT_dir
DH_dir = opt.DH_dir
GT_path = os.path.join(CURR_DIR,GT_dir)
DH_path = os.path.join(CURR_DIR,DH_dir)

psnr_list=[]
ssim_list=[]
for i in range(0,5):
    GTimg = Image.open(os.path.join(GT_path,str(i)+'.png'))
    GTimg= GTimg.resize( (480, 480))
    GTimg_convert_ndarray = np.array(GTimg)
    DHimg = Image.open(os.path.join(DH_path,str(i)+'.png'))
    DHimg_convert_ndarray = np.array(DHimg)
    psnr = compare_psnr(GTimg_convert_ndarray,DHimg_convert_ndarray)
    ssim = compare_ssim(GTimg_convert_ndarray,DHimg_convert_ndarray,multichannel = True)
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print('psnr=%f' % (psnr),end='  ')
    print('ssim=%f' % (ssim))
print('meanpsnr=%f' % (statistics.mean(psnr_list)) )
print('meanssim=%f' % (statistics.mean(ssim_list)) )
print('stdevpsnr=%f' % (statistics.stdev(psnr_list)) )
print('stdevssim=%f' % (statistics.stdev(ssim_list)) )
# for i in range(0,1):
#     GTimg = Image.open(os.path.join(GT_path,str(i)+'_indoor_GT.jpg'))
#     GTimg_convert_ndarray = np.array(GTimg)
#     DHimg = Image.open(os.path.join(DH_path,str(i)+'.png'))
#     DHimg_convert_ndarray = np.array(DHimg)
#     psnr = compare_psnr(GTimg_convert_ndarray,DHimg_convert_ndarray)
#     ssim = compare_ssim(GTimg_convert_ndarray,DHimg_convert_ndarray,multichannel = True)
#     print('PSNR %d: %f' % (i, psnr))
#     print('SSIM %d: %f\n' % (i, ssim))
