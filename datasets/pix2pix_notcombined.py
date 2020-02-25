import torch.utils.data as data

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import os.path
import numpy as np
# from guidedfilter import guidedfilter
# import guidedfilter.guidedfilter as guidedfilter
import numpy as np
import torchvision.transforms as transforms



IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
  images = []
  if not os.path.isdir(dir):
    raise Exception('Check dataroot')
  for root, _, fnames in sorted(os.walk(dir)):
    for fname in fnames:
      if is_image_file(fname):
        path = os.path.join(dir, fname)
        item = path
        images.append(item)
  return images

def default_loader(path):
    return Image.open(path).convert('RGB')
def np_loader(path):
    img=np.load(path)
    return img

class pix2pix_notcombined(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, split_points=None, seed=None):
    imgs = make_dataset(root)  # remember make_dataset(root) is 2 times length
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    self.root = root
    self.imgs = imgs
    self.transform = transform
    self.loader = loader
    
    if seed is not None:
      np.random.seed(seed)

  def __getitem__(self, index): 
    # hazy_path=self.root+'/'+str(index+1)+'_hazy.png' # remember make_dataset is 2 times length
    # gt_path=self.root+'/'+str(index+1)+'_gt.png'
    hazy_path=self.root+'/I/I_'+str(index)+'.png'
    # gt_path=self.root+'/J/J_'+str(index)+'.png'
    A_path=self.root+'/A/A_'+str(index)+'.npy'
    t_path=self.root+'/t/t_'+str(index)+'.npy'
    imgA = self.loader(hazy_path)
    imgB=np_loader(A_path)
    imgC=np_loader(t_path)
    
    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      # imgA, imgB, imgC = self.transform(imgA, imgB, imgC)
      imgA =self.transform(imgA)
      # imgB=transforms.ToTensor()(imgB)
      # imgC=transforms.ToTensor()(imgC)
      imgB = np.transpose(imgB, (2, 0, 1)) # hwc to chw
      imgC = np.transpose(imgC, (2, 0,1)) 

      # imgA, imgB = self.transform(imgA, imgB)

    return imgA, imgB,imgC

  def __len__(self):
    # return len(self.imgs)//2
    return len(self.imgs)//4

