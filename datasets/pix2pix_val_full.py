import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
# from guidedfilter import guidedfilter
# import guidedfilter.guidedfilter as guidedfilter





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

class pix2pix_val_full(data.Dataset):
  def __init__(self, root, transform=None, loader=default_loader, seed=None):
    imgs = make_dataset(root)
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
    # index = np.random.randint(self.__len__(), size=1)[0]
    # index = np.random.randint(self.__len__(), size=1)[0]

    path = self.imgs[index]

    path=self.root+'/'+str(index)+'.png'

    # path=self.root+'/'+str(index)+'.jpg'



    index_folder = np.random.randint(0,4)
    label=index_folder

    img = self.loader(path)

    w, h = img.size
    print('123')
    img=self.my_get_image_for_test(img)
    imgA,imgB= img,img
    if self.transform is not None:
      # NOTE preprocessing for each pair of images
      # imgA, imgB, imgC = self.transform(imgA, imgB, imgC)
      imgA, imgB = self.transform(imgA, imgB)

    return imgA, imgB, path
  def my_get_image_for_test(self,img,pad=6):
    img=np.array(img)
    img = img.astype(np.float32)
    H, W, C = img.shape
    Wk = W
    Hk = H
    if W % 32:
        Wk = W + (32 * pad - W % 32)
    if H % 32:
        Hk = H + (32 * pad - H % 32)
    img = np.pad(img, ((32*pad, Hk - H), (32*pad, Wk - W), (0, 0)), 'reflect')
    im_input = img / 255.0
    # im_input = np.expand_dims(np.rollaxis(im_input, 2), axis=0)
    return im_input
  def __len__(self):
    return len(self.imgs)
