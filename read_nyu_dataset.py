"""
  Filename       [ read_nyu_dataset.py ]
  PackageName    [ AtJ_DH ]
"""

import h5py
import numpy as np
import skimage.io as io

def main():
    # data path
    path_to_depth = './nyu_depth_v2_labeled.mat'

    # read mat file
    f = h5py.File(path_to_depth)

    # read 0-th image. original format is [3 x 640 x 480], uint8

    img = f['images'][0]

    # reshape
    img_ = np.empty([480, 640, 3])
    img_[:,:,0] = img[0,:,:].T
    img_[:,:,1] = img[1,:,:].T
    img_[:,:,2] = img[2,:,:].T

    # imshow
    img__ = img_.astype('float32')

    io.imsave('rgb.png',img__)
    # io.show()


    # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
    depth = f['depths'][0]

    # reshape for imshow
    depth_ = np.empty([480, 640, 3])
    depth_[:,:,0] = depth[:,:].T
    depth_[:,:,1] = depth[:,:].T
    depth_[:,:,2] = depth[:,:].T
    io.imsave('depth.png',depth_)
    io.imshow(depth_/4.0)
    # io.imsave()

if __name__ == '__main__':
    main()