"""
  Filename      [ gen_dataset.py ]
  PackageName   [ AtJ_DH.script ]
  Synopsis      [ Generate dataset by atmosphere transmission model with Nyu dataset. ]
"""

# β is in range [0.5, 1.5]
# t(x) = exp(−βd(x))
# A = [a;a;a] (a ∈ [0.7,1.0])
# I(x) = J(x)t(x) + A(1 − t(x))

import argparse
import os
import random
import warnings
# from tqdm import trange

import h5py
import numpy as np
import skimage.io as io

def synthesis(J, depth):
    """ Synthesis image by atmosphere transmission model """
    beta = random.uniform(0.5, 1.5)
    t = np.exp(-1 * beta * depth)
    A = random.uniform(0.5, 1.0) * np.ones(J.shape)
    I = (J / 255 * t + A * (1 - t)) * 255

    return I, A, t

def main():
    # warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, 
        default='../dataset/AtJ_DH/nyu')
    parser.add_argument('--random', 
        default=False, action='store_true')
    parser.add_argument('--number', type=str, 
        default=5)
    opt = parser.parse_args()

    matModel = os.path.join(opt.directory, 'nyu_depth_v2_labeled.mat')

    if not os.path.isdir(opt.directory):
        raise ValueError("Root {} doesn't exist. Please check --directory".format(opt.directory))

    if not os.path.exists(matModel):
        raise ValueError("Nyu model doesn't exist ({})".format(matModel))

    for subfolder in ['A', 'I', 'J', 'depth', 't']:
        if subfolder not in os.listdir(opt.directory):
            os.makedirs(os.path.join(opt.directory, subfolder))

    # Output format
    postfix = '.png'

    # Read .mat File (Nyu Dataset)
    f = h5py.File(matModel)

    for i in range(len(f['images'])):
        # Read images and depth. 
        # Original shape of image is [3 x 640 x 480], uint8
        # Depth of image is (aligned to the image, in-painted) in of size [640 x 480], float
        img, depth_img = f['images'][i], f['depths'][i]

        # Reshape
        J = np.empty([480, 640, 3])
        J[:,:,0] = img[0,:,:].T
        J[:,:,1] = img[1,:,:].T
        J[:,:,2] = img[2,:,:].T

        depth = np.empty([480, 640, 3])
        depth[:,:,0] = depth_img[:,:].T
        depth[:,:,1] = depth_img[:,:].T
        depth[:,:,2] = depth_img[:,:].T

        for j in range(opt.number):
            I, A, t = synthesis(J, depth)

            # Save Training Data
            io.imsave(os.path.join(opt.directory, 'I', 'I_' + str(i) + '_' + str(j) + postfix), I)
            io.imsave(os.path.join(opt.directory, 'J', 'J_' + str(i) + '_' + str(j) + postfix), J)
            np.save(os.path.join(opt.directory, 'A', 'A_' + str(i) + '_' + str(j)), A)
            np.save(os.path.join(opt.directory, 't', 't_' + str(i) + '_' + str(j)), t)

            # Generate Demo File
            io.imsave(os.path.join(opt.directory, 'depth', 'depth_' + str(i) + '_'+ str(j) + postfix), depth)

if __name__ == "__main__":
    main()
