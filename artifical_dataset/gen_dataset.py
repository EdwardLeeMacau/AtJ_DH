import skimage.io as io
import numpy as np
import h5py
import os
import random

for i in ['A', 'I', 'J', 'depth', 't']:
    if i not in os.listdir():
        os.makedirs(i)

# paths
path_to_depth = './nyu_depth_v2_labeled.mat'
J_dir = 'J/J_'
depth_dir = 'depth/depth_'
A_dir = 'A/A_'
t_dir = 't/t_'
I_dir = 'I/I_'
postfix = '.png'

f = h5py.File(path_to_depth) # read mat file
num_synth_per_img = 5

for i in range(len(f['images'])):
    # Read images. original format is [3 x 640 x 480], uint8
    img = f['images'][i]

    # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float
    depth_img = f['depths'][i]

    # reshape
    J = np.empty([480, 640, 3])
    J[:,:,0] = img[0,:,:].T
    J[:,:,1] = img[1,:,:].T
    J[:,:,2] = img[2,:,:].T
    depth = np.empty([480, 640, 3])
    depth[:,:,0] = depth_img[:,:].T
    depth[:,:,1] = depth_img[:,:].T
    depth[:,:,2] = depth_img[:,:].T

    for j in range(num_synth_per_img):
        # β ∈ [0.7,1.2]
        # t(x) = exp(−βd(x))
        # A = [a;a;a] (a ∈ [0.7,1.0])
        # I(x) = J(x)t(x) + A(1 − t(x))
        t = np.exp(-random.uniform(0.5, 1.5) * depth)
        A = random.uniform(0.5, 1.0) * np.ones(J.shape)
        I = (J / 255 * t + A * (1 - t)) * 255
        # save image io.save
        io.imsave(I_dir + str(i) + '_' + str(j) + postfix, I)
        io.imsave(J_dir + str(i) + '_' + str(j) + postfix, J)
        np.save(A_dir + str(i) + '_' + str(j),A)
        np.save(t_dir + str(i) + '_' + str(j),t)
        io.imsave(depth_dir + str(i) + '_'+str(j) + postfix, depth) # just for demo