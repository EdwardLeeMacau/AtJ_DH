"""
  Filename      [ gen_dataset.py ]
  PackageName   [ AtJ_DH.script ]
  Synopsis      [ Generate dataset by atmosphere transmission model with Nyu dataset. ]
"""

# β is in range [0.5, 1.5]
# t(x) = exp(−βd(x))
# A = [a;a;a] (a ∈ [0.7,1.0])
# I(x) = J(x)t(x) + A(1 − t(x))

# Read images and depth. 
# Original shape of image is [3 x 640 x 480], uint8
# Depth of image is (aligned to the image, in-painted) in of size [640 x 480], float

import json
import argparse
import os
import random
import warnings
# from tqdm import trange

import h5py
import numpy as np
import skimage.io as io

def normalize(ary, minimum, maximum):
    """ Normalize **t** with parameter **range** """
    ary = (ary - minimum) / (maximum - minimum)
    return ary

def synthesis(J, A, beta, depth):
    """ 
    Synthesis image by atmosphere transmission model 

    Parameters
    ----------
    J: np.array

    A: float
        A is in range [0, 1]

    beta: float

    depth: np.array

    Return
    ------
    I, t: np.array
        t is in range [0, 1]

    A: float
    """
    t = np.exp(-1 * beta * depth)
    t = np.tile(np.expand_dims(t, axis=-1), (1, 1, 3))

    I = np.clip(
        (J / 255 * t + A * np.ones(J.shape) * (1 - t)) * 255, 
        a_min=0,
        a_max=255
    ).astype(np.uint8)

    return I, t[:, :, 0]

def crop(ary, boundary):
    ary = ary[boundary:-boundary, boundary:-boundary]
    return ary

def gen_description(dataset, beta=1.0):
    with open('./nyu-v2-depth.json', 'w') as jsonfile:
        depth_dict = {}

        for i, depth in enumerate(dataset['depths'], 1):
            depth_max, depth_min = np.amax(depth), np.amin(depth)

            print("[{:4d}/{:4d}] MAX: {:.4f} MIN: {:.4f}".format(i, len(dataset['depths']), depth_max, depth_min))
            depth_dict[str(i)] = {'max': float(depth_max), 'min': float(depth_min)}

        json.dump(depth_dict, jsonfile)

    return

def main():
    # warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, 
        default='../dataset/nyu-v2', help="Output directory")
    parser.add_argument('--matModel', type=str,
        default='../dataset/nyu_depth_v2_labeled.mat', help="Matrix File")
    parser.add_argument('--randomBeta', action='store_true',
        default=False)
    parser.add_argument('--randomAtmosphere', action='store_true',
        default=False)
    parser.add_argument('--number', type=int, 
        default=9)
    parser.add_argument('--scale', type=float, nargs='*',
        default=[2, 6])
    parser.add_argument('--atmosphere', type=float, 
        default=0.5)
    opt = parser.parse_args()

    if not os.path.isdir(opt.directory):
        os.makedirs(opt.directory, exist_ok=True)

    if not os.path.exists(opt.matModel):
        raise ValueError("Nyu model doesn't exist ({})".format(opt.matModel))

    if len(opt.scale) != 2:
        raise ValueError("--scale should have 2 value (min / max)")

    for key, values in vars(opt).items():
        print('{:16} {:>50}'.format(key, str(values)))

    for subfolder in ['A', 'I', 'J', 'depth', 't']:
        if subfolder not in os.listdir(opt.directory):
            os.makedirs(os.path.join(opt.directory, subfolder))

    # Output format
    postfix = '.png'

    # Read .mat File (Nyu Dataset)
    f = h5py.File(opt.matModel)

    record = {}
    A_recorder = np.zeros((opt.number * len(f['images']), 1), dtype=float)
    B_recorder = np.zeros((opt.number * len(f['images']), 1), dtype=float)

    # Generate beta
    beta = np.linspace(opt.scale[0], opt.scale[1], opt.number)

    # gen_description(f)
    # raise NotImplementedError

    try:
    
        for i in range(len(f['images'])):
            img, depth_img = f['images'][i], f['depths'][i]

            # Reshape (HWC)
            J = np.empty([480, 640, 3])
            J[:, :, 0] = img[0, :, :].T
            J[:, :, 1] = img[1, :, :].T
            J[:, :, 2] = img[2, :, :].T
            J = crop(J, 10)

            depth = normalize(depth_img.T, depth_img.min(), depth_img.max())
            depth = crop(depth, 10)

            if opt.randomBeta:
                beta = [ random.uniform(opt.scale[0], opt.scale[1]) for _ in range(opt.number) ]

            for j in range(opt.number):
                A = random.uniform(0.7, 0.85) if opt.randomAtmosphere else opt.atmosphere
                I, t = synthesis(J, A, beta[j], depth)

                # Print histogram
                # print('>> T: ', np.histogram(t, bins=20, range=(0.0, 1.0))[0])
                record[str((i, j))] = np.histogram(t)[0].tolist()

                # Save Training Data
                io.imsave(os.path.join(opt.directory, 'I', 'I_' + str(i) + '_' + str(j) + postfix), I)
                np.save(os.path.join(opt.directory, 't', 't_' + str(i) + '_' + str(j)), t)
                A_recorder[i * opt.number + j, 0] = A
                B_recorder[i * opt.number + j, 0] = beta[j]

            depth = (depth * 255).astype(np.uint8)
            J = J.astype(np.uint8)

            # Generate Demo File
            io.imsave(os.path.join(opt.directory, 'J', 'J_' + str(i) + postfix), J)
            io.imsave(os.path.join(opt.directory, 'depth', 'depth_' + str(i) + postfix), depth)

            print('>> [{:4d}/{:4d}]'.format(i + 1, len(f['images'])))

    except KeyboardInterrupt:

        print()

    finally:

        # Save Training Data
        np.save(os.path.join(opt.directory, 'A'), A_recorder)
        np.save(os.path.join(opt.directory, 'B'), B_recorder)

        with open(os.path.join(opt.directory, 'record.json'), 'w') as jsonfile:
            json.dump(record, jsonfile)

if __name__ == "__main__":
    main()
