import argparse

parser = argparse.ArgumentParser()

# ----------------------------------------------------- #
# Root of directories                                   #
# ----------------------------------------------------- #
parser.add_argument('--dataroot', nargs='*', type=str,
    default=['../dataset/DS5_2020/train512'], help='Path to train dataset')
parser.add_argument('--nyuDataroot',
    default='../dataset/nyu/train', help='Path to train dataset (with t(x))')
parser.add_argument('--valDataroot', nargs='*', type=str,
    default=['../dataset/DS5_2020/val512'], help='Path to validation dataset')
parser.add_argument('--outdir',
    default='./pretrained-model', help='path to saved model')

# ----------------------------------------------------- #
# Training curve setting                                #
# ----------------------------------------------------- #
parser.add_argument('--print_every', type=int,
    default=100, help="Number of iterations to print")
parser.add_argument('--val_every', type=int,
    default=2000, help="Number of iterations to validate")
parser.add_argument('--comment', type=str,
    default='AtJ_DH', help='Folder to output images and model checkpoints') 

# ----------------------------------------------------- #
# Optimizer setting                                     #
# ----------------------------------------------------- #
parser.add_argument('--learningRate', type=float,
    default=1e-4, help='Initial Learning Rate')
parser.add_argument('--weight_decay', type=float,
    default=5e-5, help='L2 regularization ratio.')
parser.add_argument('--step', type=int,
    default=20, help='Step size of Learning Rate schedular')
parser.add_argument('--gamma', type=float, 
    default=0.5, help='Ratio of Learning Rate decay')

# ----------------------------------------------------- #
# Learing property setting                              #
# ----------------------------------------------------- #
parser.add_argument('--batchSize', type=int, 
    default=2, help='Input batch size when training')
parser.add_argument('--valBatchSize', type=int, 
    default=1, help='Input batch size when validation')
parser.add_argument('--niter', type=int, 
    default=120, help='Number of epochs to train')
parser.add_argument('--lambdaG', type=float, 
    default=0.0, help='Proportion of Rehaze Loss')
parser.add_argument('--lambdaK', type=float, 
    default=0.0, help='Proportion of Perceptual Loss')
# parser.add_argument('--poolSize', type=int, 
#     default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--netG', type=str,
    default=None, help="Path to netG (to continue training)")

# ----------------------------------------------------- #
# Addition property                                     #
# ----------------------------------------------------- #
parser.add_argument('--workers', type=int, 
    default=8, help='Number of data loading workers')
parser.add_argument('--gpus', nargs='*', type=int,
    default=[0], help='To use the DataParallel method')
parser.add_argument('--verbose', action='store_true',
    default=False, help='Save the validation dehaze image')
