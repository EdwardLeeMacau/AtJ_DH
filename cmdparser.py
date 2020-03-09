import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
    default='pix2pix_notcombined',  help='')
parser.add_argument('--dataroot', 
    default='../dataset/DS4_2020/train', help='Path to train dataset')
parser.add_argument('--nyuDataroot',
    default='../dataset/nyu/train', help='Path to train dataset (with t(x))')
parser.add_argument('--valDataroot',
    default='../dataset/DS4_2020/val', help='Path to validation dataset')
parser.add_argument('--outdir',
    default='./pretrained-model', help='path to saved model')

parser.add_argument('--exp', type=str,
    default='sample', help='Folder to output images and model checkpoints') 
# parser.add_argument('--outdir_CKPT', required=False, 
#     default='./pretrained-model/AtJ_DH_CKPT.pth', help='Path to checkpoint')
# parser.add_argument('--outdir_max', required=False, 
#     default='./pretrained-model/AtJ_DH_MaxCKPT.pth', help='Path to max checkpoint')
parser.add_argument('--batchSize', type=int, 
    default=2, help='Input batch size when training')
parser.add_argument('--valBatchSize', type=int, 
    default=1, help='Input batch size when validation')
# parser.add_argument('--originalSize', type=int, 
#     default=480, help='the height / width of the original input image')
# parser.add_argument('--imageSize', type=int, 
#     default=480, help='the height / width of the cropped input image to network')
# parser.add_argument('--inputChannelSize', type=int, 
#     default=3, help='size of the input channels')
# parser.add_argument('--outputChannelSize', type=int, 
#     default=3, help='size of the output channels')
# parser.add_argument('--ngf', type=int, 
#     default=64)
# parser.add_argument('--ndf', type=int, 
#     default=64)
parser.add_argument('--niter', type=int, 
    default=120, help='Number of epochs to train')
parser.add_argument('--lambda2', type=float, 
    default=1, help='Proportion of L2 Loss')
parser.add_argument('--lambdaP', type=float, 
    default=0.2, help='Proportion of Lp Loss')
parser.add_argument('--lambdaG', type=float, 
    default=0, help='Proportion of Lp Loss')
parser.add_argument('--lambdaK', type=float, 
    default=0, help='Proportion of Lp Loss')
parser.add_argument('--poolSize', type=int, 
    default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--netG', type=str,
    default=None, help="Path to netG (to continue training)")
# parser.add_argument('--netD', 
#     default=None, help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, 
    default=8, help='Number of data loading workers')
parser.add_argument('--print_every', type=int,
    default=10, help="Number of iterations to print")
parser.add_argument('--val_every', type=int,
    default=500, help="Number of iterations to validate")
# parser.add_argument('--display', type=int, 
#     default=5, help='interval for displaying train-logs')
# parser.add_argument('--evalIter', type=int, 
#     default=500, help='interval for evauating(generating) images from valDataroot')

