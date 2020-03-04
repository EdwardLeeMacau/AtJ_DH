import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, 
    default='pix2pix_notcombined',  help='')
parser.add_argument('--dataroot', required=False, 
    default='../dataset/AtJ/trainData', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False, 
    default='../dataset/AtJ/valData', help='path to val dataset')
parser.add_argument('--outdir', required=False, 
    default='./pretrained-model/nyu_final_at.pth', help='path to saved model')
  
parser.add_argument('--outdir_CKPT', required=False, 
    default='./pretrained-model/nyu_finalCKPT_at.pth', help='path to checkpoint')
parser.add_argument('--outdir_max', required=False, 
    default='./pretrained-model/nyu_maxCKPT_at_cont.pth', help='path to max checkpoint')
parser.add_argument('--batchSize', type=int, 
    default=4, help='input batch size')
parser.add_argument('--valBatchSize', type=int, 
    default=1, help='input batch size')
parser.add_argument('--originalSize', type=int, 
    default=480, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int, 
    default=480, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int, 
    default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int, 
    default=3, help='size of the output channels')
# parser.add_argument('--ngf', type=int, 
#     default=64)
# parser.add_argument('--ndf', type=int, 
#     default=64)
parser.add_argument('--niter', type=int, 
    default=120, help='number of epochs to train for')
parser.add_argument('--lambda2', type=float, 
    default=1, help='proportion of L2 Loss')
parser.add_argument('--lambdaP', type=float, 
    default=0.2, help='proportion of Lp Loss')
parser.add_argument('--poolSize', type=int, 
    default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--netG', type=str,
    default=None, help="path to netG (to continue training)")
# parser.add_argument('--netD', 
#     default=None, help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, 
    default=4, help='number of data loading workers')
parser.add_argument('--exp', type=str,
    default='sample', help='folder to output images and model checkpoints')
# parser.add_argument('--display', type=int, 
#     default=5, help='interval for displaying train-logs')
# parser.add_argument('--evalIter', type=int, 
#     default=500, help='interval for evauating(generating) images from valDataroot')

# opt = parser.parse_args()
# print(opt)
