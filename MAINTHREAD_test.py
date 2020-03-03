"""
  Filename       [ MAINTHREAD_test.py ]
  PackageName    [ AtJ_DH ]
"""

import os
import argparse
import time
import glob

from PIL import Image
from torch.autograd import Variable
import torch

from utils import utils
import model.AtJ_At as net

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img

def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
    return norm_ip(t, t.min(), t.max())


def main():
    print("Testing model ........ ")

    parser = argparse.ArgumentParser(description="Pytorch AtJ_model Evaluation")
    parser.add_argument("--cuda", action="store_true", 
        default=True, help="use cuda? Default is True")
    parser.add_argument("--model", type=str, 
        default="./model/At_model2.pth")
    parser.add_argument("--test", type=str, 
        default="./test_images/NTIRE2019_RAW/test/Hazy", help="testset path")
    parser.add_argument('--outdir', type=str,
        default='./test_images/NTIRE2019_RAW/test/DeHazy', help='path to output folder')
    opt = parser.parse_args()

    cuda = opt.cuda
    device_label = 'GPU' if opt.cuda else 'CPU'

    if cuda and not torch.cuda.is_available():
        raise Exception(">>No GPU found, please run without --cuda")

    if not cuda:
        print(">>Run on *CPU*, the running time will be longer than reported GPU run time. \n"
              ">>To run on GPU, please run the script with --cuda option")

    save_path = opt.outdir
    utils.checkdirctexist(save_path)

    model = net.AtJ()
    pretrained_state_dict = torch.load(opt.model)
    model.load_state_dict(pretrained_state_dict['model'])

    image_list = glob.glob(os.path.join(opt.test, '*.png'))

    print(">>Start testing...\n"
        "\t\t Model: {0}\n"
        "\t\t Test on: {1}\n"
        "\t\t Results save in: {2}".format(opt.model, opt.test, save_path))

    avg_elapsed_time = 0.0
    count = 0
    i = 0

    for image_name in image_list:
        count += 1
        print(">>Processing ./{}".format(image_name))
        im_input, W, H = utils.get_image_for_test(image_name)

        with torch.no_grad():
            im_input = Variable(torch.from_numpy(im_input).float())

            if cuda:
                model.cuda()
                model.train(False)
                im_input = im_input.cuda()
            else:
                im_input = im_input.cpu()
                model.cpu()
            
            model.train(False)
            model.eval()
            start_time = time.time()
            # feeding forward
            im_output = model(im_input)
            im_output = im_output[0]
            
            # compute running time
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

        tensor = im_output.data.cpu()
        tensor = torch.squeeze(tensor)
        tensor = norm_range(tensor, None)
        ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        im = Image.fromarray(ndarr)
        filename = opt.outdir+'/' + str(i) + '.png'
        i += 1
        im.save(filename)

    print(">>Finished!"
        "It takes average {}s for processing single image on {}\n"
        "Results are saved at ./{}".format(avg_elapsed_time / count, device_label, save_path))

if __name__ == "__main__":
    main()