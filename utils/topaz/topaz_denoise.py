#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
import glob

import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loader import load_image
import utils.mrc as mrc

name = 'denoise'
help = 'denoise micrographs with various denoising algorithms'

def add_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(help)

    ## only describe the model
    # set GPU and number of worker threads
    parser.add_argument('-d', '--device', default=0, type=int, help='which device to use, set to -1 to force CPU (default: 0)')

    parser.add_argument('micrographs', nargs='*', help='micrographs to denoise')

    parser.add_argument('-o', '--output', help='directory to save denoised micrographs')
    parser.add_argument('--suffix', default='', help='add this suffix to each output file name. if no output directory is specified, denoised micrographs are written to the same location as the input with a default suffix of ".denoised" (default: none)')
    parser.add_argument('--format', dest='format_', default='jpg', help='output format for the images (default: mrc)') 
    parser.add_argument('--normalize', action='store_true', help='normalize the micrographs')

    parser.add_argument('--stack', action='store_true', help='denoise a MRC stack rather than list of micorgraphs')

    parser.add_argument('--save-prefix', help='path prefix to save denoising model')
    parser.add_argument('-m', '--model', nargs='+', default=['unet'], help='use pretrained denoising model(s). can accept arguments for multiple models the outputs of which will be averaged. pretrained model options are: unet, unet-small, fcnn, affine. to use older unet version specify unet-v0.2.1 (default: unet)')

    parser.add_argument('-a', '--dir-a', nargs='+', help='directory of training images part A')
    parser.add_argument('-b', '--dir-b', nargs='+', help='directory of training images part B')
    parser.add_argument('--hdf', help='path to HDF5 file containing training image stack as an alternative to dirA/dirB')
    parser.add_argument('--preload', action='store_true', help='preload micrographs into RAM')
    parser.add_argument('--holdout', type=float, default=0.1, help='fraction of training micrograph pairs to holdout for validation (default: 0.1)')

    parser.add_argument('--lowpass', type=float, default=1, help='lowpass filter micrographs by this amount (in pixels) before applying the denoising filter. uses a hard lowpass filter (i.e. sinc) (default: no lowpass filtering)')
    parser.add_argument('--gaussian', type=float, default=0, help='Gaussian filter micrographs with this standard deviation (in pixels) before applying the denoising filter (default: 0)')
    parser.add_argument('--inv-gaussian', type=float, default=0, help='Inverse Gaussian filter micrographs with this standard deviation (in pixels) before applying the denoising filter (default: 0)')

    parser.add_argument('--deconvolve', action='store_true', help='apply optimal Gaussian deconvolution filter to each micrograph before denoising')
    parser.add_argument('--deconv-patch', type=int, default=1, help='apply spatial covariance correction to micrograph to this many patches (default: 1)')

    parser.add_argument('--pixel-cutoff', type=float, default=0, help='set pixels >= this number of standard deviations away from the mean to the mean. only used when set > 0 (default: 0)')
    parser.add_argument('-s', '--patch-size', type=int, default=1024, help='denoises micrographs in patches of this size. not used if < 1 (default: 1024)')
    parser.add_argument('-p', '--patch-padding', type=int, default=500, help='padding around each patch to remove edge artifacts (default: 500)')

    parser.add_argument('--method', choices=['noise2noise', 'masked'], default='noise2noise', help='denoising training method (default: noise2noise)')
    parser.add_argument('--arch', choices=['unet', 'unet-small', 'unet2', 'unet3', 'fcnet', 'fcnet2', 'affine'], default='unet', help='denoising model architecture (default: unet)')


    parser.add_argument('--optim', choices=['adam', 'adagrad', 'sgd'], default='adagrad', help='optimizer (default: adagrad)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--criteria', default='L2', choices=['L0', 'L1', 'L2'], help='training criteria (default: L2)')

    parser.add_argument('-c', '--crop', type=int, default=800, help='training crop size (default: 800)')
    parser.add_argument('--batch-size', type=int, default=4, help='training batch size (default: 4)')

    parser.add_argument('--num-epochs', default=100, type=int, help='number of training epochs (default: 100)') 

    parser.add_argument('--num-workers', default=16, type=int, help='number of threads to use for loading data during training (default: 16)')
    parser.add_argument('-j', '--num-threads', type=int, default=0, help='number of threads for pytorch, 0 uses pytorch defaults, <0 uses all cores (default: 0)')

    return parser


import utils.denoise as dn
from utils.image import save_image



# wget https://calla.rnet.missouri.edu/cryoppp_lite/10389.tar.gz
# tar -zxvf 10947.tar.gz -C /home/user/Topaz/data

# python topaz_denoise.py -o /home/user/Topaz/data/10947/denoised/ /home/user/Topaz/data/10947/micrographs/*.jpg

def denoise_image(mic, models, deconvolve=False, deconv_patch=1, patch_size=-1, padding=0, normalize=False, use_cuda=True):

    mic = torch.from_numpy(mic)    # from numpy to torch
    if use_cuda:
        mic = mic.cuda()

    # normalize and remove outliers
    mu = mic.mean()
    std = mic.std()
    x = (mic - mu)/std

    # apply guassian/inverse gaussian filter
    if deconvolve:
        # estimate optimal filter and correct spatial correlation
        x = dn.correct_spatial_covariance(x, patch=deconv_patch)

    # denoise
    mic = 0
    for model in models:
        mic += dn.denoise(model, x, patch_size=patch_size, padding=padding)   # default：patch_size=1024，padding=500
    mic /= len(models)

    # restore pixel scaling
    if normalize:
        mic = (mic - mic.mean())/mic.std()
    else:
        # add back std. dev. and mean
        mic = std*mic + mu

    # back to numpy/cpu
    mic = mic.cpu().numpy()

    return mic


def main(args):

    # set the number of threads
    num_threads = args.num_threads
    from utils.torch import set_num_threads
    set_num_threads(num_threads)

    # ## set the device
    # use_cuda = topaz.cuda.set_device(args.device)
    # print('# using device={} with cuda={}'.format(args.device, use_cuda), file=sys.stderr)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.device)
        print('# using device={} with cuda={}'.format(args.device, use_cuda), file=sys.stderr)
    else:
        print('CUDA is not available, using CPU.', file=sys.stderr)

    do_train = False
    
    if do_train == False: # load the saved model(s)
        models = []
        for arg in args.model:
            if arg == 'none':
                print('# Warning: no denoising model will be used', file=sys.stderr)
            else:
                print('# Loading model:', arg, file=sys.stderr)
            model = dn.load_model(arg)

            model.eval()
            if use_cuda:
                model.cuda()

            models.append(model)

    # using trained model
    # denoise the images

    normalize = args.normalize
    if args.format_ == 'png' or args.format_ == 'jpg':
        # always normalize png and jpg format
        normalize = True

    format_ = args.format_
    suffix = args.suffix

    deconvolve = args.deconvolve
    deconv_patch = args.deconv_patch

    ps = args.patch_size
    padding = args.patch_padding

    count = 0

    # we are denoising a single MRC stack
    if args.stack:
        with open(args.micrographs[0], 'rb') as f:
            content = f.read()
        stack,_,_ = mrc.parse(content)
        print('# denoising stack with shape:', stack.shape, file=sys.stderr)
        total = len(stack)

        denoised = np.zeros_like(stack)
        for i in range(len(stack)):
            mic = stack[i]
            # process and denoise the micrograph
            mic = denoise_image(mic, models, deconvolve=deconvolve
                               , deconv_patch=deconv_patch
                               , patch_size=ps, padding=padding, normalize=normalize
                               , use_cuda=use_cuda
                               )
            denoised[i] = mic

            count += 1
            print('# {} of {} completed.'.format(count, total), file=sys.stderr, end='\r')

        print('', file=sys.stderr)
        # write the denoised stack
        path = args.output
        print('# writing', path, file=sys.stderr)
        with open(path, 'wb') as f:
            mrc.write(f, denoised)
    
    else:
        # stream the micrographs and denoise them
        total = len(args.micrographs)
        if total < 1:
            # if no micrographs are input for denoising, quit
            return

        # make the output directory if it doesn't exist
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        for path in args.micrographs:
            name,_ = os.path.splitext(os.path.basename(path))
            mic = np.array(load_image(path), copy=False).astype(np.float32)

            # process and denoise the micrograph
            mic = denoise_image(mic, models, deconvolve=deconvolve
                               , deconv_patch=deconv_patch
                               , patch_size=ps, padding=padding, normalize=normalize
                               , use_cuda=use_cuda
                               )

            # write the micrograph
            if not args.output:
                if suffix == '' or suffix is None:
                    suffix = '.denoised'
                # write the file to the same location as input
                no_ext,ext = os.path.splitext(path)
                outpath = no_ext + suffix + '.' + format_
            else:
                outpath = args.output + os.sep + name + suffix + '.' + format_
            save_image(mic, outpath) #, mi=None, ma=None)

            count += 1
            print('# {} of {} completed.'.format(count, total), file=sys.stderr, end='\r')
        print('', file=sys.stderr)



if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    main(args)
