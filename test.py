import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization
from function import coral
from model import Encoder4_2, SmallDecoder4_4x, SmallEncoder4_2_4x, Decoder4_2
from model import SmallEncoder4_64x_aux, SmallEncoder4_16x_aux, SmallEncoder4_FP16x_aux

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    if args.mode in ["SE64x+BD", "SE16x+BD", "E2D1"]:
      content_f = vgg.forward_aux(content, False)[-1]
      style_f = vgg.forward_aux(style, False)[-1]
    else:
      content_f = vgg(content)
      style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=0,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')
parser.add_argument('-m', '--mode', type=str)

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)

if args.content:
    content_paths = [args.content]
else:
    content_paths = [os.path.join(args.content_dir, f) for f in
                     os.listdir(args.content_dir)]

if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [args.style]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_paths = [os.path.join(args.style_dir, f) for f in
                   os.listdir(args.style_dir)]

if not os.path.exists(args.output):
    os.mkdir(args.output)

##---------------------------------------------
# use converted decoder from .t7 model
if args.mode == "original": 
  decoder = net.decoder
  vgg = net.vgg
  decoder.eval()
  vgg.eval()
  decoder.load_state_dict(torch.load(args.decoder))
  vgg.load_state_dict(torch.load(args.vgg))
  vgg = nn.Sequential(*list(vgg.children())[:31])

# this decoder is trained by the author using this code
if args.mode == "original2":
  decoder = net.decoder
  vgg = net.vgg
  decoder.eval()
  vgg.eval()
  decoder.load_state_dict(torch.load("decoder_iter_160000.pth.tar"))
  vgg.load_state_dict(torch.load(args.vgg))
  vgg = nn.Sequential(*list(vgg.children())[:31])
  
##########################################
# Note: this rebuttal running is wrong, since the decoder is not trained by AdaIN, just the decoder of WCT.
# # 2019/01 rebuttal
# if args.mode == "16x":
  # args.decoder = "../Experiments/d5_ploss0.01_conv12345_QA/weights/12-20190131-0539_5SD_16x_E20S10000-3.pth"
  # args.vgg = "../Experiments/e5_ploss0.05_conv12345_QA/weights/12-20181105-1644_5SE_16x_QA_E20S10000-2.pth"
  # decoder = net.SmallDecoder5_16x(args.decoder)
  # vgg = net.SmallEncoder5_16x_plus(args.vgg)
# elif args.mode == "original":
  # args.decoder = "../PytorchWCT/models/feature_invertor_conv5_1.t7"
  # args.vgg = "../PytorchWCT/models/vgg_normalised_conv5_1.t7"
  # decoder = net.Decoder5(args.decoder)
  # vgg = net.Encoder5(args.vgg)
##########################################
if args.mode == "BE4_2+BD4_2": # my trained Conv4_2 BE and BD
  args.decoder = "experiments_Decoder4_2/decoder_iter_160000.pth.tar"
  args.vgg = "../PytorchWCT/models/vgg_normalised_conv5_1.t7"
  decoder = Decoder4_2(args.decoder)
  vgg = Encoder4_2(args.vgg).vgg

if args.mode == "SE4x+BD4_2":
  args.decoder = "experiments_Decoder4_2/decoder_iter_160000.pth.tar" # BD
  decoder = Decoder4_2(args.decoder)
  args.vgg = "../Bin/Experiments/SERVER138-20191109-092157_run/weights/20191109-092157_E20.pth" # SE4x
  vgg = SmallEncoder4_2_4x(args.vgg).vgg

if args.mode == "SE64x+BD":
  args.decoder = "decoder_iter_160000.pth.tar" # BD
  decoder = net.decoder
  decoder.load_state_dict(torch.load(args.decoder))
  args.vgg = "../Bin/Experiments/SERVER138-20191112-012212_adain_64x_se/weights/20191112-012212_E18.pth" # SE64x
  vgg = SmallEncoder4_64x_aux(args.vgg)

if args.mode == "SE16x+BD":
  args.decoder = "decoder_iter_160000.pth.tar" # BD
  decoder = net.decoder
  decoder.load_state_dict(torch.load(args.decoder))
  args.vgg = "../Bin/Experiments/SERVER138-20191112-012137_adain_16x_se/weights/20191112-012137_E17.pth" # SE16x
  vgg = SmallEncoder4_16x_aux(args.vgg)

if args.mode == "SE+SD":
  args.decoder = "experiments_sd_4x/decoder_iter_160000.pth.tar" # 4x SD
  decoder = SmallDecoder4_4x(args.decoder)
  args.vgg = "../Bin/Experiments/SERVER138-20191109-092157_run/weights/20191109-092157_E20.pth" # SE
  vgg = SmallEncoder4_2_4x(args.vgg).vgg[:31] # 31 is relu4_1

if args.mode == "FP_SE16x+BD":
  decoder_path = "FP_SE16x+BD/decoder_iter_160000.pth.tar"
  decoder = net.decoder
  decoder.load_state_dict(torch.load(decoder_path))
  args.vgg = "../Bin/models/normalise_fp16x/fp16x_normalised_FP16x_4E_for_adain.pth"
  vgg = SmallEncoder4_FP16x_aux(args.vgg).vgg

# Demonstrate the collaboration phenomenon
# E1 = SE4x, D1 = Decoder4_2
# E2 = SE16x, D2 = decoder_iter_160000.pth.tar
if args.mode == "E1D2":
  args.vgg = "../Bin/Experiments/SERVER138-20191109-092157_run/weights/20191109-092157_E20.pth" # SE4x
  vgg = SmallEncoder4_2_4x(args.vgg).vgg
  args.decoder = "decoder_iter_160000.pth.tar" # BD
  decoder = net.decoder
  decoder.load_state_dict(torch.load(args.decoder))

if args.mode == "E2D1":
  args.vgg = "../Bin/Experiments/SERVER138-20191112-012137_adain_16x_se/weights/20191112-012137_E17.pth" # SE16x
  vgg = SmallEncoder4_16x_aux(args.vgg)
  args.decoder = "experiments_Decoder4_2/decoder_iter_160000.pth.tar"
  decoder = Decoder4_2(args.decoder)

if args.mode == "E1D1":
  pass # this is the same as "SE4x+BD4_2"

if args.mode == "E2D2":
  pass # this is the same as "SE16x+BD"
##---------------------------------------------

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(p)) for p in style_paths])
        content = content_tf(Image.open(content_path)) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = '{:s}/{:s}_interpolation{:s}'.format(
            args.output, splitext(basename(content_path))[0], args.save_ext)
        save_image(output, output_name)

    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(content_path))
            style = style_tf(Image.open(style_path))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha)
            output = output.cpu()

            output_name = '{:s}/{:s}_stylized_{:s}_{:s}{:s}'.format(
                args.output, splitext(basename(content_path))[0],
                splitext(basename(style_path))[0], args.mode, args.save_ext,
            )
            save_image(output, output_name)
