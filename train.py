import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import torchvision.utils as vutils

import net
from sampler import InfiniteSamplerWrapper
from model import SmallEncoder4_2, SmallEncoder4_2_4x, SmallDecoder4_4x # SmallEncoder4_2 is 16x
from model import SmallEncoder4_FP16x_aux
from data_loader import Dataset_npy

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, help='Directory path to a batch of content images', default="../../../Dataset/COCO/train2014")
parser.add_argument('--style_dir', type=str, help='Directory path to a batch of style images', default="../../../Dataset/WikiArt/train")
parser.add_argument('--content_dir_npy', type=str, help='Directory path to a batch of style images', default="../../../Dataset/COCO/train2014_npy")
parser.add_argument('--style_dir_npy', type=str, help='Directory path to a batch of style images', default="../../../Dataset/WikiArt/train_npy")
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
# parser.add_argument('--log_dir', default='./logs',
                    # help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()
args.log_dir = args.save_dir

device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

# --------------------------------------------------------------
# # use my previous SE model
# decoder = net.SmallDecoder4_16x()
# args.vgg = "../Experiments/e4_ploss0.05_conv1234_QA/weights/192-20181114-0458_4SE_16x_QA_E20S10000-2.pth"
# vgg = net.SmallEncoder4_16x_aux(args.vgg, fixed=True)

# 2019-09-15
# encoder: VGG19 up to Conv4_2. train its decoder
# decoder = net.Decoder4_2()
# args.vgg = "../PytorchWCT/models/vgg_normalised_conv5_1.t7" # use conv5_1 to include conv4_2
# vgg = net.Encoder4_2(args.vgg, fixed=True)

# # train my new SD model
# decoder = SmallDecoder4_4x()
# SE_path = "../Bin/Experiments/SERVER138-20191109-092157_run/weights/20191109-092157_E20.pth"
# vgg = SmallEncoder4_2_4x(SE_path, fixed=True).vgg[:31] 

# original VGG19
# decoder = net.decoder
# vgg = net.vgg
# vgg.load_state_dict(torch.load(args.vgg))
# vgg = nn.Sequential(*list(vgg.children())[:31])

# 2019-11-13: train the BD for FP-slimmed 16x SE
decoder = net.decoder
SE_path = "../Bin/models/normalise_fp16x/fp16x_normalised_FP16x_4E_for_adain.pth" # fp16x_normalised.t7
vgg = SmallEncoder4_FP16x_aux(SE_path)
# --------------------------------------------------------------
network = net.Net(vgg, decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)
# content_dataset = Dataset_npy(args.content_dir_npy)
# style_dataset = Dataset_npy(args.style_dir_npy)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)
# optimizer = torch.optim.Adam(network.parameters(), lr=args.lr) # for training FP16x_BD, the SE still has a few params which need update

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    loss_c, loss_s, stylized_img = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth.tar'.format(args.save_dir,
                                                           i + 1))
    
    if (i + 1) % 100 == 0: # save image samples
        save_img = torch.cat([content_images, style_images, stylized_img], dim=0)
        path = os.path.join(args.save_dir, "%s_iter_%s.jpg" % (args.save_dir, i+1))
        vutils.save_image(save_img, path, nrow=args.batch_size)
    
writer.close()
