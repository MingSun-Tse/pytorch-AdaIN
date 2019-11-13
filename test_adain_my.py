from model import Encoder4_2, Decoder4_2, SmallEncoder4_2_4x, SmallDecoder4_4x
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils

# ref: AdaIN impel. (https://github.com/naoto0804/pytorch-AdaIN/blob/master/function.py)
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

BE_model = "../PytorchWCT/models/vgg_normalised_conv5_1.t7"
SE_model = "../Bin/Experiments/SERVER138-20191109-092157_run/weights/20191109-092157_E20.pth"
BD_model = "experiments_Decoder4_2/decoder_iter_160000.pth.tar"
SD_model = ""

BE = Encoder4_2(BE_model).vgg.cuda()
SE = SmallEncoder4_2_4x(SE_model).cuda()
BD = Decoder4_2(BD_model).cuda()

c_path = "input/content/avril.jpg"
s_path = "input/style/contrast_of_forms.jpg"

shorter_side = 512
def load_image(path):
  img = Image.open(path).convert("RGB")
  w, h = img.size
  if w < h: # resize the shorter side to `shorter_side`
    neww = shorter_side
    newh = int(h * neww / w)
  else:
    newh = shorter_side
    neww = int(w * newh / h)
  img = img.resize((neww, newh))
  img = transforms.ToTensor()(img)
  return img.unsqueeze(0).cuda()

imgC = load_image(c_path)
imgS = load_image(s_path)
cF = SE(imgC)
sF = SE(imgS)
stylizedcF = adaptive_instance_normalization(cF, sF)
sd = BD(stylizedcF)
vutils.save_image(sd, c_path.replace(".jpg", "_stylized_SE+BD.jpg"))



