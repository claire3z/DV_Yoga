import torch
import torchvision

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch_size(=1); b=number of feature maps; (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix by dividing by the number of element in each feature maps.
    gram = G.div(a * b * c * d)
    return gram

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        vgg19.eval()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[23:].eval())

        # for i in range(len(vgg19.features)):
        #     blocks.append(vgg19.features[i].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        content_loss = 0.0
        style_loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            content_loss += torch.nn.functional.l1_loss(x, y)
            g_x = gram_matrix(x)
            g_y = gram_matrix(y)
            style_loss += torch.nn.functional.mse_loss(g_x,g_y)

        perceptual_loss = content_loss*0.1 + style_loss*1. # different weights
        return perceptual_loss

vgg19_loss = VGGPerceptualLoss()

# Modified from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49#file-vgg_perceptual_loss-py

# # Test
# import numpy as np
# import matplotlib.pyplot as plt
# img = plt.imread('Image/reverse_warrior.jpg')
# plt.imshow(img)
# plt.show()
# img.shape #(977, 1489, 3)
#
# img_left = img[:,:745,:]
# img_right = img[:,745:,:]
#
# plt.subplot(1,2,1)
# plt.imshow(img_left)
# plt.subplot(1,2,2)
# plt.imshow(img_right)
# plt.show()
#
# loss = VGGPerceptualLoss()
# orig = torch.Tensor(img/255).unsqueeze_(0).permute(0,3,1,2)
# pred = torch.Tensor(img_left/255).unsqueeze_(0).permute(0,3,1,2)
# target = torch.Tensor(img_right/255).unsqueeze_(0).permute(0,3,1,2)
#
# loss(orig,target).item()
#


# Source: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.models as models
# import copy
#
# class ContentLoss(nn.Module):
#     def __init__(self, target,):
#         super(ContentLoss, self).__init__()
#         # we 'detach' the target content from the tree used to dynamically compute the gradient: this is a stated value,
#         # not a variable. Otherwise the forward method of the criterion will throw an error.
#         self.target = target.detach()
#
#     def forward(self, input):
#         self.loss = F.mse_loss(input, self.target)
#         return input
#
#
# def gram_matrix(input):
#     a, b, c, d = input.size()  # a=batch_size(=1) # b=number of feature maps # (c,d)=dimensions of a f. map (N=c*d)
#     features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
#     G = torch.mm(features, features.t())  # compute the gram product
#     # we 'normalize' the values of the gram matrix by dividing by the number of element in each feature maps.
#     return G.div(a * b * c * d)
#
#
# class StyleLoss(nn.Module):
#     def __init__(self, target_feature):
#         super(StyleLoss, self).__init__()
#         self.target = gram_matrix(target_feature).detach()
#
#     def forward(self, input):
#         G = gram_matrix(input)
#         self.loss = F.mse_loss(G, self.target)
#         return input
#
#
# cnn = models.vgg19(pretrained=True).features.eval()
#
# cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
# cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
#
# # create a module to normalize input image so we can easily put it in a nn.Sequential
# class Normalization(nn.Module):
#     def __init__(self, mean, std):
#         super(Normalization, self).__init__()
#         # .view the mean and std to make them [C x 1 x 1] so that they can directly work with image Tensor of shape [B x C x H x W].
#         # B is batch size. C is number of channels. H is height and W is width.
#         self.mean = torch.tensor(mean).view(-1, 1, 1)
#         self.std = torch.tensor(std).view(-1, 1, 1)
#
#     def forward(self, img):
#         # normalize img
#         return (img - self.mean) / self.std
#
# # desired depth layers to compute style/content losses :
# content_layers_default = ['conv_4']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
#
#
# def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
#                                style_img, content_img,
#                                content_layers=content_layers_default,
#                                style_layers=style_layers_default):
#     cnn = copy.deepcopy(cnn)
#
#     # normalization module
#     normalization = Normalization(normalization_mean, normalization_std).to(device)
#
#     # just in order to have an iterable access to or list of content/syle
#     # losses
#     content_losses = []
#     style_losses = []
#
#     # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
#     # to put in modules that are supposed to be activated sequentially
#     model = nn.Sequential(normalization)
#
#     i = 0  # increment every time we see a conv
#     for layer in cnn.children():
#         if isinstance(layer, nn.Conv2d):
#             i += 1
#             name = 'conv_{}'.format(i)
#         elif isinstance(layer, nn.ReLU):
#             name = 'relu_{}'.format(i)
#             # The in-place version doesn't play very nicely with the ContentLoss
#             # and StyleLoss we insert below. So we replace with out-of-place
#             # ones here.
#             layer = nn.ReLU(inplace=False)
#         elif isinstance(layer, nn.MaxPool2d):
#             name = 'pool_{}'.format(i)
#         elif isinstance(layer, nn.BatchNorm2d):
#             name = 'bn_{}'.format(i)
#         else:
#             raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
#
#         model.add_module(name, layer)
#
#         if name in content_layers:
#             # add content loss:
#             target = model(content_img).detach()
#             content_loss = ContentLoss(target)
#             model.add_module("content_loss_{}".format(i), content_loss)
#             content_losses.append(content_loss)
#
#         if name in style_layers:
#             # add style loss:
#             target_feature = model(style_img).detach()
#             style_loss = StyleLoss(target_feature)
#             model.add_module("style_loss_{}".format(i), style_loss)
#             style_losses.append(style_loss)
#
#     # now we trim off the layers after the last content and style losses
#     for i in range(len(model) - 1, -1, -1):
#         if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
#             break
#
#     model = model[:(i + 1)]
#
#     return model, style_losses, content_losses