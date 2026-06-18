import sys

import numpy as np
import torch
import torch.nn as nn
import scipy.io as scio

from loss import loss_function
from utils import image_normalization, image_restoration


class Attack_Method(nn.Module):
    def __init__(self, method, dataset, bit):
        super(Attack_Method, self).__init__()
        self.method = method
        self.dataset = dataset
        self.bit = bit

    def AE_image_generate(self, model, images, texts, S,
                          Bi, Bt,
                          epsilon=8 / 255, step=3):
        adv_image = torch.zeros_like(images.type(torch.float))

        if self.method == 'PGD':
            iteration = 20
            randomize = 10
            query = image_normalization(images).type(torch.float)
            delta = torch.zeros_like(images).cuda()
            if randomize:
                delta.uniform_(-epsilon, epsilon)
                delta.data = (query.data + delta.data).clamp(-1, 1) - query.data
            delta.requires_grad = True
            for i in range(iteration):
                noisy_output = model.image_model(image_restoration(query + delta))
                loss = loss_function(noisy_output, Bt, Bi, 1 - S, S)

                loss.backward()
                delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
                delta.data = delta.data.clamp(-epsilon, epsilon)
                delta.data = (query.data + delta.data).clamp(-1, 1) - query.data
                delta.grad.zero_()

            adv_image = image_restoration(query + delta.detach())

        return adv_image

    def AE_text_generate(self, model, images, texts, S,
                         Bi, Bt):
        adv_text = torch.zeros_like(texts.type(torch.float))

        if self.method == 'PGD':
            flag = -1  # if COCO: -1 not: 0
            epsilon = 0.05
            step = 3
            iteration = 20
            randomize = True
            query = texts.type(torch.float)
            delta = torch.zeros_like(texts).cuda()

            if randomize:
                delta.uniform_(-epsilon, epsilon)
                # if dataset == 'MS-COCO': -1,1 else 0,1
                delta.data = (query.data + delta.data).clamp(flag, 1) - query.data
            delta.requires_grad = True
            for i in range(iteration):
                noisy_output = model.text_model((query + delta))
                loss = loss_function(noisy_output, Bi, Bt, 1 - S, S)

                loss.backward()
                delta.data = delta - step * delta.grad.detach()
                delta.data = delta.data.clamp(-epsilon, epsilon)
                # if dataset == 'MS-COCO':-1,1 else 0,1
                delta.data = (query.data + delta.data).clamp(flag, 1) - query.data
                delta.grad.zero_()

            adv_text = query + delta.detach()

        return adv_text
