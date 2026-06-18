import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils import spectral_norm as SpectralNorm
import torch.nn.functional as F


class AutoEncoder_Image(nn.Module):
    def __init__(self, label_size, bit):
        super(AutoEncoder_Image, self).__init__()
        self.firstencoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.secondencoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.thridencoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fourencoder = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.fiveencoder = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.sixencoder = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        self.sevenencoder = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
        )

        self.firstdecoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        self.seconddecoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        self.thirddecoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.fourdecoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fivedecoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.sixdecoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.sevendecoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.robust = nn.Sequential(
            nn.Conv2d(2048, 4096, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(True),
        )
        self.senstive = nn.Sequential(
            nn.Conv2d(2048, 4096, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(True),
        )
        self.trans = nn.Sequential(
            nn.ConvTranspose2d(4096, 2048, kernel_size=1),
            nn.ReLU()
        )
        self.residual = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.classifier = nn.Linear(in_features=4096, out_features=label_size)
        self.hashing = nn.Linear(in_features=4096, out_features=bit)

    def forward(self, x):
        tmp_tensor = self.firstencoder(x)
        tmp_tensor_1 = tmp_tensor
        tmp_tensor = self.secondencoder(tmp_tensor)
        tmp_tensor_2 = tmp_tensor
        tmp_tensor = self.thridencoder(tmp_tensor)
        tmp_tensor_3 = tmp_tensor
        tmp_tensor = self.fourencoder(tmp_tensor)
        tmp_tensor_4 = tmp_tensor
        tmp_tensor = self.fiveencoder(tmp_tensor)
        tmp_tensor_5 = tmp_tensor
        tmp_tensor = self.sixencoder(tmp_tensor)
        tmp_tensor_6 = tmp_tensor
        original_feature = self.sevenencoder(tmp_tensor)

        robust_feature = self.robust(original_feature)
        robust_feature = robust_feature.view(robust_feature.size(0), -1)

        predict_robust = self.classifier(robust_feature)
        predict_robust = F.softmax(predict_robust, dim=1)
        code_robust = self.hashing(robust_feature)

        senstive_feature = self.senstive(original_feature)
        senstive_feature = senstive_feature.view(senstive_feature.size(0), -1)

        predict_senstive = self.classifier(senstive_feature)
        predict_senstive = F.softmax(predict_senstive, dim=1)
        code_senstive = self.hashing(senstive_feature)

        robust_feature = robust_feature.unsqueeze(-1).unsqueeze(-1)
        senstive_feature = senstive_feature.unsqueeze(-1).unsqueeze(-1)

        robust_feature = self.trans(robust_feature)
        senstive_feature = self.trans(senstive_feature)

        tmp_tensor = self.firstdecoder(robust_feature)
        tmp_tensor = torch.cat((tmp_tensor_6, tmp_tensor), dim=1)
        tmp_tensor = self.seconddecoder(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_5, tmp_tensor), dim=1)
        tmp_tensor = self.thirddecoder(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_4, tmp_tensor), dim=1)
        tmp_tensor = self.fourdecoder(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_3, tmp_tensor), dim=1)
        tmp_tensor = self.fivedecoder(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_2, tmp_tensor), dim=1)
        tmp_tensor = self.sixdecoder(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_1, tmp_tensor), dim=1)
        tmp_tensor = self.sevendecoder(tmp_tensor)

        decode_x = torch.cat([tmp_tensor, x], dim=1)
        return_x = self.residual(decode_x)

        return return_x, robust_feature, senstive_feature, code_robust, code_senstive, predict_robust, predict_senstive


class Discriminator_Image(nn.Module):
    def __init__(self, image_size=224, conv_dim=64, repeat_num=5):
        super(Discriminator_Image, self).__init__()
        layers = []
        layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / (2 ** repeat_num))
        self.main = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Conv2d(curr_dim, 1, kernel_size=kernel_size, bias=False),
            nn.Sigmoid())

    def forward(self, x, flag=0):
        h = self.main(x)
        out = self.fc(h)
        if flag == 1:
            out = out.squeeze(-1).squeeze(-1)
            if out.data > 0.5:
                return 1
            else:
                return 0
        else:
            return out.squeeze(-1).squeeze(-1)


class GANLoss(nn.Module):
    def __init__(self, target_real_label=0.0, target_fake_label=1.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class AutoEncoder_Text(nn.Module):
    def __init__(self, input_size, label_size, bit):
        super(AutoEncoder_Text, self, ).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 4096),
            nn.ReLU(True)
        )
        self.robust = nn.Sequential(
            nn.Linear(in_features=4096, out_features=8192),
            nn.ReLU(True)
        )
        self.senstive = nn.Sequential(
            nn.Linear(in_features=4096, out_features=8192),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size),
            nn.ReLU()
        )
        self.trans = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU()
        )
        self.residual = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            # if dataset == 'MS-COCO': Tanh  else: ReLU
            nn.ReLU()
        )
        self.classifier = nn.Linear(in_features=8192, out_features=label_size)
        self.hashing = nn.Linear(in_features=8192, out_features=bit)

    def forward(self, x):
        origin_feature = self.encoder(x)

        robust_f = self.robust(origin_feature)
        senstive_f = self.senstive(origin_feature)

        predict_robust = self.classifier(robust_f)
        predict_robust = F.softmax(predict_robust, dim=1)
        code_robust = self.hashing(robust_f)

        predict_senstive = self.classifier(senstive_f)
        predict_senstive = F.softmax(predict_senstive, dim=1)
        code_senstive = self.hashing(senstive_f)

        robust_f = self.trans(robust_f)
        senstive_f = self.trans(senstive_f)
        tmp_tensor = self.decoder(robust_f + origin_feature)

        decode_x = torch.cat([x, tmp_tensor], dim=1)
        return_x = self.residual(decode_x)

        return return_x, robust_f, senstive_f, code_robust, code_senstive, predict_robust, predict_senstive


class Discriminator_Text(nn.Module):
    def __init__(self, tag_dim):
        super(Discriminator_Text, self).__init__()
        layers = []
        curr_dim = tag_dim
        layers.append(nn.Linear(curr_dim, 512))
        layers.append(nn.Linear(512, 128))

        self.main = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x, flag=0):
        h = self.main(x)
        out = self.fc(h)

        if flag == 1:
            out = out.squeeze(-1).squeeze(-1)
            if out.data > 0.5:
                return 1
            else:
                return 0
        else:
            return out.squeeze(-1).squeeze(-1)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count -
                             opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.n_epochs,
                                                   eta_min=0)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
