from torchvision import transforms
import torch
import numpy as np
import random


class Transformer():

    def __init__(self,
                 scale_tsamp=0.006291456,
                 scale_fsamp=1.9531,
                 fch1=1499.93896484375,
                 fend=1000,
                 padding=1):
        self.scale_tsamp = scale_tsamp
        self.scale_fsamp = scale_fsamp
        self.fch1 = fch1
        self.fend = fend
        self.padding = padding
        self.toTensor = transforms.ToTensor()
        self.delay_constant = 4148.808

        self.augment = transforms.Compose(
            [transforms.ColorJitter(brightness=0.5),
             transforms.ColorJitter(contrast=0.5)
             ]
        )
        # self.delay = 4148.808 * (1 / fend ** 2 - 1 / fch1 ** 2) / self.scale_tsamp
        # chan_center = fch1 + (fend-fch1)/2
        # self.center_delay = (4148.808 * (1 / chan_center** 2 - 1 / fch1 ** 2) / self.scale_tsamp)

    # def gamma_augment(self,img):
    #     gamma = random.random() * 4.7+0.7
    #     img = img**gamma
    #     return img

    def img_trans(self, img, augment=False):
        if augment:
            img = self.augment(img)
            # TODO: transform img with padding if necessary
        return self.toTensor(img).float()
        # return img

    def img_detrans(self, img):
        # img = img * self.norm_param[1] + self.norm_param[0]
        img = img*255
        img = img.astype(np.uint8)
        return img

    def ann_trans(self, anns, num_samps=None):
        '''

        :param anns: (numpy.array) N * 5 [t0, DM, f0, f1, isfake]
        :return: (tensor)
        '''
        if len(anns) == 0:
            return torch.tensor(anns)
        f0 = anns[:, 2]
        f1 = anns[:, 3]
        if num_samps is not None:
            f_steps = (f0-f1)/num_samps
            f_arrays = f0[:,None] - np.mgrid[0:len(f0),0:num_samps][1] * f_steps[:,None]
            dt_arrays = self.delay_constant * (1/f_arrays**2 - 1/self.fch1**2) * anns[:, 1,None]
            x_array = torch.tensor(anns[:,0,None] + dt_arrays,dtype=torch.float)/self.scale_tsamp
            y_array = torch.tensor(self.fch1 - f_arrays,dtype=torch.float) / self.scale_fsamp
            gt_points = torch.stack((x_array, y_array), dim=-1)
        dt0 = self.delay_constant * (1/f0**2 - 1/self.fch1**2) * anns[:,1]
        x0 = np.round((anns[:, 0] + dt0) / self.scale_tsamp)

        y1 = np.round((self.fch1 - f1) / self.scale_fsamp)
        y0 = np.round((self.fch1 - f0) / self.scale_fsamp)

        dt = self.delay_constant * (1/f1**2 - 1/self.fch1**2) * anns[:,1]
        x1 = np.round((anns[:,0] + dt) / self.scale_tsamp)

        f_center = (f0 + f1) /2
        y_center = np.round((self.fch1-f_center) / self.scale_fsamp)
        x_center = np.round((anns[:,0] + self.delay_constant * (1/f_center**2 - 1/self.fch1**2)) / self.scale_tsamp)
        anns = torch.tensor([x0,y0, x1, y1,x_center, y_center,np.ones_like(x_center)], dtype=torch.float).transpose(1, 0)
        # keep = anns[:, 0] < 4000
        # anns = anns[keep].clamp_max_(4000)
        if num_samps is not None:
            # gt_points = gt_points[keep].clamp_max_(4000)
            return anns, gt_points
        return anns
