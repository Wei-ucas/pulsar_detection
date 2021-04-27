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
        # self.delay = 4148.808 * (1 / fend ** 2 - 1 / fch1 ** 2) / self.scale_tsamp
        # chan_center = fch1 + (fend-fch1)/2
        # self.center_delay = (4148.808 * (1 / chan_center** 2 - 1 / fch1 ** 2) / self.scale_tsamp)

    def gamma_augment(self,img):
        gamma = random.random() * 4.7+0.7
        img = img**gamma
        return img

    def img_trans(self, img, train=True):
        img = np.array(img, dtype=np.uint8)
        # if train:
        #     img = img.astype('float')
        #     img = img[:,:4000]
        #     img = img / img.max()
        if self.padding > 0:
            img = img
            # TODO: transform img with padding if necessary
        return self.toTensor(img).float()
        # return img

    def img_detrans(self, img):
        # img = img * self.norm_param[1] + self.norm_param[0]
        img = img*255
        img = img.astype(np.uint8)
        return img

    def ann_trans(self, anns):
        '''

        :param anns: (numpy.array) N * 5 [t0, DM, f0, f1, isfake]
        :return: (tensor)
        '''
        if len(anns) == 0:
            return torch.tensor(anns)

        f0 = anns[:, 2]
        f1 = anns[:, 3]
        dt0 = self.delay_constant * (1/f0**2 - 1/self.fch1**2) * anns[:,1]
        x0 = np.round((anns[:, 0] + dt0) / self.scale_tsamp)

        y1 = np.round((self.fch1 - f1) / self.scale_fsamp)
        y0 = np.round((self.fch1 - f0) / self.scale_fsamp)

        dt = self.delay_constant * (1/f1**2 - 1/self.fch1**2) * anns[:,1]
        x1 = np.round((anns[:,0] + dt) / self.scale_tsamp)

        f_center = (f0 + f1) /2
        y_center = np.round(f_center / self.scale_fsamp)
        x_center = np.round((anns[:,0] + self.delay_constant * (1/f_center**2 - 1/f0**2)) / self.scale_tsamp)
        # end_pos = start_pos + np.round(anns[:, 1] * self.delay)
        # center[end_pos > 4096] = 4096
        # end_pos[end_pos > 4096] = 4096
        # if self.dm_norm_param is not None:
            # anns[:, 1] = (anns[:, 1] - self.dm_norm_param[0]) / self.dm_norm_param[1]
            # anns[:,1] = (end_pos - start_pos)
        # anns = torch.tensor([start_pos,center, end_pos,  anns[:, 1], anns[:, 2]], dtype=torch.float).transpose(1, 0)
        # anns = torch.tensor([start_pos, end_pos, np.ones_like(start_pos)], dtype=torch.float).transpose(1, 0)
        anns = torch.tensor([x0,y0, x1, y1,x_center, y_center,np.ones_like(x_center)], dtype=torch.float).transpose(1, 0)
        anns = anns[anns[:,0]<4000]
        return anns.clamp_max_(4000)
