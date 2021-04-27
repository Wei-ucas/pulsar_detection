from torch.utils.data import Dataset, DataLoader
from .transform import Transformer
import os
import json
from PIL import Image
import numpy as np
import torch


def pulse_collate(batch):
    batch = filter(lambda x: x is not None, batch)
    images, anns = zip(*batch)

    images = torch.stack(images, dim=0)

    return images, anns


class TrainSet(Dataset):

    def __init__(self, filpath, annpath, transformer, augment=True):
        self.filpath = filpath
        self.annpath = annpath
        self.transformer = transformer
        self.fils = os.listdir(filpath)
        self.anns = json.load(open(annpath, 'r'))
        self.augment = augment
        assert len(self.anns) >= len(self.fils)

    def __getitem__(self, index):
        filname = self.fils[index]
        filimg = Image.open(os.path.join(self.filpath, filname), 'r').convert('L')
        ann_dict = self.anns[filname[:-4]]
        isfake = filname[:4] == 'fake'
        ann = []
        for a in ann_dict:
            ann.append([a["time"],a["DM"], a['start_freq'], a['end_freq'], isfake])
        ann = np.array(ann, dtype=np.float)
        filimg = self.transformer.img_trans(filimg)
        if self.augment:
            filimg = self.transformer.gamma_augment(filimg)
        ann = self.transformer.ann_trans(ann)
        return filimg, ann

    def __len__(self):
        return len(self.fils)


class TrainLoader:
    # The main DataLoader, including image and annotation transformation
    def __init__(self,cfg):
        filpath = cfg['file_path']
        annpath = cfg['ann_path']
        batch_size = cfg['batch_size']
        num_works = cfg['num_works']
        pin_memory = cfg['pin_memory']

        self.filpath = filpath
        self.annpath = annpath
        self.batch_size = batch_size
        self.num_works = num_works
        self.pin_memory = pin_memory

        self.transformer = Transformer(
                                     scale_tsamp=cfg['scale_tsamp'],
                                     fch1=cfg['fch1'],
                                     fend=cfg['fend'],
                                     padding=1
                                    )
        self.filset = TrainSet(filpath, annpath, self.transformer)

        self.filLoader = DataLoader(self.filset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_works,
                                    pin_memory=pin_memory,
                                    collate_fn=pulse_collate)

    def __len__(self):
        return len(self.filLoader)

    def __iter__(self):
        return iter(self.filLoader)


if __name__ == '__main__':
    dataset = TrainSet('/ssd/wangw/Ai_pulsar_search/detection/pngs_cv','/ssd/wangw/Ai_pulsar_search/detection/annotations/ann.json')
    print(len(dataset))
    f, a = dataset[1]
    print(a)