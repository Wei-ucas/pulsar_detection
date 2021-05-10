import torch
from torch.utils.data import Dataset
import glob
from sigpyproc.Readers import FilReader as filterbank
import os
from datasets.transform import Transformer
from utils import data2pic

dt = 0.006291456
df = -1.953125

class FilSet(Dataset):
    def __init__(self, fil_path, f_downsamp=16, t_downsamp=32):
        self.fil_root = fil_path
        self.f_downsamp = f_downsamp
        self.t_downsamp = t_downsamp
        self.fil_lists = glob.glob(fil_path + '/*.fil')
        self.transform = Transformer()
        # self.fil_iter = iter(self.fil_lists)

    def __len__(self):
        return len(self.fil_lists)

    # use sigpyproc to read filterbank data
    def load_fil(self, fil_name):
        fil_id = os.path.split(fil_name)[1].rstrip('.fil')
        fil = filterbank(fil_name)

        fch1 = fil.header['fch1']
        df = fil.header['foff']
        fmin = fil.header['fbottom']
        fmax = fil.header['ftop']
        nsamp = fil.header['nsamples']
        tsamp = fil.header['tsamp']
        nf = fil.header['nchans']
        tstart = fil.header['tstart']
        nchans = fil.header['nchans']
        hdrlen = fil.header['hdrlen']
        fil_info = {
            'fch1': fch1,
            'df': df,
            'fmin': fmin,
            'fmax': fmax,
            'nsamp': nsamp,
            'tsamp': tsamp,
            'nf': nf,
            'tstart': tstart,
            'nchans': nchans,
            'fil_name': fil_id
        }
        fil._file.seek(hdrlen)
        return fil, fil_info

    # normalize data to image
    def fil2pic(self, fil, fil_info, downsamp=True):
        nsamp = fil_info['nsamp']
        t_downsamp = int(round(dt/fil_info['tsamp']))
        f_downsamp = int(round(df/fil_info['df']))
        data = fil.readBlock(0, nsamp)
        dst = data2pic(data, f_downsamp, t_downsamp, downsamp)
        return dst

    def __getitem__(self, item):
        fil_path = self.fil_lists[item]
        fil, fil_info = self.load_fil(fil_path)
        pic = self.fil2pic(fil,fil_info)
        pic = self.transform.img_trans(pic,augment=False)
        return pic, fil_info


def fil_collate(batch):
    batch = filter(lambda x: x is not None, batch)
    images, fil_infos = zip(*batch)

    images = torch.stack(images, dim=0)

    return images, fil_infos