"""
Create fake pulses on real observation data
"""

import numpy as np
from sigpyproc.Readers import FilReader
import os
import cv2
import json
from utils import data2pic,create_folder
import random
import collections
import glob
import tqdm
import multiprocessing


# def run(i):
#     pass


# nsamp = 262144.0
# tsamp = 0.000196608
# fch1 = 1499.93896484375
# df = -0.1220703125
# fmin = 1000.0
# fmax = 1500.0
D = 4148.808
dt = 0.006291456


def filplot(data, source_name, tds, out_path):
    fdown_samp = 4
    tdown_samp = int(tds)
    dst = data2pic(data,fdown_samp,tdown_samp)
    cv2.imwrite(os.path.join(out_path, source_name + '.png'), dst)


class PulseCreator:

    def __init__(self, fil):

        self.fil = fil
        self.D = 4148.808
        self.tsamp = fil.header['tsamp']
        self.df = fil.header['foff']
        # self.nsamp = fil.header['nsamples']
        self.nsamp = 4000*32 # 限制训练图片的大小，没必要太大
        self.nbits = fil.header['nbits']
        self.fch1 = fil.header['fch1']
        self.fmax = fil.header['ftop']
        self.fmin = fil.header['fbottom']
        self.nchans = fil.header['nchans']
        self.data = fil.readBlock(0, self.nsamp)
        nf0 = 1 / self.fch1 ** 2
        self.chans_df = np.array([1 / (self.fch1 + self.df * i) ** 2 - nf0 for i in range(self.nchans)],
                                 dtype='float') * self.D / self.tsamp
        self.fake_count = 0
        self.timelen = self.nsamp * self.tsamp

        # self.chans_freq
        self.tds = round(dt / self.tsamp)
        # self.fil._file.close()

    def randommask(self):
        mask_length = random.randint(0, int(self.nchans * 0.6))
        mask_start = random.randint(0, self.nchans)
        mask = np.arange(mask_start, mask_start + mask_length)
        return mask

    def create_fake_pulse(self, time_array, dm, period, width, signal_rate, start_freq, end_freq, fil_path):
        bit_time_array = time_array / self.tsamp
        bit_width = int(round(width / 1000 / self.tsamp))
        bit_delays = self.chans_df * dm
        # bit_chan_time = np.round(bit_delays + bit_time - bit_width / 2).astype('int')
        # bit_chan_time = np.round(bit_delays + bit_time).astype('int')
        start_chan = (start_freq - self.fmax) / self.df
        end_chan = (end_freq - self.fmax) / self.df
        # bit_period = int(round(period / self.tsamp))

        fake_data = self.data.copy()
        for i in range(self.nchans): # 在每个频率通道上依次添加信号
            chan_delay = bit_delays[i]
            for j in range(len(time_array)):
                if not (start_chan[j] < i < end_chan[j]):
                    continue
                bit_time = int(bit_time_array[j] + chan_delay) #每个通道上的信号延迟
                pulse = []
                pulse_pos = np.round(# 简单假定信号服从正态分布
                    np.random.normal(bit_time, bit_width / 2, int(round(signal_rate * bit_width)))).astype('int')
                for pos in pulse_pos:
                    if 0 <= pos < self.nsamp:
                        pulse.append(pos)
                fake_data[i, pulse] = 1
        # fake_data = fake_data.astype('uint8')
        # fake_header = self.fil.header.new({
        #     "source_name": "fake_data",
        #     "filename": '/ssd/wangw/Ai_pulsar_search/detection/fakefils/fake_{}_{}_{}_{}_{}_{}'\
        #         .format(dm,time,width,sigma,period,self.fake_count)
        # })
        update = {
            "source_name": "fake_{:.1f}_{:.2f}_{:.2f}_{:.1f}_{:.2f}_{}".format(dm, time_array[0], width, signal_rate,
                                                                               period, self.fake_count),
            "filename": os.path.join(fil_path, 'fake_{:.1f}_{:.2f}_{:.2f}_{:.1f}_{:.2f}_{}.fil') \
                .format(dm, time_array[0], width, signal_rate, period, self.fake_count)
        }

        # 保存生成的虚拟pulsar的filbank 文件，如果是用来训练模型，可以不保存此文件，直接绘制成png图片， 如果是生成数据用来inference的流程，则需要保存filbank文件
        # fake_outfile = self.fil.header.prepOutfile(update['filename'], update)
        # fake_outfile.cwrite(fake_data.transpose().reshape(-1))

        self.fake_count += 1
        return fake_data, update['source_name']

    def get_random_para(self):
        time = random.uniform(0, self.timelen / 2)
        width = random.uniform(3, 50)
        # width = 10
        signal_rate = random.uniform(0.1, 1.5)
        period = random.uniform(1.5, self.timelen / 5)
        dm = random.uniform(20, 1000)
        time_array = []
        start_freq = []
        end_freq = []
        while time < self.timelen - 1000 * self.tsamp:
            time_array.append(time)
            time += period
            s_f = random.uniform(self.fmin + 100, self.fmax)
            e_f = random.uniform(self.fmin, s_f - 100)
            # s_f = 1450
            # e_f = 1002
            start_freq.append(s_f)
            end_freq.append(e_f)
        time_array = np.array(time_array)
        start_freq = np.array(start_freq)
        end_freq = np.array(end_freq)
        return time_array, dm, period, width, signal_rate, start_freq, end_freq

    def get_fake_ann(self, pulse_para, tds, nsamp):
        time_array, dm, period, width, signal_rate, start_freq, end_freq = pulse_para
        img_nsamp = nsamp / tds
        # times = []
        # while time < self.timelen-1000*self.tsamp:
        #     times.append(time)
        #     time += period
        ann = []
        for i in range(len(time_array)):
            ann.append(
                {
                    "pid": i,
                    "time": time_array[i],
                    "DM": dm,
                    "width": width,
                    "sigma": width * signal_rate,
                    "img_size": img_nsamp,
                    "dt": dt,
                    'start_freq': start_freq[i],
                    'end_freq': end_freq[i]
                }
            )
        return ann


def run(creator, fil_path, png_path, fake_anns):
    pulse_para = creator.get_random_para()
    # print(pulse_para)
    fake_data, name = creator.create_fake_pulse(*pulse_para, fil_path)
    fake_anns.update(
        {
            name: creator.get_fake_ann(pulse_para, creator.tds, creator.nsamp)
        }
    )
    # 将生成的fake pulsar数据绘制成图片，训练时直接读取这些图片
    filplot(fake_data, name, creator.tds, png_path)
    # pass


def main():
    out_name = 'fake_pulsar_train'
    png_path = os.path.join('./fake_images', out_name)
    if not os.path.exists(png_path):
        create_folder(png_path)
    fil_path = os.path.join('./fake_fils', out_name)
    if not os.path.exists(fil_path):
        create_folder(fil_path)
    ann_path = './annotations'
    if not os.path.exists(ann_path):
        create_folder(ann_path)
    nopulse_files = glob.glob('./nopulse_fils/*.fil')
    fake_anns = multiprocessing.Manager().dict()
    count = 0
    pool = multiprocessing.Pool(12) # option: 使用多线程加速
    for f in tqdm.tqdm(nopulse_files):
        fil = FilReader(f)

        creator = PulseCreator(fil)
        for i in range(10): #每个背景数据生成10个不同参数的fake pulsar文件
            pool.apply_async(run, args=(creator, fil_path, png_path, fake_anns))
            # run(creator, fil_path, png_path, fake_anns)
        # count += 1
        # if count == 10:
        #     break
    pool.close()
    pool.join()
    # 保存标注文件
    fake_anns = dict(fake_anns)
    with open('./annotations/{}_ann.json'.format(out_name), 'w') as jf:
        json.dump(fake_anns, jf, indent=4)


if __name__ == '__main__':
    main()
