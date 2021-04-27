"""
Create fake pulses on real observation data
"""

import numpy as np
from sigpyproc.Readers import FilReader
import os
import cv2
import json
from utils import data2pic
import random
import collections
import glob
import tqdm
import multiprocessing


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

        # self.fil = fil
        self.D = 4148.808
        self.tsamp = fil.header['tsamp']
        self.df = fil.header['foff']
        self.nsamp = fil.header['nsamples']
        # self.nsamp = 4000*32
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
        start_chan = (start_freq - self.fmax) / self.df
        end_chan = (end_freq - self.fmax) / self.df
        fake_data = self.data.copy()
        for i in range(self.nchans): # 在每个频率通道上依次添加信号
            chan_delay = bit_delays[i]
            for j in range(len(time_array)):
                # 只在一部分通道上显示信号，模拟真实数据中的不完整效果 TODO: 模拟真实数据中信号断断续续的效果
                if not (start_chan[j] < i < end_chan[j]):
                    continue
                bit_time = int(bit_time_array[j] + chan_delay) #每个通道上的信号延迟
                pulse = []
                pulse_pos = np.round(# 假定信号服从正态分布
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
        # fil文件比较大，只生成训练图片可以不保存
        # fake_outfile = self.fil.header.prepOutfile(update['filename'], update)
        # fake_outfile.cwrite(fake_data.transpose().reshape(-1))
        self.fake_count += 1
        return fake_data, update['source_name']

    def get_random_para(self):
        time = random.uniform(0, self.timelen / 2)
        width = random.uniform(3, 50)
        # width = 10
        signal_rate = random.uniform(0.1, 1.5)
        # width signal 两个参数决定了信号强度和宽度
        period = random.uniform(1.5, self.timelen / 5)
        dm = random.uniform(20, 1500)
        time_array = []
        start_freq = []
        end_freq = []
        while time < self.timelen - 1000 * self.tsamp: # 防止信号落在太靠边缘的位置
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
    filplot(fake_data, name, creator.tds, png_path)
    # pass


def main():
    # 在没有信号的数据上添加虚假的随机脉冲信号，用来训练模型
    # rootpath = '/ssd/wangw/Ai_pulsar_search/detection/nopulse_fils/'
    out_name = 'fake_pulsar'
    png_path = os.path.join('/ssd/wangw/Ai_pulsar_search/detection/fake_image_sets', out_name + '_png')
    if not os.path.exists(png_path):
        os.mkdir(png_path)
    fil_path = os.path.join('/ssd/wangw/Ai_pulsar_search/detection/fake_fils', out_name)
    if not os.path.exists(fil_path):
        os.mkdir(fil_path)
    nopulse_files = glob.glob('/ssd/wangw/Ai_pulsar_search/detection/nopulse_fils/*.fil')
    # fil = FilReader('/ssd/wangw/Ai_pulsar_search/detection/nopulse_fils/Dec+3557_arcdrift+23.4-M01_0037_1k.fil')
    # fake_anns = collections.OrderedDict()
    fake_anns = multiprocessing.Manager().dict()
    count = 0
    pool = multiprocessing.Pool(24)
    for f in tqdm.tqdm(nopulse_files):
        fil = FilReader(f)

        creator = PulseCreator(fil)
        for i in range(20):
            pool.apply_async(run, args=(creator, fil_path, png_path, fake_anns))
            # run(creator, fil_path, png_path, fake_anns)
        count += 1
        # if count == 10:
        #     break
    pool.close()
    pool.join()
    fake_anns = dict(fake_anns)
    with open('annotations/{}_ann.json'.format(out_name), 'w') as jf:
        json.dump(fake_anns, jf, indent=4)


if __name__ == '__main__':
    main()
