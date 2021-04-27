import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import argparse
from models import PulseDectector
from datasets.fils_loader import FilSet,fil_collate
import os
import cv2
from utils import create_logger, config_from_py
import json


def main():
    parser = argparse.ArgumentParser(description="Detect pulses from fils")
    parser.add_argument('fil_path', help='fils path to be detected')
    parser.add_argument('--config', default='configs/default.py', help='config file')
    parser.add_argument('--model_param',
                        default='./checkpoints/resnet-50-fcos.pth',
                        help='model parameters to be used')
    parser.add_argument('--tdownsamp', default=32, help='time downsample rate',type=int)
    parser.add_argument('--fdownsamp', default=16, help='frequency downsample rate',type=int)
    parser.add_argument('--num_gpus',type=int, default=torch.cuda.device_count())
    parser.add_argument('--output', help='output path', default='./output')
    parser.add_argument('--batch_size',type=int, help='how many fils to be detected at the same time', default=4)
    parser.add_argument('--num_works',type=int, help='number of process to load data', default=4)
    parser.add_argument('--cut_image', default=True, help='cut each pulse to single image')
    parser.add_argument('--prefix', default='resnet-50-fcos',help='detection name prefix')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    logger = create_logger('Detect-' + args.prefix, args.output)

    fil_set_name = os.path.split(args.fil_path)[-1]
    out_path = os.path.join(args.output, fil_set_name+'-'+args.prefix)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    logger.info('Results are saved in {}'.format(out_path))
    logger.info('Cnfigs are from {}'.format(args.config))
    logger.info('Runing on {} gpus'.format(args.num_gpus))

    cfg = config_from_py(args.config)
    logger.info('-' * 20)
    logger.info('loading model...')
    model = PulseDectector(cfg)
    logger.info('model parameters comes from ' + args.model_param)
    model.load_state_dict(torch.load(args.model_param)['model'], strict=True)
    model.eval()
    logger.info('model is ready!')
    logger.info('-' * 20)
    logger.info('preparing fils')
    fils = FilSet(args.fil_path, args.fdownsamp,args.tdownsamp)
    logger.info('There are {} fils to be detected'.format(len(fils)))
    fils_loader = DataLoader(
        dataset=fils,
        collate_fn=fil_collate,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_works
    )
    logger.info('-' * 20)

    if args.num_gpus != 0:
        device = torch.device('cuda')
        model = model.cuda()
        if args.num_gpus > 1:
            model = DataParallel(model)
    else:
        device = torch.device('cpu')

    model.eval()
    logger.info('Start detecting...')
    results = dict()
    for i, (images, fil_infos) in enumerate(fils_loader):
        images = images.to(device)
        try:
            with torch.no_grad():
                predict_pulses, num_pulses_per_image = model(images)
            for j in range(images.shape[0]):
                logger.info('{} pulses are detected from {}'.format(num_pulses_per_image[j], fil_infos[j]['fil_name']))

            results.update(result_save(fils, images,fil_infos, predict_pulses,num_pulses_per_image,args.tdownsamp,args.fdownsamp,output_path=out_path))
        except Exception as e:
            logger.info('Error: {}'.format(e))
            continue
    with open(os.path.join(args.output, 'detect_result-{}.json'.format(args.prefix)), 'w') as f:
        json.dump(results, f, indent=2)


def result_save(fil_set, images,fil_infos, predict_pulses,num_pulses_per_image,tdownsamp,fdownsamp, output_path, cut_images=True):
    images = images.cpu().numpy()
    n = 0
    result_dict = {}
    for i in range(images.shape[0]):
        pulses_position = (predict_pulses[n:n+num_pulses_per_image[i], :4].cpu().numpy()).astype('int')
        pulses_position = pulses_position.clip(0,8192)
        n+=num_pulses_per_image[i]
        image = images[i]
        fil_info = fil_infos[i]
        pulse_list = []
        for j in range(pulses_position.shape[0]):
            if cut_images:
                pulse_image = image[:,pulses_position[j,1]:pulses_position[j,3],pulses_position[j,0]:pulses_position[j,2]]
                pulse_image = cv2.resize(pulse_image.transpose((1,2,0)), (128,128),interpolation=cv2.INTER_LINEAR)
                pulse_image = fil_set.transform.img_detrans(pulse_image)
                cv2.imwrite(os.path.join(output_path, fil_info['fil_name']+'_'+str(j))+'.png', pulse_image)
            # 将检测到的位置还原为观测数据中的时间频率
            pulse_list.append({
                'pid':j,
                'time':pulses_position[j,0] * tdownsamp * fil_info['tsamp'],
                'time_end':pulses_position[j,2] * tdownsamp * fil_info['tsamp'],
                'topfreq':fil_info['fmax'] + pulses_position[j,1] * fdownsamp * fil_info['df'],
                'lowfreq':fil_info['fmax'] + pulses_position[j,3] * fdownsamp * fil_info['df'],
            })
        result_dict[fil_info['fil_name']] = pulse_list
    return result_dict


if __name__ == '__main__':
    main()
