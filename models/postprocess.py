import torch
import numpy as np
import torch.nn.functional as f


def iou(bbox_1, bbox_2, area_1, area_2):
    l = max(bbox_1[0], bbox_2[0])
    r = min(bbox_1[2], bbox_2[2])
    t = max(bbox_1[1], bbox_2[1])
    b = min(bbox_1[3], bbox_2[3])
    union_area = max(r - l, 0) * max(b - t, 0)
    return union_area / (area_1 + area_2 - union_area)


def nms_cpu(dets, scores, thresh, score_thresh):
    dets = dets.cpu().numpy()
    scores = scores.cpu().numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    order = order[scores[order] > score_thresh]

    keep = []
    inds = 0
    while order.size > 0:
        i = order.item(0)
        if areas[i] > 100:
            keep.append(i)
        else:
            order = order[inds + 1]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1+1)
        h = np.maximum(0.0, yy2 - yy1+1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return torch.LongTensor(keep)
    # return dets[keep], scores[keep]


class GetPulsar(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.score_thr = cfg['score']
        self.iou_thr = cfg['iou']
        self.max_num_per_img = cfg['max_num_per_img']

    def iou(self,bbox_1,bbox_2, area_1, area_2):
        l = max(bbox_1[0], bbox_2[0])
        r = min(bbox_1[2], bbox_2[2])
        t = max(bbox_1[1], bbox_2[1])
        b = min(bbox_1[3], bbox_2[3])
        union_area = max(r-l,0) * max(b-t, 0)
        return union_area/(area_1 + area_2 - union_area)


    def nms(self, bbox, bbox_score):
        '''

        :param bbox: N * 4
        :param bbox_score: N
        :return:
        '''
        keep_flag = np.ones(bbox_score.shape, dtype=np.bool)
        sort = torch.sort(bbox_score, descending=True)[1]
        bbox = bbox[sort]
        bbox_score = bbox_score[sort]
        keep_flag = keep_flag * bbox_score.cpu().numpy() > self.score_thr
        bbox_area = (bbox[:,2] - bbox[:,0]) * (bbox[:,3] - bbox[:,1])
        for i in range(len(bbox_score)):
            if not keep_flag[i]:
                continue
            for j in range(i+1, len(bbox_score)):
                if not keep_flag[j]:
                    continue
                if self.iou(bbox[i], bbox[j], bbox_area[i], bbox_area[j]) > self.iou_thr:
                    keep_flag[j] = False
        indexs = np.arange(0,len(bbox_score))
        indexs = indexs[keep_flag]
        if len(indexs) > self.max_num_per_img:
            indexs = indexs[:self.max_num_per_img]
        return bbox[indexs], bbox_score[indexs]

    def __call__(self, centerpoints, bbox_cls, bbox_reg, img_info):
        '''

        :param centerpoints: list l [H * W]
        :param bbox_cls: list l [batchsize * 2 * H * W]
        :param bbox_reg: list l [batchsize * 4 * H * W]
        :param img_info: list batchsize[list N [dict()]]
        :return: pulsar locations: list batchsize[ N * 2]
        '''
        # bbox_cls = bbox_cls.unsqueeze(dim=2).permute(0,2,1)
        # bbox_reg = bbox_reg.unsqueeze(dim=2).permute(0,2,1)

        # from bbox_reg to real coordinate
        # pulsar_score
        bbox = []
        bbox_score = []
        N  = bbox_cls[0].shape[0]
        for i in range(len(centerpoints)):
            # delta_l = bbox_reg[i][:, 0, 0, :]
            # delta_t = bbox_reg[i][:, 1, 0, :]
            # delta_r = bbox_reg[i][:, 2, 0, :]
            # delta_b = bbox_reg[i][:, 3, 0, :]
            delta_box = bbox_reg[i].permute(0, 2, 3, 1).reshape(N,-1, 4)
            bbox.append(torch.stack([centerpoints[i][None,:,0] - delta_box[:,:,0], centerpoints[i][None,:,1] -delta_box[:,:,1],
                                     centerpoints[i][None,:,0] + delta_box[:,:,2], centerpoints[i][None,:,1] + delta_box[:,:,3]], dim=2))
            score = f.softmax(bbox_cls[i][:,:,:,:].permute(0,2,3,1).reshape(N,-1, 2),dim=-1)[:,:,1]
            bbox_score.append(score)
        bbox = torch.cat(bbox, dim=1)  # batchsize * N * 4
        bbox_score = torch.cat(bbox_score, dim=1)  # batchsize * N
        pulsars = []
        num_pulsars_per_img = []
        for i in range(bbox.shape[0]):
            keep = nms_cpu(bbox[i,:,:], bbox_score[i,:],self.iou_thr, self.score_thr)
            keep = keep[:self.max_num_per_img]
            keep_bbox = bbox[i,keep]
            keep_bbox_score = bbox_score[i,keep]
            # keep_bbox, keep_bbox_score = self.nms(bbox[i,:,:],bbox_score[i,:])
            pulsars.append(torch.cat([keep_bbox, keep_bbox_score.unsqueeze(-1)],dim=1))
            num_pulsars_per_img.append(keep_bbox.shape[0])
        return torch.cat(pulsars,dim=0), torch.tensor(num_pulsars_per_img,device=keep_bbox.device)