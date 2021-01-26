import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos
import data.mytransforms as mytransforms
import torchvision.transforms as transforms
from data.constant import tusimple_row_anchor

def loader_func(path):
    return Image.open(path)

class LaneClsDataset():
    def __init__(self, griding_num=100, load_name=False, use_aux=False):
        super(LaneClsDataset, self).__init__()

        self.target_transform = transforms.Compose([
            mytransforms.FreeScaleMask((288, 800)),
            mytransforms.MaskToTensor(),
        ])

        self.segment_transform = transforms.Compose([
            mytransforms.FreeScaleMask((36, 100)),
            mytransforms.MaskToTensor(),
        ])

        self.img_transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.simu_transform = mytransforms.Compose2([
            mytransforms.RandomRotate(6),
            mytransforms.RandomUDoffsetLABEL(100),
            mytransforms.RandomLROffsetLABEL(200)
        ])

        # self.img_transform = img_transform
        # self.target_transform = target_transform
        # self.segment_transform = segment_transform
        self.simu_transform = None
        # self.path = path
        self.griding_num = griding_num
        self.load_name = load_name
        self.use_aux = use_aux
        #
        # with open(list_path, 'r') as f:
        #     self.list = f.readlines()

        self.row_anchor = tusimple_row_anchor
        self.row_anchor.sort()
        idx = np.where(np.array(tusimple_row_anchor) != np.array(self.row_anchor))
        return

    def getitem(self, label_path, img_path):
        # l = self.list[index]
        # l_info = l.split()
        # img_name, label_name = l_info[0], l_info[1]
        # if img_name[0] == '/':
        #     img_name = img_name[1:]
        #     label_name = label_name[1:]
        #
        #label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)
        #
        # img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        lane_pts = self._get_index(label)

        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_aux:
            return img, cls_label, seg_label
        if self.load_name:
            return img, cls_label, img_name
        return img, cls_label

    # def __len__(self):
    #     return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size

        if h != 288:
            scale_f = lambda x: int((x * 1.0 / 288) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))

        all_idx = np.zeros((4, len(sample_tmp), 2))
        for i, r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, 5):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        all_idx_cp = all_idx.copy()
        for i in range(4):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue

            valid = all_idx_cp[i, :, 1] != -1
            valid_idx = all_idx_cp[i, valid, :]
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                continue
            if len(valid_idx) < 6:
                continue

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp
