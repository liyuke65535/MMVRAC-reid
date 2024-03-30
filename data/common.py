import torch
from torch.utils.data import Dataset
# import cv2
import numpy as np
import PIL.Image as Image
from data.transforms.mask_generator import PartAwareMaskGenerator, RandomMaskingGenerator
# from data.transforms.transforms import rand_rotate

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, cfg, img_items, transform=None, relabel=True, domain_names=None, transform_384=None):
        self.cfg = cfg
        self.img_items = img_items
        self.transform = transform
        self.transform_384 = transform_384
        self.relabel = relabel


        self.pid_dict = {}
        pids = list()
        for i, item in enumerate(img_items):
            if item[1] in pids: continue
            pids.append(item[1])
        self.pids = pids
        if self.relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

        if domain_names is None: return

        self.pid_dict_domain_wise = dict()
        pids_domain_wise = list(list() for _ in range(len(domain_names)))
        for p in pids:
            idx = [str in p for str in domain_names].index(True)
            if p in pids_domain_wise[idx]: continue
            pids_domain_wise[idx].append(p)
        if self.relabel:
            for p in pids:
                for dom in pids_domain_wise:
                    if p in dom:
                        self.pid_dict_domain_wise[p] = dom.index(p)
        # print("A")

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        if len(self.img_items[index]) > 3:
            img_path, pid, camid, others = self.img_items[index]
            # if self.mask and self.relabel: ##### add by me
            #     others['mask'] = self.mask_generator() ##### add by me
        else:
            img_path, pid, camid = self.img_items[index]
            others = ''
        ori_img = read_image(img_path)
        h,w = ori_img.size

        img = self.transform(ori_img)
        img_384 = self.transform_384(ori_img)

        pid_domain_wise = None
        if self.relabel:
            pid_domain_wise = self.pid_dict_domain_wise[pid]
            pid = self.pid_dict[pid]

        return {
            "images": img,
            "images_384": img_384,
            "targets": pid,
            "camid": camid,
            "img_path": img_path,
            "ori_label": pid_domain_wise,
            "others": others,
            "resolutions": h*w,
        }

    @property
    def num_classes(self):
        return len(self.pids)
