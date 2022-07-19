import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from tqdm import tqdm
import os
import logging
LOGGER = logging.getLogger(__name__)

coco_stuff_colormap = np.load('coco_stuff_colormap.npy')


def coco_stuff2png(json_path, target_dir, label_shift=0):
    LOGGER.info('convert coco stuff json to png')
    color_map = True if label_shift == 0 else False

    def mask_generator(width, height, anns):
        res = np.zeros((height, width), dtype=int)
        for ann in anns:
            rle = [ann['segmentation']]
            cat = ann['category_id'] + label_shift
            m = coco_mask.decode(rle)
            res += m.reshape(m.shape[:2]) * cat
        return res

    coco = COCO(json_path)
    imgids = coco.getImgIds()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for imgid in tqdm(imgids):
        img_dict = coco.loadImgs(imgid)[0]
        file_name = img_dict['file_name'].replace('.jpg', '.png')
        w, h = img_dict['width'], img_dict['height']
        annsids = coco.getAnnIds(imgid)
        anns = coco.loadAnns(annsids)
        mask = mask_generator(w, h, anns)
        img = Image.fromarray(np.uint8(mask))
        if color_map:
            img.putpalette(coco_stuff_colormap.tolist())
        img.save(os.path.join(target_dir, file_name))
        

def gen_data_list(img_list_path, label_dir, target_path):
    with open(img_list_path) as f:
        lines = f.readlines()
    with open(target_path,"w") as f:
        for line in lines:
            img_path = line.strip('\n').strip('\r')
            label_name = img_path.rsplit(os.sep, 1)[1].replace('.jpg','.png')
            label_path = os.path.abspath(os.path.join(label_dir, label_name))
            if not os.path.exists(label_path):
                LOGGER.warning(f'{label_path}not exist')
            f.write(img_path+"\t"+label_path+"\n")


if __name__ == '__main__':
    # coco_stuff2png('/home/ucprer/datasets/coco/annotations/stuff_val2017.json',
    #                './labels/val/', label_shift=-91)
    gen_data_list('/home/ucprer/datasets/coco/val.txt', './work_root/labels/val/', './work_root/coco.val')
    
    # coco_stuff2png('/home/ucprer/datasets/coco/annotations/stuff_train2017.json',
    #                './labels/train/', label_shift=-91)
    gen_data_list('/home/ucprer/datasets/coco/train.txt', './work_root/labels/train/', './work_root/coco.train')
