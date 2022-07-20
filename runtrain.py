import argparse
from pathlib import Path
import sys
import json
from converter import coco_stuff2png, gen_data_list
import logging
import src_v2.train as train
from pycocotools.coco import COCO
from utils.helpers import suppress_stdout
from utils.logger import Logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="", help="where to store temp files")
    parser.add_argument("--log-enable", action="store_true", help="output log to file")
    parser.add_argument("--label-type", default="coco", choices=["voc", "coco", "png"], help="dataset labeling method")
    parser.add_argument("--train-list", default="", help="path of training image list")
    parser.add_argument("--val-list", default="", help="path of validation image list")
    parser.add_argument("--train-annotation", default="", help="path of training set annotation")
    parser.add_argument("--val-annotation", default="", help="path of validation set annotation")
    parser.add_argument("--config", default="", help="path of training config(Hyperparameters, JSON file)")
    parser.add_argument("--device", default="", help="cpu or cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--ignore-label", type=int, default=0, help="Ignore this label in the training loss")
    parser.add_argument(
        "--label-value-shift",
        type=int,
        default=0,
        help="Add or subtract a number to the label value(both train and val)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("--development", action="store_true", help="development mode")

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, unknown_args):
    LOGGER = logging.getLogger(__name__)
    if args.verbose:
        LOGGER.info(f"args:{args}")
        LOGGER.info(f"unknown_args:{unknown_args}")
    assert args.work_dir, "work-dir is required"
    root = Path(args.work_dir).absolute()

    if args.log_enable:
        out_file = open(root / "out.txt", "w")
        sys.stdout = out_file
        sys.stderr = out_file

    if args.label_type == "coco":
        assert args.train_list, "train-list is required"
        assert args.val_list, "val-list is required"
        assert args.train_annotation, "train-annotation is required"
        assert args.val_annotation, "val-annatation is required"

        train_label_path = "labels/train"
        val_label_path = "labels/val"
        train_list_path = "coco.train"
        val_list_path = "coco.val"
        
        with suppress_stdout(not args.verbose):
            coco = COCO(args.val_annotation)
            if not args.development:
                coco_stuff2png(
                    json_path=args.train_annotation, target_dir=root / train_label_path, label_shift=args.label_value_shift
                )
                coco_stuff2png(
                    json_path=args.val_annotation, target_dir=root / val_label_path, label_shift=args.label_value_shift
                )
        gen_data_list(
            img_list_path=args.train_list, label_dir=root / train_label_path, target_path=root / train_list_path
        )
        gen_data_list(img_list_path=args.val_list, label_dir=root / val_label_path, target_path=root / val_list_path)

        # 导入超参数
        hyp = dict()
        if args.config:
            with open(args.config) as f:
                hyp = json.load(f)

        train.run(
            unknown_args,
            enc_backbone="50",
            train_dir=["/"],
            val_dir="/",
            train_list_path=[root / train_list_path],
            val_list_path=root / val_list_path,
            num_stages=1,
            num_classes=len(coco.getCatIds()) + 1,  # 0
            ignore_label=args.ignore_label,
            verbose=args.verbose,
            device=args.device,
            ckpt_dir=root / "checkpoints",
            ckpt_path=root / "checkpoints/checkpoint.pth.tar",
            logger=Logger(root),
            **hyp,
        )


if __name__ == "__main__":
    args, unknown_args = get_args()
    if args.development:
        fstr = "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    else:
        fstr = "[%(asctime)s][%(levelname)s] %(message)s"
    logging.basicConfig(
        format=fstr,
        level=logging.INFO,
    )
    main(args, unknown_args)