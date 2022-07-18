import argparse
from pathlib import Path
import sys
import json
from converter import coco_stuff2png, gen_data_list
import logging
import os
import src_v2.train as train


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
        "--label-value-shift", type=int, default=0, help="Increase or decrease the value of the each label"
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    assert args.work_dir, "work-dir is required"
    root = Path(args.work_dir)

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
        if not os.path.exists(root / train_label_path):  # TODO: MD5 check
            coco_stuff2png(
                json_path=args.train_annotation, save_path=root / train_label_path, label_shift=args.label_value_shift
            )
        if not os.path.exists(root / val_label_path):
            coco_stuff2png(
                json_path=args.val_annotation, save_path=root / val_label_path, label_shift=args.label_value_shift
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
            enc_backbone="50",
            train_dir=["/"],
            val_dir="/",
            train_list_path=[os.path.abspath(root/train_list_path)],
            val_list_path=os.path.abspath(root/val_list_path),
            num_stages=1,
            num_classes=93,
            ignore_label=args.ignore_label,
            **hyp
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
        level=logging.INFO,
    )
    main()
