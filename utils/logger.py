from pathlib import Path

class Logger:
    # YOLOv5 Loggers class
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.keys = [
            "train/loss",
            "val/mIoU",
        ]  # params

    def on_train_epoch_end(self, loss, epoch):
        # Callback runs at the end of each fit (train+val) epoch
        file = self.save_dir / "results.csv"
        n = len(self.keys) + 1  # number of cols
        s = "" if file.exists() else (("%20s," * n % tuple(["epoch"] + self.keys)).rstrip(",") + "\n")  # add header
        n = len(loss) + 1
        with open(file, "a") as f:
            f.write(s + ("%20.5g," * n % tuple([epoch] + loss)))

    def on_val_end(self, val):
        n = len(val)
        file = self.save_dir / "results.csv"
        with open(file ,"a") as f:
            f.write(("%20.5g," * n % tuple(val)).rstrip(",")+'\n')  # no newline
