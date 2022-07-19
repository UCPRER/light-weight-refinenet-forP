import numpy as np
import torch
import os
import logging
import sys
from contextlib import contextmanager
LOGGER = logging.getLogger(__name__)

IMG_SCALE = 1.0 / 255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))


@contextmanager
def suppress_stdout(enable=True):
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        if enable:
            sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os
    import sys
    from six.moves import urllib

    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "models"))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = "{}.pth.tar".format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD


def select_device(device='', batch_size=0, verbose=False):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    if verbose:
        LOGGER.info(f"device:{device}")
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        arg = 'cuda:0'
    else:  # revert to CPU
        arg = 'cpu'
    if verbose:
        LOGGER.info(f"device:{arg}")
        LOGGER.info(f'device number:{torch.cuda.device_count()}')
    return torch.device(arg)
