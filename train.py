import torch
from generic_utils import load_config

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--restore_path', type=str,
                    help='Folder path to checkpoints', default=0)
parser.add_argument('--config_path', type=str,
                    help='path to config file for training',)
parser.add_argument('--debug', type=bool, default=False,
                    help='do not ask for git has before run.')
args = parser.parse_args()
C = load_config(args.config)

def train():
    pass


def evaluate():
    pass

def main():
    if C.max_wav_len < model.receptive_field:
        raise RuntimeError(" > Max wav length {} cannot be smaller then\
                           the model receptive field {}.".format(c.max_wav_len,
                                                                 model.receptive_field))


if __name__ == "__main__":
    print(" > Starting a new training")