import os
import sys
import time
import glob
import argparse
import soundfile as sf
import librosa
import numpy as np
import tqdm
import pyworld as pw
from audio import AudioProcessor
from generic_utils import load_config

from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='Folder path to checkpoints.')
parser.add_argument('--out_path', type=str,
                    help='path to config file for training.')
parser.add_argument('--config', type=str,
                    help='conf.json file for run settings.')
args = parser.parse_args()

args.out_path = os.path.join(*args.out_path.split('/'))
args.data_path = os.path.join(*args.data_path.split('/'))

DATA_PATH = args.data_path
OUT_PATH = args.out_path
CONFIG = load_config(args.config)
ap = AudioProcessor(CONFIG.sample_rate, CONFIG.num_mels, CONFIG.num_freq, CONFIG.min_level_db,
                    CONFIG.frame_shift_ms, CONFIG.frame_length_ms, CONFIG.preemphasis,
                    CONFIG.ref_level_db)         
        
def extract_mel(file_path):
    x, fs = sf.read(file_path)
    mel = ap.melspectrogram(x.astype('float32'))
    file_name = os.path.basename(file_path).replace(".wav","")
    mel_file = file_name + ".mel"
    np.save(os.path.join(OUT_PATH, mel_file), mel, allow_pickle=False)
    mel_len = mel.shape[1]
    wav_len = x.shape[0]
    return file_path, mel_file, str(wav_len), str(mel_len)

glob_path = os.path.join(DATA_PATH, "**/*.wav")
file_names = glob.glob(glob_path, recursive=True)

if __name__ == "__main__":
    print(" > Number of files: %i"%(len(file_names)))
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
        print(" > A new folder created at {}".format(OUT_PATH))

    with Pool(20) as p:
        r = list(tqdm.tqdm(p.imap(extract_mel, file_names), total=len(file_names)))
    
    file_path = os.path.join(OUT_PATH, "meta_fftnet.csv")
    file = open(file_path, "w")
    for line in r:
        line = ", ".join(line) 
        file.write(line+'\n')
    file.close()
