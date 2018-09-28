import os
import numpy as np
import collections
import librosa
import torch
import pickle
from torch.utils.data import Dataset

from audio import AudioProcessor

# from TTS.utils.data import (prepare_data, pad_per_step,
# prepare_tensor, prepare_stop_target)


class LJSpeechDataset(Dataset):
    def __init__(self,
                 dataset_ids,
                 data_dir,
                 num_quant,
                 min_wav_len=0,
                 max_wav_len=8000,
                 rand_offset=True):

        self.dataset_ids = dataset_ids
        self.data_dir = data_dir
        self.min_wav_len = min_wav_len
        self.max_wav_len = max_wav_len
        self.rand_offset = rand_offset
        self.receptive_field = 2**num_quant
        print(" > Reading LJSpeech from - {}".format(data_dir))
        print(" | > Number of instances : {}".format(len(self.dataset_ids)))
        print(" | > Max wav length: {}".format(self.max_wav_len))
        print(" | > Min wav length: {}".format(self.min_wav_len))
        print(" | > Receptive field: {}".format(self.receptive_field))

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        file = self.dataset_ids[idx]
        mel_file = f"{self.data_dir}mel/{file}.npy"
        wav_file = f"{self.data_dir}quant/{file}.npy"
        m = np.load(mel_file)
        x = np.load(wav_file)
        sample = {}
        sample['mel'] = m.transpose(1, 0)
        sample['wav'] = x
        sample['mel_file'] = mel_file
        sample['wav_file'] = wav_file
        return sample

    def align_feats(self, wav, feat):
        """Align audio signal and fetures. Audio signal needs to be
        quantized.
        """
        assert len(wav.shape) == 1
        assert len(feat.shape) == 2

        factor = wav.size // feat.shape[0]
        feat = np.repeat(feat, factor, axis=0)
        n_pad = wav.size - feat.shape[0]
        if n_pad != 0:
            assert n_pad > 0
            feat = np.pad(
                feat, [(0, n_pad), (0, 0)], mode="constant", constant_values=0)
        return feat

    def collate_fn(self, batch):
        r"""
            Perform preprocessing and create a final data batch:
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.Mapping):
            keys = list()
            B = len(batch)
            mel_lens = [d['mel'].shape[0] for d in batch]
            pred_lens = [
                np.minimum(d['wav'].shape[0] - 1, self.max_wav_len - 1)
                for d in batch
            ]
            wav_files = [item['wav_file'] for item in batch]
            mel_files = [item['mel_file'] for item in batch]
            max_len = np.max(pred_lens) + self.receptive_field - 1
            if max_len > self.max_wav_len:
                max_len = self.max_wav_len
            wavs = np.zeros([B, max_len + self.receptive_field - 1])
            mels = np.zeros([
                B, max_len + self.receptive_field - 1, batch[0]['mel'].shape[1]
            ])
            for idx, d in enumerate(batch):
                wav = d['wav']
                mel = d['mel']  # D x T
                # align mel specs with wav by cloning frames
                mel = self.align_feats(wav, mel)
                # if wav len is long, sample a starting offset
                if wav.shape[0] > self.max_wav_len:
                    gap = wav.shape[0] - self.max_wav_len
                    if self.rand_offset:
                        offset = np.random.randint(0, gap)
                    else:
                        offset = 0
                    wav = wav[offset:offset + self.max_wav_len]
                    mel = mel[offset:offset + self.max_wav_len]
                pad_w = max_len - wav.shape[0]
                assert wav.shape[0] == mel.shape[0]
                assert wav.shape[0] <= self.max_wav_len
                # pad left with receptive field and right with max_len in the batch
                wav = np.pad(
                    wav, [self.receptive_field - 1, pad_w],
                    mode='constant',
                    constant_values=0.0)
                mel = np.pad(
                    mel, [[self.receptive_field - 1, pad_w], [0, 0]],
                    mode='constant',
                    constant_values=0.0)
                wavs[idx] += wav
                mels[idx] += mel
            # convert things to pytorch
            # B x T x D
            mels = torch.FloatTensor(mels[:, 1:])
            # B x T
            targets = torch.LongTensor(wavs[:, self.receptive_field:])
            inputs = torch.FloatTensor(wavs[:, :-1])
            pred_lens = torch.LongTensor(pred_lens)
            return inputs, mels, pred_lens, targets, wav_files, mel_files

        raise TypeError(("batch must contain tensors, numbers, dicts or lists;\
                         found {}".format(type(batch[0]))))
