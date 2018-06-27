import os
import torch
import unittest
import time
import copy
from torch import optim
from torch.utils.data import DataLoader
from generic_utils import load_config
from model import FFTNet, FFTNetModel
from dataset import LJSpeechDataset

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestLayers(unittest.TestCase):
    def test_FFTNet(self):
        print(" ---- Test FFTNet ----")
        # test only input
        net = FFTNet(in_channels=1, out_channels=25, hid_channels=20, layer_id=1)
        inp = torch.rand(2, 1, 8)
        out = net(inp)
        assert out.shape[0] == 2
        assert out.shape[1] == 25
        assert out.shape[2] == 7
        # test cond input
        net = FFTNet(in_channels=1, out_channels=25, hid_channels=20, cond_channels=5, layer_id=1)
        inp = torch.rand(2, 1, 8)
        c_inp = torch.rand(2, 5, 8)
        out = net(inp, c_inp)
        assert out.shape[0] == 2
        assert out.shape[1] == 25
        assert out.shape[2] == 7

        net = FFTNet(in_channels=1, out_channels=25, hid_channels=20, cond_channels=5, layer_id=3)
        inp = torch.rand(2, 1, 8)
        c_inp = torch.rand(2, 5, 8)
        out = net(inp, c_inp)
        assert out.shape[0] == 2
        assert out.shape[1] == 25
        assert out.shape[2] == 4

    def test_FFTNetModel(self):
        print(" ---- Test FFTNetModel ----")
        # test only inputs
        net = FFTNetModel(hid_channels=256, out_channels=256, n_layers=11, cond_channels=None)
        inp = torch.rand(2, 1, 2048)
        out = net(inp)
        assert out.shape[0] == 2
        assert out.shape[1] == 1
        assert out.shape[2] == 256
        # test cond input
        net = FFTNetModel(hid_channels=256, out_channels=256, n_layers=11, cond_channels=80)
        inp = torch.rand(2, 1, 2048)
        c_inp = torch.rand(2, 80, 2048)
        out = net(inp, c_inp)
        assert out.shape[0] == 2
        assert out.shape[1] == 1
        assert out.shape[2] == 256
        # test cond input
        net = FFTNetModel(hid_channels=256, out_channels=256, n_layers=10, cond_channels=80)
        inp = torch.rand(2, 1, 2048)
        c_inp = torch.rand(2, 80, 2048)
        out = net(inp, c_inp)
        assert out.shape[0] == 2
        assert out.shape[1] == 1025
        assert out.shape[2] == 256

    def test_FFTNetModelStep(self):
        print(" ---- Test FFTNetModel step forward ----")
        net = FFTNetModel(hid_channels=256, out_channels=256, n_layers=11, cond_channels=80)
        time_start = time.time()
        for i in range(1024):
            x = torch.rand(1, 1, 1)
            cx = torch.rand(1, 80, 1)
            out = net.forward_step(x, cx)
        time_avg = (time.time() - time_start) / 1024
        print("> Avg time per step inference on CPU: {}".format(time_avg))
        assert abs(net.layers[0].buffer.queue1.sum().item()) > 0
        assert abs(net.layers[0].buffer.queue2.sum().item()) == 0

        # on GPU
        net = FFTNetModel(hid_channels=256, out_channels=256, n_layers=11, cond_channels=80)
        net.cuda()
        time_start = time.time()
        for i in range(1024):
            x = torch.rand(1, 1, 1)
            cx = torch.rand(1, 80, 1)
            out = net.forward_step(x.cuda(), cx.cuda())
        time_avg = (time.time() - time_start) / 1024
        print("> Avg time per step inference on GPU: {}".format(time_avg))
        assert abs(net.layers[0].buffer.queue1.sum().item()) > 0
        assert abs(net.layers[0].buffer.queue2.sum().item()) == 0

        # check the second queue
        net = FFTNetModel(hid_channels=256, out_channels=256, n_layers=11, cond_channels=80)
        time_start = time.time()
        for i in range(1025):
            x = torch.rand(1, 1, 1)
            cx = torch.rand(1, 80, 1)
            out = net.forward_step(x, cx)
        assert abs(net.layers[0].buffer.queue1.sum().item()) > 0
        assert abs(net.layers[0].buffer.queue2.sum().item()) > 0
        assert abs(net.layers[0].buffer.queue2[:, :, :-1].sum().item()) == 0

    def test_train_step(self):
        print(" ---- Test the network backpropagation ----")
        model = FFTNetModel(hid_channels=256, out_channels=256, n_layers=11, cond_channels=80)
        inp = torch.rand(2, 1, 2048)
        c_inp = torch.rand(2, 80, 2048)

        criterion = torch.nn.L1Loss().to(device)

        model.train()
        model_ref = copy.deepcopy(model)
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            assert (param - param_ref).sum() == 0, param
            count += 1
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        for i in range(5):
            out = model(inp, c_inp)
            optimizer.zero_grad()
            loss = criterion(out, torch.zeros(out.shape))
            loss.backward()
            optimizer.step()
        # check parameter changes
        count = 0
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            # ignore pre-higway layer since it works conditional
            assert (param != param_ref).any(), "param {} with shape {} not updated!! \n{}\n{}".format(count, param.shape, param, param_ref)
            count += 1


class TestLoaders(unittest.TestCase):
    def test_ljspeech_loader(self):
        print(" ---- Run data loader for 100 iterations ----")
        MAX = 10
        RF = 11
        C = load_config('test_conf.json')
        dataset = LJSpeechDataset(os.path.join(C.data_path, "mels", "meta_fftnet.csv"),
                                  os.path.join(C.data_path, "mels"),
                                  C.sample_rate,
                                  C.num_mels, C.num_freq,
                                  C.min_level_db, C.frame_shift_ms,
                                  C.frame_length_ms, C.preemphasis, C.ref_level_db,
                                  RF, C.min_wav_len, C.max_wav_len)
        dataloader = DataLoader(dataset, batch_size=2,
                                shuffle=False, collate_fn=dataset.collate_fn,
                                drop_last=True, num_workers=2)

        count = 0
        last_T = 0
        for data in dataloader:
            wavs = data[0]
            mels = data[1]
            print(" > iter: ", count)
            assert wavs.shape[1] >= last_T
            last_T = wavs.shape[1]
            assert wavs.shape[1] == mels.shape[1]
            assert wavs.shape[0] == mels.shape[0]
            assert wavs.shape[1] > RF
            assert wavs.max() > 0 and wavs.mean() > 0
            count += 1
            if count == MAX:
                break
