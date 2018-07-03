import numpy as np
import torch
from torch import nn


class FFTNetQueue(object):
    def __init__(self, batch_size, size, num_channels, cuda=True):
        super(FFTNetQueue, self).__init__()
        self.size = size
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.cuda = cuda
        self.queue1 = []
        self.queue2 = []
        self.reset()

    def reset(self):
        self.queue1 = torch.zeros([self.batch_size, self.num_channels, self.size])
        self.queue2 = torch.zeros([self.batch_size, self.num_channels, self.size])
        if self.cuda:
            self.queue1 = self.queue1.cuda()
            self.queue2 = self.queue2.cuda()

    def enqueue(self, x):
        self.queue2[:, :, :-1] = self.queue2[:, :, 1:]
        self.queue2[:, :, -1] = self.queue1[:, :, 0]
        self.queue1[:, :, :-1] = self.queue1[:, :, 1:]
        self.queue1[:, :, -1] = x.view(x.shape[0], x.shape[1])


class FFTNet(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, layer_id,
                 cond_channels=None, std_f=1.0):
        super(FFTNet, self).__init__()
        self.layer_id = layer_id
        self.receptive_field = 2**layer_id
        self.K = self.receptive_field // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels
        self.cond_channels = cond_channels
        self.conv1_1 = nn.Conv1d(in_channels, hid_channels, 1, stride=1)
        self.conv1_2 = nn.Conv1d(in_channels, hid_channels, 1, stride=1)
        if cond_channels is not None:
            self.convc1 = nn.Conv1d(cond_channels, hid_channels, 1)
            self.convc2 = nn.Conv1d(cond_channels, hid_channels, 1)
        self.conv2 = nn.Conv1d(hid_channels, out_channels, 1)
        self.relu = nn.ReLU()
        self.init_weights(std_f)
        self.buffer = None
        self.cond_buffer = None

    def init_weights(self, std_f):
        std = np.sqrt(std_f / self.in_channels)
        self.conv1_1.weight.data.normal_(mean=0, std=std)
        self.conv1_1.bias.data.zero_()
        self.conv1_2.weight.data.normal_(mean=0, std=std)
        self.conv1_2.bias.data.zero_()
        if self.cond_channels is not None:
            self.convc1.weight.data.normal_(mean=0, std=std)
            self.convc1.bias.data.zero_()
            self.convc2.weight.data.normal_(mean=0, std=std)
            self.convc2.bias.data.zero_()


    def forward(self, x, cx=None):
        """
        Shapes:
            inputs: batch x channels x time
            cx: batch x cond_channels x time
            out: batch x out_chennels x time - receptive_field/2
        """
        T = x.shape[2]
        x1 = x[:, :, :-self.K]
        x2 = x[:, :, self.K:]
        z1 = self.conv1_1(x1)
        z2 = self.conv1_2(x2)
        z = z1 + z2
        # conditional input
        if cx is not None:
            cx1 = cx[:, :, :-self.K]
            cx2 = cx[:, :, self.K:]
            cz1 = self.convc1(cx1)
            cz2 = self.convc2(cx2)
            z = z + cz1 + cz2
        out = self.relu(z)
        out = self.conv2(out)
        out = self.relu(out)
        return out

    def forward_step(self, x, cx=None):
        T = x.shape[2]
        B = x.shape[0]
        if self.buffer is None:
            self.buffer = FFTNetQueue(B, self.K, self.in_channels, x.is_cuda)
        if self.cond_channels is not None and self.cond_buffer is None:
            self.cond_buffer = FFTNetQueue(B, self.K, self.cond_channels, x.is_cuda)
        self.buffer.enqueue(x)
        if self.cond_channels is not None:
            self.cond_buffer.enqueue(cx)
        x_input = self.buffer.queue1[:, :, 0].view([B, -1]).data
        x_input2 = self.buffer.queue2[:, :, 0].view([B, -1]).data
        w1_1 = self._convert_to_fc_weights(self.conv1_1)
        w1_2 = self._convert_to_fc_weights(self.conv1_2)
        z1 = torch.nn.functional.linear(x_input, w1_1, self.conv1_1.bias)
        z2 = torch.nn.functional.linear(x_input2, w1_2, self.conv1_2.bias)
        z = z1 + z2
        if cx is not None:
            cx1 = self.cond_buffer.queue1[:, :, 0].view([B, -1]).data
            cx2 = self.cond_buffer.queue2[:, :,  0].view([B, -1]).data
            wc1_1 = self._convert_to_fc_weights(self.convc1)
            wc1_2 = self._convert_to_fc_weights(self.convc2)
            cz1 = torch.nn.functional.linear(cx1, wc1_1, self.convc1.bias)
            cz2 = torch.nn.functional.linear(cx2, wc1_2, self.convc2.bias)
            z = z + cz1 + cz2
        z = self.relu(z)
        w2 = self._convert_to_fc_weights(self.conv2)
        z = torch.nn.functional.linear(z, w2, self.conv2.bias)
        z = self.relu(z)
        z = z.view(B, -1, 1)
        return z

    def _convert_to_fc_weights(self, conv):
        w = conv.weight
        out_channels, in_channels, filter_size = w.shape
        nw = w.transpose(1, 2).view(out_channels, -1).contiguous()
        return nw


class FFTNetModel(nn.Module):
    def __init__(self, hid_channels=256, out_channels=256, n_layers=11,
                 cond_channels=None):
        super(FFTNetModel, self).__init__()
        self.cond_channels = cond_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.receptive_field = 2 ** n_layers

        self.layers = []
        for idx in range(self.n_layers):
            layer_id = n_layers - idx
            if idx == 0:
                layer = FFTNet(1, hid_channels, hid_channels, layer_id=layer_id, cond_channels=cond_channels)
            else:
                layer = FFTNet(hid_channels, hid_channels, hid_channels, layer_id=layer_id)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        self.fc = nn.Linear(hid_channels, out_channels)

    def forward(self, x, cx=None):
        """
        Shapes:
            x: batch x 1 x time
            cx: batch x dim x time
        """
        # FFTNet modules
        out = x
        for idx, layer in enumerate(self.layers):
            if idx == 0 and cx is not None:
                out = layer(out, cx)
            else:
                out = layer(out)
        out = out.transpose(1, 2)
        out = self.fc(out)
        return out

    def forward_step(self, x, cx=None):
        # FFTNet modules
        out = x
        for idx, layer in enumerate(self.layers):
            if idx == 0 and cx is not None:
                out = layer.forward_step(out, cx)
            else:
                out = layer.forward_step(out)
        out = out.transpose(1, 2)
        out = self.fc(out)
        return out


def sequence_mask(sequence_length):
    max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, input, target, lengths=None):
        if lengths is None:
            raise RuntimeError(" > Provide lengths for the loss function")
        mask = sequence_mask(lengths)
        if target.is_cuda:
            mask = mask.cuda()
        input = input.view([input.shape[0] * input.shape[1], -1])
        target = target.view([target.shape[0] * target.shape[1]])
        mask_ = mask.view([mask.shape[0] * mask.shape[1]])
        losses = self.criterion(input, target)
        _, pred = torch.max(input, 1)
        f = (pred != target).type(torch.FloatTensor)
        t = (pred == target).type(torch.FloatTensor)
        if input.is_cuda:
            f = f.cuda()
            t = t.cuda()
        f = (f.squeeze() * mask_).sum()
        t = (t.squeeze() * mask_).sum()
        return ((losses * mask_).sum()) / mask_.sum(), f.item(), t.item()

# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
class EMA(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta

    def assign_ema_model(self, model, new_model, cuda):
       new_model.load_state_dict(model.state_dict())
       for name, param in new_model.named_parameters():
           if name in self.shadow:
               param.data = self.shadow[name].clone()
       if cuda:
           new_model.cuda()
       return new_model
