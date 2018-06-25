import numpy as np
import torch
from torch import nn


class FFTNetQueue(object):
    def __init__(self, size, num_channels):
        super(FFTNetQueue, self).__init__()
        self.size = size
        self.num_channels = num_channels
        self.queue1 = []
        self.queue2 = []
        self.reset()
        
    def reset(self):
        self.queue1 = torch.zeros([self.size, self.num_channels])
        self.queue2 = torch.zeros([self.size, self.num_channels])
        
    def enqueue(self, item1, item2):
        self.queue1[:-1] = self.queue1[1:]
        self.queue1[-1] = item1
        self.queue2[:-1] = self.queue2[1:]
        self.queue2[-1] = item2
        
    
class FFTNet(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, layer_id, cond_channels=None, std_f=1.0):
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
            self.cond_queue = FFTNetQueue(self.K, hid_channels)
        self.conv2 = nn.Conv1d(hid_channels, out_channels, 1)
        self.relu = nn.ReLU()
        self.init_weights(std_f)
        self.input_queue = FFTNetQueue(self.K, hid_channels)
        
    
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
    
    def forward_step(self):
    
    
class FFTNetModel(nn.Module):
    def __init__(self, hid_channels=256, out_channels=256, n_layers=11, cond_channels=None):
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