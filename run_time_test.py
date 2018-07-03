import torch
import time
from tqdm import tqdm
from model import FFTNet, FFTNetModel
from generic_utils import count_parameters


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if use_cuda:
    torch.backends.cudnn.benchmark = False

print(" ---- Test FFTNetModel step forward ----")
net = FFTNetModel(hid_channels=256, out_channels=256, n_layers=11, cond_channels=80)
net.eval()
print(" > Number of model params: ", count_parameters(net))
x = torch.rand(1, 1, 1)
cx = torch.rand(1, 80, 1)
time_start = time.time()
with torch.no_grad():
    for i in tqdm(range(20000)):
        out = net.forward_step(x, cx)
    time_avg = (time.time() - time_start) / 20000
    print("> Avg time per step inference on CPU: {}".format(time_avg))

# on GPU
net = FFTNetModel(hid_channels=256, out_channels=256, n_layers=11, cond_channels=80)
net.cuda()
net.eval()
x = torch.rand(1, 1, 1).cuda()
cx = torch.rand(1, 80, 1).cuda()
time_start = time.time()
for i in tqdm(range(20000)):
    out = net.forward_step(x, cx)
time_avg = (time.time() - time_start) / 20000
print("> Avg time per step inference on GPU: {}".format(time_avg))
