'''
Ensure our LSTM got same value as original LSTM.
'''

from models import MyLSTM   # ,BayesLSTM

import torch
from torch import nn
# import torch.nn.functional as F

lstm_bb = nn.LSTM(6, 4, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)
lstm_mock = MyLSTM(lstm_bb.weight_ih_l0.clone(), lstm_bb.weight_hh_l0.clone(),
                   lstm_bb.bias_ih_l0.clone(), lstm_bb.bias_hh_l0.clone())

x_dummy = torch.randn(10, 1, 6)

out_bb = lstm_bb(x_dummy)
out_mock = lstm_mock(x_dummy)

output_bb, (h_bb, c_bb) = out_bb
output_mock, (h_mock, c_mock) = out_mock

for a, b in zip([output_bb, h_bb, c_bb], [output_mock, h_mock, c_mock]):
    print("Allclosed: ", a.allclose(b), " Norm of diff:", (a-b).norm())

assert lstm_bb.weight_ih_l0.grad is None
h_bb.sum().backward()
assert lstm_mock.weight_ih.grad is None
h_mock.sum().backward()

optimizer_bb = torch.optim.SGD(lstm_bb.parameters(), 0.1)
optimizer_mock = torch.optim.SGD(lstm_mock.parameters(), 0.1)

assert lstm_bb.weight_hh_l0.allclose(lstm_mock.weight_hh)

optimizer_bb.step()
optimizer_mock.step()

assert lstm_bb.weight_hh_l0.allclose(lstm_mock.weight_hh)

print("Test pass")
