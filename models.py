import torch
from torch import nn
# import torch.nn.functional as F


class MyLSTM(nn.Module):
    '''
    Move core logic from `_VF` to pure PyTorch code though it suffer from poor performance
    '''

    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh):

        super().__init__()

        self.weight_ih = nn.Parameter(weight_ih)
        self.weight_hh = nn.Parameter(weight_hh)
        self.bias_ih = nn.Parameter(bias_ih)
        self.bias_hh = nn.Parameter(bias_hh)

        self.hidden_size = self.weight_hh.shape[1]
        self.input_size = self.weight_ih.shape[1]

        self.W_ii = self.weight_ih[self.hidden_size*0:self.hidden_size*1, :]
        self.W_if = self.weight_ih[self.hidden_size*1:self.hidden_size*2, :]
        self.W_ig = self.weight_ih[self.hidden_size*2:self.hidden_size*3, :]
        self.W_io = self.weight_ih[self.hidden_size*3:self.hidden_size*4, :]

        self.b_ii = self.bias_ih[self.hidden_size*0:self.hidden_size*1]
        self.b_if = self.bias_ih[self.hidden_size*1:self.hidden_size*2]
        self.b_ig = self.bias_ih[self.hidden_size*2:self.hidden_size*3]
        self.b_io = self.bias_ih[self.hidden_size*3:self.hidden_size*4]

        self.W_hi = self.weight_hh[self.hidden_size*0:self.hidden_size*1, :]
        self.W_hf = self.weight_hh[self.hidden_size*1:self.hidden_size*2, :]
        self.W_hg = self.weight_hh[self.hidden_size*2:self.hidden_size*3, :]
        self.W_ho = self.weight_hh[self.hidden_size*3:self.hidden_size*4, :]

        self.b_hi = self.bias_hh[self.hidden_size*0:self.hidden_size*1]
        self.b_hf = self.bias_hh[self.hidden_size*1:self.hidden_size*2]
        self.b_hg = self.bias_hh[self.hidden_size*2:self.hidden_size*3]
        self.b_ho = self.bias_hh[self.hidden_size*3:self.hidden_size*4]

        self.W_ii_T = self.W_ii.transpose(0, 1)
        self.W_if_T = self.W_if.transpose(0, 1)
        self.W_ig_T = self.W_ig.transpose(0, 1)
        self.W_io_T = self.W_io.transpose(0, 1)

        self.W_hi_T = self.W_hi.transpose(0, 1)
        self.W_hf_T = self.W_hf.transpose(0, 1)
        self.W_hg_T = self.W_hg.transpose(0, 1)
        self.W_ho_T = self.W_ho.transpose(0, 1)

    def forward(self, x, hx=None):
        '''
        x: seq_length x batch_size x features
        hx: initial hidden state, if it isn't specified, a zero vector will be used.
        '''
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        if hx is None:
            zeros = torch.zeros(1, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            hx = (zeros, zeros)

        output = torch.empty(seq_len, batch_size, self.hidden_size)
        h, c = hx

        for t in range(seq_len):
            i = torch.sigmoid(x[t, :, :] @ self.W_ii_T + self.b_ii + h @ self.W_hi_T + self.b_hi)
            f = torch.sigmoid(x[t, :, :] @ self.W_if_T + self.b_if + h @ self.W_hf_T + self.b_hf)
            g = torch.tanh(x[t, :, :] @ self.W_ig_T + self.b_ig + h @ self.W_hg_T + self.b_hg)
            o = torch.sigmoid(x[t, :, :] @ self.W_io_T + self.b_io + h @  self.W_ho_T + self.b_ho)
            c = f * c + i * g
            h = o * torch.tanh(c)

            output[t, :, :] = h

        return output, (h, c)


def apply_dropout_batch(W, prob, batch_size, device=None):
    '''
    Expect input X:
        (batch, 1, in_feature)
    Expect process:
        X @ W + b

    W: (in_features, out_features) -> (batch, in_features, out_features)
    b: (out_features) -> (batch, 1, out_features)
    '''
    mask_size = (batch_size,) + W.shape
    mask = (torch.rand(mask_size, device=device) < prob).type(W.dtype)
    masked_W = mask * W

    if len(W.shape) == 1:  # bias
        masked_W = masked_W.unsqueeze(1)
    return masked_W


class BayesLSTM(MyLSTM):
    '''
    Dropout operation will be applied to all parameter for every batch.
    It keep same value in a sequence. Naive altnative, appling independent dropout in every step in a sequence
    doesn't work as verified by original paper.
    '''
    def __init__(self, dropout_prob, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.dropout_prob = dropout_prob

    def forward(self, x, hx=None):
        '''
        x: seq_length x batch_size x features
        hx: initial hidden state, if it isn't specified, a zero vector will be used.
        '''
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        if hx is None:
            zeros = torch.zeros(1, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            hx = (zeros, zeros)

        output = torch.empty(seq_len, batch_size, self.hidden_size)
        h, c = hx

        # T = lambda W: apply_dropout_batch(W, self.dropout_prob, batch_size, device=x.device)
        def T(W):
            return apply_dropout_batch(W, self.dropout_prob, batch_size, device=x.device)

        # (features, out_features) -> (batch_size, features, out_features)

        W_ii_T = T(self.W_ii_T)
        W_if_T = T(self.W_if_T)
        W_ig_T = T(self.W_ig_T)
        W_io_T = T(self.W_io_T)

        b_ii = T(self.b_ii)
        b_if = T(self.b_if)
        b_ig = T(self.b_ig)
        b_io = T(self.b_io)

        W_hi_T = T(self.W_hi_T)
        W_hf_T = T(self.W_hf_T)
        W_hg_T = T(self.W_hg_T)
        W_ho_T = T(self.W_ho_T)

        b_hi = T(self.b_hi)
        b_hf = T(self.b_hf)
        b_hg = T(self.b_hg)
        b_ho = T(self.b_ho)

        x = x.unsqueeze(2)  # seq_len x batch x features -> seq_len x batch x 1 x features
        h = h.transpose(0, 1)  # (1, batch, hidden) -> (batch, 1, hidden)
        c = c.transpose(0, 1)  # (1, batch, hidden) -> (batch, 1, hidden)

        for t in range(seq_len):
            i = torch.sigmoid(x[t] @ W_ii_T + b_ii + h @ W_hi_T + b_hi)
            f = torch.sigmoid(x[t] @ W_if_T + b_if + h @ W_hf_T + b_hf)
            g = torch.tanh(x[t] @ W_ig_T + b_ig + h @ W_hg_T + b_hg)
            o = torch.sigmoid(x[t] @ W_io_T + b_io + h @  W_ho_T + b_ho)
            c = f * c + i * g
            h = o * torch.tanh(c)

            output[t, :, :] = h.squeeze(1)

        return output, (h.squeeze(1), c.squeeze(1))


class RegressionLSTMBayes(nn.Module):

    def __init__(self, input_features, hidden_dims, dropout_prob=0.5):
        super().__init__()

        lstm_bb = nn.LSTM(input_features, hidden_dims)

        self.rnn = BayesLSTM(dropout_prob, lstm_bb.weight_ih_l0.clone(), lstm_bb.weight_hh_l0.clone(),
                             lstm_bb.bias_ih_l0.clone(), lstm_bb.bias_hh_l0.clone())
        self.linear = nn.Linear(hidden_dims, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.linear(x[-1])[:, 0]
