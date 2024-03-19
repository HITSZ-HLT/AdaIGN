import os
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_GCN import GCN_2Layers, GCNLayer1, GCNII, TextCNN
from model_mm_use import MM_GCN, MM_GCN2
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
# from torch_geometric.nn import RGCNConv, GraphConv
from scipy.special import softmax

class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]
        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim) cand_dim == mem_dim?
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            M_ = M.permute(1, 2, 0)
            x_ = x.unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)
            x_ = self.transform(x).unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)
            x_ = self.transform(x).unsqueeze(1)
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_) * mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            alpha_masked = alpha_ * mask.unsqueeze(1)
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)
            alpha = alpha_masked / alpha_sum
        else:
            M_ = M.transpose(0, 1)
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)
            M_x_ = torch.cat([M_, x_], 2)
            mx_a = F.tanh(self.transform(M_x_))
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


class DialogueRNNCell(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell = nn.GRUCell(D_m + D_p, D_g)
        self.p_cell = nn.GRUCell(D_m + D_g, D_p)
        self.e_cell = nn.GRUCell(D_p, D_e)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m + D_p, D_p)

        self.dropout = nn.Dropout(dropout)

        if context_attention == 'simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel, 0)
        return q0_sel

    def forward(self, U, qmask, g_hist, q0, e0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(torch.cat([U, q0_sel], dim=1),
                         torch.zeros(U.size()[0], self.D_g).type(U.type()) if g_hist.size()[0] == 0 else
                         g_hist[-1])
        g_ = self.dropout(g_)
        if g_hist.size()[0] == 0:
            c_ = torch.zeros(U.size()[0], self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist, U)
        U_c_ = torch.cat([U, c_], dim=1).unsqueeze(1).expand(-1, qmask.size()[1], -1)
        qs_ = self.p_cell(U_c_.contiguous().view(-1, self.D_m + self.D_g),
                          q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = U.unsqueeze(1).expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1). \
                expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_p)
            U_ss_ = torch.cat([U_, ss_], 1)
            ql_ = self.l_cell(U_ss_, q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_ * (1 - qmask_) + qs_ * qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0] == 0 \
            else e0
        e_ = self.e_cell(self._select_parties(q_, qm_idx), e0)
        e_ = self.dropout(e_)
        return g_, q_, e_, alpha


class DialogueRNN(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e,
                                             listener_state, context_attention, D_a, dropout)

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type())
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2],
                         self.D_p).type(U.type())
        e_ = torch.zeros(0).type(U.type())
        e = e_

        alpha = []
        for u_, qmask_ in zip(U, qmask):
            g_, q_, e_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)], 0)
            e = torch.cat([e, e_.unsqueeze(0)], 0)
            if type(alpha_) != type(None):
                alpha.append(alpha_[:, 0, :])

        return e, alpha


class GRUModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5, att2=True):

        super(GRUModel, self).__init__()

        self.att2 = att2
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')
        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.gru(U)
        alpha, alpha_f, alpha_b = [], [], []

        if self.att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5, att2=True):

        super(LSTMModel, self).__init__()

        self.att2 = att2
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')
        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []

        if self.att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class DialogRNNModel(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, D_h, D_a=100, n_classes=7, listener_state=False,
                 context_attention='simple', dropout_rec=0.5, dropout=0.5, att2=True):

        super(DialogRNNModel, self).__init__()

        self.att2 = att2
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout + 0.15)
        self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                        context_attention, D_a, dropout_rec)
        self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                        context_attention, D_a, dropout_rec)
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')
        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)
        return pad_sequence(xfs)

    def forward(self, U, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)
        emotions_f = self.dropout_rec(emotions_f)
        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
        emotions_b = self._reverse_seq(emotions_b, umask)
        emotions_b = self.dropout_rec(emotions_b)
        emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        if self.att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class MaskedEdgeAttention(nn.Module):

    def __init__(self, input_dim, max_seq_len, no_cuda):
        """
        Method to compute the edge weights, as in Equation 1. in the paper. 
        attn_type = 'attn1' refers to the equation in the paper.
        For slightly different attention mechanisms refer to attn_type = 'attn2' or attn_type = 'attn3'
        """

        super(MaskedEdgeAttention, self).__init__()

        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.matchatt = MatchingAttention(self.input_dim, self.input_dim, att_type='general2')
        self.simpleatt = SimpleAttention(self.input_dim)
        self.att = Attention(self.input_dim, score_function='mlp')
        self.no_cuda = no_cuda

    def forward(self, M, lengths, edge_ind):
        """
        M -> (seq_len, batch, vector)
        lengths -> length of the sequences in the batch
        edge_idn -> edge_idn是边的index的集合   
        上下文边权重计算
        """
        attn_type = 'attn1'

        if attn_type == 'attn1':
            #(seq_len,batch_size，max_seq_len)
            scale = self.scalar(M)
            #(batch_size,max_seq_len,seq_len)
            alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
            if not self.no_cuda:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach().cuda()
                mask_copy = Variable(torch.zeros(alpha.size())).detach().cuda()

            else:
                mask = Variable(torch.ones(alpha.size()) * 1e-10).detach()
                mask_copy = Variable(torch.zeros(alpha.size())).detach()

            edge_ind_ = []
            for i, j in enumerate(edge_ind):
                for x in j:
                    #(样本序号，当前话语，相关上下文)
                    edge_ind_.append([i, x[0], x[1]])
            #转置成三列：样本序号，当前话语，上下文
            edge_ind_ = np.array(edge_ind_).transpose()
            mask[edge_ind_] = 1
            mask_copy[edge_ind_] = 1
            masked_alpha = alpha * mask
            _sums = masked_alpha.sum(-1, keepdim=True)
            scores = masked_alpha.div(_sums) * mask_copy

            return scores

        elif attn_type == 'attn2':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1)
                    t = M[node, j, :].unsqueeze(0)
                    _, alpha_ = self.simpleatt(M_, t)
                    scores[j, node, neighbour] = alpha_

        elif attn_type == 'attn3':
            scores = torch.zeros(M.size(1), self.max_seq_len, self.max_seq_len, requires_grad=True)

            if not self.no_cuda:
                scores = scores.cuda()

            for j in range(M.size(1)):

                ei = np.array(edge_ind[j])

                for node in range(lengths[j]):
                    neighbour = ei[ei[:, 0] == node, 1]

                    M_ = M[neighbour, j, :].unsqueeze(1).transpose(0, 1)
                    t = M[node, j, :].unsqueeze(0).unsqueeze(0).repeat(len(neighbour), 1, 1).transpose(0, 1)
                    _, alpha_ = self.att(M_, t)
                    scores[j, node, neighbour] = alpha_[0, :, 0]

        return scores


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def edge_perms(l, window_past, window_future):
    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j + window_future + 1)]
        elif window_future == -1:
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(l, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


def simple_batch_graphify(features, lengths, no_cuda):
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.cuda()

    return node_features, None, None, None, None


def batch_graphify(features, qmask, lengths, window_past, window_future, edge_type_mapping, att_model, no_cuda):
    '''
    feature:(seq_len,batch_size,vector_dim)
    '''
    edge_index, edge_norm, edge_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        #当前话语的前后上下文连边标识
        edge_ind.append(edge_perms(lengths[j], window_past, window_future))
    #上下文边的权重分数
    scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0] + length_sum, item[1] + length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))

        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))
            edge_norm.append(scores[j, item1[0], item1[1]])
            speaker0 = (qmask[item1[0], j, :] == 1).nonzero()[0][0].tolist()
            speaker1 = (qmask[item1[1], j, :] == 1).nonzero()[0][0].tolist()

            if item1[0] < item1[1]:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '0'])
            else:
                edge_type.append(edge_type_mapping[str(speaker0) + str(speaker1) + '1'])
    #一个batch的数据合并
    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_norm = torch.stack(edge_norm)
    edge_type = torch.tensor(edge_type)

    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_norm = edge_norm.cuda()
        edge_type = edge_type.cuda()

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda):
    """
    Method to obtain attentive node features over the graph convoluted features, as in Equation 4, 5, 6. in the paper.
    """

    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()

    if not no_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)

    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)

    emotions = torch.stack([pad(emotions.narrow(0, s, l), max_len, no_cuda)
                            for s, l in zip(start.data.tolist(),
                                            input_conversation_length.data.tolist())], 0).transpose(0, 1)

    alpha, alpha_f, alpha_b = [], [], []
    att_emotions = []

    for t in emotions:
        att_em, alpha_ = matchatt_layer(emotions, t, mask=umask)
        att_emotions.append(att_em.unsqueeze(0))
        alpha.append(alpha_[:, 0, :])

    att_emotions = torch.cat(att_emotions, dim=0)

    return att_emotions


def classify_node_features(emotions, seq_lengths, umask, matchatt_layer, linear_layer, dropout_layer, smax_fc_layer, nodal_attn, avec, no_cuda):
    if nodal_attn:

        emotions = attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda)
        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return torch.cat([hidden[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])

        log_prob = F.log_softmax(hidden, 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob

    else:

        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return hidden

        log_prob = F.log_softmax(hidden, 1)
        return log_prob


# class GraphNetwork(torch.nn.Module):
#     def __init__(self, num_features, num_classes, num_relations, max_seq_len, hidden_size=64, dropout=0.5, no_cuda=False, use_GCN=False, return_feature=False):
#         super(GraphNetwork, self).__init__()

#         self.return_feature = return_feature
#         self.no_cuda = no_cuda
#         self.use_GCN = use_GCN
#         self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
#         self.conv2 = GraphConv(hidden_size, hidden_size)
#         if not self.return_feature:
#             self.matchatt = MatchingAttention(num_features + hidden_size, num_features + hidden_size, att_type='general2')
#             self.linear = nn.Linear(num_features + hidden_size, hidden_size)
#             self.dropout = nn.Dropout(dropout)
#             self.smax_fc = nn.Linear(hidden_size, num_classes)
#         if self.use_GCN:
#             self.conv3 = GCNLayer1(num_features, hidden_size, False)  # index
#             self.conv4 = GCNLayer1(hidden_size, hidden_size, False)
#             self.linear = nn.Linear(num_features + hidden_size * 2, hidden_size)
#             self.matchatt = MatchingAttention(num_features + hidden_size * 2, num_features + hidden_size * 2, att_type='general2')

#     def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, qmask,umask, nodal_attn, avec):
#         if self.use_GCN:
#             topicLabel = []
#             out1 = self.conv1(x, edge_index, edge_type)
#             out1 = self.conv2(out1, edge_index, edge_norm)
#             out2 = self.conv3(x, seq_lengths, topicLabel,qmask)
#             out2 = self.conv4(out2, seq_lengths, topicLabel,qmask)
#             emotions = torch.cat([x, out1, out2], dim=-1)
#             if self.return_feature:
#                 return emotions
#             log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout, self.smax_fc, nodal_attn, avec,
#                                               self.no_cuda)
#         else:
#             out = self.conv1(x, edge_index, edge_type)
#             out = self.conv2(out, edge_index, edge_norm)
#             emotions = torch.cat([x, out], dim=-1)
#             if self.return_feature:
#                 return emotions
#             log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout, self.smax_fc, nodal_attn, avec,
#                                               self.no_cuda)
#         return log_prob


class MMGatedAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, att_type='general'):
        super(MMGatedAttention, self).__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        self.dropouta = nn.Dropout(0.5)
        self.dropoutv = nn.Dropout(0.5)
        self.dropoutl = nn.Dropout(0.5)
        if att_type == 'av_bg_fusion':
            self.transform_al = nn.Linear(mem_dim * 2, cand_dim, bias=True)
            self.scalar_al = nn.Linear(mem_dim, cand_dim)
            self.transform_vl = nn.Linear(mem_dim * 2, cand_dim, bias=True)
            self.scalar_vl = nn.Linear(mem_dim, cand_dim)
        elif att_type == 'general':
            self.transform_l = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_v = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_a = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_av = nn.Linear(mem_dim * 3, 1)
            self.transform_al = nn.Linear(mem_dim * 3, 1)
            self.transform_vl = nn.Linear(mem_dim * 3, 1)

    def forward(self, a, v, l, modals=None):
        a = self.dropouta(a) if len(a) != 0 else a
        v = self.dropoutv(v) if len(v) != 0 else v
        l = self.dropoutl(l) if len(l) != 0 else l
        if self.att_type == 'av_bg_fusion':
            if 'a' in modals:
                fal = torch.cat([a, l], dim=-1)
                Wa = torch.sigmoid(self.transform_al(fal))
                hma = Wa * (self.scalar_al(a))
            if 'v' in modals:
                fvl = torch.cat([v, l], dim=-1)
                Wv = torch.sigmoid(self.transform_vl(fvl))
                hmv = Wv * (self.scalar_vl(v))
            if len(modals) == 3:
                hmf = torch.cat([l, hma, hmv], dim=-1)
            elif 'a' in modals:
                hmf = torch.cat([l, hma], dim=-1)
            elif 'v' in modals:
                hmf = torch.cat([l, hmv], dim=-1)
            return hmf
        elif self.att_type == 'general':
            ha = torch.tanh(self.transform_a(a)) if 'a' in modals else a
            hv = torch.tanh(self.transform_v(v)) if 'v' in modals else v
            hl = torch.tanh(self.transform_l(l)) if 'l' in modals else l

            if 'a' in modals and 'v' in modals:
                z_av = torch.sigmoid(self.transform_av(torch.cat([a, v, a * v], dim=-1)))
                h_av = z_av * ha + (1 - z_av) * hv
                if 'l' not in modals:
                    return h_av
            if 'a' in modals and 'l' in modals:
                z_al = torch.sigmoid(self.transform_al(torch.cat([a, l, a * l], dim=-1)))
                h_al = z_al * ha + (1 - z_al) * hl
                if 'v' not in modals:
                    return h_al
            if 'v' in modals and 'l' in modals:
                z_vl = torch.sigmoid(self.transform_vl(torch.cat([v, l, v * l], dim=-1)))
                h_vl = z_vl * hv + (1 - z_vl) * hl
                if 'a' not in modals:
                    return h_vl
            return torch.cat([h_av, h_al, h_vl], dim=-1)


class DialogueGNNModel_ada(nn.Module):

    def __init__(self, base_model,temp, D_m, D_g, D_p, D_e, D_h, D_a, batch_size,graph_hidden_size, n_speakers, max_seq_len, window_past, window_future,n_classes=7,
                 listener_state=False, context_attention='simple', dropout_rec=0.5, dropout=0.5, nodal_attention=True, avec=False, no_cuda=False,
                 graph_type='relation', use_topic=False, alpha=0.1, lamda=0.5, multiheads=6, graph_construct='direct', use_GCN=False, use_residue=True,
                 dynamic_edge_w=False, D_m_v=512, D_m_a=100, modals='avl', att_type='gated', av_using_lstm=False, Deep_GCN_nlayers=64, dataset='IEMOCAP',
                 use_speaker=True, use_modal=False, reason_flag=False, multi_modal=True, use_crn_speaker=False, speaker_weights='1-1-1', modal_weight=1.0,temp_decay=0.965):

        super(DialogueGNNModel_ada, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_type = graph_type
        self.alpha = alpha
        self.lamda = lamda
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.use_topic = use_topic
        self.dropout = dropout
        self.use_GCN = use_GCN
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]  # [a, v, l]
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        self.reason_flag = reason_flag
        self.multi_modal = multi_modal
        self.n_speakers = n_speakers
        self.use_crn_speaker = use_crn_speaker
        self.speaker_weights = list(map(float, speaker_weights.split('-')))
        self.modal_weight = modal_weight
        self.dataset = dataset
        self.temp = temp
        self.max_seq_len=max_seq_len
        self.losses={}
        self._tem_decay=temp_decay
        self.reset_logits(batch_size)
        self.dm=D_m
        hidden_a = 200
        hidden_v = 200
        hidden_l = 200
        #encoder
        if 'a' in self.modals:
            self.linear_a = nn.Linear(D_m_a, hidden_a)
            self.linear_a1 = nn.Linear(D_m_a, hidden_a)
        if 'v' in self.modals:
            self.linear_v = nn.Linear(D_m_v, hidden_v)    
            self.linear_v1 = nn.Linear(D_m_v, hidden_v)    
        if 'l' in self.modals: 
            self.linear_l = nn.Linear(D_m, hidden_l)
            self.linear_l1 = nn.Linear(D_m, hidden_l)
            self.lstm_l = nn.GRU(input_size=hidden_l, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            self.lstm_l1 = nn.GRU(input_size=hidden_l, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
                # self.rnn_parties = nn.LSTM(input_size=hidden_l, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
                
        self.rnn_parties = nn.GRU(input_size=hidden_l, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        n_relations = 2 * n_speakers ** 2
        self.window_past = window_past
        self.window_future = window_future
        #计算边权重的模型
        self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention                                          
        self.graph_model1 = MM_GCN(a_dim=2 * D_e, v_dim=2 * D_e, l_dim=2 * D_e, n_dim=2 * D_e, nlayers=Deep_GCN_nlayers, nhidden=graph_hidden_size,
                                          nclass=n_classes, dropout=self.dropout, lamda=self.lamda, alpha=self.alpha, variant=True,
                                          return_feature=self.return_feature, use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals,
                                          use_speaker=self.use_speaker, use_modal=self.use_modal, reason_flag=self.reason_flag, modal_weight=self.modal_weight,
                                          )
        self.graph_model2 = MM_GCN(a_dim=2 * D_e, v_dim=2 * D_e, l_dim=2 * D_e, n_dim=2 * D_e, nlayers=Deep_GCN_nlayers, nhidden=graph_hidden_size,
                                          nclass=n_classes, dropout=self.dropout, lamda=self.lamda, alpha=self.alpha, variant=True,
                                          return_feature=self.return_feature, use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals,
                                          use_speaker=self.use_speaker, use_modal=self.use_modal, reason_flag=self.reason_flag, modal_weight=self.modal_weight,
                                          )
           
        print("construct " + self.graph_type)

        edge_type_mapping = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping

        self.gatedatt = MMGatedAttention(3* D_e + graph_hidden_size, graph_hidden_size, att_type='general')
        self.dropout_ = nn.Dropout(self.dropout)

        self.smax_fc = nn.Linear(900, n_classes) if self.use_residue else nn.Linear(100 * len(self.modals), n_classes)
        if dataset=='MELD':
            self.fc1 = nn.Linear(1024,D_m)
            self.fc2 = nn.Linear(300,D_m_a)
            self.fc3 = nn.Linear(342,D_m_v)
            self.normLNa = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNb = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNc = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNd = nn.LayerNorm(D_m, elementwise_affine=True)
        if dataset=='IEMOCAP':
            self.fc1 = nn.Linear(1024,D_m)
            self.fc2 = nn.Linear(1582,D_m_a)
            self.fc3 = nn.Linear(342,D_m_v)
            self.normLNa = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNb = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNc = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNd = nn.LayerNorm(D_m, elementwise_affine=True)

    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)
    
    def reset_logits(self,batch_size=32): #modal%d_logits的初始值
        self._arch_parameters = []
        if self.dataset=='MELD':
            for j in range(3):
                for i in range(batch_size): 
                #modal_logits = 1/2 * torch.ones(33*3,2) #数值全0.5
                    modal_logits =abs(1e-1 * torch.randn(200,2))
                    self._arch_parameters = []
                    self.register_parameter( 'modal%(c)d_sample%(l)d_logits' %{'c':j+1,'l':i}, nn.Parameter(modal_logits))
                    self._arch_parameters.append(getattr(self, 'modal%(c)d_sample%(l)d_logits' %{'c':j+1,'l':i}))
                    context_logits=abs(1e-1 * torch.randn(200,2))
                    self.register_parameter( 'modal%(c)d_context%(l)d_logits' %{'c':j+1,'l':i}, nn.Parameter(context_logits))
                    self._arch_parameters.append(getattr(self, 'modal%(c)d_context%(l)d_logits' %{'c':j+1,'l':i}))
            
            modal_logits =abs(1e-1 * torch.randn(1,2))
            self.register_parameter( 'modal_logits' , nn.Parameter(modal_logits))
            self._arch_parameters.append(getattr(self, 'modal_logits'))
        else:
            for j in range(3):
                for i in range(batch_size): 
                #modal_logits = 1/2 * torch.ones(33*3,2) #数值全0.5
                    modal_logits =(1e-1 * torch.randn(200,2))
                    self._arch_parameters = []
                    self.register_parameter( 'modal%(c)d_sample%(l)d_logits' %{'c':j+1,'l':i}, nn.Parameter(modal_logits))
                    self._arch_parameters.append(getattr(self, 'modal%(c)d_sample%(l)d_logits' %{'c':j+1,'l':i}))
                    context_logits=(1e-1 * torch.randn(200,2))
                    self.register_parameter( 'modal%(c)d_context%(l)d_logits' %{'c':j+1,'l':i}, nn.Parameter(context_logits))
                    self._arch_parameters.append(getattr(self, 'modal%(c)d_context%(l)d_logits' %{'c':j+1,'l':i}))
            
            modal_logits =(1e-1 * torch.randn(1,2))
            self.register_parameter( 'modal_logits' , nn.Parameter(modal_logits))
            self._arch_parameters.append(getattr(self, 'modal_logits'))

            
            
            # modal_logits =1/2 * torch.ones(self.max_seq_len,2)
            # self.register_parameter( 'modal%d_logits' %{'c':j+1,'l':i}, nn.Parameter(modal_logits))
            # self._arch_parameters.append(getattr(self, 'modal%d_logits' %{'c':j+1,'l':i}))


    def arch_parameters(self):  #返回名字中有'modal'和'logits'对应的param
        params = []
        for name, param in self.named_parameters():
            if 'modal' in name and 'logits' in name:
                params.append(param)
        return params

    def network_parameters(self):   #返回名字中除了有'modal'和'logits'之外其他的param
        params = []
        for name, param in self.named_parameters():
            if not ('modal' in name and 'logits' in name):
                params.append(param)
        return params

    def fix_w(self):
        network_params = self.network_parameters()
        for param in network_params:
            param.requires_grad = False

    def free_w(self):
        network_params = self.network_parameters() 
        for param in network_params:
            param.requires_grad = True

    def fix_alpha(self):   
        arch_parameters = self.arch_parameters() 
        for param in arch_parameters:
            param.requires_grad = False

    def free_alpha(self): 
        arch_parameters = self.arch_parameters()
        for param in arch_parameters:
            param.requires_grad = True

    def get_importance(self,num_train_layers,batch): #Sharing loss eq.4
        self.losses['importance'] = {}
        self.losses['importance']['total'] = 0

        for i in range(batch):
            t_i=2
            logits = getattr(self,'modal%(c)d_sample%(l)d_logits' %{'c':t_i+1,'l':i})
            # num_policy_layers = logits.shape[0]   #num_layers
            num_policy_layers = self.seq_len
            loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).cuda()#把1.0均匀分成num_layers份 若=5,就是 tensor([0.2000, 0.4000, 0.6000, 0.8000, 1.0000])
                #loss_weights = torch.ones(1).cuda()
            gt = torch.zeros((num_policy_layers)).long().cuda()
            self.losses['importance']['total'] += nn.CrossEntropyLoss()(logits[-num_policy_layers:], gt)
            for t_j in range(2):
                    for z in range(batch):
                        logits_j = getattr(self, 'modal%(c)d_sample%(l)d_logits' %{'c':t_j+1,'l':z})
                            #print('logits_j[:, 0]:',logits_j[:, 0])
                        self.losses['importance']['total'] += torch.sum(loss_weights * torch.abs(logits[-num_policy_layers:][:, 0] - logits_j[-num_policy_layers:][:, 0]))
        
        for t_i in range(3):
            for i in range(batch):
                logits = getattr(self,'modal%(c)d_context%(l)d_logits' %{'c':t_i+1,'l':i})
                num_policy_layers = logits.shape[0]   #num_layers
                num_policy_layers = self.seq_len
                loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).cuda()#把1.0均匀分成num_layers份 若=5,就是 tensor([0.2000, 0.4000, 0.6000, 0.8000, 1.0000])
                        #loss_weights = torch.ones(1).cuda()
                gt = torch.zeros((num_policy_layers)).long().cuda()
                self.losses['importance']['total'] += nn.CrossEntropyLoss()(logits[-num_policy_layers:], gt)
                for t_j in range(t_i, 3):
                    for z in range(batch):
                        logits_j = getattr(self, 'modal%(c)d_context%(l)d_logits' %{'c':t_j+1,'l':z})
                                    #print('logits_j[:, 0]:',logits_j[:, 0])
                        self.losses['importance']['total'] += torch.sum(loss_weights * torch.abs(logits[-num_policy_layers:][:, 0] - logits_j[-num_policy_layers:][:, 0]))
    

    def get_maintain(self,num_train_layers,batch): #Sharing loss eq.4
        self.losses['hamming'] = {}
        self.losses['hamming']['total'] = 0
        
        for t_i in range(2):
            for i in range(batch):
                logits = getattr(self,'modal%(c)d_sample%(l)d_logits' %{'c':t_i+1,'l':i})
                # num_policy_layers = logits.shape[0]   #num_layers
                num_policy_layers = self.seq_len
                loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).cuda()#把1.0均匀分成num_layers份 若=5,就是 tensor([0.2000, 0.4000, 0.6000, 0.8000, 1.0000])
                    #loss_weights = torch.ones(1).cuda()
                gt = torch.zeros((num_policy_layers)).long().cuda()
                self.losses['hamming']['total'] += nn.CrossEntropyLoss()(logits[-num_policy_layers:], gt)
                for t_j in range(t_i, 2):
                     for z in range(batch):
                        logits_j = getattr(self, 'modal%(c)d_sample%(l)d_logits' %{'c':t_j+1,'l':z})
                            #print('logits_j[:, 0]:',logits_j[:, 0])
                        self.losses['hamming']['total'] += torch.sum(loss_weights * torch.abs(logits[-num_policy_layers:][:, 0] - logits_j[-num_policy_layers:][:, 0]))
                        
        logits = getattr(self,  'modal_logits')
                #print('t_id:',t_id,'logits:',logits[-10:])
        num_blocks = 1
        gt = torch.zeros((num_blocks)).long().cuda()
        self.losses['hamming']['modal'] = nn.CrossEntropyLoss()(logits[-num_blocks:], gt)
                #如果num_blocks==logits.shape[0]，那么logits[-num_blocks:]=logits
        self.losses['hamming']['total'] += self.losses['hamming']['modal']

                
                
                
    def get_drop(self,num_train_layers,batch):   #sparsity loss eq.3
        self.losses['sparsity'] = {}
        self.losses['sparsity']['total'] = 0

        
        # for i in range(batch):
        #     t_id=2
        for t_id in range(2):
            for i in range(batch):
                logits = getattr(self,  'modal%(c)d_sample%(l)d_logits'  %{'c':t_id+1,'l':i})
                        #print('t_id:',t_id,'logits:',logits[-10:])
                num_blocks = self.seq_len
                gt = torch.ones((num_blocks)).long().cuda()
                self.losses['sparsity']['modal%(c)d_sample%(l)d_logits'  %{'c':t_id+1,'l':i}] = nn.CrossEntropyLoss()(logits[-num_blocks:], gt)
                        #如果num_blocks==logits.shape[0]，那么logits[-num_blocks:]=logits
                self.losses['sparsity']['total'] += self.losses['sparsity']['modal%(c)d_sample%(l)d_logits'  %{'c':t_id+1,'l':i}]

        # for t_id in range(3):
        #     for i in range(batch):
        #         logits = getattr(self,  'modal%(c)d_context%(l)d_logits'%{'c':t_id+1,'l':i})
        #                 #print('t_id:',t_id,'logits:',logits[-10:])
        #         num_blocks = min(num_train_layers, logits.shape[0])
        #         gt = torch.ones((num_blocks)).long().cuda()
        #         self.losses['sparsity']['modal%(c)d_context%(l)d_logits'%{'c':t_id+1,'l':i}] = nn.CrossEntropyLoss()(logits[-num_blocks:], gt)
        #                 #如果num_blocks==logits.shape[0]，那么logits[-num_blocks:]=logits
        #         self.losses['sparsity']['total'] += self.losses['sparsity']['modal%(c)d_context%(l)d_logits'%{'c':t_id+1,'l':i}]
            
       
        logits = getattr(self,  'modal_logits')
                #print('t_id:',t_id,'logits:',logits[-10:])
        num_blocks = 1
        gt = torch.ones((num_blocks)).long().cuda()
        self.losses['sparsity']['modal'] = nn.CrossEntropyLoss()(logits[-num_blocks:], gt)
                #如果num_blocks==logits.shape[0]，那么logits[-num_blocks:]=logits
        self.losses['sparsity']['total'] += self.losses['sparsity']['modal']

    def get_not_importance(self,num_train_layers,batch): #Sharing loss eq.4
        self.losses['not_importance'] = {}
        self.losses['not_importance']['total'] = 0

        for i in range(batch):
            t_i=2
            logits = getattr(self,'modal%(c)d_sample%(l)d_logits' %{'c':t_i+1,'l':i})
            # num_policy_layers = logits.shape[0]   #num_layers
            num_policy_layers = self.seq_len
            # loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).cuda()#把1.0均匀分成num_layers份 若=5,就是 tensor([0.2000, 0.4000, 0.6000, 0.8000, 1.0000])
                #loss_weights = torch.ones(1).cuda()
            gt = torch.ones((num_policy_layers)).long().cuda()
            self.losses['not_importance']['total'] += nn.CrossEntropyLoss()(logits[-num_policy_layers:], gt)
        
        for t_i in range(3):
            for i in range(batch):
                logits = getattr(self,'modal%(c)d_context%(l)d_logits' %{'c':t_i+1,'l':i})
                num_policy_layers = logits.shape[0]   #num_layers
                num_policy_layers = self.seq_len
                # loss_weights = ((torch.arange(0, num_policy_layers, 1) + 1).float() / num_policy_layers).cuda()#把1.0均匀分成num_layers份 若=5,就是 tensor([0.2000, 0.4000, 0.6000, 0.8000, 1.0000])
                        #loss_weights = torch.ones(1).cuda()
                gt = torch.ones((num_policy_layers)).long().cuda()
                self.losses['not_importance']['total'] += nn.CrossEntropyLoss()(logits[-num_policy_layers:], gt)
                
    


    def decay_temperature(self, decay_ratio=None):
        tmp = self.temp
        if decay_ratio is None:
            self.temp *= self._tem_decay
        else:
            self.temp *= decay_ratio
        print("Change temperature from %.5f to %.5f" % (tmp, self.temp))

    def get_policy_prob(self,batch):   #输出modal%d_logits经过了softmax之后的distribution
        distributions1 = []
        distributions2 = []
        for t_id in range(3):
            for i in range(batch):
                logits = getattr(self, 'modal%(c)d_sample%(l)d_logits'  %{'c':t_id+1,'l':i}).detach().cpu().numpy()
                distributions1.append(softmax(logits, axis=-1))
            
                logits = getattr(self,  'modal%(c)d_context%(l)d_logits'%{'c':t_id+1,'l':i}).detach().cpu().numpy()
                distributions1.append(softmax(logits, axis=-1))
        
        logits = getattr(self,  'modal_logits').detach().cpu().numpy()
        distributions2.append(softmax(logits, axis=-1))
        return distributions1,distributions2
    
    def load_policy(self, label):
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join('experiments/', save_filename)
        with open(save_path, 'rb') as handle:
            policy = pickle.load(handle)
        for t_id in range(3):
            setattr(self, 'policy%d' % (t_id + 1), policy['modal%d_policy' % (t_id+1)])
            #print(getattr(self, 'policy%d' % (t_id + 1)))

    def save_policy(self, label):
        policy = {}
        for t_id in range(2):
            tmp = getattr(self, 'policy%d' % (t_id + 1))
            policy['modal%d_policy' % (t_id + 1)] = tmp.cpu().data
        save_filename = 'policy%s.pickle' % str(label)
        save_path = os.path.join('experiments/', save_filename)
        with open(save_path, 'wb') as handle:
            pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def sample_policy(self):
        for t_id in range(3):
            p = self.test_sample_policy(t_id)
            setattr(self, 'policy%d' % (t_id+1), p)
    
    def train_sample_policy(self, temperature, batch=30,hard_sampling=False): #对modal%d_logits进行gumbel softmax操作 得到policys
        #modal%d_logits 在update w 过程中是不会改变的 在update alpha过程中会参与反向传播 同一个数据gumbel softmax操作每次不一样
        policys1 = []
        policys2 = []
        policys3 = []
  
        for t_id in range(3):
            for i in range(batch):
                policy = F.gumbel_softmax(getattr(self, 'modal%(c)d_sample%(l)d_logits'  %{'c':t_id+1,'l':i}), temperature, hard=hard_sampling)
                policys1.append(policy) 
        
                policy = F.gumbel_softmax(getattr(self,  'modal%(c)d_context%(l)d_logits'%{'c':t_id+1,'l':i}), temperature, hard=hard_sampling)
                policys2.append(policy)
   
        policy = F.gumbel_softmax(getattr(self, 'modal_logits' ), temperature, hard=hard_sampling)
        policys3.append(policy)
        return policys1,policys2,policys3


    def test_sample_policy(self, temperature,batch=30):          #由logits得到[1,0]或是[0,1]的policy 
        self.policys1 = []
        self.policys2 = []
        self.policys3 = []
        
        for t_id in range(3):
            for i in range(batch):
                modal_logits = getattr(self,  'modal%(c)d_sample%(l)d_logits'  %{'c':t_id+1,'l':i})
                cuda_device = modal_logits.get_device()
                logits = modal_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).cuda()
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                    # setattr(self, 'policy%d' % t_id, policy)
                self.policys1.append(policy)


                modal_context_logits =getattr(self,  'modal%(c)d_context%(l)d_logits'%{'c':t_id+1,'l':i})
                cuda_device =  modal_context_logits.get_device()
                logits =  modal_context_logits.detach().cpu().numpy()
                distribution = softmax(logits, axis=-1)
                single_policys = []
                for tmp_d in distribution:
                    sampled = np.random.choice((1, 0), p=tmp_d)
                    policy = [sampled, 1 - sampled]
                    single_policys.append(policy)
                if cuda_device != -1:
                    policy = torch.from_numpy(np.array(single_policys)).cuda()
                else:
                    policy = torch.from_numpy(np.array(single_policys))
                        # setattr(self, 'policy%d' % t_id, policy)
                self.policys2.append(policy)
 
       
        modal_logits = getattr(self,  'modal_logits' )
        cuda_device = modal_logits.get_device()
        logits = modal_logits.detach().cpu().numpy()
        distribution = softmax(logits, axis=-1)
        single_policys = []
        for tmp_d in distribution:
            sampled = np.random.choice((1, 0), p=tmp_d)
            policy = [sampled, 1 - sampled]
            single_policys.append(policy)
        if cuda_device != -1:
            policy = torch.from_numpy(np.array(single_policys)).cuda()
        else:
            policy = torch.from_numpy(np.array(single_policys))
        #setattr(self, 'policy%d' % t_id, policy)
        # policy = F.gumbel_softmax(getattr(self, 'modal_logits' ), temperature, hard=False)
        self.policys3.append(policy)

        return self.policys1,self.policys2,self.policys3

    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, test_label=False,num_train_layers=0, flag=None, train_flag=None,p_c_a=None,p_c_v=None,p_c_t=None,p_a=None,p_t=None,p_v=None):
        emotions_a, emotions_v, emotions_l, features_a, features_v, features_l = None, None, None, None, None, None
        self.seq_len=U_a.size(0)
        seq_len=self.seq_len
        batch_size=U_a.size(1)
        if train_flag:
            self.policys1,self.policys2,self.policys3 = self.train_sample_policy( temperature=self.temp,batch=batch_size)
        else:
            self.policys1,self.policys2,self.policys3 = self.test_sample_policy(temperature=self.temp,batch=batch_size)
        for t_id in range(3):     
            for i in range(batch_size):  
                self.policys1[t_id+i] = self.policys1[t_id+i].cuda()
            
                self.policys2[t_id+i] = self.policys2[t_id+i].cuda()
              
        self.policys3[0] = self.policys3[0].cuda()
                
        #---------------curriculum
        num_train_layers = min(num_train_layers, self.max_seq_len)
        skip_layer = self.max_seq_len- num_train_layers
        padding = torch.ones(skip_layer, 2).cuda()
        padding[:, 1] = 0
        context_outputs=[]
        modal_outputs=[]
        for j in range(3):
            modal_padding_policys=[]
            context_padding_policys = []
            for i in range(batch_size):
                modal_padding_policy = torch.cat((padding.float(), self.policys1[j][-num_train_layers:].float()), dim=0)
                modal_padding_policys.append(modal_padding_policy)
                context_padding_policy=torch.cat((padding.float(), self.policys2[j][-num_train_layers:].float()), dim=0)
                context_padding_policys.append(context_padding_policy)
            output=torch.stack(modal_padding_policys,dim=1)
            modal_outputs.append(output)
            output=torch.stack(context_padding_policys,dim=1)
            context_outputs.append(output)
            
        if isinstance(U,list):
            [r1,r2,r3,r4]=U
            seq_len, _, _ = r1.size()
            r1=self.fc1(r1)
            r2=self.fc1(r2)
            r3=self.fc1(r3)
            r4=self.fc1(r4)
            if self.dataset=='IEMOCAP':
                r1 = self.normLNa(r1.transpose(0, 1).reshape(-1, self.dm)).reshape(-1, seq_len, self.dm).transpose(1, 0)
                r2 = self.normLNb(r2.transpose(0, 1).reshape(-1, self.dm)).reshape(-1, seq_len, self.dm).transpose(1, 0)
                r3 = self.normLNc(r3.transpose(0, 1).reshape(-1, self.dm)).reshape(-1, seq_len, self.dm).transpose(1, 0)
                r4 = self.normLNd(r4.transpose(0, 1).reshape(-1, self.dm)).reshape(-1, seq_len, self.dm).transpose(1, 0)
            else:
                r1 = self.normLNa(r1.transpose(0, 1).reshape(-1, 600)).reshape(-1, seq_len, 600).transpose(1, 0)
                r2 = self.normLNb(r2.transpose(0, 1).reshape(-1, 600)).reshape(-1, seq_len, 600).transpose(1, 0)
                r3 = self.normLNc(r3.transpose(0, 1).reshape(-1, 600)).reshape(-1, seq_len,600).transpose(1, 0)
                r4 = self.normLNd(r4.transpose(0, 1).reshape(-1, 600)).reshape(-1, seq_len, 600).transpose(1, 0)

            U = (r1+r2+r3+r4)/4
            U_a=self.fc2(U_a)
            U_v=self.fc3(U_v)
        if 'a' in self.modals:
            # (seq_len,batch_size,hidden_a):(77,32,2)
            modal_padding_policy_a=modal_outputs[0][self.max_seq_len-seq_len:, :, 0]
            # padding_policy_a=modal_padding_policy_a.unsqueeze(dim=1).repeat(1,U_a.size(1))
            modal_policy_a=modal_padding_policy_a.unsqueeze(dim=2).repeat(1,1,U_a.size(2))
            if p_a==None:
                # padding_policy=torch.ones_like(padding_policy)
                pass
            else:
                for i in p_a:
                    modal_policy_a[i[0],i[1],:]=1
            #(77,32,2)
            # context_padding_policy_a=context_outputs[0][self.max_seq_len-seq_len:,:,0]
            # #(77,32,dm)
            # context_policy_a=context_padding_policy_a.unsqueeze(dim=2).repeat(1,1,U_a.size(2))
            
            context_padding_policy_a=context_outputs[0][self.max_seq_len-seq_len:,:,:]
            if p_c_a==None:
                pass
            else:
                for i in p_c_a:
                    context_padding_policy_a[i[0],i[1],:]=1
            U_a1=torch.mul(U_a,modal_policy_a)
            U_a = self.linear_a(U_a)
            U_a1 = self.linear_a1(U_a1)
            emotions_a = U_a
            emotions_a1 = U_a1
                

            if self.use_crn_speaker:

                # (32,77,200) (32,77,2)
                U_, qmask_ = U_a.transpose(0, 1), qmask.transpose(0, 1)
                U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U_a.type())
                U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
                        #按照说话人拆分样本
                for b in range(U_.size(0)):
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        if index_i.size(0) > 0:
                            U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
                        #编码相应话语的说话人信息：(32,77,200)
                E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]
                        #截断到原本seq_len,补充的位置为0
                for b in range(U_p_.size(0)):
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                        # (77,32,200)
                U_p = U_p_.transpose(0, 1)
                        #x_i^a
                emotions_a = emotions_a + self.speaker_weights[0] * U_p

                U_, qmask_ = U_a1.transpose(0, 1), qmask.transpose(0, 1)
                U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U_a1.type())
                U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
                        #按照说话人拆分样本
                for b in range(U_.size(0)):
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        if index_i.size(0) > 0:
                            U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
                        #编码相应话语的说话人信息：(32,77,200)
                E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]
                        #截断到原本seq_len,补充的位置为0
                for b in range(U_p_.size(0)):
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                        # (77,32,200)
                U_p = U_p_.transpose(0, 1)
                        #x_i^a
                emotions_a1 = emotions_a1 + self.speaker_weights[0] * torch.mul(U_p,modal_policy_a[:,:,:U_p.size(2)])
            
            features_a, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
            features_a1, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_a1, seq_lengths, self.no_cuda)
        if 'v' in self.modals:
                # (21.32,200)
            modal_padding_policy_v=modal_outputs[1][self.max_seq_len-seq_len: ,:,0]
            # padding_policy_v=modal_padding_policy_v.unsqueeze(dim=1).repeat(1,U_v.size(1))
            modal_policy_v=modal_padding_policy_v.unsqueeze(dim=2).repeat(1,1,U_v.size(2))
            if p_v==None:
                # padding_policy=torch.ones_like(padding_policy)
                pass
            else:
                for i in p_v:
                    modal_policy_v[i[0],i[1],:]=1
            # context_padding_policy_v=context_outputs[1][self.max_seq_len-seq_len: ,:,0]
            # context_policy_v=context_padding_policy_v.unsqueeze(dim=2).repeat(1,1,U_v.size(2))
            
            context_padding_policy_v=context_outputs[1][self.max_seq_len-seq_len: ,:,:]
            if p_c_v==None:
                pass
            else:
                for i in p_c_v:
                    context_padding_policy_v[i[0],i[1],:]=1
            U_v1=torch.mul(U_v,modal_policy_v)
            U_v = self.linear_v(U_v)
            U_v1 = self.linear_v1(U_v1)
            emotions_v = U_v
            emotions_v1 = U_v1
                    
                    # TODO

            if self.use_crn_speaker:
                    # (32,21,200), (32,21,9)
                U_, qmask_ = U_v.transpose(0, 1), qmask.transpose(0, 1)
                U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U_v.type())
                U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
                for b in range(U_.size(0)):
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        if index_i.size(0) > 0:
                            U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]

                E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]

                for b in range(U_p_.size(0)):
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                        # (21,32,200)
                U_p = U_p_.transpose(0, 1)
                emotions_v = emotions_v + self.speaker_weights[1] * U_p

                U_, qmask_ = U_v1.transpose(0, 1), qmask.transpose(0, 1)
                U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U_v1.type())
                U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
                for b in range(U_.size(0)):
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        if index_i.size(0) > 0:
                            U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]

                E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]

                for b in range(U_p_.size(0)):
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                        # (21,32,200)
                U_p = U_p_.transpose(0, 1)

                emotions_v1 = emotions_v1 + self.speaker_weights[1] * torch.mul(U_p,modal_policy_v[:,:,:U_p.size(2)])

            features_v, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            features_v1, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_v1, seq_lengths, self.no_cuda)
        if 'l' in self.modals:
            modal_padding_policy_l=modal_outputs[2][self.max_seq_len-seq_len: ,:,0]
            # padding_policy_l=modal_padding_policy_l.unsqueeze(dim=1).repeat(1,U.size(1))
            modal_policy_l=modal_padding_policy_l.unsqueeze(dim=2).repeat(1,1,U.size(2))
            if p_t==None:
                modal_policy_l=torch.ones_like(modal_policy_l)
            
            else:
                for i in p_t:
                    modal_policy_l[i[0],i[1],:]=1
            # context_padding_policy_l=context_outputs[2][self.max_seq_len-seq_len: ,:,0]
            # context_policy_l=context_padding_policy_l.unsqueeze(dim=2).repeat(1,1,U.size(2))
            # if p_c_t==None:
            #     context_policy_l=torch.ones_like(context_policy_l)
            # else:
            #     for i in p_c_t:
            #         context_policy_l[i[0],i[1],:]=1
            context_padding_policy_l=context_outputs[2][self.max_seq_len-seq_len: ,:,:]
            if p_c_t==None:
                pass
            else:
                for i in p_c_t:
                    context_padding_policy_l[i[0],i[1],:]=1
            U1=torch.mul(U,modal_policy_l)
            U = self.linear_l(U)
            U1 = self.linear_l1(U1)

            # (21,32,200)
            emotions_l, hidden_l = self.lstm_l(U)
            emotions_l1, hidden_l1 = self.lstm_l1(U1)
            
            speaker_masks=[]
            if self.use_crn_speaker:
                        # (32,21,200), (32,21,9)
                U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)
                U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U.type())
                U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
                for b in range(U_.size(0)):
                    speaker_mask=[]
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        mask=torch.zeros([200,1]).cuda()
                        mask[index_i]=1
                        speaker_mask.append(mask)
                        if index_i.size(0) > 0:
                            U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
                    speaker_masks.append(speaker_mask)
                
                E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]

                for b in range(U_p_.size(0)):
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                        # (21,32,200)
                U_p = U_p_.transpose(0, 1)

                emotions_l = emotions_l + self.speaker_weights[2] * U_p

                U_, qmask_ = U1.transpose(0, 1), qmask.transpose(0, 1)
                U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U1.type())
                U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
                for b in range(U_.size(0)):
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        if index_i.size(0) > 0:
                            U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]

                E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]

                for b in range(U_p_.size(0)):
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                        # (21,32,200)
                U_p = U_p_.transpose(0, 1)
                mask_l=modal_padding_policy_l.unsqueeze(dim=2).repeat(1,1,U_p.size(2))
                emotions_l1 = emotions_l1 + self.speaker_weights[2] * torch.mul(U_p,mask_l)
            else:
                U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)
                U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U.type())
                U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
                for b in range(U_.size(0)):
                    speaker_mask=[]
                    for p in range(len(U_parties_)):
                        index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                        mask=torch.zeros([200,1]).cuda()
                        mask[index_i]=1
                        speaker_mask.append(mask)
                        if index_i.size(0) > 0:
                            U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
                    speaker_masks.append(speaker_mask)


            features_l, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)

            features_l1, edge_index, edge_norm, edge_type, edge_index_lengths = simple_batch_graphify(emotions_l1, seq_lengths, self.no_cuda)
        features_i = torch.cat([0.1*features_a, 0.1*features_v, 0.8*features_l], dim=1).cuda()
        emotions_feat1 = self.graph_model2(features_a1, features_v1, features_l1, seq_lengths, qmask, False,test_label,context_padding_policy_a,context_padding_policy_v,context_padding_policy_l,modal_padding_policy_a,modal_padding_policy_v,modal_padding_policy_l,speaker_masks)
        emotions_feat = self.graph_model1(features_a, features_v, features_l, seq_lengths, qmask, False,test_label)
        
        emotions_feat=emotions_feat*self.policys3[0][:,0]+emotions_feat1*(1-self.policys3[0][:,0])
        #emotions_feat=emotions_feat*1+emotions_feat1*1
        emotions_feat=torch.cat((emotions_feat,features_i),dim=1)
        if test_label:
            index = 15
            print('# deepGCN layer ' + str(index))
            if not os.path.isdir('../context_padding_policys/iemocap/'): os.makedirs('../context_padding_policys/iemocap/')
            np.save("../context_padding_policys/iemocap/1080_v2_test_output_multi_{}".format(index), emotions_feat.data.cpu().numpy())
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
        emotions_feat = self.smax_fc(emotions_feat)
        if test_label:
            index = 15
            print('# deepGCN layer ' + str(index))
            if not os.path.isdir('../context_padding_policys/iemocap/'): os.makedirs('../context_padding_policys/iemocap/')
            np.save("../context_padding_policys/iemocap/1080_v3_test_output_multi_after_relu-fc_{}".format(index), emotions_feat.data.cpu().numpy())

        log_prob = F.log_softmax(emotions_feat, 1)
        if flag == 'update_alpha':
            
            self.get_maintain(num_train_layers,batch_size) #Sharing loss eq.4
            # loss += 0.6 * self.losses['hamming']['total']
            self.get_drop(num_train_layers,batch_size)
            # loss += 0.4 * self.losses['sparsity']['total']
            self.get_importance(num_train_layers,batch_size)
            self.get_not_importance(num_train_layers,batch_size)

            return log_prob, [self.losses['importance']['total'],self.losses['hamming']['total'],self.losses['sparsity']['total'],self.losses['not_importance']['total']]
        else:
            return log_prob, [None,None,None,None]


