import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet
import math


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class SelfAttention(nn.Module):
    def __init__(self, in_dim, num_hid, dropout=0.2):
        super(SelfAttention, self).__init__()

        self.input_proj = FCNet([in_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, input_vec):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(input_vec)
        # w = nn.functional.softmax(logits, 1)
        # return w
        return logits

    def logits(self, input_vec):
        batch, k, _ = input_vec.size()
        input_proj = self.input_proj(input_vec)  # [batch, k, qdim]
        input_repr = self.dropout(input_proj)
        logits = self.linear(input_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, 1)
        # return w
        return logits

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


class TestAttention(nn.Module):
    def __init__(self, in_dim, dropout):
        super(TestAttention, self).__init__()
        self.hidden_size = in_dim
        self.num_attention_heads = 12
        self.attention_heads_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_heads_size
        self.ctx_dim = in_dim

        self.query = nn.Linear(in_dim, self.all_head_size)
        self.key = nn.Linear(self.ctx_dim, self.all_head_size)
        self.vlaue = nn.Linear(self.ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_heads_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_features, context):
        mixed_query_layer = self.query(input_features)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.vlaue(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_heads_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class CrossAttention(nn.Module):
    def __init__(self, in_dim, dropout):
        super(CrossAttention, self).__init__()
        self.att = TestAttention(in_dim, dropout)
        self.att_output = nn.Sequential(
            nn.Linear(1020, in_dim),
            nn.LayerNorm(in_dim, eps=1e-12),
            nn.Dropout(dropout)
        )

    def cross_att(self, text_input, visn_input):
        lang_att_output = self.att(text_input, visn_input)
        visn_att_output = self.att(visn_input, text_input)
        lang_att_output = self.att_output(lang_att_output)
        visn_att_output = self.att_output(visn_att_output)
        return lang_att_output, visn_att_output

    def self_att(self, text_input, visn_input):
        lang_att_output = self.att(text_input, text_input)
        visn_att_output = self.att(visn_input, visn_input)
        lang_att_output = self.att_output(lang_att_output)
        visn_att_output = self.att_output(visn_att_output)
        return lang_att_output, visn_att_output

    def forward(self, text_feature, visn_feature, key=None):
        text_att_output = text_feature
        visn_att_output = visn_feature
        # text_att_output, visn_att_output = self.self_att(text_att_output, visn_att_output)
        if key is None:
            text_att_output, visn_att_output = self.cross_att(text_att_output, visn_att_output)
            text_att_output, visn_att_output = self.self_att(text_att_output, visn_att_output)
        else:
            text_att_output, visn_att_output = self.self_att(text_att_output, visn_att_output)
        return text_att_output, visn_att_output
