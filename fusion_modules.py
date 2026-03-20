import torch
import torch.nn as nn
import pdb
from torch.nn.utils.weight_norm import weight_norm


class NewFusion(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.2):
        super(NewFusion, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            # nn.Dropout(dropout, inplace=True),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.fc_out = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.fc_out(x)
        return logits


class NewConCatFusion(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(NewConCatFusion, self).__init__()
        layers = [
            nn.Dropout(dropout),
            weight_norm(nn.Linear(in_dim, out_dim), dim=None)
        ]
        self.fc_out = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.fc_out(x)
        return logits


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = weight_norm(nn.Linear(input_dim, output_dim),dim=None)
        self.fc_y = weight_norm(nn.Linear(input_dim, output_dim),dim=None)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output


class ConcatFusion3(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(ConcatFusion3, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y, z):
        output = torch.cat((x, y, z), dim=1)

        output = self.fc_out(output)
        return x, y, z, output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = weight_norm(nn.Linear(input_dim, 2 * dim),dim=None)
        self.fc_out = weight_norm(nn.Linear(dim, output_dim),dim=None)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = weight_norm(nn.Linear(input_dim, dim),dim=None)
        self.fc_y = weight_norm(nn.Linear(input_dim, dim),dim=None)
        self.fc_out = weight_norm(nn.Linear(dim, output_dim),dim=None)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output