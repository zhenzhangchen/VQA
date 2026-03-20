import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

# class SimpleClassifier(nn.Module):
#     def __init__(self, in_dim, hid_dim, dropout=0.5):
#         super(SimpleClassifier, self).__init__()
#         layers = [
#             weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
#             nn.ReLU(),
#             # nn.Dropout(dropout, inplace=True),
#             nn.Dropout(dropout),
#         ]
#         self.main = nn.Sequential(*layers)

#     def forward(self, x):
#         logits = self.main(x)
#         return logits


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.5):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            # nn.Dropout(dropout, inplace=True),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class SimpleLinearNet(nn.Module):
    def __init__(self):
        super(SimpleLinearNet, self).__init__()
        layers = [
            weight_norm(nn.Linear(36, 1), dim=None),
            nn.ReLU(),
            # nn.Dropout(0.25),
            weight_norm(nn.Linear(1, 2274), dim=None),
            # weight_norm(nn.Linear(1, 2410), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class VisualEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout):
        super(VisualEncoder, self).__init__()
        self.visn_fc = nn.Linear(in_dim, hid_dim)
        self.visn_layer_norm = nn.LayerNorm(hid_dim, eps=1e-12)

        self.box_fc = nn.Linear(7, hid_dim)
        self.box_layer_norm = nn.LayerNorm(hid_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, spatial):
        x = self.visn_fc(feats) # [512,36,2048]->[512,36,1024]
        x = self.visn_layer_norm(x)
        y = self.box_fc(spatial) # [512,36,7]->[512,36,1024]
        y = self.box_layer_norm(y)
        out = (x + y) / 2
        out = self.dropout(out)
        return out
