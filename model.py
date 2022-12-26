import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Encoder, self).__init__()
        self.conv_1st = nn.Conv2d(3, 32, 3) # (32-3)/1+1 = 30
        self.conv_2nd = nn.Conv2d(32, 32, 3) # (30-3)/1+1 = 28
        self.conv_3rd = nn.Conv2d(32, 64, 3) # (28-3)/1+1 = 26
        self.conv_4th = nn.Conv2d(64, 64, 3) # (26-3)/1+1 = 24
        self.conv_5th = nn.Conv2d(64, 128, 3) # (24-3)/1+1 = 22
        self.conv_6th = nn.Conv2d(128, 128, 3) # (22-3)/1+1 = 20
        self.flat = nn.Flatten()
        self.hidden = nn.Linear(128*20*20, hidden_dim)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.batch_norm_3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_1st(x)
        x = self.conv_2nd(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        x = self.conv_3rd(x)      
        x = self.conv_4th(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        x = self.conv_5th(x)
        x = self.conv_6th(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)
        x = self.flat(x)
        x = self.relu(x)
        x = self.hidden(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Decoder, self).__init__()
        self.convt_1st = nn.ConvTranspose2d(128, 128, 3)
        self.convt_2nd = nn.ConvTranspose2d(128, 64, 3)
        self.convt_3rd = nn.ConvTranspose2d(64, 64, 3)
        self.convt_4th = nn.ConvTranspose2d(64, 32, 3)
        self.convt_5th = nn.ConvTranspose2d(32, 32, 3)
        self.convt_6th = nn.ConvTranspose2d(32, 3, 3)
        self.hidden = nn.Linear(hidden_dim, 128*20*20)
        self.unflat = nn.Unflatten(dim=1, unflattened_size=(128, 20, 20))
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.unflat(x)
        x = self.convt_1st(x)
        x = self.convt_2nd(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        x = self.convt_3rd(x)
        x = self.convt_4th(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        x = self.convt_5th(x)
        x = self.convt_6th(x)
        x = self.relu(x)
        return x

class ClassifModel(nn.Module):
    def __init__(self, input_dim):
        super(ClassifModel, self).__init__()
        self.fc_1st = nn.Linear(input_dim, 1024)
        self.fc_2nd = nn.Linear(1024, 1024)
        self.fc_3rd = nn.Linear(1024, 10)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc_1st(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc_2nd(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc_3rd(x)
        return x

class FinalModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(FinalModel, self).__init__()
        self.enc = encoder
        self.dec = decoder
        
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x
        