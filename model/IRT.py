import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class IRT(nn.Module):
    def __init__(self, student_num, exercise_num):
        super(IRT, self).__init__()
        self.theta = nn.Embedding(student_num, 1)
        self.a = nn.Embedding(exercise_num, 1)
        self.b = nn.Embedding(exercise_num, 1)
        self.c = nn.Embedding(exercise_num, 1)

    def forward(self, student_id, exercise_id):
        theta = self.theta(student_id).squeeze(-1)
        a = self.a(exercise_id).squeeze(-1)
        b = self.b(exercise_id).squeeze(-1)
        c = self.c(exercise_id).squeeze(-1)
        c = torch.sigmoid_(c)
        a = F.softplus(a)
        D = 1.702
        return c + (1 - c) / (1 + torch.exp(-D * a * (theta - b)))
