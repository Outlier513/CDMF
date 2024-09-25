import torch
import torch.nn as nn
import torch.nn.functional as F


class MIRT(nn.Module):
    def __init__(self, student_num, exercise_num, knowledge_num) -> None:
        super(MIRT, self).__init__()
        self.theta = nn.Embedding(student_num, knowledge_num)
        self.a = nn.Embedding(exercise_num, knowledge_num)
        self.b = nn.Embedding(exercise_num, 1)

    def forward(self, student_id, exercise_id):
        theta = self.theta(student_id).squeeze(-1)
        a = self.a(exercise_id).squeeze(-1)
        a = torch.sigmoid_(a)
        #a = F.softplus(a)
        b = self.b(exercise_id).squeeze(-1)
        return 1 / (1 + torch.exp(- torch.sum(torch.multiply(a, theta), axis=-1) + b))
