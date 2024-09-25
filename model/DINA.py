import torch
import numpy as np
import torch.nn as nn


class DINA(nn.Module):
    def __init__(self, student_num, problem_num, hidden_dim, max_slip=0.4, max_guess=0.4) -> None:
        super(DINA, self).__init__()
        self.max_slip = max_slip
        self.max_guess = max_guess
        self.step = 0
        self.max_step = 1000
        self.guess = nn.Embedding(problem_num, 1)
        self.slip = nn.Embedding(problem_num, 1)
        self.theta = nn.Embedding(student_num, hidden_dim)

    def forward(self, student_id, problem_id, knowledge_emb):
        theta = self.theta(student_id)
        slip = torch.sigmoid(self.slip(problem_id)*self.max_slip).squeeze(-1)
        guess = torch.sigmoid(self.guess(problem_id)*self.max_guess).squeeze(-1)
        if self.training:
            n = torch.sum(knowledge_emb*(torch.sigmoid(theta)-0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                               1e-6), self.step + 1 if self.step < self.max_step else 0
            return torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(
                    torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            n = torch.prod(knowledge_emb * (theta >= 0) +
                           (1 - knowledge_emb), dim=1)
            return (1-slip)**n*guess**(1-n)
