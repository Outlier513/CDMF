import torch
import torch.nn as nn


class MCD(nn.Module):
    def __init__(self, student_num, exercise_num, knowledge_num) -> None:
        super(MCD, self).__init__()
        self.student_embedding = nn.Embedding(student_num, knowledge_num)
        self.exercise_embedding = nn.Embedding(exercise_num, knowledge_num)
        self.response = nn.Linear(2*knowledge_num, 1)

    def forward(self, student_id, exercise_id):
        student = self.student_embedding(student_id)
        exercise = self.exercise_embedding(exercise_id)
        response = self.response(torch.cat([student, exercise], dim=-1))
        response = torch.sigmoid(response).squeeze(-1)
        return response
        
