import torch
import torch.nn as nn
import torch.nn.functional as F


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * torch.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class NeuralCDM(nn.Module):
    def __init__(
        self,
        student_num: int,
        exercise_num: int,
        knowledge_num: int,
        hid_dim1=512,
        hid_dim2=256,
        p=0.5,
    ) -> None:
        """
        :param int student_num: the number of student
        :param int exercise_num: the number of exercise
        :param int knowledge_num: the number of knowledge concept
        :param int hid_dim1: hidden layer dimension, defaults to 512
        :param int hid_dim2: hidden layer dimension, defaults to 256
        """
        super(NeuralCDM, self).__init__()
        self.stu_emb = nn.Embedding(student_num, knowledge_num)
        self.k_difficulty = nn.Embedding(exercise_num, knowledge_num)
        self.e_discrimination = nn.Embedding(exercise_num, 1)
        self.fc_1 = PosLinear(knowledge_num, hid_dim1)
        self.drop_1 = nn.Dropout(p)
        self.fc_2 = PosLinear(hid_dim1, hid_dim2)
        self.drop_2 = nn.Dropout(p)
        self.fc_3 = PosLinear(hid_dim2, 1)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(
        self,
        stu_id: torch.LongTensor,
        exer_id: torch.LongTensor,
        kn_emb: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        :param torch.LongTensor stu_id: student id
        :param torch.LongTensor exer_id: exercies id
        :param torch.FloatTensor kn_emb: knowledge concept embedding
        """
        stu_emb = self.stu_emb(stu_id)
        stu_emb = torch.sigmoid(stu_emb)
        k_difficulty = self.k_difficulty(exer_id)
        k_difficulty = torch.sigmoid(k_difficulty)
        e_discrimination = self.e_discrimination(exer_id)
        e_discrimination = torch.sigmoid(e_discrimination)
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.fc_1(input_x)
        input_x = torch.sigmoid_(input_x)
        input_x = self.drop_1(input_x)
        input_x = self.fc_2(input_x)
        input_x = torch.sigmoid_(input_x)
        input_x = self.drop_2(input_x)
        input_x = self.fc_3(input_x)
        output = torch.sigmoid_(input_x)
        return output.view(-1)


    def get_knowledge_status(self, stu_id: int) -> torch.Tensor:
        """return the status of student

        :param int stu_id: student id
        """
        stat_emb = self.stu_emb(stu_id)
        stat_emb = torch.sigmoid_(stat_emb)
        return stat_emb.data

    def get_exer_params(self, exer_id: int):
        """return the params of exercise

        :param int exer_id: exercise id
        """
        k_difficulty = self.k_difficulty(exer_id)
        k_difficulty = torch.sigmoid_(k_difficulty)
        e_discrimination = self.e_discrimination(exer_id)
        e_discrimination = torch.sigmoid_(e_discrimination) * 10
        return k_difficulty.data, e_discrimination.data
