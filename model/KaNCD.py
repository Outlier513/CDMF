import torch
import torch.nn as nn
import torch.nn.functional as F
from NCDM import PosLinear


class KaNCD(nn.Module):
    def __init__(
        self,
        student_num: int,
        exercise_num: int,
        knowledge_num: int,
        emb_dim: int,
        mf_type: str,
        hid_dim1: int = 512,
        hid_dim2: int = 256,
        p=0.5,
    ) -> None:
        """
        Parameters
        ----------
        student_num : int
            the number of student
        exercise_num : int
            the number of exercise
        knowledge_num : int
            the number of knowledge concept
        emb_dim : int
            the dimension of embedding
        mf_type : str
            the type of method
        hid_dim1 : int
            hidden layer dimension, by defaults 512
        hid_dim2 : int
            hidden layer dimension, by defaults 256
        p : float
            dropout rate, by default 0.5
        """
        super().__init__()
        self.stu_emb = nn.Embedding(student_num, emb_dim)
        self.exer_emb = nn.Embedding(exercise_num, emb_dim)
        self.e_disc = nn.Embedding(exercise_num, 1)
        self.know_emb = nn.Parameter(torch.zeros(knowledge_num, emb_dim))
        if mf_type == "gmf":
            self.k_diff = nn.Linear(emb_dim, 1)
            self.stat = nn.Linear(emb_dim, 1)
        elif mf_type == "ncf1":
            self.k_diff = nn.Linear(2 * emb_dim, 1)
            self.stat = nn.Linear(2 * emb_dim, 1)
        elif mf_type == "ncf2":
            self.k_diff1 = nn.Linear(2 * emb_dim, emb_dim)
            self.k_diff2 = nn.Linear(emb_dim, 1)
            self.stat1 = nn.Linear(2 * emb_dim, emb_dim)
            self.stat2 = nn.Linear(emb_dim, 1)
        self.fc1 = PosLinear(knowledge_num, hid_dim1)
        self.drop_1 = nn.Dropout(p)
        self.fc2 = PosLinear(hid_dim1, hid_dim2)
        self.drop_2 = nn.Dropout(p)
        self.fc3 = PosLinear(hid_dim2, 1)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(
        self,
        stu_id: torch.LongTensor,
        exer_id: torch.LongTensor,
        kn_emb: torch.LongTensor,
    ) -> torch.Tensor:
        stu_emb = self.stu_emb(stu_id)
        exer_emb = self.exer_emb(exer_id)
        batch, emb_dim = stu_emb.shape
        
