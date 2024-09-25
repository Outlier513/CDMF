from torch.utils.data import Dataset, DataLoader
from functools import partial
import json
import torch


class CognitiveDiagnosisDataset(Dataset):
    def __init__(self, record_file) -> None:
        super().__init__()
        with open(record_file, "r") as rf:
            self.data = json.load(rf)
            self.len = len(self.data)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index):
        item = self.data[index]
        user_id = item["user_id"]
        exer_id = item["exer_id"]
        score = item["score"]
        knowledge_code = item["knowledge_code"]
        return user_id, exer_id, knowledge_code, score


def collate_fn(batch, include_knowledge, knowledge_num):
    user_batch, exercise_batch, knowledge_batch, score_batch = [], [], [], []
    for user_id, exercise_id, knowledge_code, score in batch:
        user_batch.append(user_id)
        exercise_batch.append(exercise_id)
        score_batch.append(score)
        if include_knowledge:
            knowledge_emb = [0.0] * knowledge_num
            for c in knowledge_code:
                knowledge_emb[c - 1] = 1.0
            knowledge_batch.append(knowledge_emb)
    return (
        torch.tensor(user_batch, dtype=torch.int32)-1,
        torch.tensor(exercise_batch, dtype=torch.int32)-1,
        torch.tensor(knowledge_batch, dtype=torch.float32),
        torch.tensor(score_batch, dtype=torch.float32),
    )


def generate_dataloader(filename, args, include_knowledge=True):
    json_file = "data/" + args.dataset + "/" + filename
    dataset = CognitiveDiagnosisDataset(json_file)
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(
                collate_fn, include_knowledge=include_knowledge, knowledge_num=args.knowledge_num),
            num_workers=args.num_workers,
        )
    return dataloader
