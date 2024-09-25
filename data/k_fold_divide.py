import json
from tqdm import tqdm
from sklearn.model_selection import KFold
from pathlib import Path


min_log = 15
k = 5
folder_path = Path('train')
folder_path.mkdir(parents=True, exist_ok=True)
folder_path = Path('test')
folder_path.mkdir(parents=True, exist_ok=True)
train_set, test_set = [[] for i in range(k)], [[] for i in range(k)]

with open("log_data.json", encoding="utf8") as f:
    stus = json.load(f)
stus = filter(lambda x: x["log_num"] >= min_log, stus)
stus = list(stus)
kf = KFold(n_splits=k, shuffle=True, random_state=42)
for stu in tqdm(stus):
    user_id = stu["user_id"]
    logs = stu["logs"]
    for i, (train_index, test_index) in enumerate(kf.split(logs)):
        for j, log in enumerate(logs):
            if j in train_index:
                train_set[i].append(
                    {
                        "user_id": user_id,
                        "exer_id": log["exer_id"],
                        "score": log["score"],
                        "knowledge_code": log["knowledge_code"],
                    }
                )
            else:
                test_set[i].append(
                    {
                        "user_id": user_id,
                        "exer_id": log["exer_id"],
                        "score": log["score"],
                        "knowledge_code": log["knowledge_code"],
                    }
                )
for i in range(k):
    with open(f"train/train_set_{i}.json", "w", encoding="utf8") as output_file:
        json.dump(train_set[i], output_file, indent=4, ensure_ascii=False)
    with open(f"test/test_set_{i}.json", "w", encoding="utf8") as output_file:
        json.dump(test_set[i], output_file, indent=4, ensure_ascii=False)