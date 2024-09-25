import sys
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score


def validate(model, data_loader, include_knowledge, args):
    model.eval()
    pred_total, score_total, output_total = np.array([]), np.array([]), np.array([])
    with tqdm(desc=f"Validate", total=len(data_loader)) as pbar:
        for i, data in enumerate(data_loader):
            user_id, exercise_id, knowledge_emb, score = (
                data[0].to(args.device),
                data[1].to(args.device),
                data[2].to(args.device),
                data[3],
            )
            if include_knowledge:
                output = model(user_id, exercise_id, knowledge_emb)
            else:
                output = model(user_id, exercise_id)
            if output.ndim > 1:
                output = output.view(-1)
            output = output.to("cpu")
            pred = torch.where(output > 0.5, 1.0, 0.0)
            output_total = np.hstack((output_total, output.detach().numpy()))
            pred_total = np.hstack((pred_total, pred))
            score_total = np.hstack((score_total, np.array(score)))
            acc = accuracy_score(score_total, pred_total)
            rmse = np.sqrt(mean_squared_error(score_total, output_total))
            roc = roc_auc_score(score_total, output_total)
            pbar.set_postfix(
                {
                    "acc": acc,
                    "rmse": rmse,
                    "roc": roc,
                }
            )
            pbar.update()
    return acc, rmse, roc


def save_snapshot(model, filename):
    parent_folder = Path(filename).parent
    parent_folder.mkdir(parents=True, exist_ok=True)
    with open(file=f"{filename}.pth", mode="wb") as f:
        torch.save(model.state_dict(), f)


def load_snapshot(model, filename):
    with open(filename, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--device", help="the device model train, val or test", default="cuda:0"
    )
    parser.add_argument("-c", "--config", help="config file", default="config.yaml")
    parser.add_argument("-m", "--model", help="model select")
    parser.add_argument("-ds", "--dataset", help="dataset select")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        data = yaml.safe_load(f)
    for k, v in data.items():
        if not hasattr(args, k):
            args.__setattr__(k, v)
    return args


def init_log(args):
    logger.remove()
    if args.is_print:
        logger.add(sink=sys.stderr, level=args.logging_level)
    logger.add(
        sink=f"logs/traning-{args.model}-{args.dataset}.log",
        level=args.logging_level,
        rotation="1 hours",
    )
    return logger
