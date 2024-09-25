import tensorboardX
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.MIRT import MIRT
from utils.dataloader import generate_dataloader
from utils.utils import save_snapshot, validate, parse, init_log
from model.IRT import IRT
from model.DINA import DINA
from model.NCDM import NeuralCDM
from loguru import logger


def train(model, data_loader, logger, include_knowledge, args):
    writer = tensorboardX.SummaryWriter(log_dir="./logs")
    optimizer = (
        optim.Adam(model.parameters(), lr=args.lr)
        if args.optimizer == "Adam"
        else optim.SGD(model.parameter(), lr=args.lr)
    )
    loss_function = nn.NLLLoss() if args.loss_function == "nll" else nn.BCELoss()
    logger.info(
        f"Learning rate is {args.lr}, optimizer is {args.optimizer}, loss_function is {args.loss_function}"
    )
    logger.info(f"Begining training")
    best_acc = 0.0
    for epoch in range(args.epoch_n):
        model.train()
        train_loss = 0.0
        with tqdm(
            desc=f"Epoch:{epoch+1: 2d}/{args.epoch_n}", total=len(data_loader["train"])
        ) as pbar:
            for i, data in enumerate(data_loader["train"]):
                user_id, exercise_id, knowledge_emb, score = (
                    data[0].to(args.device),
                    data[1].to(args.device),
                    data[2].to(args.device),
                    data[3].to(args.device),
                )
                if include_knowledge:
                    output = model(user_id, exercise_id, knowledge_emb)
                else:
                    output = model(user_id, exercise_id)
                loss = loss_function(output, score)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if hasattr(model, "apply_clipper"):
                    model.apply_clipper()
                train_loss += loss.item()
                avg_loss = train_loss / (i + 1)
                pbar.set_postfix({"train_loss": avg_loss})
                pbar.update()
        writer.add_scalar(
            tag=f"{args.model}/train_loss", scalar_value=avg_loss, global_step=epoch
        )
        logger.info(f"Epoch:{epoch+1: 2d} loss={avg_loss:.6f}")
        acc, mse, roc = validate(model, data_loader["val"], include_knowledge, args)
        logger.info(f"acc={acc:.6f}, mse={mse:.6f} roc={roc:.6f}")
        writer.add_scalar(
            tag=f"{args.model}/val_acc", scalar_value=acc, global_step=epoch
        )
        writer.add_scalar(
            tag=f"{args.model}/val_mse", scalar_value=mse, global_step=epoch
        )
        writer.add_scalar(
            tag=f"{args.model}/val_roc", scalar_value=roc, global_step=epoch
        )
        if acc > best_acc:
            best_acc = acc
            save_snapshot(model, f"checkpoint/{args.dataset}/{args.model}/best_model")
            logger.info(f"Save best model")
        save_snapshot(
            model,
            f"checkpoint/{args.dataset}/{args.model}/model_epoch{str(epoch+1)}",
        )


if __name__ == "__main__":
    args = parse()
    logger = init_log(args)
    logger.info(f"Model is {args.model}")
    include_knowledge = True
    dataset_info = getattr(args, args.dataset)
    student_num = dataset_info["student_num"]
    exercise_num = dataset_info["exercise_num"]
    knowledge_num = dataset_info["knowledge_num"]
    logger.info(
        f"Student_num is {student_num}, Exercise_num is {exercise_num}, Knowledge_num is {knowledge_num}"
    )
    args.__setattr__("knowledge_num", knowledge_num)
    if args.model == "IRT":
        model = IRT(student_num, exercise_num).to(args.device)
        args.loss_function = "bce"
        include_knowledge = False
    elif args.model == "DINA":
        model = DINA(student_num, exercise_num, knowledge_num).to(args.device)
        args.loss_function = "bce"
    elif args.model == "NCD":
        model = NeuralCDM(student_num, exercise_num, knowledge_num).to(args.device)
        args.loss_function = "bce"
    elif args.model == "MIRT":
        model = MIRT(student_num, exercise_num, knowledge_num).to(args.device)
        include_knowledge = False
        args.loss_function = "bce"
    elif args.model == "MCD":
        pass
    data_loader = {
        "train": generate_dataloader(args.train_set, args, include_knowledge),
        "val": generate_dataloader(args.test_set, args, include_knowledge),
    }
    logger.info(
        f"Training set is {len(data_loader['train'])}, Validation set is {len(data_loader['val'])}"
    )
    train(model, data_loader, logger, include_knowledge, args)
