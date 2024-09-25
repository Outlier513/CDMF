import argparse
import pandas as pd
import json
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path

absolue_path_prefix = Path(__file__).resolve().parent

URL_DICT = {
    "assistment-2009-2010-skill": "http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip",
    "assistment-2012-2013-non-skill": "http://base.ustc.edu.cn/data/ASSISTment/2012-2013-data-with-predictions-4-final.zip",
    "assistment-2015": "http://base.ustc.edu.cn/data/ASSISTment/2015_100_skill_builders_main_problems.zip",
    "assistment-2017": "http://base.ustc.edu.cn/data/ASSISTment/anonymized_full_release_competition_dataset_name.zip",
    "junyi": "http://base.ustc.edu.cn/data/JunyiAcademy_Math_Practicing_Log/junyi.rar",
    "KDD-CUP-2010": "http://base.ustc.edu.cn/data/KDD_Cup_2010/",
    "NIPS-2020": "http://base.ustc.edu.cn/data/NIPS2020/",
    "slepemapy.cz": "http://base.ustc.edu.cn/data/slepemapy.cz/",
    "synthetic": "http://base.ustc.edu.cn/data/synthetic/",
    "psychometrics": "http://base.ustc.edu.cn/data/psychometrics/",
    "psy": "http://base.ustc.edu.cn/data/psychometrics/",
    "pisa2015": "http://base.ustc.edu.cn/data/pisa2015_science.zip",
    "workbankr": "http://base.ustc.edu.cn/data/wordbankr.zip",
    "critlangacq": "http://base.ustc.edu.cn/data/critlangacq.zip",
    "ktbd": "http://base.ustc.edu.cn/data/ktbd/",
    "ktbd-a0910": "http://base.ustc.edu.cn/data/ktbd/assistment_2009_2010/",
    "ktbd-junyi": "http://base.ustc.edu.cn/data/ktbd/junyi/",
    "ktbd-synthetic": "http://base.ustc.edu.cn/data/ktbd/synthetic/",
    "ktbd-a0910c": "http://base.ustc.edu.cn/data/ktbd/a0910c/",
    "cdbd": "http://base.ustc.edu.cn/data/cdbd/",
    "cdbd-lsat": "http://base.ustc.edu.cn/data/cdbd/LSAT/",
    "cdbd-a0910": "http://base.ustc.edu.cn/data/cdbd/a0910/",
    "math2015": "http://staff.ustc.edu.cn/~qiliuql/data/math2015.rar",
    "ednet": "http://base.ustc.edu.cn/data/EdNet/",
    "ktbd-ednet": "http://base.ustc.edu.cn/data/ktbd/EdNet/",
    "math23k": "http://base.ustc.edu.cn/data/math23k.zip",
    "OLI-Fall-2011": "http://base.ustc.edu.cn/data/OLI_data.zip",
    "open-luna": "http://base.ustc.edu.cn/data/OpenLUNA/OpenLUNA.json",
}

DATA_DICT = {
    "assistment-2009-2010-skill": "assistment-2009-2010-skill/skill_builder_data_corrected.csv",
    "junyi": "http://base.ustc.edu.cn/data/JunyiAcademy_Math_Practicing_Log/junyi.rar",
}


def decompress(compress_file):
    if compress_file.endswith("zip"):
        with zipfile.ZipFile(compress_file, "r") as zip_ref:
            zip_ref.extractall(Path(compress_file).parent)
    elif compress_file.endwith("rar"):
        pass
    print(f"The file {compress_file} has been downloaded and extracted.")


def download(dataset_name):
    url = URL_DICT[dataset_name]
    file_suffix = url.split(r"/")[-1]
    compress_file = absolue_path_prefix / Path(dataset_name + "/" + file_suffix)
    if not compress_file.parent.exists():
        compress_file.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with open(compress_file, "wb") as file, tqdm(
        desc=f"downloading {file_suffix}",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)
    decompress(str(compress_file))


def process_data(dataset_name):
    if dataset_name == "assistment-2009-2010-skill":
        df = pd.read_csv(absolue_path_prefix / DATA_DICT[dataset_name], encoding="ansi")
        df = df.loc[
            :,
            [
                "order_id",
                "user_id",
                "problem_id",
                "correct",
                "answer_type",
                "skill_id",
            ],
        ]
        df = df[df.answer_type != "open_response"]
        df = df.dropna(subset=["skill_id"])
        df = df.drop("answer_type", axis=1)
        df["user_id"] = pd.factorize(df["user_id"])[0] + 1
        df["problem_id"] = pd.factorize(df["problem_id"])[0] + 1
        df["skill_id"] = pd.factorize(df["skill_id"])[0] + 1
        df["correct"] = df["correct"].astype(float)
        merged_df = (
            df.groupby(["user_id", "problem_id"])
            .agg({"skill_id": list, "correct": "first"})
            .reset_index()
        )
        merged_df["logs"] = merged_df.apply(
            lambda x: {"exer_id": x[1], "score": x[3], "knowledge_code": x[2]}, axis=1
        )
        merged_df = merged_df.drop(["problem_id", "skill_id", "correct"], axis=1)
        merged_df = merged_df.groupby("user_id").agg({"logs": list}).reset_index()
        merged_df["log_num"] = merged_df.apply(lambda x: len(x[1]), axis=1)
        merged_df = merged_df[["user_id", "log_num", "logs"]]
        data_dict = merged_df.to_dict(orient="records")
        with open("log_data.json", "w", encoding="utf-8") as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=4)
        print(f"The dataset {dataset_name} was successfully processed")
    else:
        print("TODO")
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", help="dataset select")
    dataset_name = parser.dataset
    data_file = absolue_path_prefix / Path(DATA_DICT[dataset_name])
    if not data_file.exists():
        print(f"The dataset {dataset_name} does not exist, start to download")
        download(dataset_name)
    else:
        print(f"The dataset {dataset_name} already exists")
    process_data(dataset_name)
