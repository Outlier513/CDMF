import argparse
import pandas as pd
import json
import requests
import zipfile
import rarfile
from tqdm import tqdm
from pathlib import Path


absolue_path_prefix = Path(__file__).resolve().parent

URL_DICT = {
    "assistment-2009-2010-skill": "http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip",
    "junyi": "http://base.ustc.edu.cn/data/JunyiAcademy_Math_Practicing_Log/junyi.rar",
}

DATA_DICT = {
    "assistment-2009-2010-skill": "assistment-2009-2010-skill/skill_builder_data_corrected.csv",
    "junyi": "junyi/junyi_ProblemLog_original.csv",
}


def min_log_filter(df, min_log):
    log_counts = df["user_id"].value_counts()
    valid_users = log_counts[log_counts >= min_log].index
    filtered_df = df[df["user_id"].isin(valid_users)]
    return filtered_df


def encode_id(df):
    df.loc[:, "user_id"] = pd.factorize(df["user_id"])[0] + 1
    df.loc[:, "problem_id"] = pd.factorize(df["problem_id"])[0] + 1
    flat_list = [item for sublist in df["skill_id"] for item in sublist]
    unique_values = pd.Series(flat_list).unique()
    value_mapping = {val: idx for idx, val in enumerate(unique_values)}
    df.loc[:, "skill_id"] = df["skill_id"].apply(
        lambda x: [value_mapping[val] for val in x]
    )
    return df, len(unique_values)


def print_ds_detail(df, num_concept):
    num_student = df["user_id"].nunique()
    num_exercise = df["problem_id"].nunique()
    num_response = len(df)
    e = df.drop_duplicates(subset=["problem_id"])
    sum = 0
    for concepts in e["skill_id"]:
        sum += len(concepts)
    concepts_per_exercise = sum / len(e)
    response_per_student = len(df) / num_student
    print(
        num_student,
        num_exercise,
        num_concept,
        num_response,
        concepts_per_exercise,
        response_per_student,
    )


def decompress(compress_file, dataset_name):
    output_path = Path(compress_file).parent
    if compress_file.endswith("zip"):
        with zipfile.ZipFile(compress_file, "r") as zip_ref:
            zip_ref.extractall(output_path)
    elif compress_file.endswith("rar"):
        with rarfile.RarFile(compress_file, "r") as rar_ref:
            if dataset_name == "junyi":
                rar_ref.extract(
                    member="junyi_ProblemLog_original.csv", path=output_path
                )
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
    decompress(str(compress_file), dataset_name)


def process_data(dataset_name, min_log):
    if dataset_name == "assistment-2009-2010-skill":
        df = pd.read_csv(
            absolue_path_prefix / DATA_DICT[dataset_name],
            encoding="ansi",
            low_memory=False,
        )
        df = df[df.answer_type != "open_response"]
        df = df.dropna(subset=["skill_id"])
        df = df.drop("answer_type", axis=1)
        df = df.sort_values(by="order_id")
        df.loc[:, "correct"] = df["correct"].astype(float)
        merged_df = (
            df.groupby(["user_id", "problem_id"])
            .agg({"skill_id": list, "correct": "first"})
            .reset_index()
        )
        filtered_df = min_log_filter(merged_df, min_log)

    elif dataset_name == "junyi":
        df = pd.read_csv(
            absolue_path_prefix / DATA_DICT[dataset_name],
            encoding="utf-8",
            low_memory=False,
        )
        df = df[df["problem_number"] == 1]
        df = df[df["count_attempts"] == 1]
        df.loc[:, "correct"] = df["correct"].astype(float)
        df = df.rename(columns={"exercise": "problem_id"})
        df.loc[:, "skill_id"] = df["problem_id"].apply(lambda x: [x])
        filtered_df = min_log_filter(df, min_log=min_log)
        random_user_ids = filtered_df['user_id'].drop_duplicates().sample(n=10000, random_state=42)
        filtered_df = filtered_df[filtered_df['user_id'].isin(random_user_ids)]
    else:
        print("TODO")

    filtered_df, num_concept = encode_id(filtered_df)
    print_ds_detail(filtered_df, num_concept)
    filtered_df = filtered_df.loc[
        :,
        ["user_id", "problem_id", "skill_id", "correct"],
    ]
    filtered_df.loc[:, "logs"] = filtered_df.apply(
        lambda x: {"exer_id": x[1], "score": x[3], "knowledge_code": x[2]}, axis=1
    )
    result_df = (
        filtered_df.groupby("user_id")
        .agg(
            log_num=("logs", "count"),
            logs=("logs", list),
        )
        .reset_index()
    )
    data_dict = result_df.to_dict(orient="records")
    with open(
        absolue_path_prefix / f"{dataset_name}/log_data.json", "w", encoding="utf-8"
    ) as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)
    print(f"The dataset {dataset_name} was successfully processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset_name", required=True)
    parser.add_argument("-m", "--min_log", default=15)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    min_log = args.min_log
    data_file = absolue_path_prefix / Path(DATA_DICT[dataset_name])
    if not data_file.exists():
        print(f"The dataset {dataset_name} does not exist, start to download")
        download(dataset_name)
    else:
        print(f"The dataset {dataset_name} already exists")
    process_data(dataset_name, min_log)
