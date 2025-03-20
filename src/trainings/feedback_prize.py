import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.cuda
import wandb
from datasets import Dataset
from omegaconf import DictConfig
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    ModernBertForTokenClassification,
)

from utils.ner_utils import get_tags, tokenize_and_align_labels, beam_search
from utils.hf_utils import compute_ner_metrics, SaveArtifactCallback


def prepare_df(df: pd.DataFrame, txt_path: str):
    # group labels by txt file
    df1 = df.groupby("id")["discourse_type"].apply(list).reset_index(name="entities")
    df2 = df.groupby("id")["discourse_start"].apply(list).reset_index(name="starts")
    df3 = df.groupby("id")["discourse_end"].apply(list).reset_index(name="ends")
    df4 = (
        df.groupby("id")["predictionstring"]
        .apply(list)
        .reset_index(name="predictionstrings")
    )

    df = pd.merge(df1, df2, how="inner", on="id")
    df = pd.merge(df, df3, how="inner", on="id")
    df = pd.merge(df, df4, how="inner", on="id")

    path = Path(txt_path)
    read_txt = lambda x: open(path / f"{x}.txt", "r").read()
    df["text"] = df["id"].apply(read_txt)

    return df


# TODO: class weights to cross entropy
# TODO: end of entity token tagging option
def train(cfg: DictConfig):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_df = pd.read_csv(cfg.dataset.csv_path)
    entities = train_df.discourse_type.unique().tolist()

    label2id = get_tags(entities)
    id2label = {v: k for k, v in label2id.items()}

    num_labels = len(id2label) - 1
    train_df = prepare_df(df=train_df, txt_path=cfg.dataset.txt_path)

    hf_ds = Dataset.from_pandas(train_df)
    datasets = hf_ds.train_test_split(
        test_size=cfg.dataset.test_size, shuffle=True, seed=cfg.trainer.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_path, add_prefix_space=True
    )
    tokenized_datasets = datasets.map(
        partial(
            tokenize_and_align_labels,
            cfg=cfg,
            label2id=label2id,
            tokenizer=tokenizer,
            auto_max_length=cfg.trainer.auto_max_length,
            end_token=cfg.trainer.end_token,
        ),
        batched=True,
        batch_size=20000,
        remove_columns=datasets["train"].column_names,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model.checkpoint_path, num_labels=num_labels
    )
    model.to(device)

    if isinstance(model, ModernBertForTokenClassification):
        if cfg.model.freeze_emb:
            for p in model.model.embeddings.parameters():
                p.requires_grad = False

        for idx, layer in enumerate(model.model.layers):
            if idx in range(0, cfg.model.freeze_attn):
                for p in layer.parameters():
                    p.requires_grad = False

    compute_metrics = partial(compute_ner_metrics, id2label=id2label)
    trainer = Trainer(
        model=model,
        args=cfg.trainer.hf_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            SaveArtifactCallback(model_name=cfg.trainer.name, save_interval=1000)
        ],
    )

    trainer.train()
    wandb.finish()

    trainer.save_model(cfg.trainer.save_final_path)


def get_pred_string(
    pred: np.array,
    example: dict[str, Any],
    test_txt_path: str,
    id2label: dict[int, str],
    num_entities: int,
):
    example_id = example["id"]
    n_tokens = len(example["input_ids"])

    l2id = {v: k for k, v in id2label.items()}
    print(f"{id2label.keys()}")
    def get_class(x):
        print(f"X: {x}, Type: {type(x)}")
        if x != 14 and x != "14":
           return id2label[x][2:]
        else:
            return "Other"

    # get_class = (
    #     lambda x: id2label[x][2:] if x != 14 and x != "14" else "Other"
    # )  # remove B-, I-

    entities = []
    all_span = []
    cur_span = None

    for i, c in enumerate(pred.tolist()):
        if i == n_tokens - 1:
            break
        if i == 0:
            cur_span = example["offset_mapping"][i]
            entities.append(get_class(c))

        elif i > 0 and (
            c == pred[i - 1] or (c - num_entities) == pred[i - 1]
        ):  # beginning
            cur_span[1] = example["offset_mapping"][i][1]

        else:
            all_span.append(cur_span)
            cur_span = example["offset_mapping"][i]
            entities.append(get_class(c))

    all_span.append(cur_span)

    text = open(Path(test_txt_path) / f"{example_id}.txt", "r").read()

    # map token ids to word (split by whitespace) ids
    pred_strings = []
    for span in all_span:
        span_start, span_end = span[0], span[1]
        before = text[:span_start]

        word_start = len(before.split())
        if before[-1] != " ":
            word_start -= 1

        num_words = len(text[span_start : span_end + 1].split())
        word_ids = [str(x) for x in range(word_start, word_start + num_words)]
        pred_strings.append(" ".join(word_ids))

    rows = []
    for e, span, pred_string in zip(entities, all_span, pred_strings):
        row = {
            "id": example_id,
            "discourse_type": e,
            "predictionstring": pred_string,
            "discourse_start": span[0],
            "discourse_end": span[1],
            "discourse": text[span[0] : span[1] + 1],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def run_inference(
    test_txt_path: str,
    train_csv_path: str,
    checkpoint_path: str,
    max_length: int,
    min_tokens: int,
    beam_size: int = None,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"

    train_df = pd.read_csv(train_csv_path)
    entities = train_df.discourse_type.unique().tolist()

    id2label = {v: k for k, v in get_tags(entities).items()}

    path = Path(test_txt_path)
    read_txt = lambda x: open(path / f"{x}.txt", "r").read()
    test_df = pd.DataFrame()
    test_df["id"] = [x.split(".")[0] for x in os.listdir(test_txt_path)]
    test_df["text"] = test_df["id"].apply(read_txt)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, add_prefix_space=True)
    tokenize = lambda x: tokenizer(
        x["text"], truncation=True, return_offsets_mapping=True, max_length=max_length
    )
    tokenized_test = test_ds.map(tokenize)

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint_path, num_labels=len(id2label) - 1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if beam_size is not None:
        preds = beam_search(
            trainer=trainer, model=model, dataset=tokenized_test, beam_size=beam_size
        )
    else:
        preds, _, _ = trainer.predict(tokenized_test)
        preds = np.argmax(preds, axis=-1)

    submission_data = []
    for i in range(len(tokenized_test)):
        df = get_pred_string(
            preds[i],
            tokenized_test[i],
            test_txt_path,
            id2label,
            num_entities=len(entities),
        )
        df["len"] = df["discourse"].apply(lambda t: len(t.split()))

        # remove very short discourses (likely false positives)
        df = df[df.length > min_tokens].reset_index(drop=True)

        submission_data.append(df)

    submission_df = pd.concat(submission_data, axis=0)
    submission_df["class"] = submission_df["discourse_type"]
    submission_df = submission_df[["id", "class", "predictionstring"]]
    submission_df.to_csv("submission.csv", index=False)
