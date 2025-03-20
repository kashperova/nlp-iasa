from typing import Callable, Any

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, BatchEncoding, Trainer, PreTrainedModel

get_special_tokens: Callable[[PreTrainedTokenizer], list[int]] = lambda tokenizer: [
    tokenizer.pad_token_id,
    tokenizer.cls_token_id,
    tokenizer.sep_token_id,
]


def get_tags(entities: list[str], end_tag: bool = False) -> dict[str, int]:
    tags = dict()

    for i, entity in enumerate(entities):
        tags[f"B-{entity}"] = i
        tags[f"I-{entity}"] = i + len(entities)
        if end_tag:
            tags[f"E-{entity}"] = i + len(entities) + len(entities)

    tags["O"] = len(entities) * 2 if end_tag else len(entities) * 3
    tags["Special"] = -100

    return tags


def tokenize_and_align_labels(
    examples: dict[str, Any],
    tokenizer: PreTrainedTokenizer = None,
    cfg: DictConfig = None,
    label2id: dict[str, int] = None,
    auto_max_length: bool = False,
    end_token: bool = False,
) -> BatchEncoding:
    kwargs = {
        "truncation": True,
        "return_offsets_mapping": True,
        "return_overflowing_tokens": True,
    }

    if auto_max_length:
        kwargs["padding"] = "max_length"
    else:
        kwargs.update(
            {
                "padding": True,
                "max_length": cfg.dataset.max_length,
                "stride": cfg.dataset.stride,
            }
        )
    outputs = tokenizer(examples["text"], **kwargs)

    sample_mapping = outputs["overflow_to_sample_mapping"]
    offset_mapping = outputs["offset_mapping"]
    outputs["labels"] = []

    for i in range(len(offset_mapping)):
        sample_index = sample_mapping[i]

        labels = [label2id["O"] for i in range(len(outputs["input_ids"][i]))]

        for label_start, label_end, label in zip(
            examples["starts"][sample_index],
            examples["ends"][sample_index],
            examples["entities"][sample_index],
        ):
            for j in range(len(labels)):
                token_start = offset_mapping[i][j][0]
                token_end = offset_mapping[i][j][1]

                if token_start == label_start:
                    labels[j] = label2id[f"B-{label}"]

                elif token_start > label_start and token_end < label_end:
                    labels[j] = label2id[f"I-{label}"]

                elif end_token and token_end == label_end:
                    labels[j] = label2id[f"E-{label}"]

                elif token_end == label_end:
                    labels[j] = label2id[f"I-{label}"]

        for k, input_id in enumerate(outputs["input_ids"][i]):
            if input_id in get_special_tokens(tokenizer):
                labels[k] = -100

        outputs["labels"].append(labels)

    return outputs


def beam_search(trainer: Trainer, model: PreTrainedModel, dataset, beam_size: int):
    preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in trainer.get_eval_dataloader(dataset):
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask"]
        }
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits  # shape: (batch_size, seq_len, num_labels)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        # take top-k labels for each token
        topk_probs, topk_indices = torch.topk(
            probs, beam_size, dim=-1
        )  # shape: (batch_size, seq_len, beam_width)

        # choose the most probable sequence from beam search
        best_seq = topk_indices[:, :, 0].cpu().numpy()

        preds.extend(best_seq)

    return preds
