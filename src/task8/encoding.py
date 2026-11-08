"""Turn dataframe rows into token-id lists, and token-id lists into padded batches.

Input layout for encoder models (XLM-R, mDeBERTa, ...):

    <cls> comment <sep> title | description <sep>

The comment comes FIRST so that, when the text is too long, only the description is cut.
Decoder LLMs (Qwen) get a plain-text template instead:  "Video: <context>\nComment: <comment><eos>".
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig

from .data import context_text

# Model families whose pair separator is two </s> tokens (RoBERTa-style).
ROBERTA_LIKE = ("xlm-roberta", "roberta", "camembert", "xlm-roberta-xl")


def load_tokenizer(name: str):
    """Fast tokenizer if possible; otherwise the slow SentencePiece one (mDeBERTa's fast tokenizer needs a
    protobuf version that some environments do not have)."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    except Exception:  # noqa: BLE001
        tok = AutoTokenizer.from_pretrained(name, use_fast=False)
    if tok.pad_token_id is None:  # decoder LLMs have no pad token
        tok.pad_token = tok.eos_token
    return tok


def build_encodings(df: pd.DataFrame, tok, cfg: dict) -> List[List[int]]:
    """Return one list of token ids per row of `df`."""
    use_context = cfg["context"] != "none"
    max_len = cfg["max_len"]

    # 1) tokenise comment and context separately, without special tokens
    comment_max = min(cfg["max_comment_len"], max_len - 8)  # keep room for special tokens + some context
    comments = df["yt_comment"].astype(str).str.strip().tolist()
    comment_ids = tok(comments, add_special_tokens=False, truncation=True, max_length=comment_max)["input_ids"]
    if use_context:
        contexts = [context_text(row, cfg["context"]) for _, row in df.iterrows()]
        context_ids = tok(contexts, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    else:
        context_ids = [[] for _ in comments]

    # 2) glue them together with the special tokens of the model family
    if cfg["arch"] == "decoder":
        return _glue_decoder(tok, comment_ids, context_ids, use_context, max_len)
    return _glue_encoder(tok, cfg, comment_ids, context_ids, use_context, max_len)


def _glue_encoder(tok, cfg, comment_ids, context_ids, use_context, max_len):
    # Written by hand (not tok.build_inputs_with_special_tokens) so it works in every transformers version.
    model_type = AutoConfig.from_pretrained(cfg["backbone"]).model_type
    cls, sep = tok.cls_token_id, tok.sep_token_id
    middle = [sep, sep] if model_type in ROBERTA_LIKE else [sep]
    out = []
    for c, x in zip(comment_ids, context_ids):
        if not use_context:
            out.append([cls] + c[: max_len - 2] + [sep])
        else:
            room_for_context = max_len - len(c) - (2 + len(middle))
            out.append([cls] + c + middle + x[: max(room_for_context, 0)] + [sep])
    return out


def _glue_decoder(tok, comment_ids, context_ids, use_context, max_len):
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    video_prefix = tok("Video: ", add_special_tokens=False)["input_ids"]
    comment_prefix = tok("\nComment: ", add_special_tokens=False)["input_ids"]
    out = []
    for c, x in zip(comment_ids, context_ids):
        room = max_len - len(c) - len(video_prefix) - len(comment_prefix) - 1
        head = (video_prefix + x[: max(room, 0)]) if use_context else []
        out.append(head + comment_prefix + c + [eos])
    return out


def make_items(df: pd.DataFrame, encodings: List[List[int]], with_labels: bool) -> List[dict]:
    """One dict per example: token ids (+ the four label arrays when training)."""
    items = []
    y_identity = np.stack(df["y_i"].values) if with_labels else None
    for k, ids in enumerate(encodings):
        item = {"ids": ids}
        if with_labels:
            item.update(y_a=int(df["y_a"].iat[k]), y_b=int(df["y_b"].iat[k]),
                        y_t=int(df["y_t"].iat[k]), y_i=y_identity[k])
        items.append(item)
    return items


def collate(items: List[dict], pad_id: int) -> dict:
    """Right-pad a list of items into tensors."""
    longest = max(len(it["ids"]) for it in items)
    input_ids = torch.full((len(items), longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(items), longest), dtype=torch.long)
    for k, it in enumerate(items):
        input_ids[k, : len(it["ids"])] = torch.tensor(it["ids"])
        attention_mask[k, : len(it["ids"])] = 1
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    for key in ("y_a", "y_b", "y_t"):
        if key in items[0]:
            batch[key] = torch.tensor([it[key] for it in items], dtype=torch.long)
    if "y_i" in items[0]:
        batch["y_i"] = torch.tensor(np.stack([it["y_i"] for it in items]), dtype=torch.float32)
    return batch
