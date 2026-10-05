"""
src/model.py - BERT + BiLSTM + softmax word-level LID tagger (Fig. 9).

Forward pass, with shapes (B = batch, L = subtokens, W = words,
H = 768 for base/mBERT, K = number of tags):

  input_ids (B, L) --BERT--> subtoken vectors (B, L, H)
                   --pool to words--> word vectors (B, W, H)
                     pooling="first": vector of each word's first subtoken
                     pooling="mean" : average of all subtokens of the word
                   --dropout--> BiLSTM (packed) --> (B, W, 2*lstm_hidden)
                   --dropout--> Linear --> logits (B, W, K)
Softmax is applied inside CrossEntropyLoss during training and via argmax at
prediction time; each word is classified independently (no CRF).
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

from .data import IGNORE_INDEX


class LIDTagger(nn.Module):
    def __init__(self, encoder_name, num_labels, use_bilstm=True,
                 lstm_hidden=256, lstm_layers=1, dropout=0.5, pooling="first"):
        super().__init__()
        if pooling not in ("first", "mean"):
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling
        # Pretrained encoder; all its weights are fine-tuned (paper setting)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size  # 768 (base) / 1024 (large)

        # nn.LSTM's own `dropout` argument only acts BETWEEN stacked layers,
        # so with 1 layer it does nothing (PyTorch even warns). To apply the
        # paper's 0.5 dropout around the recurrent layer, we use an explicit
        # Dropout on the LSTM input and output instead.
        self.dropout = nn.Dropout(dropout)

        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.lstm = nn.LSTM(
                input_size=hidden,
                hidden_size=lstm_hidden,   # 256 per direction
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=True,
            )
            out_dim = 2 * lstm_hidden      # forward ++ backward = 512 (Eq. 15)
        else:
            out_dim = hidden               # "BERT + softmax" ablation

        self.classifier = nn.Linear(out_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # ------------------------------------------------------------------ #
    def forward(self, input_ids, attention_mask, word_starts, word_mask,
                labels=None, word_index=None, **_):
        # 1) Contextual subtoken vectors from BERT: (B, L, H)
        sub = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state

        # 2) One vector per word: (B, W, H)
        if self.pooling == "first":
            # vector of the word's FIRST subtoken (gather needs the index
            # expanded over the hidden dimension)
            idx = word_starts.unsqueeze(-1).expand(-1, -1, sub.size(-1))
            words = sub.gather(dim=1, index=idx)
        else:
            # MEAN of all subtokens of the word.
            # assign[b, w, l] = 1 if subtoken l belongs to word w, else 0
            # ([CLS]/[SEP]/padding have word_index -1, so they match no word)
            W = word_starts.size(1)
            word_ids = torch.arange(W, device=sub.device).view(1, W, 1)
            assign = (word_index.unsqueeze(1) == word_ids).to(sub.dtype)  # (B, W, L)
            counts = assign.sum(dim=-1, keepdim=True).clamp(min=1.0)      # (B, W, 1)
            words = torch.bmm(assign, sub) / counts                       # (B, W, H)
        words = self.dropout(words)

        # 3) BiLSTM over the word sequence
        if self.use_bilstm:
            lengths = word_mask.sum(dim=1)  # real words per sentence
            # Packing matters for the BACKWARD direction: without it, the
            # backward LSTM of a short sentence would start reading at the
            # padding slots and carry that noise into every real word.
            packed = pack_padded_sequence(words, lengths.cpu(),
                                          batch_first=True,
                                          enforce_sorted=False)
            out, _ = self.lstm(packed)
            words, _ = pad_packed_sequence(out, batch_first=True,
                                           total_length=word_starts.size(1))
            words = self.dropout(words)

        # 4) Tag scores: (B, W, K)
        logits = self.classifier(words)

        result = {"logits": logits}
        if labels is not None:
            # flatten words across the batch; -100 (padding) is ignored
            result["loss"] = self.loss_fn(
                logits.view(-1, logits.size(-1)), labels.view(-1))
        return result

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, batch):
        """Argmax tag ids per word; padded slots set to -100."""
        logits = self.forward(**batch)["logits"]
        preds = logits.argmax(dim=-1)
        return preds.masked_fill(~batch["word_mask"], IGNORE_INDEX)


def build_model(cfg, num_labels):
    m = cfg["model"]
    if m.get("head", "softmax") != "softmax":
        raise NotImplementedError("CRF head comes later")
    return LIDTagger(
        encoder_name=m["encoder_name"],
        num_labels=num_labels,
        use_bilstm=m["use_bilstm"],
        lstm_hidden=m["lstm_hidden"],
        lstm_layers=m["lstm_layers"],
        dropout=m["dropout"],
        pooling=m.get("pooling", "first"),  # older configs: first subtoken
    )


def get_param_groups(model, lr_encoder, lr_head, weight_decay):
    """Two learning rates (small for pretrained BERT, larger for the freshly
    initialised BiLSTM + classifier) and no weight decay on biases and
    LayerNorm weights (standard for BERT fine-tuning)."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = {
        ("enc", True): [], ("enc", False): [],
        ("head", True): [], ("head", False): [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "enc" if name.startswith("encoder.") else "head"
        decay = not any(nd in name for nd in no_decay)
        groups[(part, decay)].append(p)

    lr = {"enc": lr_encoder, "head": lr_head}
    return [
        {"params": ps, "lr": lr[part],
         "weight_decay": weight_decay if decay else 0.0}
        for (part, decay), ps in groups.items() if ps
    ]