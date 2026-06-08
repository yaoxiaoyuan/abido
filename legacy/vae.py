"""
vae_lm.py — VAE Language Model

Usage:
    # Train on PTB
    python vae_lm.py --dataset ptb --train --gpu 0

    # Evaluate a saved checkpoint on Yahoo
    python vae_lm.py --dataset yahoo --gpu 0

    # Generate reconstructions from stdin on Yelp
    echo "the food was great" | python vae_lm.py --dataset yelp --sample --gpu 0
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
import math
import random
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions.normal import Normal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOLS = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

# ---------------------------------------------------------------------------
# Config — argument parsers
# ---------------------------------------------------------------------------

import argparse


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register hyperparameter arguments shared by all three dataset parsers."""
    # RNN cell type
    parser.add_argument("--cell_type", default="lstm", type=str, choices=["lstm", "gru"])
    # Latent space size
    parser.add_argument("--latent_size", default=32, type=int)
    # Encoder architecture
    parser.add_argument("--enc_embedding_size", default=256, type=int)
    parser.add_argument("--enc_hidden_size", default=256, type=int)
    parser.add_argument("--num_enc_layers", default=1, type=int)
    # Decoder architecture
    parser.add_argument("--dec_embedding_size", default=256, type=int)
    parser.add_argument("--dec_hidden_size", default=256, type=int)
    parser.add_argument("--num_dec_layers", default=1, type=int)
    # Regularisation
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--dropout", default=0.5, type=float)
    # Optimisation
    parser.add_argument("--num_epoch", default=40, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_decay", default=0.5, type=float)
    parser.add_argument("--max_decay", default=5, type=float)
    parser.add_argument("--grad_clip", default=5, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("--word_dropout", default=0, type=float)
    # Runtime
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--print_every", default=100, type=int)
    # Mode flags
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sample", action="store_true")


def parse_ptb_args(arguments: list) -> argparse.Namespace:
    """Return parsed arguments for the PTB (Penn Treebank) dataset."""
    parser = argparse.ArgumentParser(description="VAE-LM on PTB")
    parser.add_argument("--train_file", default="../data/ptb/train.txt")
    parser.add_argument("--val_file", default="../data/ptb/val.txt")
    parser.add_argument("--test_file", default="../data/ptb/test.txt")
    parser.add_argument("--vocab_file", default="../data/ptb/vocab.txt")
    parser.add_argument("--vocab_size", default=10002, type=int)
    parser.add_argument("--save_model", default="../model/ptb")
    # PTB sentences are short; cap at 82 tokens
    parser.add_argument("--max_len", default=82, type=int)
    _add_common_args(parser)
    return parser.parse_args(arguments)


def parse_yahoo_args(arguments: list) -> argparse.Namespace:
    """Return parsed arguments for the Yahoo Answers dataset."""
    parser = argparse.ArgumentParser(description="VAE-LM on Yahoo")
    parser.add_argument("--train_file", default="../data/yahoo/train.txt")
    parser.add_argument("--val_file", default="../data/yahoo/val.txt")
    parser.add_argument("--test_file", default="../data/yahoo/test.txt")
    parser.add_argument("--vocab_file", default="../data/yahoo/vocab.txt")
    parser.add_argument("--vocab_size", default=20000, type=int)
    parser.add_argument("--save_model", default="../model/yahoo")
    # Yahoo sentences tend to be longer
    parser.add_argument("--max_len", default=202, type=int)
    _add_common_args(parser)
    # Override defaults that differ for Yahoo (larger model)
    parser.set_defaults(enc_embedding_size=512, enc_hidden_size=1024,
                        dec_embedding_size=512, dec_hidden_size=1024)
    return parser.parse_args(arguments)


def parse_customize_args(arguments: list) -> argparse.Namespace:
    """
    Return parsed arguments for a custom dataset.

    All data paths, vocabulary size, and sequence length must be supplied
    explicitly via command-line flags; there are no hard-coded defaults for
    dataset-specific values.

    Required flags:
        --train_file   Path to the training corpus.
        --val_file     Path to the validation corpus.
        --test_file    Path to the test corpus.
        --vocab_file   Path to the vocabulary file.
        --vocab_size   Vocabulary size (must match the vocab file).
        --save_model   Path prefix for saving / loading the model checkpoint.

    All other hyperparameters fall back to the common defaults defined in
    _add_common_args (enc_hidden_size=256, latent_size=32, etc.).
    """
    parser = argparse.ArgumentParser(description="VAE-LM on a custom dataset")
    parser.add_argument("--train_file", required=True,
                        help="Path to the training corpus (one sentence per line).")
    parser.add_argument("--val_file", required=True,
                        help="Path to the validation corpus.")
    parser.add_argument("--test_file", required=True,
                        help="Path to the test corpus.")
    parser.add_argument("--vocab_file", required=True,
                        help="Path to the vocabulary file (word<TAB>id format).")
    parser.add_argument("--vocab_size", required=True, type=int,
                        help="Number of tokens in the vocabulary.")
    parser.add_argument("--save_model", default="./model/customize",
                        help="File path prefix for saving the best checkpoint.")
    parser.add_argument("--max_len", default=200, type=int,
                        help="Maximum sequence length; longer sentences are truncated.")
    _add_common_args(parser)
    return parser.parse_args(arguments)


def parse_yelp_args(arguments: list) -> argparse.Namespace:
    """Return parsed arguments for the Yelp review dataset."""
    parser = argparse.ArgumentParser(description="VAE-LM on Yelp")
    parser.add_argument("--train_file", default="../data/yelp/yelp.train.txt")
    parser.add_argument("--val_file", default="../data/yelp/yelp.valid.txt")
    parser.add_argument("--test_file", default="../data/yelp/yelp.test.txt")
    parser.add_argument("--vocab_file", default="../data/yelp/vocab.txt")
    parser.add_argument("--vocab_size", default=20000, type=int)
    parser.add_argument("--save_model", default="../model/yelp")
    parser.add_argument("--max_len", default=200, type=int)
    _add_common_args(parser)
    parser.set_defaults(enc_embedding_size=512, enc_hidden_size=1024,
                        dec_embedding_size=512, dec_hidden_size=1024)
    return parser.parse_args(arguments)


# ---------------------------------------------------------------------------
# Data — vocabulary and DataLoader
# ---------------------------------------------------------------------------

def load_vocab(vocab_path: str) -> dict:
    """
    Load a tab-separated vocabulary file and return a word-to-id mapping.

    File format (one entry per line):
        <word>\\t<id>

    Args:
        vocab_path: Path to the vocabulary file.

    Returns:
        vocab: Dict mapping each word string to its integer token id.
    """
    vocab = {}
    for line in open(vocab_path, "rb"):
        line = line.decode("utf-8").strip()
        word, word_id = line.split("\t")
        vocab[word] = int(word_id)
    return vocab


class DataLoader:
    """
    Iterable data loader that groups text sequences into same-length batches.

    Sequences are sorted by length and grouped so every sequence in a batch
    has the same length, eliminating the need for intra-batch padding.
    Batches are shuffled before each epoch.
    """

    def __init__(self, file_path: str, symbols: dict, word2id: dict,
                 batch_size: int, max_len: int, gpu: int, word_dropout: float):
        """
        Pre-cache all batches from the corpus during initialisation.

        Args:
            file_path:    Path to the text corpus (one sentence per line).
            symbols:      Special-token id mapping, e.g. SYMBOLS constant above.
            word2id:      Word-to-id mapping from load_vocab.
            batch_size:   Number of sequences per batch.
            max_len:      Maximum sequence length (API compatibility; not used during caching).
            gpu:          GPU device index; -1 means CPU.
            word_dropout: Probability of replacing a token with <UNK> at each step.
        """
        self.PAD = symbols["<PAD>"]
        self.BOS = symbols["<BOS>"]
        self.EOS = symbols["<EOS>"]
        self.UNK = symbols["<UNK>"]
        self.word2id = word2id
        self.max_len = max_len
        self.gpu = gpu
        self.word_dropout = word_dropout

        self.cache = []
        lines = [line.strip().split()
                 for line in open(file_path, "r", encoding="utf-8")]
        random.shuffle(lines)
        lines.sort(key=lambda sentence: len(sentence))

        current_length = len(lines[0])
        pending_batch = []
        for line in lines:
            token_ids = [word2id.get(word, self.UNK) for word in line]

            if len(token_ids) == current_length:
                pending_batch.append([self.BOS] + token_ids + [self.EOS])
                if len(pending_batch) == batch_size:
                    self.cache.append(pending_batch)
                    pending_batch = []
            else:
                if pending_batch:
                    self.cache.append(pending_batch)
                current_length = len(token_ids)
                pending_batch = [[self.BOS] + token_ids + [self.EOS]]

        if pending_batch:
            self.cache.append(pending_batch)

    def __call__(self):
        """
        Iterate over all batches in random order for one epoch.

        Yields:
            (seq, target): LongTensor pair of shape [batch, seq_len].
                - seq:    Input tokens  (BOS … last token).
                - target: Target tokens (first token … EOS) for teacher forcing.
        """
        random.shuffle(self.cache)
        for batch_data in self.cache:
            yield self._build_tensors(batch_data)

    def _build_tensors(self, batch_data: list) -> tuple:
        """Convert a list of token-id sequences into (seq, target) tensor pair."""
        max_seq_len = max(len(sentence) for sentence in batch_data) - 1
        seq_array = np.zeros([len(batch_data), max_seq_len]) + self.PAD
        target_array = np.zeros([len(batch_data), max_seq_len]) + self.PAD

        for row_index, sentence in enumerate(batch_data):
            seq_len = len(sentence) - 1
            if self.word_dropout > 0:
                # Randomly replace encoder input tokens with <UNK>
                noisy_input = [
                    token if random.random() > self.word_dropout else self.UNK
                    for token in sentence[:-1]
                ]
                seq_array[row_index, :seq_len] = noisy_input
            else:
                seq_array[row_index, :seq_len] = sentence[:-1]
            target_array[row_index, :seq_len] = sentence[1:]

        seq = torch.tensor(seq_array, dtype=torch.long)
        target = torch.tensor(target_array, dtype=torch.long)

        if self.gpu >= 0:
            seq = seq.cuda()
            target = target.cuda()

        return seq, target


# ---------------------------------------------------------------------------
# Model — Encoder / Decoder / VAE
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    RNN encoder that maps a token sequence to a latent Gaussian distribution.

    The final RNN hidden state is projected to produce the mean (mu) and
    log-variance (logvar) of the approximate posterior q(z|x).
    """

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int,
                 latent_size: int, num_layers: int,
                 cell_type: str = "lstm", bidirectional: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional = bool(bidirectional)

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # When bidirectional the RNN output width doubles
        rnn_output_size = hidden_size * (2 if self.bidirectional else 1)
        # Project RNN output → (mu, logvar) concatenated
        self.hidden2mulogvar = nn.Linear(rnn_output_size, 2 * latent_size)

        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def sample_z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Draw a latent sample via the reparameterisation trick: z = mu + eps * std.

        Args:
            mu:     Posterior mean,          shape [batch, latent_size].
            logvar: Posterior log-variance,  shape [batch, latent_size].

        Returns:
            z: Sampled latent vector, shape [batch, latent_size].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encode a sequence to (mu, logvar) without sampling.

        Args:
            x: Token id tensor of shape [batch, seq_len].

        Returns:
            (mu, logvar): Each of shape [batch, latent_size].
        """
        embedded = self.embedding(x)
        # Use only the final time-step output as the sentence representation
        rnn_out = self.rnn(embedded)[0][:, -1, :]
        mu, logvar = torch.chunk(self.hidden2mulogvar(rnn_out), 2, dim=-1)
        return mu, logvar

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Encode and return (mu, logvar, z).

        Args:
            x: Token id tensor of shape [batch, seq_len].

        Returns:
            (mu, logvar, z): mu and logvar each [batch, latent_size]; z sampled.
        """
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        return mu, logvar, z


class Decoder(nn.Module):
    """
    RNN decoder that reconstructs sequences conditioned on a latent vector z.

    z influences decoding in two complementary ways:
      1. It initialises the top RNN hidden state via z2h.
      2. It is added directly to the output logits at every step via z2logits,
         giving the decoder a constant reminder of the global latent context.
    """

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int,
                 latent_size: int, num_layers: int,
                 cell_type: str = "lstm", dropout: float = 0, gamma: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.gamma = gamma

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # z → initial hidden state
        self.z2h = nn.Linear(latent_size, hidden_size)
        # RNN hidden state → vocabulary logits
        self.h2logits = nn.Linear(hidden_size, vocab_size)
        # z → vocabulary logits (broadcast over all time steps)
        self.z2logits = nn.Linear(latent_size, vocab_size)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Decode with teacher forcing: ground-truth tokens are fed at each step.

        Args:
            x: Input token ids, shape [batch, seq_len].
            z: Latent vector,   shape [batch, latent_size].

        Returns:
            pred: Log-softmax distribution over vocabulary,
                  shape [batch, seq_len, vocab_size].
        """
        batch_size, seq_len = x.size()

        embedded = self.dropout(self.embedding(x))
        # Broadcast z across time steps for the direct latent contribution
        z_expanded = z.unsqueeze(1).expand(batch_size, seq_len, self.latent_size)

        z_hidden = self.z2h(z)
        if self.cell_type == "lstm":
            init_h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            init_c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            init_h[-1] = z_hidden
            rnn_out, _ = self.rnn(embedded, (init_h, init_c))
        else:
            init_h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            init_h[-1] = z_hidden
            rnn_out, _ = self.rnn(embedded, init_h)

        rnn_out = self.dropout(rnn_out)
        logits = self.h2logits(rnn_out) + self.z2logits(z_expanded)
        return torch.log_softmax(logits, dim=-1)


class VAE(nn.Module):
    """
    Variational Autoencoder for text generation.

    Combines Encoder and Decoder into a single trainable module.
    Training objective: ELBO = E[log p(x|z)] − KL(q(z|x) ‖ p(z))
    where p(z) = N(0, I).
    """

    def __init__(self, symbols: dict, vocab_size: int,
                 enc_embedding_size: int, enc_hidden_size: int, num_enc_layers: int,
                 dec_embedding_size: int, dec_hidden_size: int, num_dec_layers: int,
                 latent_size: int, cell_type: str = "lstm",
                 bidirectional: bool = False, dropout: float = 0,
                 dynamic: bool = True, gamma: int = 0):
        super().__init__()
        self.PAD = symbols["<PAD>"]
        self.BOS = symbols["<BOS>"]
        self.EOS = symbols["<EOS>"]
        self.UNK = symbols["<UNK>"]

        self.encoder = Encoder(vocab_size, enc_embedding_size, enc_hidden_size,
                               latent_size, num_enc_layers, cell_type, bidirectional)
        self.decoder = Decoder(vocab_size, dec_embedding_size, dec_hidden_size,
                               latent_size, num_dec_layers, cell_type, dropout, gamma)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Encode and decode with teacher forcing (training).

        Returns:
            (pred, mu, logvar): pred is log-softmax over vocab per step.
        """
        mu, logvar, z = self.encoder(x)
        pred = self.decoder(x, z)
        return pred, mu, logvar

    def decode(self, x: torch.Tensor, max_steps: int) -> torch.Tensor:
        """
        Autoregressively decode starting from <BOS> (inference).

        Encodes x to obtain z, then greedily generates tokens step-by-step
        until every sequence in the batch has emitted <EOS> or max_steps is reached.

        Args:
            x:         Conditioning token ids, shape [batch, seq_len].
            max_steps: Maximum decoding steps.

        Returns:
            hyp: Generated token ids, shape [batch, num_steps].
        """
        mu, logvar, z = self.encoder(x)
        batch_size = x.size(0)
        device = x.device

        hyp = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        current_token = torch.full((batch_size, 1), self.BOS, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, 1, dtype=torch.uint8, device=device)

        # Initialise decoder hidden state from z
        z_hidden = self.decoder.z2h(z)
        if self.decoder.cell_type == "lstm":
            init_h = torch.zeros(self.decoder.num_layers, batch_size,
                                 self.decoder.hidden_size, device=device)
            init_c = torch.zeros(self.decoder.num_layers, batch_size,
                                 self.decoder.hidden_size, device=device)
            init_h[-1] = z_hidden
            hidden_state = (init_h, init_c)
        else:
            init_h = torch.zeros(self.decoder.num_layers, batch_size,
                                 self.decoder.hidden_size, device=device)
            init_h[-1] = z_hidden
            hidden_state = init_h

        # Pre-compute the constant latent contribution to logits
        z_logit_bias = self.decoder.z2logits(z.unsqueeze(1))

        for _ in range(max_steps):
            embedded = self.decoder.dropout(self.decoder.embedding(current_token))

            if self.decoder.cell_type == "lstm":
                rnn_out, hidden_state = self.decoder.rnn(embedded, hidden_state)
            else:
                rnn_out, hidden_state = self.decoder.rnn(embedded, hidden_state)

            rnn_out = self.decoder.dropout(rnn_out)
            logits = self.decoder.h2logits(rnn_out) + z_logit_bias
            pred = torch.log_softmax(logits, dim=-1)

            current_token = pred.argmax(dim=-1)
            hyp = torch.cat([hyp, current_token], dim=-1)

            finished = finished | current_token.eq(self.EOS).byte()
            if finished.all():
                break

        return hyp


# ---------------------------------------------------------------------------
# Utils — loss, evaluation metrics, and logging
# ---------------------------------------------------------------------------

def gaussian_kld(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL(N(mu, exp(logvar)) ‖ N(0, I)) per sample.

    Returns:
        kld: Per-sample KL divergence, shape [batch].
    """
    return -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, dim=1)


def loss_fn(pred: torch.Tensor, mu: torch.Tensor,
            logvar: torch.Tensor, target: torch.Tensor) -> tuple:
    """
    Compute reconstruction loss and mean KL divergence.

    Args:
        pred:   Log-softmax probabilities, shape [batch, seq_len, vocab_size].
        mu:     Posterior mean,            shape [batch, latent_size].
        logvar: Posterior log-variance,    shape [batch, latent_size].
        target: Ground-truth token ids,    shape [batch, seq_len].

    Returns:
        (rec, kld): rec is total NLL summed over tokens; kld is mean over batch.
    """
    vocab_size = pred.size(-1)
    rec = F.nll_loss(pred.view(-1, vocab_size), target.flatten(), reduction="sum")
    kld = gaussian_kld(mu, logvar).mean()
    return rec, kld


def eval_fn(vae: VAE, data_loader: DataLoader) -> tuple:
    """
    Evaluate the VAE and return (rec, kld, loss, elbo, ppl).

    All values are averaged per sentence except ppl, which is word-level.
    """
    vae.eval()
    total_rec = total_kl = total_sents = total_words = 0

    with torch.no_grad():
        for seq, target in data_loader():
            pred, mu, logvar = vae(seq)
            rec, kld = loss_fn(pred, mu, logvar, target)

            num_sents = seq.size(0)
            rec = rec / num_sents

            total_rec += rec.item() * num_sents
            total_kl += kld.item() * num_sents
            total_sents += num_sents
            total_words += seq.size(0) * seq.size(1)

    rec = total_rec / total_sents
    kld = total_kl / total_sents
    loss = rec + kld
    elbo = -loss
    ppl = np.exp((total_rec + total_kl) / total_words)
    return rec, kld, loss, elbo, ppl


def calc_au(vae: VAE, data_loader: DataLoader, delta: float = 0.01) -> int:
    """
    Count active units in the latent space.

    A unit is active if the variance of its posterior mean across the dataset
    exceeds `delta`. High AU count indicates the model avoids posterior collapse.

    Returns:
        au: Number of active latent dimensions.
    """
    vae.eval()
    with torch.no_grad():
        mu_sum = None
        num_samples = 0
        for seq, _ in data_loader():
            mu, _, __ = vae.encoder(seq)
            mu_sum = mu.sum(dim=0, keepdim=True) if mu_sum is None else mu_sum + mu.sum(dim=0, keepdim=True)
            num_samples += mu.size(0)

        mu_mean = mu_sum / num_samples

        var_sum = None
        num_samples = 0
        for seq, _ in data_loader():
            mu, _, __ = vae.encoder(seq)
            delta_sq = (mu - mu_mean) ** 2
            var_sum = delta_sq.sum(dim=0) if var_sum is None else var_sum + delta_sq.sum(dim=0)
            num_samples += mu.size(0)

        au_var = var_sum / (num_samples - 1)
        return int((au_var >= delta).sum().item())


def calc_iw_nll(vae: VAE, data_loader: DataLoader,
                n_samples: int = 512, batch_size: int = 128) -> tuple:
    """
    Estimate the importance-weighted NLL (tighter bound than ELBO).

    Uses log(1/K * sum_k [p(x|z_k) p(z_k) / q(z_k|x)]) with K=n_samples.

    Returns:
        (nll, ppl): Average IW-NLL per sentence and corresponding perplexity.
    """
    vae.eval()
    with torch.no_grad():
        num_sents = num_words = 0
        total_nll = 0.0

        for seq, target in data_loader():
            for sent_idx in range(seq.size(0)):
                seq_i = seq[sent_idx:sent_idx + 1]
                target_i = target[sent_idx:sent_idx + 1]
                num_sents += 1
                num_words += seq.size(1)

                mu, logvar = vae.encoder.encode(seq_i)

                seq_i = seq_i.expand(batch_size, -1)
                target_i = target_i.expand(batch_size, -1)
                mu = mu.expand(batch_size, -1)
                logvar = logvar.expand(batch_size, -1)
                std = torch.exp(0.5 * logvar)

                prior = Normal(torch.zeros_like(mu), torch.ones_like(std))
                posterior = Normal(mu, std)

                log_weights = []
                for _ in range(n_samples // batch_size):
                    z = vae.encoder.sample_z(mu, logvar)
                    log_pz = prior.log_prob(z).sum(-1)
                    log_qz = posterior.log_prob(z).sum(-1)
                    pred = vae.decoder(seq_i, z).view(-1, vae.decoder.vocab_size)
                    log_pxz = -F.nll_loss(pred, target_i.flatten(), reduction="none")
                    log_pxz = log_pxz.view(batch_size, -1).sum(-1)
                    log_weights.append(log_pxz + log_pz - log_qz)

                log_p = torch.logsumexp(torch.cat(log_weights), dim=-1) - math.log(n_samples)
                total_nll -= log_p.item()

        nll = total_nll / num_sents
        ppl = np.exp(nll * num_sents / num_words)
    return nll, ppl


def calc_mi(vae: VAE, data_loader: DataLoader) -> float:
    """
    Estimate mutual information I(x; z) = E[log q(z|x)] − E[log q(z)].

    Returns:
        mi: Average MI per sentence.
    """
    total_mi = 0.0
    num_sents = 0
    vae.eval()

    with torch.no_grad():
        for seq, _ in data_loader():
            mu, logvar = vae.encoder.encode(seq)
            batch_size, latent_size = mu.size()

            post_dist = Normal(mu, torch.exp(0.5 * logvar))
            neg_entropy = -post_dist.entropy().sum(-1).mean()

            z = vae.encoder.sample_z(mu, logvar)

            # Expand to compute aggregate posterior q(z) over the batch
            mu_expanded = mu.repeat(batch_size, 1)
            logvar_expanded = logvar.repeat(batch_size, 1)
            z_expanded = z.unsqueeze(1).repeat(1, batch_size, 1).view(-1, latent_size)

            expanded_dist = Normal(mu_expanded, torch.exp(0.5 * logvar_expanded))
            log_density = expanded_dist.log_prob(z_expanded).sum(-1).view(batch_size, batch_size)
            log_qz = torch.logsumexp(log_density, dim=1) - math.log(batch_size)

            total_mi += (neg_entropy - log_qz.mean()).item() * batch_size
            num_sents += batch_size

    return total_mi / num_sents


def build_logger(log_dir: str = "log") -> logging.Logger:
    """
    Build a logger that writes to both a timestamped file and stdout.

    Args:
        log_dir: Directory in which to create the log file.

    Returns:
        logger: Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir,
        datetime.today().strftime("%Y-%m-%d-%H-%M-%S.log"),
    )
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format=format_str)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(format_str))
    logging.getLogger("").addHandler(console_handler)

    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Train the VAE with KL annealing and learning-rate decay.

    The ELBO is optimised with a linearly annealed KL weight beta that rises
    from 0.1 to 1.0 over `args.warmup` epochs. The model checkpoint is saved
    whenever validation loss improves. Early stopping fires when the learning
    rate has been decayed more than `args.max_decay` times consecutively.

    Args:
        args:   Parsed argument namespace.
        logger: Logger instance for progress reporting.
    """
    logger.info(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    word2id = load_vocab(args.vocab_file)
    train_loader = DataLoader(args.train_file, SYMBOLS, word2id,
                              args.batch_size, args.max_len, args.gpu, args.word_dropout)
    val_loader = DataLoader(args.val_file, SYMBOLS, word2id,
                            args.batch_size, args.max_len, args.gpu, 0)
    test_loader = DataLoader(args.test_file, SYMBOLS, word2id,
                             args.batch_size, args.max_len, args.gpu, 0)

    vae = VAE(SYMBOLS, args.vocab_size,
              args.enc_embedding_size, args.enc_hidden_size, args.num_enc_layers,
              args.dec_embedding_size, args.dec_hidden_size, args.num_dec_layers,
              args.latent_size, args.cell_type, args.bidirectional, args.dropout)

    for param in vae.parameters():
        param.data.uniform_(-0.1, 0.1)

    if args.gpu >= 0:
        vae = vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    num_train_batches = len(train_loader.cache)
    current_lr = args.lr
    decay_count = 0
    global_step = 1
    best_val_loss = float("inf")

    # beta rises from 0.1 → 1.0 linearly over warmup * num_batches steps
    anneal_rate = 0.9 / max(1, args.warmup * num_train_batches)

    for epoch in range(args.num_epoch):
        vae.train()

        for seq, target in train_loader():
            beta = min(1.0, 0.1 + 0.9 * global_step * anneal_rate)

            pred, mu, logvar = vae(seq)
            rec, kld = loss_fn(pred, mu, logvar, target)
            rec = rec / seq.size(0)
            loss = rec + beta * kld

            if global_step % args.print_every == 0:
                logger.info(
                    "epoch %d  step %d  beta %.2f  lr %.5f  rec %.2f  kld %.2f",
                    epoch, global_step % num_train_batches, beta, current_lr,
                    rec.item(), kld.item(),
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        val_rec, val_kld, val_loss, val_elbo, val_ppl = eval_fn(vae, val_loader)
        logger.info("epoch %d  val  rec %.2f  kld %.2f  elbo %.2f  ppl %.2f",
                    epoch, val_rec, val_kld, val_elbo, val_ppl)

        test_rec, test_kld, test_loss, test_elbo, test_ppl = eval_fn(vae, test_loader)
        logger.info("epoch %d  test rec %.2f  kld %.2f  elbo %.2f  ppl %.2f",
                    epoch, test_rec, test_kld, test_elbo, test_ppl)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vae.state_dict(), args.save_model)
        elif args.lr_decay > 0:
            decay_count += 1
            current_lr *= args.lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        if decay_count > args.max_decay:
            logger.info("Learning rate decayed %d times — stopping early.", decay_count)
            break


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_model(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Load a saved checkpoint and report comprehensive test-set metrics.

    Metrics: rec, kld, elbo, ppl, active-units (AU),
             importance-weighted NLL/PPL, and mutual information (MI).
    """
    logger.info(args)
    random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    word2id = load_vocab(args.vocab_file)
    test_loader = DataLoader(args.test_file, SYMBOLS, word2id,
                             args.batch_size, args.max_len, args.gpu, 0)

    vae = VAE(SYMBOLS, args.vocab_size,
              args.enc_embedding_size, args.enc_hidden_size, args.num_enc_layers,
              args.dec_embedding_size, args.dec_hidden_size, args.num_dec_layers,
              args.latent_size, args.cell_type, args.bidirectional, args.dropout)

    for param in vae.parameters():
        param.data.uniform_(-0.1, 0.1)

    if args.gpu >= 0:
        vae = vae.cuda()

    vae.load_state_dict(
        torch.load(args.save_model, map_location=lambda storage, loc: storage)
    )

    rec, kld, _, elbo, ppl = eval_fn(vae, test_loader)
    au = calc_au(vae, test_loader)
    iw_nll, iw_ppl = calc_iw_nll(vae, test_loader)
    mi = calc_mi(vae, test_loader)

    logger.info(
        "test  rec %.2f  kld %.2f  elbo %.2f  ppl %.2f  "
        "au %d  iw_nll %.2f  iw_ppl %.2f  mi %.2f",
        rec, kld, elbo, ppl, au, iw_nll, iw_ppl, mi,
    )


# ---------------------------------------------------------------------------
# Inference / sampling
# ---------------------------------------------------------------------------

def sample_text(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Load a trained VAE and print reconstructions for sentences read from stdin.

    For each whitespace-tokenised input line the function:
      1. Converts tokens to ids (<UNK> for out-of-vocabulary words).
      2. Encodes with the VAE encoder.
      3. Decodes autoregressively up to args.max_len steps.
      4. Converts output ids back to words and prints to stdout.
    """
    logger.info(args)
    random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    word2id = load_vocab(args.vocab_file)
    id2word = {token_id: word for word, token_id in word2id.items()}

    vae = VAE(SYMBOLS, args.vocab_size,
              args.enc_embedding_size, args.enc_hidden_size, args.num_enc_layers,
              args.dec_embedding_size, args.dec_hidden_size, args.num_dec_layers,
              args.latent_size, args.cell_type, args.bidirectional, args.dropout)

    for param in vae.parameters():
        param.data.uniform_(-0.1, 0.1)

    if args.gpu >= 0:
        vae = vae.cuda()

    vae.load_state_dict(
        torch.load(args.save_model, map_location=lambda storage, loc: storage)
    )

    vae.eval()
    with torch.no_grad():
        for line in sys.stdin:
            token_ids = [word2id.get(w, SYMBOLS["<UNK>"]) for w in line.split()]
            x = torch.tensor([[SYMBOLS["<BOS>"]] + token_ids], dtype=torch.long)

            if args.gpu >= 0:
                x = x.cuda()

            generated_ids = vae.decode(x, args.max_len)[0].cpu().numpy()
            print(" ".join(id2word[token_id] for token_id in generated_ids))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(arguments: list) -> None:
    """
    Parse arguments and dispatch to train / eval / sample based on --dataset flag.

    --dataset choices: ptb | yahoo | yelp | customize
    --train:           Train the model from scratch.
    --sample:          Read from stdin and print reconstructions.
    (no flag):         Evaluate a saved checkpoint.

    When --dataset customize is used, all data paths and vocab_size must be
    provided explicitly:
        --train_file  --val_file  --test_file  --vocab_file  --vocab_size
    """
    # Peek at --dataset before full parsing so we can select the right parser
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--dataset", default="ptb",
                            choices=["ptb", "yahoo", "yelp", "customize"])
    pre_args, remaining = pre_parser.parse_known_args(arguments)

    dataset_parsers = {
        "ptb": parse_ptb_args,
        "yahoo": parse_yahoo_args,
        "yelp": parse_yelp_args,
        "customize": parse_customize_args,
    }
    args = dataset_parsers[pre_args.dataset](remaining)
    logger = build_logger()

    if args.train:
        train(args, logger)
    elif args.sample:
        sample_text(args, logger)
    else:
        eval_model(args, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
