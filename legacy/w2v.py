"""
PACE — Position-Aware Context Embedding

A self-supervised word embedding training framework that learns position-sensitive
representations via multi-head attention with Rotary Position Embedding (RoPE).

Architecture overview:
  - Training objective: self-reconstruction. Given a token sequence, the model predicts
    each position's word from the context of all other positions in the same chunk.
  - Embedding layer: context_embeddings serves as both key and value; no extra projection.
  - Query vector: a learnable mask vector (one per head), shared across all positions.
  - Positional encoding: optional RoPE (Rotary Position Embedding), which encodes relative
    position implicitly by rotating query/key vectors before the dot-product; no extra
    parameters. Can be disabled (--no_rope) for plain dot-product attention.
  - Aggregation: multi-head attention (use_attention=True) or simple mean pooling over
    all other positions (use_attention=False, no learned query/mask parameters).
  - Output projection: an independent linear layer mapping context representations to
    vocabulary logits.

Key classes:
  - PACEDataset:                 dataset supporting multiple input texts,
                                 low-frequency word filtering, and non-overlapping chunking.
  - PACEModel:                   core encoder supporting multi-head attention + RoPE or
                                 mean-pooling aggregation.
  - PACETrainer:                 trainer encapsulating data loading, optimisation,
                                 checkpointing, and similarity testing.
  - PACEClassifier:              downstream classification head built on top of PACEModel;
                                 reads the <cls> position as the sequence representation.
  - PACEClassificationTrainer:   fine-tuning trainer for PACEClassifier.

Command-line usage:
  python positional_attention_embedding.py --mode train     --data_path <path> [options]
  python positional_attention_embedding.py --mode test      --model_path <path>
  python positional_attention_embedding.py --mode finetune  --model_path <path> --cls_data_path <path>

  --no_attention  disable attention; use mean pooling (no learnable query/mask)
  --no_rope       disable RoPE; use plain dot-product attention scores
  --use_cls       prepend a <cls> token to each chunk (required for fine-tuning)
  See parse_args() for the full list of options.
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import json

# ---------------------------------------------------------------------------
# Special token strings and their fixed vocabulary indices.
# The vocabulary always assigns these five tokens first, in this exact order,
# so the indices below are stable across all datasets.
# ---------------------------------------------------------------------------
SPECIAL_TOKEN_UNK = '<unk>'
SPECIAL_TOKEN_BOS = '<bos>'
SPECIAL_TOKEN_EOS = '<eos>'
SPECIAL_TOKEN_PAD = '<pad>'
SPECIAL_TOKEN_CLS = '<cls>'

SPECIAL_TOKEN_ID_UNK = 0
SPECIAL_TOKEN_ID_BOS = 1
SPECIAL_TOKEN_ID_EOS = 2
SPECIAL_TOKEN_ID_PAD = 3
SPECIAL_TOKEN_ID_CLS = 4


class PACEDataset(Dataset):
    """PACE training dataset. Each sample is a fixed-window chunk of word indices.

    The training objective is self-reconstruction: given a word sequence, PACEModel
    predicts every position in parallel using the context of all other positions.

    Supports multiple input texts (list of token lists). Each text is tokenised
    independently into a shared vocabulary. Chunks do not cross text boundaries.
    Low-frequency words are mapped to '<unk>' rather than discarded, preserving
    sequence length and boundary integrity.

    When use_cls=True, a '<cls>' token is prepended to every chunk so that the
    position-0 context vector serves as a sequence-level representation for
    downstream tasks (see PACEClassifier).
    """

    def __init__(
        self,
        data_path: str,
        max_records: int = 50000,
        max_ctx_size: int = 4,
        min_ctx_size: int = 2,
        min_count: int = 5,
        use_cls: bool = False,
        vocab_path: str = "",
        positive_window: int = 2,
    ):
        """
        Args:
            data_path: path to a jsonl file; each line must be a JSON object with a
                ``"content"`` key whose value is a list of token strings.
            max_records: maximum number of lines to read from *data_path*.
            max_ctx_size: maximum number of tokens per sample chunk.
            min_ctx_size: minimum number of tokens per sample; shorter chunks are discarded.
            min_count: words with frequency below this threshold are replaced with '<unk>'.
                       Only used when *vocab_path* is not provided.
            use_cls: if True, prepend a '<cls>' token at position 0 of every training chunk;
                     the cls representation can be used as a sequence-level feature for
                     classification.  When enabled, effective content length per chunk is
                     max_ctx_size - 1.
            vocab_path: optional path to a JSON file containing a list of word strings
                        (no special tokens).  When provided the vocabulary is built
                        directly from that list in order, skipping frequency counting.
                        When omitted, the vocabulary is derived from the training data
                        using *min_count*.
            positive_window: maximum offset for contrastive positive pair sampling;
                        for each segment, segments within ±positive_window positions in
                        the same document are considered positive candidates.
        """
        self.max_ctx_size = max_ctx_size
        self.min_ctx_size = min_ctx_size
        self.use_cls = use_cls
        self.positive_window = positive_window

        # ── Load texts from disk ──────────────────────────────────────────────
        tokenized_texts = []
        with open(data_path, encoding="utf-8") as data_file:
            for i, line in enumerate(data_file):
                tokenized_texts.append(json.loads(line)["content"])
                if i % 10000 == 0:
                    print(f"Loaded {i + 1} records")
                if i >= max_records - 1:
                    break
        print(f"Total records loaded: {len(tokenized_texts)}")

        # ── Build vocabulary ──────────────────────────────────────────────────
        # Special tokens always occupy the first five fixed indices; '<cls>' is
        # always registered (index 4) so downstream classifiers can reference it
        # even when use_cls=False.
        self.vocab = {
            SPECIAL_TOKEN_UNK: SPECIAL_TOKEN_ID_UNK,
            SPECIAL_TOKEN_BOS: SPECIAL_TOKEN_ID_BOS,
            SPECIAL_TOKEN_EOS: SPECIAL_TOKEN_ID_EOS,
            SPECIAL_TOKEN_PAD: SPECIAL_TOKEN_ID_PAD,
            SPECIAL_TOKEN_CLS: SPECIAL_TOKEN_ID_CLS,
        }

        if vocab_path:
            # Pre-built vocab: read an ordered word list and assign indices in order.
            with open(vocab_path, encoding="utf-8") as vocab_file:
                word_list = json.load(vocab_file)
            for word in word_list:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
            print(f"Vocab loaded from '{vocab_path}': {len(self.vocab)} entries (incl. special tokens)")
        else:
            # Derive vocab by counting word frequencies in the training data.
            all_words = [word for words in tokenized_texts for word in words]
            word_counts = Counter(all_words)
            for word, count in word_counts.items():
                if count >= min_count:
                    self.vocab[word] = len(self.vocab)
            print(f"Vocab built from data: {len(self.vocab)} entries (incl. special tokens)")

        self.cls_token_id = SPECIAL_TOKEN_ID_CLS
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # Replace low-frequency words with '<unk>' while counting unk rate in a single pass
        unk_id = SPECIAL_TOKEN_ID_UNK
        n_total_words = 0
        n_unk_words = 0
        # Store texts as int arrays (numpy) for fast downstream chunking
        self.tokenized_texts = []
        for words in tokenized_texts:
            n_total_words += len(words)
            indices = np.empty(len(words), dtype=np.int32)
            for i, word in enumerate(words):
                idx = self.vocab.get(word, unk_id)
                indices[i] = idx
                if idx == unk_id and word != SPECIAL_TOKEN_UNK:
                    n_unk_words += 1
            self.tokenized_texts.append(indices)

        unk_rate = n_unk_words / n_total_words if n_total_words else 0.0
        print(f"{n_unk_words} out of {n_total_words} words replaced with '<unk>', {unk_rate:.2%}")

        # Generate training samples. Each sample has a global segment id.
        # doc_ids[i] = doc id of segment i (numpy int32 array)
        # positive_candidates[i] = global ids of positive partners, padded with -1 (numpy int32 2D)
        self.data, self.doc_ids, self.positive_candidates = self._generate_training_data()

    def _generate_training_data(self):
        """Generate training chunks and pre-compute positive candidates for contrastive learning.

        Pre-allocates numpy arrays based on estimated maximum segment count to avoid
        repeated list resizing and small-object allocation overhead.

        Returns:
            data:                list of numpy arrays (token-index chunks)
            doc_ids:             numpy int32 array of shape [N], doc_ids[i] = document id of segment i
            positive_candidates: numpy int32 array of shape [N, 2*window], padded with -1;
                                 positive_candidates[i] contains global segment ids that are
                                 valid positive partners (entries == -1 are invalid padding)
        """
        content_size = self.max_ctx_size - 1 if self.use_cls else self.max_ctx_size
        min_size = self.min_ctx_size - 1 if self.use_cls else self.min_ctx_size
        cls_id = self.cls_token_id
        use_cls = self.use_cls
        window = self.positive_window
        max_ctx = self.max_ctx_size

        # Pre-estimate max segment count: each doc produces at most ceil(len/content_size) segments
        max_segments = sum(
            len(indices) // content_size + 1 for indices in self.tokenized_texts
        )

        # Pre-allocate: data buffer (N x max_ctx) filled with pad_id, doc_ids (N,), positive_candidates (N x 2*window)
        pad_id = SPECIAL_TOKEN_ID_PAD
        data_buf = np.full((max_segments, max_ctx), pad_id, dtype=np.int32)
        doc_ids_buf = np.empty(max_segments, dtype=np.int32)
        max_pos_width = 2 * window
        pos_cand_buf = np.full((max_segments, max_pos_width), -1, dtype=np.int32)

        seg_count = 0  # actual number of valid segments produced

        for doc_id, indices in enumerate(self.tokenized_texts):
            n = len(indices)
            if n < min_size:
                continue
            doc_start_id = seg_count
            starts = np.arange(0, n, content_size)
            for start in starts:
                end = min(start + content_size, n)
                chunk_len = end - start
                if chunk_len < min_size:
                    continue
                if use_cls:
                    actual_len = chunk_len + 1
                    data_buf[seg_count, 0] = cls_id
                    data_buf[seg_count, 1:actual_len] = indices[start:end]
                else:
                    actual_len = chunk_len
                    data_buf[seg_count, :actual_len] = indices[start:end]

                doc_ids_buf[seg_count] = doc_id
                seg_count += 1

            # Backfill positive_candidates for this doc's segments
            num_segs = seg_count - doc_start_id
            for local_idx in range(num_segs):
                global_id = doc_start_id + local_idx
                low = max(0, local_idx - window)
                high = min(num_segs - 1, local_idx + window)
                col = 0
                for j in range(low, high + 1):
                    if j != local_idx:
                        pos_cand_buf[global_id, col] = doc_start_id + j
                        col += 1

        # Trim to actual segment count (data_buf is already pad-filled for short chunks)
        data = data_buf[:seg_count]  # [N, max_ctx], padded with pad_id
        doc_ids = doc_ids_buf[:seg_count]
        positive_candidates = pos_cand_buf[:seg_count]

        return data, doc_ids, positive_candidates

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return fixed-length word-index sequence and document identity.

        Returns:
            token_ids: tensor of shape [max_ctx_size], padded with SPECIAL_TOKEN_ID_PAD
            doc_id:    integer document id this segment belongs to
        """
        return torch.from_numpy(self.data[idx].astype(np.int64)), int(self.doc_ids[idx])


class PACEContrastiveBatchSampler:
    """Batch sampler that ensures each batch contains positive pairs for contrastive learning.

    Strategy: iterate through all samples sequentially (so every sample is trained exactly
    once per epoch). For each anchor sample, its adjacent segment from the same document
    (positive partner) is injected into the same batch. Duplicates are avoided — if the
    positive partner already appears as an anchor in this batch, no extra copy is added.

    Each batch yields indices of size up to `batch_size`. Approximately half the slots
    are "anchors" (sequential traversal guarantees full coverage) and the other half are
    their positive partners (adjacent segments from the same document). This guarantees
    positive pairs co-occur in a batch while negatives (cross-document) are naturally
    abundant.
    """

    def __init__(self, dataset: 'PACEDataset', batch_size: int, shuffle_docs: bool = True):
        """
        Args:
            dataset:      PACEDataset instance with positive_candidates pre-computed
            batch_size:   desired batch size (actual may be smaller for the last batch)
            shuffle_docs: if True, shuffle sample order each epoch at segment level
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_docs = shuffle_docs
        self.num_samples = len(dataset)

    def __iter__(self):
        # Build sequential order over all samples; optionally shuffle at doc level
        if self.shuffle_docs:
            all_indices = list(range(self.num_samples))
            np.random.shuffle(all_indices)
        else:
            all_indices = list(range(self.num_samples))

        # Number of anchor slots per batch: reserve ~half for positive partners
        anchors_per_batch = self.batch_size // 2
        if anchors_per_batch < 1:
            anchors_per_batch = 1

        position = 0
        while position < len(all_indices):
            # Collect anchors for this batch
            anchor_end = min(position + anchors_per_batch, len(all_indices))
            batch_set = set()
            batch = []
            for i in range(position, anchor_end):
                idx = all_indices[i]
                batch.append(idx)
                batch_set.add(idx)

            # Collect positive partners for each anchor (skip duplicates already in batch)
            positives = []
            for idx in list(batch):
                pos_idx = self._find_positive(idx)
                if pos_idx is not None and pos_idx not in batch_set:
                    positives.append(pos_idx)
                    batch_set.add(pos_idx)

            # Fill up to batch_size with positives
            remaining_slots = self.batch_size - len(batch)
            batch.extend(positives[:remaining_slots])

            yield batch
            position = anchor_end

    def _find_positive(self, idx):
        """Pick a random positive partner from pre-computed candidates.

        Uses dataset.positive_candidates[idx] which is a numpy row of global segment ids
        padded with -1. Picks one valid (non -1) entry at random.
        Returns None if no valid candidates exist.
        """
        row = self.dataset.positive_candidates[idx]
        # Valid candidates are those != -1
        valid_mask = row >= 0
        num_valid = valid_mask.sum()
        if num_valid == 0:
            return None
        valid_indices = row[valid_mask]
        return int(valid_indices[np.random.randint(num_valid)])

    def __len__(self):
        anchors_per_batch = max(self.batch_size // 2, 1)
        return (self.num_samples + anchors_per_batch - 1) // anchors_per_batch


def pace_collate_fn(batch):
    """Collate fixed-length context sequences into batched tensors.

    All sequences are already padded to max_ctx_size with SPECIAL_TOKEN_ID_PAD,
    so no dynamic padding is needed. The padding_mask is derived directly from
    comparing token ids to the pad id.

    Args:
        batch: List[Tuple[Tensor, int]], each element is (token_ids [max_ctx_size], doc_id)

    Returns:
        contexts:     word-index tensor, shape [batch_size, max_ctx_size]
        padding_mask: True indicates a padding position, shape [batch_size, max_ctx_size]
        doc_ids:      document id tensor, shape [batch_size]
    """
    contexts = torch.stack([item[0] for item in batch])
    doc_ids = torch.tensor([item[1] for item in batch], dtype=torch.long)
    padding_mask = (contexts == SPECIAL_TOKEN_ID_PAD)
    return contexts, padding_mask, doc_ids


def apply_rope(x, base=10000):
    """Apply Rotary Position Embedding (RoPE) to the input tensor.

    RoPE groups dimensions in pairs and applies a position-dependent rotation to each pair,
    so that the dot product q·k naturally encodes the relative distance between positions
    without any extra position bias matrix.

    Rotation formula for dimension pair i at position m:
        [x_{2i}, x_{2i+1}] -> [x_{2i}*cos(m*θ_i) - x_{2i+1}*sin(m*θ_i),
                                x_{2i}*sin(m*θ_i) + x_{2i+1}*cos(m*θ_i)]
    where θ_i = base^(-2i/d)

    Args:
        x:    input tensor, shape [batch_size, seq_len, embed_dim]; embed_dim must be even
        base: RoPE frequency base, default 10000

    Returns:
        rotated tensor with the same shape as input
    """
    batch_size, seq_len, embed_dim = x.shape
    assert embed_dim % 2 == 0, "embed_dim must be even to apply RoPE"

    # Compute per-pair frequencies: [embed_dim/2]
    half_dim = embed_dim // 2
    freq_indices = torch.arange(half_dim, dtype=torch.float32, device=x.device)
    theta = base ** (-freq_indices / half_dim)  # theta_i = base^(-2i/d)

    # Generate position indices and compute rotation angles: [seq_len, embed_dim/2]
    positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
    angles = torch.outer(positions, theta)  # m * theta_i

    # Expand to [1, seq_len, embed_dim/2] for broadcasting
    cos_angles = angles.cos().unsqueeze(0)
    sin_angles = angles.sin().unsqueeze(0)

    # Split x into even/odd dimension pairs: [batch_size, seq_len, embed_dim/2]
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]

    # Apply rotation
    rotated_even = x_even * cos_angles - x_odd * sin_angles
    rotated_odd  = x_even * sin_angles + x_odd * cos_angles

    # Interleave rotated even/odd dimensions back to [batch_size, seq_len, embed_dim]
    rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
    return rotated.flatten(-2)


class PACEModel(nn.Module):
    """Position-Aware Context Embedding (PACE) encoder.

    Learns position-sensitive word representations via multi-head attention and
    Rotary Position Embedding (RoPE). Trained with a self-reconstruction objective:
    each position predicts its own token from the context of all other positions.

    Architecture:
      - query:    shared learnable mask vector (one per head), fixed for all positions;
                  acts as a position-aware context scanner over the sequence
      - key/value: raw word embeddings, no extra projection
      - positional encoding: RoPE rotates query and key vectors so that the dot-product
                  naturally encodes relative distance without additional parameters
      - output:   independent linear projection from context vectors to vocabulary logits

    The model can also serve as a frozen or fine-tunable encoder for downstream tasks
    via PACEClassifier, which reads the '<cls>' position as a sequence-level embedding.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        rope_base=10000,
        n_heads=1,
        mask_unk_target=True,
        use_attention=True,
        use_rope=True,
        use_cls=False,
    ):
        """
        Args:
            vocab_size:      vocabulary size
            embedding_dim:   word embedding dimension; must be even when use_rope=True (required by RoPE)
            rope_base:       RoPE frequency base, default 10000; used only when use_rope=True
            n_heads:         number of attention heads; embedding_dim must be divisible by n_heads;
                             used only when use_attention=True
            mask_unk_target: if True, positions whose target is <unk> (index 0) are excluded from loss
            use_attention:   if True, use multi-head attention to aggregate context;
                             if False, use simple mean of all other positions' embeddings (no learned params)
            use_rope:        if True, apply RoPE to query and key before computing attention scores;
                             if False, use plain dot-product attention scores without positional encoding;
                             only effective when use_attention=True
            use_cls:         if True, the query at pos-0 (the <cls> token position) is replaced by
                             the CLS token's embedding vector instead of the shared learnable mask;
                             only effective when use_attention=True
        """
        if use_attention:
            assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
        if use_attention and use_rope:
            assert embedding_dim % 2 == 0, "embedding_dim must be even (required by RoPE)"

        super(PACEModel, self).__init__()

        self.embedding_dim   = embedding_dim
        self.n_heads         = n_heads
        self.head_dim        = embedding_dim // n_heads
        self.rope_base       = rope_base
        self.mask_unk_target = mask_unk_target
        self.use_attention   = use_attention
        self.use_rope        = use_rope
        self.use_cls         = use_cls
        self.scale           = self.head_dim ** -0.5

        # Word embedding layer (used as both key and value)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Output projection: independent parameters mapping context to vocabulary logits
        self.output_proj = nn.Linear(embedding_dim, vocab_size)

        if use_attention:
            # Learnable mask (query) vectors, one per head: shape [n_heads, head_dim]
            self.mask = nn.Parameter(torch.zeros(n_heads, self.head_dim))

        self._init_weights()

    def _init_weights(self):
        if self.use_attention:
            init_range = 1 / (self.embedding_dim ** 0.5)
            self.mask.data.uniform_(-init_range, init_range)

    def _compute_attention_scores(self, context_words, context_embeds):
        """Compute per-head attention scores using RoPE.

        query: learnable mask vector expanded to every position
        key:   raw word embeddings
        score_h(i, j) = RoPE(mask_h) · RoPE(embed_j) / sqrt(head_dim)

        Args:
            context_words:  word index tensor, shape [batch_size, ctx_len]; used for shape only
            context_embeds: word embeddings, shape [batch_size, ctx_len, embed_dim]

        Returns:
            scores_per_head: shape [batch_size, n_heads, ctx_len, ctx_len]
        """
        batch_size, ctx_len = context_words.shape

        # Expand mask to [batch_size, ctx_len, n_heads, head_dim]
        queries = self.mask.reshape(1, 1, self.n_heads, self.head_dim).expand(
            batch_size, ctx_len, -1, -1
        ).clone()

        if self.use_cls:
            # Replace the query at pos-0 (CLS position) with the CLS token embedding,
            # split into heads: [embed_dim] → [n_heads, head_dim]
            cls_embed = self.context_embeddings.weight[SPECIAL_TOKEN_ID_CLS]  # [embed_dim]
            cls_query = cls_embed.view(self.n_heads, self.head_dim)           # [n_heads, head_dim]
            queries[:, 0, :, :] = cls_query

        # Split embeddings into heads: [batch_size, ctx_len, n_heads, head_dim]
        keys = context_embeds.view(batch_size, ctx_len, self.n_heads, self.head_dim)

        # Reshape to [batch*n_heads, ctx_len, head_dim] for batched ops
        queries_flat = queries.permute(0, 2, 1, 3).reshape(batch_size * self.n_heads, ctx_len, self.head_dim)
        keys_flat    = keys.permute(0, 2, 1, 3).reshape(batch_size * self.n_heads, ctx_len, self.head_dim)

        if self.use_rope:
            # Apply RoPE to encode relative position information into q/k before dot-product
            queries_flat = apply_rope(queries_flat, base=self.rope_base)
            keys_flat    = apply_rope(keys_flat,    base=self.rope_base)

        # Batched dot-product: [batch*n_heads, ctx_len, ctx_len]
        scores_flat = torch.bmm(queries_flat, keys_flat.transpose(1, 2)) * self.scale
        return scores_flat.view(batch_size, self.n_heads, ctx_len, ctx_len)

    def compute_global_context(self, context_words, padding_mask=None):
        """Compute token-level context vectors and one sequence-level vector.

        When use_attention=True: multi-head attention with RoPE; each position attends to
        all other positions (self-position masked out), values are raw word embeddings.

        When use_attention=False: simple mean of all other positions' embeddings
        (self-position excluded, padding positions excluded via padding_mask).

        Sequence-level vector rule:
          - if use_attention=True and use_cls=True: use global_context[:, 0, :]
          - otherwise: mean-pool valid word embeddings, excluding padding and all special tokens

        Args:
            context_words: word index tensor, shape [batch_size, ctx_len]
            padding_mask:  optional bool tensor, shape [batch_size, ctx_len]; True = padding

        Returns:
            global_context:     shape [batch_size, ctx_len, embed_dim]
            cls_vectors:        shape [batch_size, embed_dim]
            attention_weights:  shape [batch_size, n_heads, ctx_len, ctx_len] when use_attention=True;
                                None when use_attention=False
        """
        batch_size, ctx_len = context_words.size(0), context_words.size(1)

        # Raw word embeddings serve as both key and value
        context_embeds = self.context_embeddings(context_words)  # [batch_size, ctx_len, embed_dim]
        
        mean_pooled_vectors = None
        if not (self.use_attention and self.use_cls):
            if padding_mask is None:
                valid_word_mask = torch.ones_like(context_words, dtype=torch.bool)
            else:
                valid_word_mask = ~padding_mask
            # Exclude all fixed special tokens: <unk>, <bos>, <eos>, <pad>, <cls>.
            valid_word_mask = valid_word_mask & (context_words > SPECIAL_TOKEN_ID_CLS)
            valid_word_counts = valid_word_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            mean_pooled_vectors = (context_embeds * valid_word_mask.unsqueeze(-1).float()).sum(dim=1)
            mean_pooled_vectors = mean_pooled_vectors / valid_word_counts
        if not self.use_attention:
            # Build a mask of positions to exclude: self-position + padding positions
            # exclude_mask[b, i, j] = True means position j should NOT contribute to position i's mean
            self_diagonal = torch.eye(ctx_len, dtype=torch.bool, device=context_words.device)
            exclude_mask = self_diagonal.unsqueeze(0)  # [1, ctx_len, ctx_len]
            if padding_mask is not None:
                # padding_mask: [batch_size, ctx_len] -> [batch_size, 1, ctx_len]
                exclude_mask = exclude_mask | padding_mask.unsqueeze(1)

            # Zero out excluded positions, then sum and divide by valid count
            valid_mask = ~exclude_mask  # [batch_size, ctx_len, ctx_len]
            valid_counts = valid_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)  # [batch_size, ctx_len, 1]
            context_sum = (context_embeds.unsqueeze(1) * valid_mask.unsqueeze(-1).float()).sum(dim=2)
            global_context = context_sum / valid_counts  # [batch_size, ctx_len, embed_dim]
            return global_context, mean_pooled_vectors, None

        scores_per_head = self._compute_attention_scores(context_words, context_embeds)

        # Mask out self-positions: True on the diagonal
        self_diagonal = torch.eye(ctx_len, dtype=torch.bool, device=context_words.device).unsqueeze(0).unsqueeze(0)
        scores_per_head = scores_per_head.masked_fill(self_diagonal, float("-inf"))

        # When use_cls=True, CLS (pos-0) should never appear as a key/value source
        # for any position — mask the entire first column across all queries and heads.
        if self.use_cls:
            cls_col_mask = torch.zeros(ctx_len, ctx_len, dtype=torch.bool, device=context_words.device)
            cls_col_mask[:, 0] = True   # all queries, key=pos-0
            scores_per_head = scores_per_head.masked_fill(cls_col_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Mask out padding key positions so padding tokens are not attended to
        if padding_mask is not None:
            # padding_mask: [batch_size, ctx_len] -> [batch_size, 1, 1, ctx_len]
            scores_per_head = scores_per_head.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attention_weights = torch.softmax(scores_per_head, dim=-1)
        
        # Values: raw embeddings split into heads [batch_size, n_heads, ctx_len, head_dim]
        values = context_embeds.view(batch_size, ctx_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Weighted sum: [batch_size, n_heads, ctx_len, head_dim]
        head_outputs = torch.matmul(attention_weights, values)

        # Concatenate heads: [batch_size, ctx_len, embed_dim]
        global_context = head_outputs.permute(0, 2, 1, 3).reshape(batch_size, ctx_len, self.embedding_dim)
        cls_vectors = global_context[:, 0, :] if self.use_cls else mean_pooled_vectors
        return global_context, cls_vectors, attention_weights

    def _contrastive_loss(
        self,
        cls_vectors: torch.Tensor,
        doc_ids: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Document-aware InfoNCE contrastive loss over CLS vectors.

        Positive pairs: segments from the same document (adjacent segments share
        semantics). Negative pairs: segments from different documents.

        For each anchor i, the positive set is all other samples j where
        doc_ids[i] == doc_ids[j]. The loss is computed via softmax over the full
        similarity matrix, with cross-entropy targeting the positive positions.

        When no positive partner exists for an anchor (single-segment doc or all
        partners filtered), that anchor is excluded from the loss.

        Args:
            cls_vectors: [batch_size, embed_dim] — sequence-level vectors
            doc_ids:     [batch_size] — document id for each sample in the batch
            temperature: softmax temperature; smaller = sharper

        Returns:
            scalar loss (mean over valid anchors)
        """
        batch_size = cls_vectors.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=cls_vectors.device)

        # L2-normalise so dot-product equals cosine similarity
        normed = F.normalize(cls_vectors, dim=-1)  # [B, D]
        similarity_matrix = torch.matmul(normed, normed.T) / temperature  # [B, B]

        # Mask out self-similarity (diagonal)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=cls_vectors.device)
        similarity_matrix = similarity_matrix.masked_fill(self_mask, float('-inf'))

        # Build positive mask: same doc_id AND not self
        # positive_mask[i, j] = True means j is a positive partner of i
        doc_ids_row = doc_ids.unsqueeze(1)  # [B, 1]
        doc_ids_col = doc_ids.unsqueeze(0)  # [1, B]
        positive_mask = (doc_ids_row == doc_ids_col) & (~self_mask)  # [B, B]

        # For each anchor, compute log-sum-exp over all non-self positions (denominator)
        # and log-sum-exp over positive positions (numerator), then loss = -log(num/denom)
        # Equivalent to: for each anchor, -log( sum_pos(exp(sim)) / sum_all(exp(sim)) )

        # Check which anchors have at least one positive
        has_positive = positive_mask.any(dim=1)  # [B]
        if not has_positive.any():
            return torch.tensor(0.0, device=cls_vectors.device)

        # log-sum-exp over all non-self entries (denominator)
        log_denom = torch.logsumexp(similarity_matrix, dim=1)  # [B]

        # For numerator: mask out non-positive positions with -inf, then logsumexp
        neg_inf_matrix = similarity_matrix.clone()
        neg_inf_matrix[~positive_mask] = float('-inf')
        log_numer = torch.logsumexp(neg_inf_matrix, dim=1)  # [B]

        # Loss per anchor: -log(sum_pos / sum_all) = -(log_numer - log_denom)
        per_anchor_loss = -(log_numer - log_denom)

        # Only average over anchors that have valid positives
        loss = per_anchor_loss[has_positive].mean()
        return loss

    def forward(self, context_words, padding_mask=None, contrastive_weight: float = 0.0, doc_ids: torch.Tensor = None):
        """
        Args:
            context_words:       context word indices, shape [batch_size, ctx_len]
            padding_mask:        True indicates a padding position to ignore, shape [batch_size, ctx_len], optional
            contrastive_weight:  weight of the NT-Xent contrastive loss added to the reconstruction loss;
                                 default: 0.0 = disabled
            doc_ids:             document id for each sample, shape [batch_size]; required when
                                 contrastive_weight > 0 to identify positive/negative pairs

        Returns:
            total_loss:        recon_loss + contrastive_weight * contrastive_loss, scalar
            recon_loss:        self-reconstruction cross-entropy loss, scalar
            contrastive_loss:  NT-Xent loss on sequence-level vectors (0.0 if disabled), scalar
            global_context:    context representations, shape [batch_size, ctx_len, embed_dim]
        """
        global_context, cls_vectors, _ = self.compute_global_context(context_words, padding_mask)
        logits = self.output_proj(global_context)

        batch_size, ctx_len, vocab_size = logits.size()
        flat_logits = logits.view(batch_size * ctx_len, vocab_size)
        flat_targets = context_words.view(batch_size * ctx_len)

        if padding_mask is not None:
            flat_padding_mask = padding_mask.view(batch_size * ctx_len)
            flat_targets = flat_targets.masked_fill(flat_padding_mask, -100)

        if self.mask_unk_target:
            flat_targets = flat_targets.masked_fill(flat_targets == SPECIAL_TOKEN_ID_UNK, -100)

        # Exclude CLS token from the reconstruction target (same approach as unk masking).
        flat_targets = flat_targets.masked_fill(flat_targets == SPECIAL_TOKEN_ID_CLS, -100)

        recon_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)
        contrastive_loss = None

        if contrastive_weight > 0.0 and batch_size > 1 and doc_ids is not None:
            contrastive_loss = self._contrastive_loss(cls_vectors, doc_ids)

        total_loss = recon_loss if contrastive_loss is None else recon_loss + contrastive_weight * contrastive_loss
        return total_loss, recon_loss, contrastive_loss, global_context

    def get_embedding(self, word_idx):
        return self.context_embeddings.weight[word_idx].detach()

    @classmethod
    def from_pretrained(cls, save_dir, device='cpu'):
        """Restore a model from a saved directory.

        Expects the directory to contain:
          - embedding_config.json: model architecture hyperparameters (saved by Trainer.save_model)
          - model_weights.pt:      model state dict

        Args:
            save_dir: path to the directory produced by PACETrainer.save_model
            device:   device to load the model onto, e.g. 'cpu' or 'cuda'

        Returns:
            model: a fully restored PACEModel instance in eval mode
        """
        config_path = os.path.join(save_dir, 'embedding_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file not found: {config_path}')

        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)

        model = cls(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            rope_base=config.get('rope_base', 10000),
            n_heads=config.get('n_heads', 1),
            mask_unk_target=config.get('mask_unk_target', True),
            use_attention=config.get('use_attention', True),
            use_rope=config.get('use_rope', True),
            use_cls=config.get('use_cls', True)
        )

        weights_path = os.path.join(save_dir, 'model_weights.pt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f'Weights file not found: {weights_path}')

        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        print(f'Model restored from: {save_dir}')
        return model

    def find_similar_words_by_vector(self, query_vector, top_k=10, idx2word=None):
        """Find the top-k most similar words to a given query vector by cosine similarity.

        Computes cosine similarity between the query vector and every word embedding in
        the vocabulary in a single batched operation, which is significantly faster than
        iterating over words one by one.

        Special tokens at indices 0, 1, 2 ('<unk>', '<bos>', '<eos>') are excluded from
        the results.

        Args:
            query_vector: numpy array of shape [embedding_dim], or a torch.Tensor
            top_k:        number of top similar words to return
            idx2word:     optional dict mapping word index (int) -> word (str);
                          when provided, returns (word, similarity) tuples;
                          when None, returns (word_index, similarity) tuples

        Returns:
            list of (word_or_index, similarity) tuples sorted by descending similarity
        """
        if isinstance(query_vector, np.ndarray):
            query_tensor = torch.from_numpy(query_vector).float()
        else:
            query_tensor = query_vector.float().clone().detach()

        query_tensor = query_tensor.to(self.context_embeddings.weight.device)

        query_norm = query_tensor.norm()
        if query_norm == 0:
            return []

        # All word embeddings: [vocab_size, embedding_dim]
        all_embeddings = self.context_embeddings.weight.detach()

        # Cosine similarity via normalised dot product: [vocab_size]
        embedding_norms = all_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalised_embeddings = all_embeddings / embedding_norms
        normalised_query = query_tensor / query_norm
        cosine_similarities = normalised_embeddings @ normalised_query  # [vocab_size]

        # Mask out special tokens (indices 0, 1, 2: <unk>, <bos>, <eos>)
        cosine_similarities[:3] = -float('inf')

        # Retrieve top-k indices and their similarity scores
        actual_top_k = min(top_k, cosine_similarities.size(0) - 3)
        top_similarities, top_indices = torch.topk(cosine_similarities, k=actual_top_k)

        top_similarities = top_similarities.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        if idx2word is not None:
            return [(idx2word[int(idx)], float(sim)) for idx, sim in zip(top_indices, top_similarities)]
        return [(int(idx), float(sim)) for idx, sim in zip(top_indices, top_similarities)]

class PACETrainer:
    """PACE pre-training trainer.

    Encapsulates data loading, optimisation, checkpointing and similarity testing
    for the self-reconstruction pre-training phase of PACEModel.
    """
    
    def __init__(self, model, dataset, device='cpu'):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        
    def train(
        self,
        epochs=10,
        batch_size=128,
        lr=0.025,
        optimizer="adam",
        save_dir=None,
        save_every_steps=None,
        test_words=None,
        test_every_steps=1000,
        contrastive_weight: float = 0.0,
    ):
        """
        Train the model

        Args:
            epochs:              number of training epochs
            batch_size:          batch size
            lr:                  learning rate
            optimizer:           optimizer type, "adam" or "sgd"
            save_dir:            root directory for saving models; no saving if None
            save_every_steps:    save a checkpoint every this many global steps; no interval saving if None.
                                 checkpoints are saved to save_dir/checkpoint-{global_step}/;
                                 the final model is saved directly to save_dir/ after training.
            contrastive_weight:  weight of the NT-Xent contrastive loss on CLS vectors;
                                 only effective when model.use_attention=True and model.use_cls=True
                                 (default: 0.0 = disabled)
        """
        # Use contrastive batch sampler when contrastive learning is active to ensure
        # positive pairs (adjacent segments from same document) co-occur in each batch.
        use_contrastive_sampler = contrastive_weight > 0.0
        if use_contrastive_sampler:
            batch_sampler = PACEContrastiveBatchSampler(
                self.dataset, batch_size=batch_size, shuffle_docs=True,
            )
            dataloader = DataLoader(
                self.dataset,
                batch_sampler=batch_sampler,
                collate_fn=pace_collate_fn,
            )
        else:
            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=pace_collate_fn,
            )

        if optimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=lr)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for step, (contexts, padding_mask, doc_ids) in enumerate(dataloader, 1):
                global_step += 1

                contexts = contexts.to(self.device)
                padding_mask = padding_mask.to(self.device)
                doc_ids = doc_ids.to(self.device)

                optimizer.zero_grad()
                step_loss, recon_loss, contrastive_loss, _ = self.model(
                    contexts, padding_mask,
                    contrastive_weight=contrastive_weight,
                    doc_ids=doc_ids,
                )
                step_loss.backward()
                optimizer.step()

                if contrastive_loss is not None:
                    print(
                        f'Epoch {epoch+1}/{epochs}, Step {step}/{len(dataloader)}, '
                        f'Loss: {step_loss.item():.4f} '
                        f'(recon={recon_loss.item():.4f}, contrast={contrastive_loss.item():.4f})'
                    )
                else:
                    print(f'Epoch {epoch+1}/{epochs}, Step {step}/{len(dataloader)}, Loss: {step_loss.item():.4f}')

                if test_words is not None and test_every_steps is not None:
                    if global_step % test_every_steps == 0:
                        for word in test_words:
                            print("-" * 20)
                            print(f'Top similar words for {word}:')
                            for similar_word, similarity in self.find_similar_words(word):
                                print(f'  {similar_word}: {similarity:.4f}')
                            print("-" * 20)

                # Save checkpoint at regular intervals
                if save_dir is not None and save_every_steps is not None:
                    if global_step % save_every_steps == 0:
                        checkpoint_dir = os.path.join(save_dir, f'checkpoint-{global_step}')
                        self.save_model(checkpoint_dir)

                epoch_loss += step_loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

        # Save the final model to the root directory after training
        if save_dir is not None:
            self.save_model(save_dir)
        
    def get_word_vector(self, word):
        """Return the embedding vector for a given word."""
        if word not in self.dataset.vocab:
            return None
        word_idx = self.dataset.vocab[word]
        return self.model.get_embedding(word_idx).cpu().numpy()
    
    def find_similar_words(self, word, top_k=10):
        """Find the top-k most similar words by cosine similarity."""
        if word not in self.dataset.vocab:
            return []

        word_vec = self.get_word_vector(word)
        idx2word = {idx: w for w, idx in self.dataset.vocab.items()}
        # Exclude the query word itself from results
        return [
            (similar_word, sim)
            for similar_word, sim in self.model.find_similar_words_by_vector(
                word_vec, top_k=top_k + 1, idx2word=idx2word
            )
            if similar_word != word
        ][:top_k]

    def save_model(self, save_dir):
        """Save the model to the specified directory.

        Saved files:
          - embedding_config.json: model architecture config (hyperparameters)
          - vocab.json:            vocabulary (word->id and id->word merged)
          - model_weights.pt:      model weights

        Args:
            save_dir: target directory path; created automatically if it does not exist
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save model architecture config
        config = {
            'vocab_size':     self.model.context_embeddings.num_embeddings,
            'embedding_dim':  self.model.embedding_dim,
            'rope_base':      self.model.rope_base,
            'n_heads':        self.model.n_heads,
            'mask_unk_target': self.model.mask_unk_target,
            'use_attention':  self.model.use_attention,
            'use_rope':       self.model.use_rope,
            'use_cls':        self.model.use_cls
        }

        config_path = os.path.join(save_dir, 'embedding_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # Save vocabulary (word->id and id->word merged)
        vocab_data = {
            'word2id': self.dataset.vocab,
            'id2word': {str(idx): word for idx, word in self.dataset.idx2word.items()},
        }
        vocab_path = os.path.join(save_dir, 'vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        # Save model weights
        weights_path = os.path.join(save_dir, 'model_weights.pt')
        torch.save(self.model.state_dict(), weights_path)

        print(f'Model saved to directory: {save_dir}')
        print(f'  embedding_config.json  model config')
        print(f'  vocab.json             vocabulary')
        print(f'  model_weights.pt       model weights')

    def load_model(self, save_dir):
        """Load model weights from the specified directory.

        Args:
            save_dir: directory path (as used in save_model)
        """
        weights_path = os.path.join(save_dir, 'model_weights.pt')
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        print(f'Model weights loaded from {weights_path}')


class PACEClassifier(nn.Module):
    """PACE downstream sequence classifier.

    Wraps a pre-trained PACEModel encoder and adds a multi-layer MLP classification
    head. The '<cls>' token at position 0 is used as the fixed-length sequence
    representation (identical in spirit to BERT's [CLS] pooling strategy).

    Architecture:
      - Base encoder: PACEModel (frozen or fine-tunable), must be run with use_cls=True
      - CLS extraction: global_context[:, 0, :] — the position-0 context vector
      - Classification head: Linear → LayerNorm → GELU → Dropout  (×n_hidden_layers)
                             → Linear(num_classes)

    Usage:
      1. Load from a pretrained PACE directory:
             model = PACEClassifier.from_pretrained_embedding(save_dir, num_classes=5)
      2. Wrap an already-constructed PACEModel:
             model = PACEClassifier(pace_model, num_classes=5)

    Every input sequence must start with the '<cls>' token (index 4 by default).
    """

    def __init__(
        self,
        base_model: PACEModel,
        num_classes: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        """
        Args:
            base_model:      pretrained PACEModel used as the encoder
            num_classes:     number of output classes
            hidden_dim:      width of each hidden layer in the classification head
            n_hidden_layers: number of hidden (Linear → LayerNorm → GELU → Dropout) blocks
            dropout:         dropout probability applied after each activation
            freeze_base:     if True, freeze all base model parameters so only the
                             classification head is trained
        """
        super(PACEClassifier, self).__init__()

        self.base_model = base_model
        self.embedding_dim = base_model.embedding_dim
        self.num_classes = num_classes

        if freeze_base:
            for parameter in self.base_model.parameters():
                parameter.requires_grad = False

        # Build the classification head: variable number of hidden layers + output projection
        layers = []
        input_dim = self.embedding_dim
        for _ in range(n_hidden_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    @classmethod
    def from_pretrained_embedding(
        cls,
        save_dir: str,
        num_classes: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        dropout: float = 0.1,
        freeze_base: bool = False,
        device: str = 'cpu',
    ) -> 'PACEClassifier':
        """Load a pretrained PACEModel and wrap it in a PACEClassifier.

        Args:
            save_dir:        path to the directory produced by PACETrainer.save_model
            num_classes:     number of output classes
            hidden_dim:      hidden layer width
            n_hidden_layers: number of hidden blocks in the classification head
            dropout:         dropout probability
            freeze_base:     freeze base model weights during fine-tuning
            device:          device string, e.g. 'cpu' or 'cuda'

        Returns:
            PACEClassifier instance with pretrained encoder weights loaded
        """
        base_model = PACEModel.from_pretrained(save_dir, device=device)
        return cls(
            base_model=base_model,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            dropout=dropout,
            freeze_base=freeze_base,
        ).to(device)

    def forward(self, context_words, labels=None, padding_mask=None):
        """Run the encoder and classify from the CLS token representation.

        Args:
            context_words: token index tensor, shape [batch_size, seq_len];
                           position 0 of every sequence must be the '<cls>' token
            labels:        optional class index tensor, shape [batch_size];
                           when provided, cross-entropy loss is computed and returned
            padding_mask:  optional bool tensor, shape [batch_size, seq_len]; True = padding

        Returns:
            If labels is provided: (loss, logits) tuple
            Otherwise:             logits tensor of shape [batch_size, num_classes]
        """
        # cls_representation: [batch_size, embedding_dim]
        _, cls_representation, _ = self.base_model.compute_global_context(context_words, padding_mask)

        logits = self.classifier(cls_representation)  # [batch_size, num_classes]

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits

        return logits


class PACEClassificationTrainer:
    """PACE fine-tuning trainer for downstream classification tasks.

    Wraps PACEClassifier and handles the training loop, step-level and epoch-level
    evaluation on both an eval split and an optional held-out test set, and
    final model checkpointing.

    Expected DataLoader format: each batch is a tuple of
        (token_ids, padding_mask, labels)
    where token_ids[:, 0] must be the '<cls>' token index.
    """

    def __init__(self, model: PACEClassifier, device: str = 'cpu'):
        """
        Args:
            model:  PACEClassifier to fine-tune
            device: device string, e.g. 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device

    def train(
        self,
        train_dataloader,
        epochs: int = 3,
        lr: float = 2e-4,
        optimizer_type: str = 'adam',
        eval_dataloader=None,
        test_dataloader=None,
        eval_every_steps: int = None,
        save_dir: str = None,
    ):
        """Fine-tune the PACEClassifier.

        Args:
            train_dataloader: DataLoader yielding (token_ids, padding_mask, labels) batches.
                              token_ids: [batch_size, seq_len]  (position 0 = '<cls>')
                              padding_mask: [batch_size, seq_len] bool (True = padding)
                              labels: [batch_size] long
            epochs:           number of fine-tuning epochs
            lr:               learning rate
            optimizer_type:   'adam' or 'sgd'
            eval_dataloader:  optional DataLoader for eval set; evaluated at every
                              eval_every_steps interval and at the end of each epoch
            test_dataloader:  optional DataLoader for held-out test set; evaluated
                              alongside eval_dataloader at every eval point
            eval_every_steps: if provided, run evaluation every this many global steps
                              in addition to end-of-epoch evaluation; no intra-epoch
                              evaluation when None
            save_dir:         if provided, save the final model weights here
        """
        if optimizer_type == 'sgd':
            optimizer = optim.SGD(
                filter(lambda param: param.requires_grad, self.model.parameters()), lr=lr
            )
        else:
            optimizer = optim.Adam(
                filter(lambda param: param.requires_grad, self.model.parameters()), lr=lr
            )

        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            total_loss = 0.0
            for step, batch in enumerate(train_dataloader, 1):
                global_step += 1

                token_ids, padding_mask, labels = batch
                token_ids    = token_ids.to(self.device)
                padding_mask = padding_mask.to(self.device)
                labels       = labels.to(self.device)

                optimizer.zero_grad()
                loss, _ = self.model(token_ids, labels=labels, padding_mask=padding_mask)
                loss.backward()
                optimizer.step()

                print(f'Epoch {epoch+1}/{epochs}, Step {step}/{len(train_dataloader)}, '
                      f'Loss: {loss.item():.4f}')
                total_loss += loss.item()

                # Intra-epoch evaluation at fixed step intervals
                if eval_every_steps is not None and global_step % eval_every_steps == 0:
                    self._run_evaluation(global_step=global_step,
                                         eval_dataloader=eval_dataloader,
                                         test_dataloader=test_dataloader)

            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch+1}/{epochs} avg loss: {avg_loss:.4f}')
            self._run_evaluation(epoch=epoch, epochs=epochs,
                                  eval_dataloader=eval_dataloader,
                                  test_dataloader=test_dataloader)

        if save_dir is not None:
            self.save_model(save_dir)

    def _run_evaluation(self, eval_dataloader=None, test_dataloader=None,
                        global_step=None, epoch=None, epochs=None):
        """Run evaluation on eval and/or test dataloader and print results.

        Called both at intra-epoch step intervals and at end-of-epoch.
        """
        if eval_dataloader is None and test_dataloader is None:
            return

        prefix = f'[Step {global_step}]' if global_step is not None else f'Epoch {epoch+1}/{epochs}'

        if eval_dataloader is not None:
            eval_accuracy = self.evaluate(eval_dataloader)
            print(f'{prefix} eval accuracy: {eval_accuracy:.4f}')

        if test_dataloader is not None:
            test_accuracy = self.evaluate(test_dataloader)
            print(f'{prefix} test accuracy: {test_accuracy:.4f}')

    @torch.no_grad()
    def evaluate(self, dataloader) -> float:
        """Compute accuracy on a dataloader.

        Args:
            dataloader: DataLoader yielding (token_ids, padding_mask, labels) batches

        Returns:
            accuracy as a float in [0, 1]
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0

        for token_ids, padding_mask, labels in dataloader:
            token_ids    = token_ids.to(self.device)
            padding_mask = padding_mask.to(self.device)
            labels       = labels.to(self.device)

            logits = self.model(token_ids, padding_mask=padding_mask)
            predicted_classes = logits.argmax(dim=-1)
            total_correct += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)

        self.model.train()
        return total_correct / total_samples if total_samples > 0 else 0.0

    def save_model(self, save_dir: str):
        """Save classification model weights and config.

        Saved files:
          - classifier_config.json: num_classes, hidden_dim, n_hidden_layers, dropout
          - classifier_weights.pt:  full model state dict (base encoder + head)

        Args:
            save_dir: target directory; created automatically if it does not exist
        """
        os.makedirs(save_dir, exist_ok=True)
        config = {
            'num_classes':     self.model.num_classes,
            'embedding_dim':   self.model.embedding_dim,
        }
        config_path = os.path.join(save_dir, 'classifier_config.json')
        with open(config_path, 'w', encoding='utf-8') as config_file:
            json.dump(config, config_file, ensure_ascii=False, indent=2)

        weights_path = os.path.join(save_dir, 'classifier_weights.pt')
        torch.save(self.model.state_dict(), weights_path)

        print(f'Classification model saved to: {save_dir}')
        print(f'  classifier_config.json  model config')
        print(f'  classifier_weights.pt   model weights')


def parse_args():
    parser = argparse.ArgumentParser(description='Train PositionalAttentionEmbedding model')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'finetune'],
                        help='train: pretrain embedding model; '
                             'test: interactive embedding query; '
                             'finetune: fine-tune a PACEClassifier on a downstream task')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])

    # ── Pretraining data ──────────────────────────────────────────────────────
    parser.add_argument('--data_path', type=str, default='data/tokenized_texts.json')
    parser.add_argument('--max_records', type=int, default=50000)
    parser.add_argument('--vocab_path', type=str, default='',
                        help='optional path to a JSON file containing an ordered list of word '
                             'strings (no special tokens).  When provided, the vocabulary is '
                             'built directly from that list in order and min_count is ignored. '
                             'When omitted, vocab is derived from training data using min_count.')

    # ── Embedding model architecture ──────────────────────────────────────────
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--rope_base', type=int, default=10000)
    parser.add_argument('--mask_unk_target', action='store_true', default=True)
    parser.add_argument('--no_attention', action='store_true', default=False,
                        help='if set, use mean pooling instead of attention (no RoPE)')
    parser.add_argument('--no_rope', action='store_true', default=False,
                        help='if set, disable RoPE and use plain dot-product attention')
    parser.add_argument('--use_cls', action='store_true', default=False,
                        help='if set, prepend a <cls> token at position 0 of every training chunk')

    # ── Dataset chunking ─────────────────────────────────────────────────────
    parser.add_argument('--max_ctx_size', type=int, default=20)
    parser.add_argument('--min_ctx_size', type=int, default=20)
    parser.add_argument('--min_count', type=int, default=200)

    # ── Training hyperparameters ──────────────────────────────────────────────
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--save_dir', type=str, default='embedding_models')
    parser.add_argument('--save_every_steps', type=int, default=10000)

    # ── Evaluation / testing ──────────────────────────────────────────────────
    parser.add_argument('--model_path', type=str, default='embedding_models',
                        help='path to pretrained embedding model directory (used in test/finetune mode)')
    parser.add_argument('--test_words', type=str, nargs='+', default=[])
    parser.add_argument('--test_every_steps', type=int, default=1000)
    parser.add_argument('--contrastive_weight', type=float, default=0.0,
                        help='Weight of NT-Xent contrastive loss on CLS vectors; '
                             'only effective when --use_cls is set (default: 0.0 = disabled)')

    # ── Classification fine-tuning (mode=finetune) ────────────────────────────
    parser.add_argument('--cls_data_path', type=str, default=None,
                        help='path to classification dataset (jsonl, each line: {"tokens": [...], "label": int})')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of output classes for the classification head')
    parser.add_argument('--classifier_hidden_dim', type=int, default=256,
                        help='hidden layer width in the classification MLP head')
    parser.add_argument('--classifier_n_hidden_layers', type=int, default=2,
                        help='number of hidden blocks in the classification MLP head')
    parser.add_argument('--classifier_dropout', type=float, default=0.1,
                        help='dropout probability in the classification head')
    parser.add_argument('--freeze_base', action='store_true', default=False,
                        help='if set, freeze base embedding model weights during fine-tuning')
    parser.add_argument('--cls_save_dir', type=str, default='classification_models',
                        help='directory to save the fine-tuned classification model')
    parser.add_argument('--test_cls_data_path', type=str, default=None,
                        help='path to held-out test dataset for classification '
                             '(jsonl, same format as --cls_data_path); '
                             'evaluated once after fine-tuning completes; '
                             'when provided the full cls_data_path is used for training '
                             'and no automatic train/eval split is performed')
    parser.add_argument('--eval_split_ratio', type=float, default=0.1,
                        help='fraction of cls_data_path to use as eval set when '
                             'test_cls_data_path is not provided (default: 0.1)')
    parser.add_argument('--eval_every_steps', type=int, default=None,
                        help='if set, evaluate on the eval split every this many global steps '
                             'during fine-tuning (in addition to end-of-epoch evaluation)')

    return parser.parse_args()


def train(args):
    """Example usage."""
    # Device setup
    device = torch.device(args.device)

    # Build dataset – loading texts from disk is handled inside PACEDataset.
    print('\nBuilding dataset...')
    dataset = PACEDataset(
        data_path=args.data_path,
        max_records=args.max_records,
        max_ctx_size=args.max_ctx_size,
        min_ctx_size=args.min_ctx_size,
        min_count=args.min_count,
        use_cls=args.use_cls,
        vocab_path=args.vocab_path,
    )
    print(f'Vocabulary size: {dataset.vocab_size}')
    print(f'Training samples: {len(dataset)}')
    
    # Build model
    print('\nBuilding model...')
    model = PACEModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=args.embedding_dim,
        rope_base=args.rope_base,
        n_heads=args.n_heads,
        mask_unk_target=args.mask_unk_target,
        use_attention=not args.no_attention,
        use_rope=not args.no_rope,
        use_cls=args.use_cls,
    )
    
    print(model)

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel Statistics:')
    print(f'  Total parameters: {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')
    
    # Build trainer
    trainer = PACETrainer(model, dataset, device=device)
    
    # Train
    print('\nStarting training...')
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        save_dir=args.save_dir,
        save_every_steps=args.save_every_steps,
        test_words=args.test_words,
        test_every_steps=args.test_every_steps,
        contrastive_weight=args.contrastive_weight)

def test(args):
    """
    """
    model = PACEModel.from_pretrained(args.model_path)

    word2id = json.load(open(args.model_path + "/vocab.json", encoding="utf-8"))["word2id"]
    id2word = json.load(open(args.model_path + "/vocab.json", encoding="utf-8"))["id2word"]
    id2word = {int(idx):word for idx,word in id2word.items()}
    
    print("Interactive word embedding query. Input format:")
    print("  <word>              find top-20 similar words")
    print("  <word1>\\t<word2>    compute cosine similarity between two words")
    print("  <word1>\\t<word2>\\t<word3>  analogy: word1 + word2 - word3")
    print("Enter Ctrl+D to exit.\n")

    for line in sys.stdin:
        words = line.strip().split("\t")
        if len(words) == 1:
            if words[0] not in word2id:
                print(f"{words[0]} is not in vocab, skip")
                continue
            vec = model.get_embedding(word2id[words[0]])
            topk = model.find_similar_words_by_vector(vec.cpu().numpy(), top_k=20, idx2word=id2word)
            topk = [[word, sim] for word, sim in topk if word not in words]
            print(topk)
        elif len(words) == 2:
            if any(word not in word2id for word in words):
                print(f"one or more words is not in vocab, skip")
                continue
            vec_1 = model.get_embedding(word2id[words[0]])
            vec_2 = model.get_embedding(word2id[words[1]])
            sim = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
            print(f"{words[0]}, {words[1]}, sim: {sim}")
        elif len(words) == 3:
            if any(word not in word2id for word in words):
                print(f"one or more words is not in vocab, skip")
                continue
            vec_1 = model.get_embedding(word2id[words[0]])
            vec_2 = model.get_embedding(word2id[words[1]])
            vec_3 = model.get_embedding(word2id[words[2]])
            vec = vec_1 + vec_2 - vec_3
            topk = model.find_similar_words_by_vector(vec.cpu().numpy(), top_k=20, idx2word=id2word)
            topk = [[word, sim] for word, sim in topk if word not in words]
            print(topk)

def finetune(args):
    """Fine-tune a PACEClassifier on a downstream classification task.

    Expected dataset format (jsonl, one record per line):
        {"tokens": ["word1", "word2", ...], "label": 0}

    The vocabulary and model weights are loaded from args.model_path (the pretrained
    embedding directory). Each token sequence is prefixed with '<cls>' before being
    fed into the encoder so that position 0 always holds the aggregate representation.
    """
    if args.cls_data_path is None:
        raise ValueError('--cls_data_path is required in finetune mode')

    # Load the pretrained vocabulary so we can map tokens -> indices
    vocab_file_path = os.path.join(args.model_path, 'vocab.json')
    with open(vocab_file_path, 'r', encoding='utf-8') as vocab_file:
        vocab_data = json.load(vocab_file)
    word2id = vocab_data['word2id']
    cls_token_id = word2id.get('<cls>', 4)
    pad_token_id = word2id.get('<pad>', 3)
    unk_token_id = word2id.get('<unk>', 0)

    # Load classification dataset
    all_token_sequences = []
    all_labels = []
    with open(args.cls_data_path, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            record = json.loads(line)
            tokens = record['tokens']
            label = record['label']
            # Map tokens to indices; prepend <cls>; truncate to max_ctx_size
            token_ids = [cls_token_id] + [
                word2id.get(token, unk_token_id) for token in tokens
            ]
            token_ids = token_ids[: args.max_ctx_size]
            all_token_sequences.append(token_ids)
            all_labels.append(label)

    print(f'Loaded {len(all_token_sequences)} classification samples')

    # Build a simple tensor dataset with padding
    def build_classification_dataloader(token_sequences, labels, batch_size, shuffle):
        max_len = max(len(seq) for seq in token_sequences)
        padded_ids = []
        padding_masks = []
        for seq in token_sequences:
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [pad_token_id] * pad_len)
            padding_masks.append([False] * len(seq) + [True] * pad_len)

        token_tensor   = torch.tensor(padded_ids,    dtype=torch.long)
        mask_tensor    = torch.tensor(padding_masks, dtype=torch.bool)
        label_tensor   = torch.tensor(labels,        dtype=torch.long)
        tensor_dataset = torch.utils.data.TensorDataset(token_tensor, mask_tensor, label_tensor)
        return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

    # Always split cls_data_path into train/eval by eval_split_ratio;
    # test_cls_data_path (if provided) is an independent held-out set evaluated after training.
    split_index = int(len(all_token_sequences) * (1.0 - args.eval_split_ratio))
    train_dataloader = build_classification_dataloader(
        all_token_sequences[:split_index], all_labels[:split_index],
        batch_size=args.batch_size, shuffle=True,
    )
    eval_dataloader = build_classification_dataloader(
        all_token_sequences[split_index:], all_labels[split_index:],
        batch_size=args.batch_size, shuffle=False,
    ) if split_index < len(all_token_sequences) else None
    print(f'Train: {split_index} samples, '
          f'Eval: {len(all_token_sequences) - split_index} samples '
          f'(eval_split_ratio={args.eval_split_ratio})')

    # Load held-out test set
    test_dataloader = None
    if args.test_cls_data_path is not None:
        test_token_sequences = []
        test_labels = []
        with open(args.test_cls_data_path, 'r', encoding='utf-8') as test_file:
            for line in test_file:
                record = json.loads(line)
                token_ids = [cls_token_id] + [
                    word2id.get(token, unk_token_id) for token in record['tokens']
                ]
                test_token_sequences.append(token_ids[: args.max_ctx_size])
                test_labels.append(record['label'])
        test_dataloader = build_classification_dataloader(
            test_token_sequences, test_labels, batch_size=args.batch_size, shuffle=False,
        )
        print(f'Loaded {len(test_token_sequences)} test samples from: {args.test_cls_data_path}')

    # Build PACEClassifier from pretrained embedding weights
    device = torch.device(args.device)
    print(f'\nLoading pretrained embedding from: {args.model_path}')
    cls_model = PACEClassifier.from_pretrained_embedding(
        save_dir=args.model_path,
        num_classes=args.num_classes,
        hidden_dim=args.classifier_hidden_dim,
        n_hidden_layers=args.classifier_n_hidden_layers,
        dropout=args.classifier_dropout,
        freeze_base=args.freeze_base,
        device=str(device),
    )
    print(cls_model)

    total_params     = sum(p.numel() for p in cls_model.parameters())
    trainable_params = sum(p.numel() for p in cls_model.parameters() if p.requires_grad)
    print(f'\nModel Statistics:')
    print(f'  Total parameters:     {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')

    # Fine-tune
    trainer = PACEClassificationTrainer(cls_model, device=str(device))
    print('\nStarting fine-tuning...')
    trainer.train(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        lr=args.lr,
        optimizer_type=args.optimizer,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader,
        eval_every_steps=args.eval_every_steps,
        save_dir=args.cls_save_dir,
    )


def main():
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'finetune':
        finetune(args)


if __name__ == '__main__':
    main()

