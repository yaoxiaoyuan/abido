"""
Tokenizer module — contains the base Tokenizer and BPETokenizer.

Tokenizer (base class)
  Greedy longest-match (Max-Match) tokenizer backed by a Trie, supporting:
  - Character-level / sub-word vocabularies
  - Atomic matching of chat-template special tokens (never split)
  - GPT-2-style pre-tokenization regex
  - N-Gram frequency-filtered training (single-pass, no iteration)
  - Serialization / deserialization

BPETokenizer (inherits Tokenizer)
  Byte Pair Encoding tokenizer.  Core data structures:
  Training  — each word in the corpus is stored as a doubly-linked list;
              a max-heap tracks the frequency of every adjacent pair;
              the highest-frequency pair is popped and merged each iteration,
              with the heap updated incrementally.
  Encoding  — a single word is stored as a doubly-linked list;
              a min-heap (lower rank = higher priority) drives greedy merging
              until no more mergeable pairs remain.
  Decoding  — look up the id→bytes mapping, concatenate, then UTF-8 decode.

Data-structure highlights:
  - SymbolNode       : doubly-linked list node; __slots__ reduces memory overhead
  - PairStats        : training heap node storing pair, frequency, and heap position
  - sift_up/sift_down: O(log n) in-place heap maintenance, no full rebuild needed
  - Encoding uses explicit iteration over the linked list instead of recursion
    to avoid Python's default recursion-depth limit
"""

from __future__ import annotations

import base64
import collections
import heapq
import itertools
import json
import os
import regex
import tempfile
from typing import Iterable, Literal

import torch


# ─────────────────────────────────────────────
# 1. Tokenizer (base class)
# ─────────────────────────────────────────────

class Tokenizer:
    """
    A character-level / sub-word tokenizer with a configurable special-token
    set and support for dialogue-format (chat template) tokens.

    By default four built-in special tokens are reserved at ids 0-3:
        <pad> = 0,  <bos> = 1,  <eos> = 2,  <unk> = 3

    Both the token strings and their ids are fully customisable via the
    ``special_tokens`` argument, so you can match any pre-existing vocabulary
    layout (e.g. GPT-2, LLaMA, Qwen) without renumbering.

    The ``vocab`` argument accepts either a ``list[str]`` (ids are assigned
    automatically, starting after the highest special-token id) or a
    ``dict[str, int]`` (ids are taken directly from the mapping, enabling
    exact reproduction of an existing vocabulary).

    **Dialogue / chat-template support**

    Role tags and control tokens used in chat templates (e.g. ``<|user|>``,
    ``<|assistant|>``, ``<|end|>``) must be registered as *extra special tokens*
    via :meth:`add_special_token`.  Extra special tokens are matched as atomic
    units during tokenisation — they are never split into individual characters
    even if the characters themselves are in the vocabulary.

    Usage — plain text (default special tokens)::

        tok = Tokenizer(["a", "b", "c", "hello", "world"])
        ids = tok.encode("hello world", add_bos=True, add_eos=True)
        text = tok.decode(ids)

    Usage — custom special tokens and ids::

        tok = Tokenizer(
            vocab={"hello": 10, "world": 11, " ": 12},
            special_tokens={
                "pad": ("<|pad|>", 0),
                "bos": ("<|startoftext|>", 1),
                "eos": ("<|endoftext|>", 2),
                "unk": ("<|unk|>", 3),
            },
        )

    Usage — dialogue / chat format::

        tok = Tokenizer(list("abcdefghijklmnopqrstuvwxyz .,!?"))
        for tag in ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]:
            tok.add_special_token(tag)

        prompt = (
            "<|system|>\nYou are a helpful assistant.<|end|>\n"
            "<|user|>\nHello!<|end|>\n"
            "<|assistant|>\n"
        )
        ids  = tok.encode(prompt, add_bos=True)
        text = tok.decode(ids, skip_special_tokens=False)
    """

    # Default special-token strings.  These are used when no custom
    # special_tokens mapping is supplied to __init__.
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    # Ordered list of the four default built-in special tokens.
    # Used only as a fallback when special_tokens is not provided.
    _DEFAULT_SPECIAL_TOKENS: list[str] = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    # Default GPT-2-style pre-tokenizer pattern.
    # Splits on: contractions ("'s", "'t", "'re" …), whitespace-prefixed words,
    # standalone punctuation, and digit runs — matching tiktoken / cl100k_base.
    _DEFAULT_PRETOK_PATTERN: str = (
        r"'s|'t|'re|'ve|'m|'ll|'d"
        r"| ?\w+"
        r"| ?[^\s\w]+"
        r"|\s+(?!\S)"
        r"|\s+"
    )

    def __init__(
        self,
        vocab: "list[str] | dict[str, int]",
        special_tokens: "dict[str, tuple[str, int]] | None" = None,
        pretok_pattern: str | None = _DEFAULT_PRETOK_PATTERN,
        extra_special_tokens: "list[str] | dict[str, int] | None" = None,
    ) -> None:
        """
        Args:
            vocab                : token vocabulary, in one of two forms:

                                   - ``list[str]``: token strings whose ids are
                                     assigned automatically, starting at
                                     ``max(special_token_ids) + 1`` and incrementing
                                     by 1.  Duplicates are silently ignored.
                                   - ``dict[str, int]``: explicit token → id mapping.
                                     Ids may be any non-negative integers and do not
                                     need to be contiguous.  Tokens that collide with
                                     a special-token id raise ``ValueError``.

            special_tokens       : mapping that customises the four built-in special
                                   tokens.  Keys are role names (``"pad"``, ``"bos"``,
                                   ``"eos"``, ``"unk"``); values are ``(token_str, id)``
                                   tuples.  Any key that is omitted falls back to the
                                   class-level default (``<pad>``/0, ``<bos>``/1,
                                   ``<eos>``/2, ``<unk>``/3).

                                   Example — match a GPT-2-style layout::

                                       special_tokens={
                                           "bos": ("<|endoftext|>", 50256),
                                           "eos": ("<|endoftext|>", 50256),
                                           "unk": ("<|endoftext|>", 50256),
                                           "pad": ("<|endoftext|>", 50256),
                                       }

            pretok_pattern       : regex pattern (string or compiled) used as a
                                   pre-tokenizer.  The text is first split into coarse
                                   "word-level" chunks; Max-Match is then applied within
                                   each chunk independently, preventing n-grams from
                                   spanning word boundaries.
                                   Defaults to a GPT-2-style pattern.  Pass ``None`` to
                                   disable pre-tokenization (pure character-level scan).

            extra_special_tokens : additional special tokens to register atomically
                                   (e.g. chat-template role tags).  Accepts either:
                                   - ``list[str]``: ids are auto-assigned after the
                                     highest existing id.
                                   - ``dict[str, int]``: explicit token → id mapping
                                     so that ids can be pinned to specific positions
                                     in the vocabulary.
        """
        # ── Resolve special tokens ────────────────────────────────────────────
        # Build the four built-in special token (string, id) pairs, merging any
        # caller-supplied overrides on top of the class-level defaults.
        _defaults: dict[str, tuple[str, int]] = {
            "pad": (self.PAD_TOKEN, 0),
            "bos": (self.BOS_TOKEN, 1),
            "eos": (self.EOS_TOKEN, 2),
            "unk": (self.UNK_TOKEN, 3),
        }
        if special_tokens:
            for role, value in special_tokens.items():
                if role not in _defaults:
                    raise ValueError(
                        f"Unknown special-token role {role!r}. "
                        "Valid roles: 'pad', 'bos', 'eos', 'unk'."
                    )
                _defaults[role] = value

        pad_str, pad_id_val = _defaults["pad"]
        bos_str, bos_id_val = _defaults["bos"]
        eos_str, eos_id_val = _defaults["eos"]
        unk_str, unk_id_val = _defaults["unk"]

        # Store resolved special-token strings as instance attributes so that
        # properties (pad_id, bos_id, …) and _build_tries() can reference them.
        self.PAD_TOKEN = pad_str  # type: ignore[assignment]
        self.BOS_TOKEN = bos_str  # type: ignore[assignment]
        self.EOS_TOKEN = eos_str  # type: ignore[assignment]
        self.UNK_TOKEN = unk_str  # type: ignore[assignment]

        # _builtin_special_tokens: set of the four built-in token strings.
        # Used by _build_tries() to exclude them from the regular vocab Trie
        # (they are never matched by the greedy scanner; only via add_special_token).
        self._builtin_special_tokens: set[str] = {pad_str, bos_str, eos_str, unk_str}

        # ── Build id ↔ token mappings ─────────────────────────────────────────
        # Start with an empty sparse mapping; fill in special tokens first,
        # then user vocab.  We use a dict for _id_to_token_map (sparse) and
        # reconstruct _id_to_token (dense list) at the end.
        id_to_token_map: dict[int, str] = {}
        token_to_id_map: dict[str, int] = {}

        # Insert the four built-in special tokens (may share a single id, e.g. GPT-2).
        for token_str, token_id in [
            (pad_str, pad_id_val),
            (bos_str, bos_id_val),
            (eos_str, eos_id_val),
            (unk_str, unk_id_val),
        ]:
            id_to_token_map[token_id] = token_str
            token_to_id_map[token_str] = token_id

        # Determine the next auto-assign id for list-mode vocab.
        # It must be strictly greater than every special-token id.
        special_ids = {pad_id_val, bos_id_val, eos_id_val, unk_id_val}
        next_auto_id = max(special_ids) + 1

        if isinstance(vocab, dict):
            # dict mode: caller provides explicit token → id mapping.
            for token_str, token_id in vocab.items():
                if token_str in token_to_id_map:
                    continue  # already a special token — skip silently
                if token_id in id_to_token_map and id_to_token_map[token_id] != token_str:
                    raise ValueError(
                        f"Vocab id {token_id} is already assigned to "
                        f"{id_to_token_map[token_id]!r}; cannot also assign it to {token_str!r}."
                    )
                id_to_token_map[token_id] = token_str
                token_to_id_map[token_str] = token_id
        else:
            # list mode: assign ids sequentially starting at next_auto_id.
            for token_str in vocab:
                if token_str in token_to_id_map:
                    continue  # duplicate or special token — skip
                id_to_token_map[next_auto_id] = token_str
                token_to_id_map[token_str] = next_auto_id
                next_auto_id += 1

        # Build the dense _id_to_token list (indexed by id).
        # Gaps in the id space are filled with UNK so that id_to_token()
        # never raises an IndexError for a valid-range id.
        max_id = max(id_to_token_map.keys()) if id_to_token_map else 0
        self._id_to_token: list[str] = [unk_str] * (max_id + 1)
        for token_id, token_str in id_to_token_map.items():
            self._id_to_token[token_id] = token_str
        self._token_to_id: dict[str, int] = token_to_id_map

        # Extra special tokens registered via add_special_token().
        # These are matched atomically in _tokenize before greedy longest-match,
        # so role tags like <|user|> are never split into individual characters.
        self._extra_special_tokens: set[str] = set()

        # Compile the pre-tokenizer pattern once at construction time.
        if pretok_pattern is None:
            self._pretok_regex = None
        elif isinstance(pretok_pattern, str):
            self._pretok_regex = regex.compile(pretok_pattern)
        else:
            self._pretok_regex = pretok_pattern

        # Cached Tries — rebuilt lazily whenever the vocabulary changes.
        # _special_trie : covers self._extra_special_tokens (Priority 1)
        # _vocab_trie   : covers the regular vocabulary (Priority 2)
        self._special_trie: dict = {}
        self._vocab_trie:   dict = {}
        self._build_tries()

        # Register extra special tokens (chat-template tags, control tokens, etc.)
        if extra_special_tokens is not None:
            self.add_special_tokens(extra_special_tokens)

    # ── Trie cache ──────────────────────────────────────────────────────────

    def _build_tries(self) -> None:
        """Rebuild both Tries from the current vocabulary and special tokens.

        Called once at construction time and again whenever the vocabulary or
        the set of extra special tokens changes (add_token / add_special_token).
        """
        # Trie 1: all special tokens (Priority 1) — builtin + extra
        # Both sets are matched atomically so they are never split by the
        # greedy vocab scanner or the BPE pre-tokenizer.
        special_trie: dict = {}
        for token in (*self._builtin_special_tokens, *self._extra_special_tokens):
            node = special_trie
            for char in token:
                node = node.setdefault(char, {})
            node[None] = token
        self._special_trie = special_trie

        # Trie 2: regular vocabulary (Priority 2, excludes built-in specials)
        vocab_trie: dict = {}
        for token in self._token_to_id:
            if token in self._builtin_special_tokens:
                continue
            node = vocab_trie
            for char in token:
                node = node.setdefault(char, {})
            node[None] = token
        self._vocab_trie = vocab_trie

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Total number of tokens including special tokens."""
        return len(self._id_to_token)

    @property
    def pad_id(self) -> int:
        return self._token_to_id[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self._token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._token_to_id[self.EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self._token_to_id[self.UNK_TOKEN]

    # ── Core API ────────────────────────────────────────────────────────────

    def token_to_id(self, token: str) -> int:
        """Return the id for *token*, or unk_id if not in vocabulary."""
        return self._token_to_id.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        """Return the token string for *token_id*."""
        if token_id < 0 or token_id >= len(self._id_to_token):
            return self.UNK_TOKEN
        return self._id_to_token[token_id]

    # ── Tokenisation helpers ────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """
        Split *text* into a list of token strings.

        Tokenisation has three priority levels applied left-to-right:

        **Priority 0 — Regex pre-tokenizer** (when ``pretok_pattern`` is set):
        The text is first split into coarse "word-level" chunks using the
        compiled regex.  Each chunk is then tokenised independently by
        priorities 1 and 2 below.  This prevents sub-word tokens from
        spanning word boundaries (e.g. the n-gram ``"dog the"`` can never
        appear as a single token).  Extra special tokens are matched *before*
        the regex split so that role tags like ``<|user|>`` are never fed into
        the regex engine.

        **Priority 1 — extra special tokens** (registered via
        :meth:`add_special_token`): checked first at every position.  If any
        registered special token starts at the current position it is emitted
        as a single atomic token, regardless of length.  This ensures that
        role tags like ``<|user|>`` or ``<|end|>`` are never split into
        individual characters even when those characters are in the vocabulary.
        The longest matching special token wins when multiple candidates start
        at the same position.

        **Priority 2 — greedy longest-match (Max-Match)** against the regular
        vocabulary: used when no special token matches.  Tries decreasing
        window sizes from *max_token_len* down to 1.  Unknown characters (not
        in the vocab at all) are emitted as-is and will be mapped to ``<unk>``
        by :meth:`token_to_id`.
        """
        # Use the pre-built Tries cached on the instance.
        # They are rebuilt automatically by _build_tries() whenever the
        # vocabulary or extra special tokens change.
        # ── Priority 0: regex pre-tokenizer ─────────────────────────────────
        if self._pretok_regex is not None:
            return self._tokenize_with_pretok(text)

        # ── No pre-tokenizer: scan the full text directly ────────────────────
        return self._max_match_chunk(text)

    def _trie_longest_match(self, trie: dict, text: str, pos: int) -> "str | None":
        """Return the longest token in *trie* starting at *pos*, or None."""
        node = trie
        last_match: "str | None" = None
        scan = pos
        text_len = len(text)
        while scan < text_len:
            char = text[scan]
            if char not in node:
                break
            node = node[char]
            scan += 1
            if None in node:
                last_match = node[None]
        return last_match

    def _max_match_chunk(self, chunk: str) -> list[str]:
        """Apply priority-1 (special tokens) + priority-2 (vocab) scan to one chunk."""
        chunk_tokens: list[str] = []
        pos = 0
        chunk_len = len(chunk)
        while pos < chunk_len:
            # Priority 1: extra special tokens via Trie — O(token_length).
            special_match = self._trie_longest_match(self._special_trie, chunk, pos)
            if special_match is not None:
                chunk_tokens.append(special_match)
                pos += len(special_match)
                continue

            # Priority 2: greedy longest-match via vocab Trie — O(token_length).
            vocab_match = self._trie_longest_match(self._vocab_trie, chunk, pos)
            if vocab_match is not None:
                chunk_tokens.append(vocab_match)
                pos += len(vocab_match)
            else:
                # Unknown character: emit as-is (will map to <unk>)
                chunk_tokens.append(chunk[pos])
                pos += 1
        return chunk_tokens

    def _tokenize_with_pretok(self, text: str) -> list[str]:
        """Apply the regex pre-tokenizer while preserving special tokens atomically.

        Special tokens (role tags etc.) are matched first via the special Trie so
        they are never split by the regex.  The remaining text between special
        tokens is segmented by the regex and each chunk is then processed by
        :meth:`_max_match_chunk` for vocab longest-match.

        Args:
            text : raw input string

        Returns:
            List of token strings.
        """
        tokens: list[str] = []
        remaining = text
        while remaining:
            # Check for a leading special token via Trie — O(token_length).
            leading_special = self._trie_longest_match(self._special_trie, remaining, 0)
            if leading_special is not None:
                tokens.append(leading_special)
                remaining = remaining[len(leading_special):]
                continue

            # Find the next special token boundary using Trie scan — O(L).
            next_special_pos = len(remaining)
            for scan_pos in range(1, len(remaining)):
                match = self._trie_longest_match(self._special_trie, remaining, scan_pos)
                if match is not None:
                    next_special_pos = scan_pos
                    break

            # Apply regex pre-tokenizer to the segment before the next special token.
            segment   = remaining[:next_special_pos]
            remaining = remaining[next_special_pos:]
            for chunk in self._pretok_regex.findall(segment):
                tokens.extend(self._max_match_chunk(chunk))
        return tokens

    # ── Core API ────────────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: bool = False,
        return_offsets: bool = False,
    ) -> "list[int] | tuple":
        """
        Encode *text* into a list of token ids.

        Tokenisation uses greedy longest-match against the vocabulary, so
        multi-character tokens (sub-words, emoji, punctuation clusters …) are
        matched before falling back to single characters.  Characters that are
        not in the vocabulary at all are mapped to <unk>.

        Args:
            text           : input string
            add_bos        : prepend BOS token
            add_eos        : append EOS token
            max_length     : if set, the output is truncated to this many tokens
                             (BOS/EOS are counted toward the limit)
            truncation     : must be True to activate max_length truncation;
                             raises ValueError if max_length is set but
                             truncation is False and the sequence is too long
            return_tensors : if True, return a 1-D torch.LongTensor instead of
                             a plain list
            return_offsets : if True, also return a list of (start, end) char
                             spans for each token in the original string
                             (BOS/EOS get span (-1, -1))

        Returns:
            ids                          when return_offsets=False
            (ids, offset_mapping)        when return_offsets=True
            In both cases ids is a list[int] or torch.LongTensor depending on
            return_tensors.
        """
        raw_tokens = self._tokenize(text)

        # Build offset mapping before adding special tokens.
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for token in raw_tokens:
            # Find the token in the original string starting from cursor.
            # Because _tokenize is left-to-right this is always a forward scan.
            start = text.index(token, cursor)
            end   = start + len(token)
            offsets.append((start, end))
            cursor = end

        ids = [self.token_to_id(t) for t in raw_tokens]

        if add_bos:
            ids     = [self.bos_id] + ids
            offsets = [(-1, -1)] + offsets
        if add_eos:
            ids     = ids + [self.eos_id]
            offsets = offsets + [(-1, -1)]

        # Truncation
        if max_length is not None:
            if len(ids) > max_length:
                if not truncation:
                    raise ValueError(
                        f"Sequence length {len(ids)} exceeds max_length={max_length}. "
                        "Pass truncation=True to enable automatic truncation."
                    )
                ids     = ids[:max_length]
                offsets = offsets[:max_length]

        if return_tensors:
            tensor_ids = torch.tensor(ids, dtype=torch.long)
            if return_offsets:
                return tensor_ids, offsets
            return tensor_ids

        if return_offsets:
            return ids, offsets
        return ids

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_max: bool = True,
        return_tensors: bool = False,
    ) -> "tuple[list[list[int]], list[int]]":
        """
        Encode a batch of strings with optional truncation and right-padding.

        Args:
            texts          : list of input strings
            add_bos        : prepend BOS to each sequence
            add_eos        : append EOS to each sequence
            max_length     : truncate each sequence to this many tokens
            truncation     : must be True to activate truncation
            pad_to_max     : right-pad all sequences to the same length with
                             pad_id (default True)
            return_tensors : if True, return a 2-D torch.LongTensor of shape
                             [batch, max_seq_len] instead of a nested list

        Returns:
            (encoded, lengths) where *lengths* holds the original (pre-pad)
            length of each sequence (including BOS/EOS, after truncation).
        """
        encoded = [
            self.encode(text, add_bos=add_bos, add_eos=add_eos,
                        max_length=max_length, truncation=truncation)
            for text in texts
        ]
        lengths = [len(seq) for seq in encoded]

        if pad_to_max and len(encoded) > 1:
            max_len = max(lengths)
            encoded = [seq + [self.pad_id] * (max_len - len(seq)) for seq in encoded]

        if return_tensors:
            return torch.tensor(encoded, dtype=torch.long), lengths

        return encoded, lengths

    def decode(
        self,
        ids,
        skip_special_tokens: bool = True,
        stop_at_eos: bool = True,
        join_with: str = "",
    ) -> str:
        """
        Decode a sequence of token ids back to a string.

        Args:
            ids                 : list[int] or torch.Tensor of token ids
            skip_special_tokens : omit PAD / BOS / UNK from the output
                                  (EOS is always handled by stop_at_eos)
            stop_at_eos         : stop decoding when EOS is encountered
                                  (default True)
            join_with           : string used to join token strings; default ""
                                  (character-level concatenation).  Use " " for
                                  space-separated word-level vocabularies.

        Returns:
            Decoded string.
        """
        # Accept torch.Tensor transparently.
        if hasattr(ids, "tolist"):
            ids = ids.tolist()

        special_ids = {self.pad_id, self.bos_id, self.unk_id}
        tokens: list[str] = []
        for token_id in ids:
            if stop_at_eos and token_id == self.eos_id:
                break
            if skip_special_tokens and token_id in special_ids:
                continue
            tokens.append(self.id_to_token(token_id))
        return join_with.join(tokens)

    def decode_batch(
        self,
        batch_ids,
        skip_special_tokens: bool = True,
        stop_at_eos: bool = True,
        join_with: str = "",
    ) -> list[str]:
        """
        Decode a batch of token-id sequences.

        Args:
            batch_ids           : list[list[int]] or 2-D torch.Tensor
            skip_special_tokens : passed through to decode()
            stop_at_eos         : passed through to decode()
            join_with           : passed through to decode()
        """
        # Accept 2-D tensors.
        if hasattr(batch_ids, "tolist"):
            batch_ids = batch_ids.tolist()
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens,
                        stop_at_eos=stop_at_eos, join_with=join_with)
            for ids in batch_ids
        ]

    # ── Vocabulary helpers ───────────────────────────────────────────────────

    def add_special_token(self, token: str, token_id: int | None = None) -> int:
        """
        Register *token* as an extra special token.

        Extra special tokens are matched atomically during tokenisation
        (Priority 1 in :meth:`_tokenize`) so they are never split into
        individual characters.  Typical use cases:

        - Chat-template role tags: ``<|user|>``, ``<|assistant|>``, ``<|end|>``
        - Control tokens: ``<|im_start|>``, ``<|im_end|>``, ``[INST]``, ``[/INST]``

        The token is also added to the regular vocabulary (so it gets an id)
        if it is not already present.

        Args:
            token    : the special token string to register
            token_id : explicit id to assign.  If the token is already in the
                       vocabulary its existing id is kept and this argument is
                       ignored.  If ``None``, the next available id is used
                       (auto-increment via :meth:`add_token`).

        Returns:
            The token id (existing, explicitly assigned, or newly auto-assigned).
        """
        if token_id is not None and token not in self._token_to_id:
            assigned_id = self._add_token_with_id(token, token_id)
        else:
            assigned_id = self.add_token(token)
        self._extra_special_tokens.add(token)
        # add_token / _add_token_with_id already rebuilds _vocab_trie;
        # rebuild _special_trie too.
        self._build_tries()
        return assigned_id

    def add_special_tokens(self, tokens: "list[str] | dict[str, int]") -> list[int]:
        """
        Register multiple extra special tokens at once.

        Args:
            tokens : either a ``list[str]`` (ids are auto-assigned) or a
                     ``dict[str, int]`` mapping token string → explicit id.

        Returns:
            List of token ids in the same order as *tokens*.
        """
        if isinstance(tokens, dict):
            return [self.add_special_token(tok, tid) for tok, tid in tokens.items()]
        return [self.add_special_token(tok) for tok in tokens]

    def _add_token_with_id(self, token: str, token_id: int) -> int:
        """
        Add *token* to the vocabulary with an explicit *token_id*.

        Raises ``ValueError`` if *token_id* is already occupied by a different
        token.  If *token* is already in the vocabulary its existing id is
        returned unchanged.

        Returns:
            The assigned token id.
        """
        if token in self._token_to_id:
            return self._token_to_id[token]
        if token_id < len(self._id_to_token) and self._id_to_token[token_id] != self.UNK_TOKEN:
            existing = self._id_to_token[token_id]
            raise ValueError(
                f"Token id {token_id} is already occupied by {existing!r}; "
                f"cannot assign it to {token!r}."
            )
        # Extend the dense list if necessary to cover token_id.
        if token_id >= len(self._id_to_token):
            self._id_to_token.extend([self.UNK_TOKEN] * (token_id - len(self._id_to_token) + 1))
        self._id_to_token[token_id] = token
        self._token_to_id[token] = token_id
        self._build_tries()
        return token_id

    def add_token(self, token: str) -> int:
        """
        Add a new token to the vocabulary if it is not already present.

        The new id is assigned as ``len(self._id_to_token)``, which equals
        ``max_existing_id + 1`` because the dense list is always extended to
        cover the full id range.

        Returns the id of the token (existing or newly assigned).
        """
        if token in self._token_to_id:
            return self._token_to_id[token]
        new_id = len(self._id_to_token)
        self._id_to_token.append(token)
        self._token_to_id[token] = new_id
        # Vocabulary changed — rebuild the cached Tries.
        self._build_tries()
        return new_id

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, "
            f"pad={self.pad_id}, bos={self.bos_id}, "
            f"eos={self.eos_id}, unk={self.unk_id})"
        )

    # ── Training (Frequency-Filtered N-Gram) ────────────────────────────────

    @classmethod
    def train(
        cls,
        texts: "list[str] | Iterable[str]",
        vocab_size: int,
        min_frequency: "int | dict[int, int]" = 50,
        max_ngram_len: int = 6,
        length_score_alpha: float = 0.75,
        chunk_size: int = 10_000,
        pretok_pattern: "str | None" = _DEFAULT_PRETOK_PATTERN,
        show_progress: bool = True,
    ) -> "Tokenizer":
        """
        Train a Frequency-Filtered N-Gram tokenizer from scratch on *texts*.

        This is significantly faster than BPE because it requires only a single
        pass over the corpus to collect n-gram frequencies, rather than BPE's
        iterative merge loop.

        The algorithm:
          1. Collect every unique character (unigram) from the corpus — these
             form the character-level fallback vocabulary.
          2. Optionally apply a regex pre-tokenizer to split each text into
             coarse word-level chunks.  N-gram statistics are collected
             *within* each chunk, so tokens never span word boundaries.
          3. Count all n-grams of length 2 … *max_ngram_len* across the corpus
             using a single streaming pass with ``collections.Counter``.
          4. Discard any n-gram that appears fewer than its length-specific
             threshold (see *min_frequency*).
          5. Rank surviving n-grams by a **length-normalised score** and add
             the top-k to the vocabulary until *vocab_size* is reached.

        **Length-aware filtering and ranking**

        Short n-grams (e.g. 2-grams) appear far more often than long ones by
        construction, so a single global frequency threshold would over-retain
        short tokens and over-prune long ones.  Two mechanisms address this:

        - *Per-length thresholds* (``min_frequency``): pass a ``dict`` mapping
          n-gram length → minimum count, e.g. ``{2: 200, 3: 80, 4: 30, 5: 10,
          6: 5}``.  A plain ``int`` applies the same threshold to all lengths.

        - *Length-normalised ranking score*: instead of sorting by raw count,
          each candidate is scored as::

              score = count / (total_count_for_this_length ** alpha)

          where ``total_count_for_this_length`` is the sum of all counts for
          n-grams of the same length and ``alpha`` (default 0.75) controls how
          aggressively the normalisation penalises short n-grams.  This puts
          n-grams of different lengths on a comparable scale so that a highly
          distinctive 5-gram can outrank a mediocre 2-gram.

        Tokenisation (at inference time) uses the existing greedy longest-match
        algorithm in :meth:`_tokenize`, which is equivalent to Max-Match and
        naturally prefers longer (higher-quality) tokens over shorter ones.

        Args:
            texts              : iterable of raw text strings (may be a generator)
            vocab_size         : target vocabulary size (including the 4 special tokens)
            min_frequency      : minimum occurrence count to keep an n-gram.
                                 - ``int``: same threshold for all lengths (default 50)
                                 - ``dict[int, int]``: per-length thresholds, e.g.
                                   ``{2: 200, 3: 80, 4: 30, 5: 10, 6: 5}``.
                                   Missing lengths fall back to the value at the
                                   nearest smaller key, or 1 if none exists.
            max_ngram_len      : maximum n-gram length to consider (default 6)
            length_score_alpha : exponent for length-normalised ranking (default 0.75).
                                 0 = raw frequency (no normalisation),
                                 1 = fully normalised by total mass of that length.
            chunk_size         : number of texts to buffer in RAM at once
            pretok_pattern     : regex pattern used to split each text into word-level
                                 chunks before n-gram counting.  N-grams are counted
                                 within each chunk independently, preventing cross-word
                                 tokens like ``"dog the"``.  Defaults to the same
                                 GPT-2-style pattern used at inference time.
                                 Pass None to count n-grams over the raw text.
            show_progress      : print a summary line after vocabulary construction

        Returns:
            A fully initialised Tokenizer whose vocabulary contains all learned
            n-gram tokens plus the 4 special tokens.
        """
        # Compile the pre-tokenizer pattern once (shared across all texts).
        pretok_regex = regex.compile(pretok_pattern) if pretok_pattern else None

        # ── Step 1: single pass — collect all n-gram counts (length 1..max) ──
        # Single characters (length-1) are counted in the same Counter as
        # longer n-grams so they go through the same frequency filter and
        # length-normalised ranking in Steps 3-5.  No special-casing needed.
        # When a pre-tokenizer is active, counts are collected *within* each
        # pre-tok chunk so tokens never span word boundaries.
        ngram_counts: collections.Counter = collections.Counter()

        def _count_ngrams_in_chunk(chunk_text: str) -> None:
            """Count all n-grams of length 1..max_ngram_len within chunk_text."""
            chunk_len = len(chunk_text)
            for ngram_len in range(1, max_ngram_len + 1):
                for start in range(chunk_len - ngram_len + 1):
                    ngram_counts[chunk_text[start : start + ngram_len]] += 1

        text_iter = iter(texts)
        num_texts_seen = 0
        while True:
            batch = list(itertools.islice(text_iter, chunk_size))
            if not batch:
                break
            num_texts_seen += len(batch)
            for text in batch:
                if pretok_regex is not None:
                    # Count within each pre-tok chunk independently.
                    for word_chunk in pretok_regex.findall(text):
                        _count_ngrams_in_chunk(word_chunk)
                else:
                    _count_ngrams_in_chunk(text)

        unique_chars = sum(1 for ng in ngram_counts if len(ng) == 1)
        if show_progress:
            print(
                f"  N-Gram scan complete | texts={num_texts_seen} | "
                f"unique chars={unique_chars} | "
                f"raw n-grams={len(ngram_counts)} | "
                f"pre-tokenizer={'on' if pretok_regex else 'off'}"
            )

        # ── Step 2: initialise tokenizer with empty vocab ────────────────────
        # All tokens (chars and n-grams alike) are selected in Steps 3-5 via
        # the unified candidates list.  ASCII printable chars (0x20-0x7E) are
        # injected into ngram_counts with a sentinel count of max+1 so they
        # always survive the frequency filter and rank at the top of length-1.
        _ASCII_PRINTABLE = set(chr(i) for i in range(0x20, 0x7F))
        _sentinel_count = max(ngram_counts.values(), default=1) + 1
        for ch in _ASCII_PRINTABLE:
            if ch not in ngram_counts:
                ngram_counts[ch] = _sentinel_count

        tokenizer = cls([], pretok_pattern=pretok_pattern)
        slots_remaining = vocab_size - tokenizer.vocab_size

        if slots_remaining <= 0:
            return tokenizer  # special tokens alone already fill the vocab

        # ── Step 3: per-length threshold lookup ─────────────────────────────
        # Build a fast length → threshold mapping from min_frequency.
        # dict input: use exact value for each length; fall back to the nearest
        # smaller key, or 1 if no smaller key exists.
        # int input: same threshold for every length.
        if isinstance(min_frequency, dict):
            sorted_threshold_keys = sorted(min_frequency.keys())

            def _threshold_for_len(ngram_len: int) -> int:
                # Binary-search for the largest key <= ngram_len.
                lo, hi = 0, len(sorted_threshold_keys) - 1
                result = 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if sorted_threshold_keys[mid] <= ngram_len:
                        result = min_frequency[sorted_threshold_keys[mid]]
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return result
        else:
            def _threshold_for_len(ngram_len: int) -> int:  # type: ignore[misc]
                return min_frequency

        # ── Step 4: filter candidates by per-length threshold ────────────────
        candidates = [
            (ngram, count)
            for ngram, count in ngram_counts.items()
            if count >= _threshold_for_len(len(ngram))
        ]

        # ── Step 5: length-normalised ranking ────────────────────────────────
        # Raw frequency is not comparable across lengths: a 2-gram with count
        # 5000 may be less "distinctive" than a 5-gram with count 200.
        # We normalise each count by the total mass of its length bucket raised
        # to the power alpha, giving a relative-frequency-like score.
        #
        #   score = count / (total_mass_for_length ** alpha)
        #
        # alpha=0   → pure raw frequency (original behaviour)
        # alpha=0.75 → partial normalisation (default, balances lengths well)
        # alpha=1   → fully normalised (proportional to relative frequency)
        length_total_mass: dict[int, float] = {}
        for ngram, count in candidates:
            ngram_len = len(ngram)
            length_total_mass[ngram_len] = length_total_mass.get(ngram_len, 0.0) + count

        def _score(ngram: str, count: int) -> float:
            total_mass = length_total_mass.get(len(ngram), 1.0)
            normaliser = total_mass ** length_score_alpha if total_mass > 0 else 1.0
            return count / normaliser

        # Sort descending by normalised score; break ties by length (longer
        # preferred) then lexicographically for full determinism.
        candidates.sort(
            key=lambda item: (_score(item[0], item[1]), len(item[0]), item[0]),
            reverse=True,
        )

        added = 0
        for ngram, count in candidates:
            if added >= slots_remaining:
                break
            tokenizer.add_token(ngram)
            added += 1

        if show_progress:
            print(
                f"  Vocabulary built | total={tokenizer.vocab_size} | "
                f"n-grams added={added} | "
                f"candidates above min_freq={len(candidates)} | "
                f"length_score_alpha={length_score_alpha}"
            )

        return tokenizer

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, save_dir: str) -> None:
        """
        Save the tokenizer vocabulary and configuration to *save_dir*.

        Creates one file:
          - ``tokenizer.json`` — vocabulary list, extra special tokens,
            EOS override, and pre-tokenizer pattern as JSON.

        The directory is created automatically if it does not exist.

        Usage::

            tok.save("checkpoints/my_model")
            # Later:
            tok = Tokenizer.from_pretrained("checkpoints/my_model")
        """
        os.makedirs(save_dir, exist_ok=True)

        # Collect the pre-tokenizer pattern string (not the compiled regex).
        pretok_pattern: str | None = None
        if self._pretok_regex is not None:
            pretok_pattern = self._pretok_regex.pattern

        tokenizer_state = {
            # Full vocabulary as a {token: id} dict (preserves sparse id spaces).
            "vocab": self._token_to_id,
            # Built-in special token configuration: role → [token_str, id].
            "special_tokens": {
                "pad": [self.PAD_TOKEN, self._token_to_id[self.PAD_TOKEN]],
                "bos": [self.BOS_TOKEN, self._token_to_id[self.BOS_TOKEN]],
                "eos": [self.EOS_TOKEN, self._token_to_id[self.EOS_TOKEN]],
                "unk": [self.UNK_TOKEN, self._token_to_id[self.UNK_TOKEN]],
            },
            # Extra special tokens (role tags, control tokens …).
            # Stored as {token_str: id} so that explicit ids survive save/load.
            "extra_special_tokens": {
                tok: self._token_to_id[tok]
                for tok in sorted(self._extra_special_tokens)
            },
            # Pre-tokenizer regex pattern string (or null).
            "pretok_pattern": pretok_pattern,
        }

        tokenizer_path = os.path.join(save_dir, "tokenizer.json")
        with open(tokenizer_path, "w", encoding="utf-8") as tokenizer_file:
            json.dump(tokenizer_state, tokenizer_file, indent=2, ensure_ascii=False)

        print(f"  Tokenizer saved → {tokenizer_path}  (vocab_size={self.vocab_size})")

    @classmethod
    def from_pretrained(cls, save_dir: str) -> "Tokenizer":
        """
        Reconstruct a :class:`Tokenizer` from a directory created by :meth:`save`.

        Loads ``tokenizer.json`` and restores the full vocabulary, extra special
        tokens, EOS override, and pre-tokenizer pattern.  No tokenizer instance
        is required beforehand.

        Args:
            save_dir : path to the directory produced by :meth:`save`

        Returns:
            A fully initialised :class:`Tokenizer` identical to the saved one.

        Usage::

            tok = Tokenizer.from_pretrained("checkpoints/my_model")
        """
        tokenizer_path = os.path.join(save_dir, "tokenizer.json")
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"tokenizer.json not found in {save_dir!r}")

        with open(tokenizer_path, "r", encoding="utf-8") as tokenizer_file:
            state = json.load(tokenizer_file)

        # Restore special-token configuration (new format) or fall back to
        # the legacy list-based format for backwards compatibility.
        raw_special = state.get("special_tokens")
        if raw_special is not None:
            special_tokens = {role: (val[0], val[1]) for role, val in raw_special.items()}
        else:
            special_tokens = None

        # Restore vocabulary.  New format stores a {token: id} dict; the legacy
        # format stored a list (ids were positional).
        raw_vocab = state["vocab"]
        if isinstance(raw_vocab, dict):
            # New format: pass the dict directly, excluding built-in special tokens
            # (they are reconstructed from special_tokens above).
            builtin_strs = (
                {v[0] for v in raw_special.values()} if raw_special
                else set(cls._DEFAULT_SPECIAL_TOKENS)
            )
            user_vocab: dict[str, int] = {
                tok: tok_id for tok, tok_id in raw_vocab.items()
                if tok not in builtin_strs
            }
        else:
            # Legacy list format: skip the first 4 entries (built-in specials).
            legacy_list: list[str] = raw_vocab
            user_vocab = legacy_list[len(cls._DEFAULT_SPECIAL_TOKENS):]  # type: ignore[assignment]

        # Restore extra special tokens: {token_str: id}
        raw_extra = state.get("extra_special_tokens", {})
        extra_special_tokens = {tok: tid for tok, tid in raw_extra.items()} or None

        tokenizer = cls(
            vocab=user_vocab,
            special_tokens=special_tokens,
            pretok_pattern=state.get("pretok_pattern"),
            extra_special_tokens=extra_special_tokens,
        )

        print(f"  Tokenizer loaded ← {tokenizer_path}  (vocab_size={tokenizer.vocab_size})")
        return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 1. Linked-list node
# ─────────────────────────────────────────────────────────────────────────────

class SymbolNode:
    """
    Doubly-linked list node representing one symbol (a byte sequence or a
    merged sub-word) during BPE processing.

    Uses __slots__ to avoid __dict__ overhead, which significantly reduces
    memory usage on large corpora.
    """
    __slots__ = ("symbol", "left", "right")

    def __init__(
        self,
        symbol: bytes,
        left: "SymbolNode | None" = None,
        right: "SymbolNode | None" = None,
    ) -> None:
        self.symbol = symbol
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"SymbolNode({self.symbol!r})"


def _linked_list_to_symbols(head: SymbolNode) -> list[bytes]:
    """
    Traverse the linked list from *head* to the tail iteratively,
    collecting every symbol into a list.

    Iterative traversal avoids Python's default recursion limit (1 000),
    making it safe for very long word sequences (e.g. base64-encoded text).
    """
    result: list[bytes] = []
    node: SymbolNode | None = head
    while node is not None:
        result.append(node.symbol)
        node = node.right
    return result


def _build_linked_list(byte_sequence: bytes) -> SymbolNode:
    """
    Build a doubly-linked list from *byte_sequence*, with each byte
    becoming one initial symbol node.

    Returns:
        The head node (corresponding to the first byte).
    """
    head = SymbolNode(byte_sequence[:1])
    prev = head
    for i in range(1, len(byte_sequence)):
        node = SymbolNode(byte_sequence[i : i + 1], left=prev)
        prev.right = node
        prev = node
    return head


# ─────────────────────────────────────────────────────────────────────────────
# 2. Training phase: max-heap node and heap maintenance
# ─────────────────────────────────────────────────────────────────────────────

class PairStats:
    """
    Node in the training max-heap, tracking the occurrence frequency of
    one symbol pair.

    Attributes:
        pair      : (left_symbol, right_symbol) bytes tuple
        frequency : current occurrence count of this pair in the corpus
        heap_pos  : current index of this node in the heap array (enables O(1) lookup)
    """
    __slots__ = ("pair", "frequency", "heap_pos")

    def __init__(self, pair: tuple[bytes, bytes], frequency: int, heap_pos: int) -> None:
        self.pair = pair
        self.frequency = frequency
        self.heap_pos = heap_pos

    @property
    def priority(self) -> int:
        """Max-heap comparison key: higher frequency = higher priority."""
        return self.frequency


def _heap_swap(heap: list[PairStats], i: int, j: int) -> None:
    """Swap two nodes in the heap and update their heap_pos fields accordingly."""
    heap[i], heap[j] = heap[j], heap[i]
    heap[i].heap_pos = i
    heap[j].heap_pos = j


def _sift_up(heap: list[PairStats], index: int) -> None:
    """Bubble the node at *index* upward to restore the max-heap invariant."""
    while index > 0:
        parent = (index - 1) // 2
        if heap[index].priority <= heap[parent].priority:
            break
        _heap_swap(heap, index, parent)
        index = parent


def _sift_down(heap: list[PairStats], index: int) -> None:
    """Sink the node at *index* downward to restore the max-heap invariant."""
    heap_size = len(heap)
    while True:
        left_child = 2 * index + 1
        right_child = 2 * index + 2
        largest = index

        if left_child < heap_size and heap[left_child].priority > heap[largest].priority:
            largest = left_child
        if right_child < heap_size and heap[right_child].priority > heap[largest].priority:
            largest = right_child

        if largest == index:
            break

        _heap_swap(heap, index, largest)
        index = largest


def _update_pair_heap(
    heap: list[PairStats],
    heap_index: dict[tuple[bytes, bytes], PairStats],
    frequency_deltas: dict[tuple[bytes, bytes], int],
) -> None:
    """
    Apply *frequency_deltas* to the heap in bulk, maintaining the max-heap
    invariant after each update.

    New pairs are inserted as fresh PairStats nodes; existing pairs are
    updated in-place and then sifted up or down as needed.

    Args:
        heap             : max-heap array
        heap_index       : pair → PairStats mapping for O(1) lookup
        frequency_deltas : pair → frequency delta (positive = increase, negative = decrease)
    """
    for pair, delta in frequency_deltas.items():
        if pair not in heap_index:
            # New pair: append to heap tail, then sift up
            new_node = PairStats(pair, delta, len(heap))
            heap.append(new_node)
            heap_index[pair] = new_node
            _sift_up(heap, new_node.heap_pos)
        else:
            node = heap_index[pair]
            node.frequency += delta
            # Frequency increased → may need to sift up; decreased → may need to sift down
            if delta > 0:
                _sift_up(heap, node.heap_pos)
            else:
                _sift_down(heap, node.heap_pos)

    frequency_deltas.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 3. BPETokenizer
# ─────────────────────────────────────────────────────────────────────────────

class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding tokenizer, inheriting from :class:`Tokenizer`.

    Training
    --------
    Call the class method :meth:`train` to learn a BPE vocabulary from raw text.
    Training uses a **doubly-linked list + max-heap**:
      - Linked list: each word's byte sequence forms a list; merges are O(1)
      - Max-heap: tracks all pair frequencies; each iteration pops the
        highest-frequency pair in O(log n)

    Encoding
    --------
    :meth:`encode` first applies the regex pre-tokenizer, then calls
    :meth:`_bpe_encode_word` on each word chunk.
    Encoding uses a **doubly-linked list + min-heap** (lower rank = higher priority):
      - Linked list: same as training; merges are O(1)
      - Min-heap: sorted by rank in ``mergeable_ranks``; greedy merging

    Decoding
    --------
    :meth:`decode` looks up the id→bytes mapping, concatenates, then UTF-8 decodes.

    Serialization
    -------------
    :meth:`save` / :meth:`from_pretrained` extend the parent class and additionally
    write ``bpe_ranks.json`` (a base64-encoded bytes→rank mapping).
    """

    # Default GPT-2-style pre-tokenization regex (same as the parent class)
    _DEFAULT_PRETOK_PATTERN: str = Tokenizer._DEFAULT_PRETOK_PATTERN

    def __init__(
        self,
        mergeable_ranks: dict[bytes, int],
        pretok_pattern: str | None = Tokenizer._DEFAULT_PRETOK_PATTERN,
        special_tokens: "dict[str, tuple[str, int]] | None" = None,
        extra_special_tokens: "list[str] | dict[str, int] | None" = None,
    ) -> None:
        """
        Args:
            mergeable_ranks      : bytes → rank mapping; lower rank = higher merge priority.
                                   Typically produced by :meth:`train`; can also be loaded
                                   from a saved file.
            pretok_pattern       : pre-tokenization regex forwarded to the parent Tokenizer.
                                   Defaults to the GPT-2-style pattern.
            special_tokens       : optional mapping to customise the four built-in special
                                   tokens (pad/bos/eos/unk); forwarded to the parent Tokenizer.
                                   Example: ``{"bos": ("<start_of_text>", 100000)}``.
            extra_special_tokens : additional special tokens to register atomically (e.g.
                                   chat-template role tags).  Accepts either:
                                   - ``list[str]``: ids are auto-assigned after the highest
                                     existing id.
                                   - ``dict[str, int]``: explicit token → id mapping so that
                                     ids can be pinned to specific positions in the vocabulary
                                     (e.g. to match a pre-existing model's embedding table).
        """
        # Pass an empty vocab to the parent: BPE tokens are bytes sequences and
        # have no meaningful string representation in the parent's char-based
        # _id_to_token / _token_to_id tables.  The parent is used only to manage
        # the four built-in special tokens (pad/bos/eos/unk) and their ids.
        super().__init__(
            [],
            special_tokens=special_tokens,
            pretok_pattern=pretok_pattern,
            extra_special_tokens=extra_special_tokens,
        )

        # BPE core data structures
        self.mergeable_ranks: dict[bytes, int] = mergeable_ranks
        self._id_to_bytes: dict[int, bytes] = {
            rank: token_bytes for token_bytes, rank in mergeable_ranks.items()
        }

    # ── Encoding core: word-level BPE ───────────────────────────────────────

    def _bpe_encode_word(self, word_bytes: bytes) -> list[int]:
        """
        BPE-encode a single word (byte sequence) and return its token id list.

        Algorithm:
          1. Build a doubly-linked list from *word_bytes* (one node per byte).
          2. Enumerate all adjacent pairs; push those present in
             ``mergeable_ranks`` onto a min-heap.
          3. Repeatedly pop the heap top (lowest rank = highest priority):
             - If the pair is stale (node contents no longer match), skip it
               (lazy deletion).
             - Otherwise merge: update the linked list and push any newly
               adjacent pairs onto the heap.
          4. Traverse the linked list iteratively to collect the final token ids.

        Args:
            word_bytes : UTF-8 byte sequence of a single pre-tokenized word

        Returns:
            List of BPE token ids for this word
        """
        if len(word_bytes) == 0:
            return []

        # If the entire word is already in the vocabulary, return it directly
        if word_bytes in self.mergeable_ranks:
            return [self.mergeable_ranks[word_bytes]]

        # ── Step 1: build doubly-linked list ────────────────────────────────
        head = _build_linked_list(word_bytes)

        # ── Step 2: initialise min-heap ──────────────────────────────────────
        # Heap element: (rank, position_tiebreak, left_snapshot, right_snapshot, left_node)
        # position_tiebreak ensures determinism when two pairs share the same rank.
        min_heap: list[tuple[int, int, bytes, bytes, SymbolNode]] = []
        node = head
        position = 0
        while node.right is not None:
            pair = node.symbol + node.right.symbol
            rank = self.mergeable_ranks.get(pair)
            if rank is not None:
                heapq.heappush(min_heap, (rank, position, node.symbol, node.right.symbol, node))
            node = node.right
            position += 1

        # ── Step 3: greedy merging ───────────────────────────────────────────
        while min_heap:
            rank, _pos, left_snapshot, right_snapshot, left_node = heapq.heappop(min_heap)

            # Lazy deletion: verify the node is still valid
            if left_node.symbol != left_snapshot:
                continue
            if left_node.right is None or left_node.right.symbol != right_snapshot:
                continue

            right_node = left_node.right

            # Perform merge: absorb right_node into left_node
            left_node.symbol = left_node.symbol + right_node.symbol

            # Update linked-list pointers to bypass right_node
            left_node.right = right_node.right
            if right_node.right is not None:
                right_node.right.left = left_node
            # Clear right_node references to assist garbage collection
            right_node.left = right_node.right = None

            # Push newly adjacent pairs produced by the merge onto the heap
            if left_node.left is not None:
                new_pair = left_node.left.symbol + left_node.symbol
                new_rank = self.mergeable_ranks.get(new_pair)
                if new_rank is not None:
                    heapq.heappush(
                        min_heap,
                        (new_rank, _pos - 1, left_node.left.symbol, left_node.symbol, left_node.left),
                    )

            if left_node.right is not None:
                new_pair = left_node.symbol + left_node.right.symbol
                new_rank = self.mergeable_ranks.get(new_pair)
                if new_rank is not None:
                    heapq.heappush(
                        min_heap,
                        (new_rank, _pos, left_node.symbol, left_node.right.symbol, left_node),
                    )

        # ── Step 4: traverse linked list iteratively to collect token ids ──
        token_ids: list[int] = []
        node = head
        while node is not None:
            token_id = self.mergeable_ranks.get(node.symbol)
            if token_id is None:
                # Should not happen in practice (all single bytes are in the vocab),
                # but fall back to unk as a safety measure
                token_id = self.unk_id
            token_ids.append(token_id)
            node = node.right

        return token_ids

    # ── Public API: encode / decode ──────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
        truncation: bool = False,
        return_tensors: bool = False,
    ) -> list[int]:
        """
        Encode *text* into a list of BPE token ids.

        Steps:
          1. Split the text into word chunks using the pre-tokenization regex.
          2. UTF-8-encode each chunk and call :meth:`_bpe_encode_word`.
          3. Optionally prepend BOS and/or append EOS.

        Args:
            text          : input string
            add_bos       : prepend BOS token
            add_eos       : append EOS token
            max_length    : maximum sequence length (including special tokens)
            truncation    : must be True to activate truncation; raises ValueError otherwise
            return_tensors: if True, return a 1-D torch.LongTensor

        Returns:
            List of token ids (or torch.LongTensor)
        """
        if not self.mergeable_ranks:
            raise ValueError("BPETokenizer.mergeable_ranks is empty; train or load a vocabulary first.")

        token_ids: list[int] = []

        # Split text into segments, preserving all special tokens (builtin + extra)
        # as atomic units — identical strategy to the parent's _tokenize_with_pretok.
        # Segments that are special tokens are looked up directly in _token_to_id;
        # all other segments are fed through the pretok regex then BPE-encoded.

        remaining = text
        while remaining:
            # Priority 1: match a leading special token via the special Trie.
            leading_special = self._trie_longest_match(self._special_trie, remaining, 0)
            if leading_special is not None:
                token_ids.append(self._token_to_id[leading_special])
                remaining = remaining[len(leading_special):]
                continue

            # Find the next special token boundary.
            next_special_pos = len(remaining)
            for scan_pos in range(1, len(remaining)):
                if self._trie_longest_match(self._special_trie, remaining, scan_pos) is not None:
                    next_special_pos = scan_pos
                    break

            # BPE-encode the segment before the next special token.
            segment = remaining[:next_special_pos]
            remaining = remaining[next_special_pos:]

            if self._pretok_regex is not None:
                word_chunks = self._pretok_regex.findall(segment)
            else:
                word_chunks = [segment]

            for word_chunk in word_chunks:
                token_ids.extend(self._bpe_encode_word(word_chunk.encode("utf-8")))

        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)

        if max_length is not None and len(token_ids) > max_length:
            if not truncation:
                raise ValueError(
                    f"Sequence length {len(token_ids)} exceeds max_length={max_length}. "
                    "Pass truncation=True to enable automatic truncation."
                )
            token_ids = token_ids[:max_length]

        if return_tensors:
            return torch.tensor(token_ids, dtype=torch.long)

        return token_ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token ids back to a text string.

        Args:
            ids                 : list of token ids
            skip_special_tokens : if True, filter out special tokens (pad/bos/eos/unk)

        Returns:
            Decoded UTF-8 string
        """
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}

        byte_chunks: list[bytes] = []
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            token_bytes = self._id_to_bytes.get(token_id)
            if token_bytes is not None:
                byte_chunks.append(token_bytes)

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    # ── Training: class method ───────────────────────────────────────────────

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        pretok_pattern: str | None = Tokenizer._DEFAULT_PRETOK_PATTERN,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> "BPETokenizer":
        """
        Train a BPE vocabulary from raw text and return an initialised
        BPETokenizer instance.

        Algorithm (linked list + max-heap):
          1. **Initialise vocabulary**: collect all words from the corpus
             (after pre-tokenization), count word frequencies, and build a
             doubly-linked list for each word's byte sequence.
             Simultaneously count all adjacent pair frequencies and build
             the max-heap.
          2. **Iterative merging**:
             a. Pop the heap top (highest-frequency pair).
             b. Stop if frequency < min_frequency.
             c. Add the pair to the vocabulary and assign a new rank.
             d. Visit every linked-list position that contains this pair
                and perform the merge.
             e. Incrementally update affected neighbour pair frequencies
                and batch-update the heap.
          3. Repeat until the vocabulary reaches *vocab_size* or no
             mergeable pairs remain.

        Args:
            texts           : iterable of raw text strings (generators are fine)
            vocab_size      : target vocabulary size (including 256 single-byte base tokens)
            pretok_pattern  : pre-tokenization regex; None disables pre-tokenization
            min_frequency   : minimum pair frequency; merging stops below this threshold
            show_progress   : whether to print training progress

        Returns:
            A fully trained BPETokenizer instance
        """
        pretok_regex = regex.compile(pretok_pattern) if pretok_pattern else None

        # ── Step 1: count word frequencies ──────────────────────────────────
        word_frequency: collections.Counter[bytes] = collections.Counter()
        num_texts = 0
        for text in texts:
            num_texts += 1
            if pretok_regex is not None:
                for word_chunk in pretok_regex.findall(text):
                    word_bytes = word_chunk.encode("utf-8")
                    if word_bytes:
                        word_frequency[word_bytes] += 1
            else:
                word_bytes = text.encode("utf-8")
                if word_bytes:
                    word_frequency[word_bytes] += 1

        if show_progress:
            print(
                f"  [BPE Train] texts={num_texts} | "
                f"unique words={len(word_frequency)} | "
                f"target vocab_size={vocab_size}"
            )

        # ── Step 2: initialise mergeable_ranks (256 single-byte base tokens) ─
        # Standard BPE practice: all 256 single bytes (0x00-0xFF) form the base vocab
        mergeable_ranks: dict[bytes, int] = {
            bytes([byte_value]): byte_value for byte_value in range(256)
        }

        # ── Step 3: build linked lists per word, count initial pair frequencies
        # word_linked_lists[word_bytes] = list of head nodes, one per occurrence
        # (the same word may appear multiple times; each occurrence gets its own list)
        word_linked_lists: dict[bytes, list[SymbolNode]] = {}
        pair_frequency: collections.Counter[tuple[bytes, bytes]] = collections.Counter()
        # pair_occurrences[pair] = [(left_node, word_bytes), ...] tracks every occurrence
        pair_occurrences: dict[tuple[bytes, bytes], list[tuple[SymbolNode, bytes]]] = (
            collections.defaultdict(list)
        )

        for word_bytes, freq in word_frequency.items():
            linked_list_instances: list[SymbolNode] = []
            for _ in range(freq):
                head = _build_linked_list(word_bytes)
                linked_list_instances.append(head)
                # Count all adjacent pairs in this linked list
                node = head
                while node.right is not None:
                    pair = (node.symbol, node.right.symbol)
                    pair_frequency[pair] += 1
                    pair_occurrences[pair].append((node, word_bytes))
                    node = node.right
            word_linked_lists[word_bytes] = linked_list_instances

        # ── Step 4: build max-heap ───────────────────────────────────────────
        heap: list[PairStats] = []
        heap_index: dict[tuple[bytes, bytes], PairStats] = {}
        for pair, freq in pair_frequency.items():
            node_stats = PairStats(pair, freq, len(heap))
            heap.append(node_stats)
            heap_index[pair] = node_stats

        # Heapify using Floyd's algorithm — O(n)
        for i in range(len(heap) // 2 - 1, -1, -1):
            _sift_down(heap, i)

        num_merges_target = vocab_size - len(mergeable_ranks)
        num_merges_done = 0

        # ── Step 5: iterative merging ────────────────────────────────────────
        while heap and num_merges_done < num_merges_target:
            # Pop the highest-frequency pair from the max-heap
            best_stats = heap[0]
            if best_stats.frequency < min_frequency:
                break

            best_pair = best_stats.pair
            merged_symbol = best_pair[0] + best_pair[1]

            # Assign a new rank to the merged symbol
            new_rank = len(mergeable_ranks)
            mergeable_ranks[merged_symbol] = new_rank
            num_merges_done += 1

            if show_progress and num_merges_done % 1000 == 0:
                print(
                    f"  [BPE Train] merges {num_merges_done}/{num_merges_target} | "
                    f"pair={merged_symbol!r} freq={best_stats.frequency}"
                )

            # Remove this pair from the heap (swap with tail, then sift down)
            last_stats = heap[-1]
            heap[0] = last_stats
            last_stats.heap_pos = 0
            heap.pop()
            del heap_index[best_pair]
            if heap:
                _sift_down(heap, 0)

            # Visit every linked-list position containing best_pair, merge, and update frequencies
            frequency_deltas: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
            new_pair_occurrences: list[tuple[SymbolNode, bytes]] = []

            for left_node, word_bytes in pair_occurrences.get(best_pair, []):
                # Validate the position is still current (lazy-deletion check)
                if left_node.symbol != best_pair[0]:
                    continue
                if left_node.right is None or left_node.right.symbol != best_pair[1]:
                    continue

                right_node = left_node.right

                # Decrement frequencies of adjacent pairs that will be broken by the merge
                if left_node.left is not None:
                    old_left_pair = (left_node.left.symbol, left_node.symbol)
                    frequency_deltas[old_left_pair] -= 1

                if right_node.right is not None:
                    old_right_pair = (right_node.symbol, right_node.right.symbol)
                    frequency_deltas[old_right_pair] -= 1

                # Perform the merge
                left_node.symbol = merged_symbol
                left_node.right = right_node.right
                if right_node.right is not None:
                    right_node.right.left = left_node
                right_node.left = right_node.right = None

                # Increment frequencies of newly adjacent pairs after the merge
                if left_node.left is not None:
                    new_left_pair = (left_node.left.symbol, left_node.symbol)
                    frequency_deltas[new_left_pair] += 1
                    new_pair_occurrences.append((left_node.left, word_bytes))

                if left_node.right is not None:
                    new_right_pair = (left_node.symbol, left_node.right.symbol)
                    frequency_deltas[new_right_pair] += 1
                    new_pair_occurrences.append((left_node, word_bytes))

            # Remove the consumed pair from pair_occurrences
            if best_pair in pair_occurrences:
                del pair_occurrences[best_pair]

            # Register newly produced pair occurrences
            for left_node, word_bytes in new_pair_occurrences:
                if left_node.right is not None:
                    new_pair = (left_node.symbol, left_node.right.symbol)
                    pair_occurrences[new_pair].append((left_node, word_bytes))

            # Batch-update the heap with accumulated frequency deltas
            _update_pair_heap(heap, heap_index, frequency_deltas)

        if show_progress:
            print(
                f"  [BPE Train] training complete | "
                f"vocab_size={len(mergeable_ranks)} | "
                f"merges={num_merges_done}"
            )

        return cls(mergeable_ranks, pretok_pattern=pretok_pattern)

    # ── Serialization: extend parent save / from_pretrained ─────────────────

    def save(self, save_dir: str) -> None:
        """
        Save the BPETokenizer to *save_dir*.

        In addition to the parent's ``tokenizer.json``, writes:
          - ``bpe_ranks.json``: base64(token_bytes) → rank JSON mapping

        Args:
            save_dir : destination directory (created automatically if absent)
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save BPE ranks (base64-encode bytes keys for JSON safety)
        bpe_ranks_serializable = {
            base64.b64encode(token_bytes).decode("ascii"): rank
            for token_bytes, rank in self.mergeable_ranks.items()
        }
        bpe_ranks_path = os.path.join(save_dir, "bpe_ranks.json")
        with open(bpe_ranks_path, "w", encoding="utf-8") as bpe_file:
            json.dump(bpe_ranks_serializable, bpe_file, indent=2, ensure_ascii=True)

        # Save pre-tokenization regex pattern
        pretok_pattern_str: str | None = None
        if self._pretok_regex_bpe is not None:
            pretok_pattern_str = self._pretok_regex_bpe.pattern

        tokenizer_state = {
            "tokenizer_type": "BPETokenizer",
            "pretok_pattern": pretok_pattern_str,
            # Persist built-in special token configuration so custom bos/eos/pad/unk
            # ids (e.g. a chat-model <start_of_text> bos token) survive save/load.
            "special_tokens": {
                "pad": [self.PAD_TOKEN, self._token_to_id[self.PAD_TOKEN]],
                "bos": [self.BOS_TOKEN, self._token_to_id[self.BOS_TOKEN]],
                "eos": [self.EOS_TOKEN, self._token_to_id[self.EOS_TOKEN]],
                "unk": [self.UNK_TOKEN, self._token_to_id[self.UNK_TOKEN]],
            },
            # Extra special tokens stored as {token_str: id} to preserve explicit ids.
            "extra_special_tokens": {
                tok: self._token_to_id[tok]
                for tok in sorted(self._extra_special_tokens)
            },
        }
        tokenizer_meta_path = os.path.join(save_dir, "tokenizer.json")
        with open(tokenizer_meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(tokenizer_state, meta_file, indent=2, ensure_ascii=False)

        print(
            f"  BPETokenizer saved → {save_dir}  "
            f"(vocab_size={len(self.mergeable_ranks)})"
        )

    @classmethod
    def from_pretrained(cls, save_dir: str) -> "BPETokenizer":
        """
        Restore a BPETokenizer from a directory created by :meth:`save`.

        Args:
            save_dir : path to the directory produced by :meth:`save`

        Returns:
            A fully initialised BPETokenizer instance
        """
        bpe_ranks_path = os.path.join(save_dir, "bpe_ranks.json")
        if not os.path.isfile(bpe_ranks_path):
            raise FileNotFoundError(f"bpe_ranks.json not found in {save_dir!r}")

        with open(bpe_ranks_path, "r", encoding="utf-8") as bpe_file:
            bpe_ranks_raw = json.load(bpe_file)

        mergeable_ranks = {
            base64.b64decode(b64_key): rank
            for b64_key, rank in bpe_ranks_raw.items()
        }

        tokenizer_meta_path = os.path.join(save_dir, "tokenizer.json")
        pretok_pattern: str | None = None
        special_tokens: "dict[str, tuple[str, int]] | None" = None
        if os.path.isfile(tokenizer_meta_path):
            with open(tokenizer_meta_path, "r", encoding="utf-8") as meta_file:
                state = json.load(meta_file)
            pretok_pattern = state.get("pretok_pattern")
            raw_special = state.get("special_tokens")
            if raw_special is not None:
                special_tokens = {role: (val[0], val[1]) for role, val in raw_special.items()}

        # Restore extra special tokens: {token_str: id}
        raw_extra = state.get("extra_special_tokens", {})
        extra_special_tokens = {tok: tid for tok, tid in raw_extra.items()} or None

        tokenizer = cls(
            mergeable_ranks,
            pretok_pattern=pretok_pattern,
            special_tokens=special_tokens,
            extra_special_tokens=extra_special_tokens,
        )

        print(
            f"  BPETokenizer loaded ← {save_dir}  "
            f"(vocab_size={len(tokenizer.mergeable_ranks)})"
        )
        return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test (executed when this file is run directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_corpus = [
        "hello world hello",
        "world is beautiful",
        "hello beautiful world",
        "BPE tokenizer is fast and efficient",
        "fast tokenizer for fast encoding",
    ] * 50  # repeat to generate sufficient pair frequency

    print("=== Training BPETokenizer ===")
    tokenizer = BPETokenizer.train(
        texts=sample_corpus,
        vocab_size=300,
        min_frequency=2,
        show_progress=True,
    )

    test_sentences = [
        "hello world",
        "beautiful tokenizer",
        "fast and efficient BPE",
    ]
    print("\n=== Encode / decode test ===")
    for sentence in test_sentences:
        ids = tokenizer.encode(sentence)
        decoded = tokenizer.decode(ids)
        print(f"  Input  : {sentence!r}")
        print(f"  Token IDs: {ids}")
        print(f"  Decoded: {decoded!r}")
        print(f"  Round-trip OK: {sentence == decoded}")
        print()

    print("=== Serialization test ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer.save(tmp_dir)
        loaded_tokenizer = BPETokenizer.from_pretrained(tmp_dir)
        ids_original = tokenizer.encode("hello world")
        ids_loaded = loaded_tokenizer.encode("hello world")
        print(f"  Original IDs: {ids_original}")
        print(f"  Loaded IDs  : {ids_loaded}")
        print(f"  Serialization consistent: {ids_original == ids_loaded}")
