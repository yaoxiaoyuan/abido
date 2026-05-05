"""
chat_cli.py — interactive text generation entry point

Supports:
  - Pre-trained language models (lm mode, encoder_decoder mode)
  - Instruction-tuned / chat models (is_chat=True, with ChatFormatter)
  - Cross-turn KV cache reuse (lm + is_chat mode)
  - Streaming token output

Main components:
  GenerationConfig     : sampling hyperparameters (temperature, top_k, top_p, …)
  ChatFormatter        : multi-turn dialogue template formatter (subclass to customise)
  generate_interactive : command-line interactive generation loop

Usage (CLI)::

    python chat_cli.py [model_dir] [--device cpu] [--max-tokens 256] \\
                       [--temperature 0.8] [--top-k 40] [--top-p 0.9] \\
                       [--repetition-penalty 1.0] [--generation-config path.json]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import torch

from transformer_minimal import Transformer, decode_stream
from tokenizer_minimal import Tokenizer, BPETokenizer


# ─────────────────────────────────────────────
# 1. GenerationConfig
# ─────────────────────────────────────────────

class GenerationConfig:
    """
    Holds all hyperparameters that control text generation.

    Can be constructed directly or loaded from / saved to a JSON file so that
    generation settings can be version-controlled alongside model checkpoints.

    Attributes:
        max_decode_len     : maximum number of new tokens to generate per turn
        temperature        : softmax temperature; lower = more deterministic
        top_k              : keep only the top-k highest-probability tokens
                             (1 = greedy, 0 = disabled)
        top_p              : nucleus sampling threshold; 1.0 = disabled
        repetition_penalty : > 1.0 penalises tokens already generated; 1.0 = off

    Example::

        cfg = GenerationConfig(temperature=0.8, top_p=0.9, repetition_penalty=1.2)
        cfg.save("my_model/generation_config.json")

        cfg2 = GenerationConfig.from_file("my_model/generation_config.json")
    """

    def __init__(
        self,
        max_decode_len: int = 256,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> None:
        self.max_decode_len     = max_decode_len
        self.temperature        = temperature
        self.top_k              = top_k
        self.top_p              = top_p
        self.repetition_penalty = repetition_penalty

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "max_decode_len":     self.max_decode_len,
            "temperature":        self.temperature,
            "top_k":              self.top_k,
            "top_p":              self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

    def save(self, path: str) -> None:
        """Write the config to *path* as a JSON file."""
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file_handle:
            json.dump(self.to_dict(), file_handle, indent=2)

    @classmethod
    def from_file(cls, path: str) -> "GenerationConfig":
        """Load a GenerationConfig from a JSON file created by save()."""
        with open(path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
        return cls(**{key: value for key, value in data.items() if key in cls.__init__.__code__.co_varnames})

    def __repr__(self) -> str:
        fields = ", ".join(f"{key}={value!r}" for key, value in self.to_dict().items())
        return f"GenerationConfig({fields})"


# ─────────────────────────────────────────────
# 2. ChatFormatter
# ─────────────────────────────────────────────

class ChatFormatter:
    """
    Protocol / base class for formatting multi-turn dialogue into a flat
    prompt string that the model can consume.

    Subclass this and override :meth:`format_prompt` to implement any
    chat template (ChatML, Llama-3, Alpaca, custom, …).

    The formatter also maintains the conversation history so that each new
    turn can include prior context if desired.

    Example — minimal ChatML formatter::

        class ChatMLFormatter(ChatFormatter):
            def format_prompt(self, user_input: str) -> str:
                history_text = ""
                for role, text in self.history:
                    tag = "user" if role == "user" else "assistant"
                    history_text += f"<|im_start|>{tag}\\n{text}<|im_end|>\\n"
                history_text += f"<|im_start|>user\\n{user_input}<|im_end|>\\n"
                history_text += "<|im_start|>assistant\\n"
                return history_text
    """

    # ── Per-role turn templates ──────────────────────────────────────────────
    # Placeholder: {user} for user text, {assistant} for assistant text.
    # The part of DEFAULT_ASSISTANT_TEMPLATE *before* {assistant} is appended
    # at generation time to open the assistant's reply slot.
    DEFAULT_SYSTEM_TEMPLATE    = "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\n"
    DEFAULT_USER_TEMPLATE      = "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\n"
    DEFAULT_ASSISTANT_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>\n"

    # ── Full prompt template ─────────────────────────────────────────────────
    # Placeholders:
    #   {system}          : rendered system block (empty string when no system prompt)
    #   {history}         : all prior turns rendered via the two turn templates above
    #   {user}            : the current user input
    #   {assistant_start} : the prefix of DEFAULT_ASSISTANT_TEMPLATE before {assistant}
    DEFAULT_CHAT_TEMPLATE = (
        "{system}"
        "{history}"
        "{user_turn}"
        "{assistant_start}"
    )

    def __init__(
        self,
        system_prompt: str = "",
        chat_template: "str | None" = None,
        system_template: "str | None" = None,
        user_template: "str | None" = None,
        assistant_template: "str | None" = None,
    ) -> None:
        """
        Args:
            system_prompt       : optional system / instruction text prepended to
                                  every conversation (empty string = no system prompt)
            chat_template       : full prompt template with placeholders ``{system}``,
                                  ``{history}``, ``{user_turn}``, ``{assistant_start}``.
                                  Defaults to ``DEFAULT_CHAT_TEMPLATE``.
            system_template     : template for the system block; placeholder ``{system}``.
                                  Defaults to ``DEFAULT_SYSTEM_TEMPLATE``.
            user_template       : template for each user history turn; placeholder ``{user}``.
                                  Defaults to ``DEFAULT_USER_TEMPLATE``.
            assistant_template  : template for each assistant history turn; placeholder
                                  ``{assistant}``.  The text *before* ``{assistant}``
                                  is also used as the generation prefix (assistant_start).
                                  Defaults to ``DEFAULT_ASSISTANT_TEMPLATE``.
        """
        self.system_prompt       = system_prompt
        self.chat_template       = chat_template       or self.DEFAULT_CHAT_TEMPLATE
        self.system_template     = system_template     or self.DEFAULT_SYSTEM_TEMPLATE
        self.user_template       = user_template       or self.DEFAULT_USER_TEMPLATE
        self.assistant_template  = assistant_template  or self.DEFAULT_ASSISTANT_TEMPLATE
        # List of (role, text) pairs accumulated across turns.
        # role is either "user" or "assistant".
        self.history: list[tuple[str, str]] = []

    @property
    def assistant_start(self) -> str:
        """The prefix that opens the assistant reply slot (text before {assistant})."""
        return self.assistant_template.split("{assistant}")[0]

    def format_prompt(self, user_input: str) -> str:
        """
        Convert the current conversation history + *user_input* into a single
        prompt string ready to be encoded by the Tokenizer.

        The prompt is built by:
        1. Rendering the system block via ``system_template`` (empty when no system prompt).
        2. Rendering each history turn via ``user_template`` / ``assistant_template``.
        3. Substituting ``{system}``, ``{history}``, ``{user_turn}``, and
           ``{assistant_start}`` into ``chat_template``.

        The default template produces::

            <|system|>
            {system_prompt}<|end|>          <- omitted when system_prompt is empty
            <|user|>
            {turn_1_user}<|end|>
            <|assistant|>
            {turn_1_assistant}<|end|>
            ...
            <|user|>
            {current_user_input}<|end|>
            <|assistant|>

        Args:
            user_input : the raw text typed by the user this turn

        Returns:
            Formatted prompt string including system prompt, full conversation
            history, and the current user turn.
        """
        # Render system block (empty string when no system prompt)
        system_block = (
            self.system_template.format(system=self.system_prompt)
            if self.system_prompt
            else ""
        )

        # Render history turns using the per-role templates
        history_parts: list[str] = []
        for role, text in self.history:
            if role == "user":
                history_parts.append(self.user_template.format(user=text))
            else:
                history_parts.append(self.assistant_template.format(assistant=text))
        history_block = "".join(history_parts)

        return self.chat_template.format(
            system=system_block,
            history=history_block,
            user_turn=self.user_template.format(user=user_input),
            assistant_start=self.assistant_start,
        )

    def add_turn(
        self,
        user_input: "str | None" = None,
        assistant_response: "str | None" = None,
    ) -> None:
        """
        Append one or both sides of a turn to the conversation history.

        Either argument may be omitted (or None) to add only one role.
        For example, call ``add_turn(user_input="hi")`` to record a user
        message without a corresponding assistant reply yet.

        Args:
            user_input         : the user message for this turn (optional)
            assistant_response : the model's decoded response for this turn (optional)
        """
        if user_input is not None:
            self.history.append(("user", user_input))
        if assistant_response is not None:
            self.history.append(("assistant", assistant_response))

    def format_history(self) -> str:
        """
        Render the full conversation history (all turns so far) as a flat
        string using the configured templates, without appending a new user
        turn or the assistant prefix.

        Useful for inspecting or logging the current dialogue context.

        Returns:
            Formatted string of all history turns.
        """
        system_block = (
            self.system_template.format(system=self.system_prompt)
            if self.system_prompt
            else ""
        )
        history_parts: list[str] = []
        for role, text in self.history:
            if role == "user":
                history_parts.append(self.user_template.format(user=text))
            else:
                history_parts.append(self.assistant_template.format(assistant=text))
        return system_block + "".join(history_parts)

    def format_single_turn(self, user_input: str) -> tuple[str, str]:
        """
        Render a single user turn and the assistant prefix **without** any
        conversation history, ignoring ``self.history`` entirely.

        This is useful for SFT data encoding where each turn's tokens are
        accumulated manually — calling :meth:`format_prompt` would
        re-render all prior history and cause duplication.

        Returns:
            ``(user_turn_text, assistant_start_text)`` — the rendered user
            block and the assistant opening prefix, as separate strings so
            the caller can encode them independently and assign loss masks.
        """
        user_turn_text       = self.user_template.format(user=user_input)
        assistant_start_text = self.assistant_start
        return user_turn_text, assistant_start_text

    def reset(self) -> None:
        """Clear the conversation history (start a new session)."""
        self.history.clear()

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return all template settings as a plain dict (JSON-serialisable)."""
        return {
            "system_prompt":      self.system_prompt,
            "chat_template":      self.chat_template,
            "system_template":    self.system_template,
            "user_template":      self.user_template,
            "assistant_template": self.assistant_template,
        }

    def save(self, path: str) -> None:
        """
        Save the formatter's template configuration to a JSON file.

        Only the template strings and system prompt are persisted;
        conversation history is intentionally excluded.

        Args:
            path : destination file path (e.g. "my_model/chat_template.json")
        """
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file_handle:
            json.dump(self.to_dict(), file_handle, indent=2, ensure_ascii=False)

    @classmethod
    def from_file(cls, path: str) -> "ChatFormatter":
        """
        Load a ChatFormatter from a JSON file created by save().

        Args:
            path : path to the JSON file

        Returns:
            A new ChatFormatter instance with the saved template settings.
        """
        with open(path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
        return cls(**data)


# ─────────────────────────────────────────────
# 3. Interactive generation
# ─────────────────────────────────────────────

def _stream_token_to_str(
    token_id: int,
    tokenizer: "Tokenizer",
    byte_buffer: bytearray,
) -> "str | None":
    """
    Convert a single streamed token id to a displayable string.

    For BPE tokenizers, raw bytes are accumulated in *byte_buffer* until a
    valid UTF-8 sequence is complete.  For char-based tokenizers the token
    string is returned directly.

    Args:
        token_id     : the token id just generated
        tokenizer    : the tokenizer used for decoding
        byte_buffer  : mutable bytearray shared across calls (BPE only);
                       callers must pass the same object on every call so
                       that incomplete multi-byte sequences are preserved

    Returns:
        A decoded string when ready, or ``None`` if more bytes are needed
        (BPE multi-byte character not yet complete).
    """
    if isinstance(tokenizer, BPETokenizer):
        token_bytes = tokenizer._id_to_bytes.get(token_id, b"")
        byte_buffer += token_bytes
        try:
            decoded = byte_buffer.decode("utf-8")
            byte_buffer.clear()
            return decoded
        except UnicodeDecodeError:
            # Incomplete multi-byte sequence — wait for more tokens
            return None
    return tokenizer.id_to_token(token_id)


def _prefill_incremental(
    model: "Transformer",
    new_tokens: torch.Tensor,
    past_kvs: "list | None",
) -> tuple[torch.Tensor, list]:
    """
    Run a forward pass over *new_tokens* only, extending *past_kvs*.

    This is the building block for cross-turn KV cache reuse: instead of
    re-processing the entire prompt on every turn, we only feed the tokens
    that were added since the last prefill.

    Args:
        model      : LM-mode Transformer
        new_tokens : [1, num_new] tensor of token ids to prefill
        past_kvs   : existing KV cache from previous turns (None for first call)

    Returns:
        (last_hidden [1, 1, d_model], updated_past_kvs)
    """
    hidden, updated_kvs = model.decode(new_tokens, past_kvs=past_kvs)
    return hidden[:, -1:, :], updated_kvs


def generate_interactive(
    model: "Transformer",
    tokenizer: "Tokenizer",
    chat_formatter: "ChatFormatter | None" = None,
    generation_config: "GenerationConfig | None" = None,
    stream_flush_interval: int = 1,
    single_turn_mode: bool = False,
    device: "str | torch.device | None" = None,
) -> None:
    """
    Interactive command-line chat loop with streaming token output and
    cross-turn KV cache reuse (LM mode only).

    **KV cache reuse strategy (LM mode)**:
    The session maintains a running KV cache and the full list of token ids
    seen so far.  On each new turn:

    1. The full prompt (history + new user turn) is encoded.
    2. Only the *new* tokens (those not yet in the cache) are prefilled
       incrementally, extending the existing KV cache.
    3. Autoregressive generation continues from the updated cache.
    4. Generated tokens are appended to the session token list so the next
       turn can reuse them without re-computation.
    5. ``/reset`` clears both the KV cache and the token history.

    Encoder-decoder mode does not support cross-turn KV cache reuse because
    the encoder output changes every turn; each turn is processed from scratch.

    If ``model.cfg.is_chat`` is True, a *chat_formatter* **must** be provided.
    The formatter converts the raw user input (plus optional conversation
    history) into the prompt template the model was trained on.

    Args:
        model                : trained Transformer (lm or encoder_decoder)
        tokenizer            : Tokenizer used to encode input and decode output
        chat_formatter       : ChatFormatter instance that converts user input
                               into the model's expected prompt format.
                               Required when model.cfg.is_chat=True;
                               ignored when is_chat=False.
        generation_config    : GenerationConfig instance controlling sampling
                               (max_decode_len, temperature, top_k, top_p,
                               repetition_penalty).  Defaults to GenerationConfig()
                               when not provided.
        stream_flush_interval: flush stdout every N tokens (default 1 = instant)
        single_turn_mode     : if True, conversation history and KV cache are
                               cleared automatically after every turn so each
                               user message is treated as an independent query.
                               Useful for single-turn chat models.
        device               : torch device string or object; defaults to the
                               model's current device.

    Raises:
        ValueError: if model.cfg.is_chat is True but chat_formatter is None.
    """
    # Resolve generation config — fall back to defaults when not provided
    gen_cfg            = generation_config if generation_config is not None else GenerationConfig()
    max_decode_len     = gen_cfg.max_decode_len
    temperature        = gen_cfg.temperature
    top_k              = gen_cfg.top_k
    top_p              = gen_cfg.top_p
    repetition_penalty = gen_cfg.repetition_penalty

    # Resolve device: explicit argument > model's current device
    if device is not None:
        device = torch.device(device) if isinstance(device, str) else device
        model.to(device)
    else:
        device = next(model.parameters()).device
    cfg = model.cfg

    if cfg.is_chat and chat_formatter is None:
        raise ValueError(
            "model.cfg.is_chat=True but no chat_formatter was provided. "
            "Pass a ChatFormatter instance (or a subclass) to generate_interactive()."
        )

    use_formatter  = cfg.is_chat and chat_formatter is not None
    supports_cache = (cfg.model_type == "lm" and cfg.is_chat)

    print("=" * 60)
    print("Chat mode  |  /quit or Ctrl-D to exit  |  /reset to clear history")
    print(f"model: {cfg.model_type} | is_chat={cfg.is_chat} | "
          f"kv_cache_reuse={supports_cache} | "
          f"temp={temperature} | top_k={top_k} | top_p={top_p}")
    print("=" * 60)

    # ── Session state (persists across turns) ────────────────────────────────
    # session_past_kvs   : accumulated KV cache for all tokens seen so far
    # session_token_ids  : flat list of all token ids fed to the model so far
    # Both are reset together when the user types /reset.
    session_past_kvs:  list | None = None
    session_token_ids: list[int]   = []

    def reset_session() -> None:
        nonlocal session_past_kvs, session_token_ids
        session_past_kvs  = None
        session_token_ids = []
        if chat_formatter is not None:
            chat_formatter.reset()

    turn = 0
    while True:
        # ── Read user input ──────────────────────────────────────────────────
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exiting chat]")
            break

        if user_input.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("[Exiting chat]")
            break
        if user_input.lower() == "/reset":
            reset_session()
            print("[Conversation history and KV cache cleared]")
            continue
        if not user_input:
            continue

        turn += 1

        # ── Format prompt ────────────────────────────────────────────────────
        if use_formatter:
            prompt_text = chat_formatter.format_prompt(user_input)
        else:
            prompt_text = user_input
        
        # ── Encode full prompt ───────────────────────────────────────────────
        if cfg.model_type == "lm":
            full_ids = tokenizer.encode(prompt_text, add_bos=(turn == 1))
        else:
            full_ids = tokenizer.encode(prompt_text)

        # ── Incremental prefill (LM only) ────────────────────────────────────
        print("Model: ", end="", flush=True)
        generated_ids: list[int] = []
        token_buffer:  list[str] = []

        if supports_cache:
            # Only feed tokens that are new since the last prefill.
            num_cached = len(session_token_ids)
            new_ids    = full_ids[num_cached:]

            if new_ids:
                new_tensor = torch.tensor([new_ids], dtype=torch.long, device=device)
                last_hidden, session_past_kvs = _prefill_incremental(
                    model, new_tensor, session_past_kvs
                )
                session_token_ids.extend(new_ids)
            else:
                # Edge case: formatter produced no new tokens (shouldn't happen normally)
                last_hidden = model.decode(
                    torch.tensor([[session_token_ids[-1]]], dtype=torch.long, device=device),
                    past_kvs=session_past_kvs,
                )[0][:, -1:, :]

            # Autoregressive generation reusing the session KV cache via decode_stream
            kv_state: dict = {}
            byte_buffer: bytearray = bytearray()
            for token_id in decode_stream(
                model, src=torch.zeros((1, 1), dtype=torch.long, device=device),
                max_decode_len=max_decode_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                past_kvs=session_past_kvs,
                last_hidden=last_hidden,
                out_state=kv_state,
            ):
                generated_ids.append(token_id)
                if token_id in {tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id}:
                    continue
                token_str = _stream_token_to_str(token_id, tokenizer, byte_buffer)
                if token_str is None:
                    continue
                token_buffer.append(token_str)
                if len(token_buffer) >= stream_flush_interval:
                    print("".join(token_buffer), end="", flush=True)
                    token_buffer.clear()

            # Retrieve updated KV cache and record generated token ids
            session_past_kvs = kv_state.get("past_kvs", session_past_kvs)
            session_token_ids.extend(generated_ids)

        else:
            # encoder_decoder: no cross-turn cache; process from scratch each turn
            src_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            byte_buffer: bytearray = bytearray()
            for token_id in decode_stream(
                model, src_tensor,
                max_decode_len=max_decode_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ):
                generated_ids.append(token_id)
                if token_id in {tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id}:
                    continue
                token_str = _stream_token_to_str(token_id, tokenizer, byte_buffer)
                if token_str is None:
                    continue
                token_buffer.append(token_str)
                if len(token_buffer) >= stream_flush_interval:
                    print("".join(token_buffer), end="", flush=True)
                    token_buffer.clear()

        # Flush remaining buffer
        if token_buffer:
            print("".join(token_buffer), end="", flush=True)
        print()  # newline after model response

        # ── Update formatter history (chat models only) ───────────────────────
        if use_formatter:
            assistant_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            chat_formatter.add_turn(user_input, assistant_response)

        # ── Single-turn mode: clear history and KV cache after every turn ────
        if single_turn_mode:
            reset_session()
            turn = 0  # reset turn counter so next turn gets BOS again


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a saved Transformer + Tokenizer and enter interactive generation."
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="model",
        help="Directory containing config.json, model.pt and tokenizer.json (default: 'model')",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="torch device string, e.g. 'cpu', 'cuda', 'mps' (default: cpu)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Maximum new tokens to generate per prompt (default: from GenerationConfig)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (default: from GenerationConfig)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Top-k sampling (default: from GenerationConfig; 1 = greedy)",
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="Nucleus sampling threshold (default: from GenerationConfig)",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=None,
        help="Repetition penalty > 1.0 discourages repeated tokens (default: 1.0)",
    )

    args = parser.parse_args()

    run_device = torch.device(args.device)

    print(f"Loading tokenizer from: {args.model_dir}")
    # Auto-detect tokenizer type: BPETokenizer saves bpe_ranks.json alongside tokenizer.json
    if os.path.isfile(os.path.join(args.model_dir, "bpe_ranks.json")):
        tokenizer = BPETokenizer.from_pretrained(args.model_dir)
    else:
        tokenizer = Tokenizer.from_pretrained(args.model_dir)

    print(f"Loading model from: {args.model_dir}")
    model = Transformer.from_pretrained(args.model_dir, device=str(run_device))
    model.eval()

    # Load generation config from model_dir if present, otherwise use defaults
    gen_config_path = os.path.join(args.model_dir, "generation_config.json")
    if os.path.isfile(gen_config_path):
        gen_cfg = GenerationConfig.from_file(gen_config_path)
        print(f"Loaded generation config from: {gen_config_path}")
    else:
        gen_cfg = GenerationConfig()

    # Individual CLI flags override file values when explicitly supplied
    # (argparse defaults are None so we can detect "not supplied")
    if args.max_tokens         is not None: gen_cfg.max_decode_len     = args.max_tokens
    if args.temperature        is not None: gen_cfg.temperature        = args.temperature
    if args.top_k              is not None: gen_cfg.top_k              = args.top_k
    if args.top_p              is not None: gen_cfg.top_p              = args.top_p
    if args.repetition_penalty is not None: gen_cfg.repetition_penalty = args.repetition_penalty

    chat_template_path = os.path.join(args.model_dir, "chat_template.json")
    if os.path.exists(chat_template_path):
        chat_formatter = ChatFormatter.from_file(chat_template_path)
        print(f"Loaded chat template from: {chat_template_path}")
    elif getattr(model.cfg, "is_chat", False):
        chat_formatter = ChatFormatter()
        print("No chat_template.json found — using default ChatFormatter")
    else:
        chat_formatter = None

    generate_interactive(
        model             = model,
        tokenizer         = tokenizer,
        generation_config = gen_cfg,
        chat_formatter    = chat_formatter,
    )
