"""
Single-file topic modeling CLI.

Includes:
- LDA with collapsed Gibbs sampling
- Labeled LDA with semi-supervised label priors
- LightLDA with Metropolis-Hastings-Walker sampling

Supports demo data, corpus files, external vocabulary, and JSON/NumPy outputs.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


class Vocabulary:
    """Build, save, load, and apply a word-to-id mapping."""

    def __init__(self, word2id=None):
        self.word2id = word2id or {}
        self.vocab = [w for w, _ in sorted(self.word2id.items(), key=lambda x: x[1])]
        self.V = len(self.vocab)

    @classmethod
    def build(cls, documents, min_freq=1, max_vocab=None, stopwords=None):
        """
        Build vocabulary from tokenized documents.

        Parameters
        ----------
        documents : list of list of str
        min_freq : int
            Discard words appearing fewer than this many times.
        max_vocab : int or None
            Keep only the top-N most frequent words.
        stopwords : set or None
            Words to exclude.
        """
        stopwords = stopwords or set()
        freq = Counter(w for doc in documents for w in doc)

        words = [
            w for w, c in freq.most_common()
            if c >= min_freq and w not in stopwords
        ]
        if max_vocab is not None:
            words = words[:max_vocab]
        words.sort()

        word2id = {w: i for i, w in enumerate(words)}
        return cls(word2id)

    @classmethod
    def load(cls, path):
        """Load vocabulary from a JSON file."""
        with open(path, "r") as f:
            word2id = json.load(f)
        return cls(word2id)

    def save(self, path):
        """Save vocabulary to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=1)

    def doc2ids(self, document):
        """Convert a single tokenized document to a list of word IDs."""
        return [self.word2id[w] for w in document if w in self.word2id]

    def docs2ids(self, documents):
        """Convert a list of tokenized documents to lists of word IDs."""
        return [self.doc2ids(doc) for doc in documents]

    def ids2words(self, ids):
        """Convert word IDs back to words."""
        return [self.vocab[i] for i in ids]

    def __len__(self):
        return self.V

    def __contains__(self, word):
        return word in self.word2id

    def __repr__(self):
        return f"Vocabulary(size={self.V})"


# ── Document I/O ────────────────────────────────────────────────────

def read_corpus(path, tokenizer=None):
    """
    Read a text corpus from a file.

    Supported formats:
      - .txt:  one document per line, whitespace-separated tokens
      - .jsonl: one JSON object per line, expects "text" or "tokens" field
    Parameters
    ----------
    path : str
        Path to corpus file.
    tokenizer : callable or None
        Function str → list[str]. If None, splits on whitespace.

    Returns
    -------
    list of list of str
    """
    if tokenizer is None:
        tokenizer = lambda s: re.findall(r"[a-z0-9]+", s.lower())

    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if path.endswith(".jsonl"):
                obj = json.loads(line)
                text = obj.get("tokens") or obj.get("text", "")
                if isinstance(text, list):
                    documents.append(text)
                else:
                    documents.append(tokenizer(text))
            else:
                documents.append(tokenizer(line))

    return documents


def read_labeled_corpus(path, tokenizer=None):
    """
    Read a labeled corpus from a JSONL file.

    Each line: {"text": "...", "labels": ["label1", "label2"]}
    Or:        {"tokens": [...], "labels": [...]}

    Returns
    -------
    documents : list of list of str
    labels : list of list of str
    """
    if tokenizer is None:
        tokenizer = lambda s: re.findall(r"[a-z0-9]+", s.lower())

    documents, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("tokens") or obj.get("text", "")
            if isinstance(text, list):
                documents.append(text)
            else:
                documents.append(tokenizer(text))
            labels.append(obj.get("labels", []))

    return documents, labels


# ── Demo corpora ────────────────────────────────────────────────────

DEMO_TOPIC_WORDS = {
    "ml": ["neural", "network", "layer", "training", "gradient",
           "loss", "weight", "backprop", "optimizer", "batch"],
    "bio": ["gene", "protein", "cell", "mutation", "dna",
            "expression", "sequence", "genome", "rna", "enzyme"],
    "astro": ["planet", "orbit", "star", "galaxy", "telescope",
              "light", "mass", "gravity", "solar", "cosmic"],
    "finance": ["market", "stock", "bond", "portfolio", "trading",
                "asset", "risk", "return", "capital", "investment"],
    "sports": ["team", "match", "score", "player", "coach",
               "league", "season", "tournament", "goal", "training"],
    "music": ["song", "melody", "rhythm", "guitar", "piano",
              "album", "concert", "singer", "chorus", "harmony"],
    "food": ["recipe", "flavor", "spice", "kitchen", "ingredient",
             "meal", "baking", "sauce", "vegetable", "dessert"],
    "travel": ["hotel", "flight", "beach", "museum", "ticket",
               "passport", "journey", "tour", "city", "luggage"],
}

DEMO_LABEL_WORDS = {
    "sports": ["game", "team", "player", "score", "win",
               "coach", "season", "match", "league", "goal"],
    "politics": ["government", "policy", "vote", "election", "party",
                 "law", "congress", "debate", "reform", "campaign"],
    "tech": ["software", "data", "algorithm", "computer", "code",
             "system", "cloud", "api", "deploy", "server"],
    "science": ["research", "experiment", "hypothesis", "theory", "paper",
                "lab", "method", "result", "study", "evidence"],
}

def make_demo_corpus(n_docs=200, doc_len=40, noise=0.15, seed=0):
    """Synthetic 3-topic corpus (ml / bio / astro)."""
    rng = np.random.RandomState(seed)
    topics = list(DEMO_TOPIC_WORDS.values())

    docs = []
    for _ in range(n_docs):
        main = rng.randint(len(topics))
        doc = []
        for _ in range(doc_len):
            t = main if rng.rand() > noise else rng.randint(len(topics))
            doc.append(rng.choice(topics[t]))
        docs.append(doc)
    return docs


def make_labeled_demo_corpus(n_docs=200, doc_len=30, seed=0):
    """Synthetic multi-label corpus (sports / politics / tech / science)."""
    rng = np.random.RandomState(seed)
    label_list = list(DEMO_LABEL_WORDS.keys())

    docs, labels = [], []
    for _ in range(n_docs):
        n_labels = rng.choice([1, 2], p=[0.5, 0.5])
        doc_labels = list(rng.choice(label_list, size=n_labels, replace=False))
        labels.append(doc_labels)

        doc = []
        for _ in range(doc_len):
            chosen = rng.choice(doc_labels)
            doc.append(rng.choice(DEMO_LABEL_WORDS[chosen]))
        docs.append(doc)

    return docs, labels


class LDA:
    """
    Collapsed Gibbs sampling integrates out phi and theta,
    samples topic assignments z directly:

        P(z_i=k | rest) ∝ (n_dk + α) · (n_kw + β) / (n_k + Vβ)

    where:
        n_dk = count of words in doc d assigned to topic k  (excluding i)
        n_kw = count of word w assigned to topic k          (excluding i)
        n_k  = total words assigned to topic k              (excluding i)
    """

    def __init__(self, n_topics, alpha=None, beta=0.01, random_state=42):
        self.K = n_topics
        self.alpha = alpha if alpha is not None else 50.0 / n_topics
        self.beta = beta
        self.rng = np.random.RandomState(random_state)

    def fit(self, documents, vocab=None, n_iter=500, verbose=True, log_every=50):
        """
        Parameters
        ----------
        documents : list of list of str
        vocab : Vocabulary or None
            If None, build vocabulary from documents.
        """
        if vocab is None:
            vocab = Vocabulary.build(documents)
        self.vocab = vocab
        self.V = vocab.V
        self.docs = vocab.docs2ids(documents)
        self.D = len(self.docs)

        self._initialize()

        for it in range(n_iter):
            self._gibbs_sweep()
            if verbose and (it + 1) % log_every == 0:
                ll = self._log_likelihood()
                print(f"  iter {it+1:4d}/{n_iter}  log-likelihood: {ll:.1f}")

        self._compute_distributions()
        return self

    def _initialize(self):
        K, V, D = self.K, self.V, self.D
        self.n_dk = np.zeros((D, K), dtype=np.int32)
        self.n_kw = np.zeros((K, V), dtype=np.int32)
        self.n_k = np.zeros(K, dtype=np.int32)

        self.z = []
        for d, doc in enumerate(self.docs):
            z_doc = self.rng.randint(0, K, size=len(doc))
            self.z.append(z_doc)
            for i, w in enumerate(doc):
                k = z_doc[i]
                self.n_dk[d, k] += 1
                self.n_kw[k, w] += 1
                self.n_k[k] += 1

    def _gibbs_sweep(self):
        K, V = self.K, self.V
        alpha, beta = self.alpha, self.beta

        beta_V = beta * V

        for d, doc in enumerate(self.docs):
            for i, w in enumerate(doc):
                k_old = self.z[d][i]

                self.n_dk[d, k_old] -= 1
                self.n_kw[k_old, w] -= 1
                self.n_k[k_old] -= 1

                p = (self.n_dk[d] + alpha) * (self.n_kw[:, w] + beta) / (self.n_k + beta_V)
                p /= p.sum()
                k_new = self.rng.choice(K, p=p)

                self.z[d][i] = k_new
                self.n_dk[d, k_new] += 1
                self.n_kw[k_new, w] += 1
                self.n_k[k_new] += 1

    def _compute_distributions(self):
        self.phi = (self.n_kw + self.beta) / (self.n_k[:, None] + self.V * self.beta)
        self.theta = (self.n_dk + self.alpha) / (
            self.n_dk.sum(axis=1, keepdims=True) + self.K * self.alpha
        )

    def _log_likelihood(self):
        ll = 0.0
        for d, doc in enumerate(self.docs):
            for i, w in enumerate(doc):
                k = self.z[d][i]
                pw = (self.n_kw[k, w] + self.beta) / (self.n_k[k] + self.V * self.beta)
                ll += np.log(pw + 1e-30)
        return ll

    def top_words(self, n=10):
        topics = []
        for k in range(self.K):
            top_ids = self.phi[k].argsort()[::-1][:n]
            topics.append(self.vocab.ids2words(top_ids))
        return topics

    def print_topics(self, n=10):
        for k, words in enumerate(self.top_words(n)):
            print(f"  Topic {k:2d}: {', '.join(words)}")

    def transform(self):
        return self.theta


class LabeledLDA:

    def __init__(
        self,
        n_topics=None,
        alpha=0.01,
        beta=0.01,
        label_prior_strength=10.0,
        random_state=42,
    ):
        self.K = n_topics
        self.alpha = alpha
        self.beta = beta
        self.label_prior_strength = label_prior_strength
        self.rng = np.random.RandomState(random_state)

    def fit(self, documents, labels, vocab=None, n_iter=500, verbose=True, log_every=50):
        """
        Parameters
        ----------
        documents : list of list of str
        labels : list of list of str
            Each document's label set.
        vocab : Vocabulary or None
            If None, build vocabulary from documents.
        """
        if vocab is None:
            vocab = Vocabulary.build(documents)
        self.vocab = vocab
        self.V = vocab.V
        self.docs = vocab.docs2ids(documents)
        self.D = len(self.docs)

        self._build_label_map(labels)
        self._initialize()

        for it in range(n_iter):
            self._gibbs_sweep()
            if verbose and (it + 1) % log_every == 0:
                ll = self._log_likelihood()
                print(f"  iter {it+1:4d}/{n_iter}  log-likelihood: {ll:.1f}")

        self._compute_distributions()
        return self

    def _build_label_map(self, labels):
        all_labels = sorted(set(label for doc_labels in labels for label in doc_labels))
        n_labeled_topics = len(all_labels)
        if self.K is None:
            if n_labeled_topics == 0:
                raise ValueError("n_topics is required when no labels are provided")
            self.K = n_labeled_topics
        if self.K < n_labeled_topics:
            raise ValueError(
                f"n_topics={self.K} must be >= number of labels={n_labeled_topics}"
            )

        self.label_names = all_labels + [
            f"unlabeled_topic_{topic_id}"
            for topic_id in range(n_labeled_topics, self.K)
        ]
        self.label2id = {label: topic_id for topic_id, label in enumerate(all_labels)}
        self.default_topic_prior = np.full(self.K, self.alpha, dtype=np.float64)

        self.doc_labels = []
        self.doc_topic_priors = []
        for doc_labels in labels:
            if len(doc_labels) == 0:
                preset_topics = np.array([], dtype=np.int32)
                topic_prior = self.default_topic_prior
            else:
                preset_topics = np.array(
                    [self.label2id[label] for label in doc_labels],
                    dtype=np.int32,
                )
                topic_prior = self._build_doc_topic_prior(preset_topics)
            self.doc_labels.append(preset_topics)
            self.doc_topic_priors.append(topic_prior)

    def _build_doc_topic_prior(self, preset_topics):
        topic_prior = self.default_topic_prior.copy()
        topic_prior[preset_topics] = self.label_prior_strength
        return topic_prior

    def _initialize(self):
        K, V = self.K, self.V

        self.n_dk = np.zeros((self.D, K), dtype=np.int32)
        self.n_kw = np.zeros((K, V), dtype=np.int32)
        self.n_k = np.zeros(K, dtype=np.int32)

        self.z = []
        for d, doc in enumerate(self.docs):
            topic_probs = self.doc_topic_priors[d] / self.doc_topic_priors[d].sum()
            z_doc = self.rng.choice(K, size=len(doc), p=topic_probs)
            self.z.append(z_doc)
            for i, w in enumerate(doc):
                k = z_doc[i]
                self.n_dk[d, k] += 1
                self.n_kw[k, w] += 1
                self.n_k[k] += 1

    def _gibbs_sweep(self):
        alpha, beta = self.alpha, self.beta
        beta_V = beta * self.V

        for d, doc in enumerate(self.docs):
            topic_prior = self.doc_topic_priors[d]

            for i, w in enumerate(doc):
                k_old = self.z[d][i]

                self.n_dk[d, k_old] -= 1
                self.n_kw[k_old, w] -= 1
                self.n_k[k_old] -= 1

                p = (
                    (self.n_dk[d] + topic_prior)
                    * (self.n_kw[:, w] + beta)
                    / (self.n_k + beta_V)
                )
                p /= p.sum()

                k_new = self.rng.choice(self.K, p=p)

                self.z[d][i] = k_new
                self.n_dk[d, k_new] += 1
                self.n_kw[k_new, w] += 1
                self.n_k[k_new] += 1

    def _compute_distributions(self):
        self.phi = (self.n_kw + self.beta) / (self.n_k[:, None] + self.V * self.beta)
        theta_prior = np.vstack(self.doc_topic_priors)
        self.theta = (self.n_dk + theta_prior) / (
            self.n_dk.sum(axis=1, keepdims=True)
            + theta_prior.sum(axis=1, keepdims=True)
        )

    def _log_likelihood(self):
        ll = 0.0
        for d, doc in enumerate(self.docs):
            for i, w in enumerate(doc):
                k = self.z[d][i]
                pw = (self.n_kw[k, w] + self.beta) / (self.n_k[k] + self.V * self.beta)
                ll += np.log(pw + 1e-30)
        return ll

    def top_words(self, n=10):
        topics = {}
        for k in range(self.K):
            top_ids = self.phi[k].argsort()[::-1][:n]
            topics[self.label_names[k]] = self.vocab.ids2words(top_ids)
        return topics

    def print_topics(self, n=10):
        for label, words in self.top_words(n).items():
            print(f"  {label:>15s}: {', '.join(words)}")

    def predict(self):
        return self.theta

    def predict_labels(self, threshold=0.1):
        predictions = []
        for d in range(self.D):
            pred = [
                self.label_names[k]
                for k in range(self.K)
                if self.theta[d, k] > threshold
            ]
            predictions.append(pred)
        return predictions


# ── Walker's Alias Table ────────────────────────────────────────────
# Allows O(K) build, O(1) sampling from any discrete distribution.
# Idea: partition the probability mass into K equal-area bins, each
# containing at most 2 outcomes. A sample is: pick a random bin,
# then flip a biased coin to choose which of the 2 outcomes.

class AliasTable:

    def __init__(self, weights):
        weights = np.asarray(weights, dtype=np.float64)
        K = len(weights)
        self.K = K
        self.prob = np.ones(K, dtype=np.float64)
        self.alias = np.arange(K, dtype=np.int32)

        total = weights.sum()
        if total <= 0:
            return
        p = weights * (K / total)

        small, large = [], []
        for i in range(K):
            (small if p[i] < 1.0 else large).append(i)

        while small and large:
            s = small.pop()
            l = large.pop()
            self.prob[s] = p[s]
            self.alias[s] = l
            p[l] -= (1.0 - p[s])
            (small if p[l] < 1.0 else large).append(l)

        for i in small + large:
            self.prob[i] = 1.0

    def sample(self, rng):
        k = rng.randint(self.K)
        return k if rng.rand() < self.prob[k] else self.alias[k]


# ── LightLDA ────────────────────────────────────────────────────────

class LightLDA:
    """
    Core decomposition
    ------------------
    Full conditional:
        p(z_i=k | rest) ∝ (n_dk + α) · (n_kw + β) / (n_k + Vβ)
                            ╰─ doc ──╯   ╰──── word ────────────╯

    Two proposals (alternate with prob 0.5 each):

    1. Doc-proposal:  q_d(k) ∝ (n_dk + α)
       Accept ratio:  (n_k'w + β)/(n_k' + Vβ)
                    / (n_kw  + β)/(n_k  + Vβ)
       → Only the word-side ratio matters (doc-side cancels).

    2. Word-proposal: q_w(k) ∝ (n_kw + β) / (n_k + Vβ)
       Accept ratio:  (n_dk' + α)
                    / (n_dk  + α)
       → Only the doc-side ratio matters (word-side cancels).

    Both proposals use alias tables → O(1) sampling.
    MH correction guarantees the correct stationary distribution
    even with stale alias tables (just lower acceptance rate).
    """

    def __init__(self, n_topics, alpha=None, beta=0.01, random_state=42):
        self.K = n_topics
        self.alpha = alpha if alpha is not None else 50.0 / n_topics
        self.beta = beta
        self.rng = np.random.RandomState(random_state)

    def fit(
        self,
        documents,
        vocab=None,
        n_iter=500,
        max_stale=50,
        mh_steps=2,
        verbose=True,
        log_every=50,
    ):
        """
        Parameters
        ----------
        documents : list of list of str
        vocab : Vocabulary or None
            If None, build vocabulary from documents.
        n_iter : int
            Number of MH sweeps.
        max_stale : int
            Rebuild a word's alias table after it has been sampled this many
            times. Common words hit this sooner → rebuild more often.
            Rare words stay fresh longer → fewer wasted rebuilds.
        mh_steps : int
            Number of MH sub-steps per token in each sweep. Each sub-step runs
            one doc proposal followed by one word proposal.
        """
        if vocab is None:
            vocab = Vocabulary.build(documents)
        self.vocab = vocab
        self.V = vocab.V
        self.docs = vocab.docs2ids(documents)
        self.D = len(self.docs)

        self._initialize()
        self.mh_steps = mh_steps
        self._max_stale = max_stale
        self._word_alias_samples = np.zeros(self.V, dtype=np.int32)
        self._word_alias = {}
        for w in range(self.V):
            self._rebuild_word_alias(w)

        for it in range(n_iter):
            self._mhw_sweep()

            if verbose and (it + 1) % log_every == 0:
                ll = self._log_likelihood()
                print(f"  iter {it+1:4d}/{n_iter}  log-likelihood: {ll:.1f}")

        self._compute_distributions()
        return self

    def _initialize(self):
        K, V, D = self.K, self.V, self.D

        self.n_dk = np.zeros((D, K), dtype=np.int32)
        self.n_kw = np.zeros((K, V), dtype=np.int32)
        self.n_k = np.zeros(K, dtype=np.int32)

        self._doc_len = [len(doc) for doc in self.docs]
        self._K_alpha = K * self.alpha

        self.z = []
        for d, doc in enumerate(self.docs):
            z_doc = self.rng.randint(0, K, size=len(doc))
            self.z.append(z_doc)
            for i, w in enumerate(doc):
                k = z_doc[i]
                self.n_dk[d, k] += 1
                self.n_kw[k, w] += 1
                self.n_k[k] += 1

    def _rebuild_word_alias(self, w):
        """Rebuild alias table for a single word and reset its sample counter."""
        weights = (self.n_kw[:, w] + self.beta) / (self.n_k + self.beta * self.V)
        self._word_alias[w] = AliasTable(weights)
        self._word_alias_samples[w] = 0

    def _sample_doc_proposal(self, d):
        # q_d(k) ∝ n_dk + α + 𝟙(k == old_topic)
        #   smooth (total K*α):   uniform → randint(K)
        #   word mass (total n_d): pick any position j (including stale z[d][i])
        n_d = self._doc_len[d]
        if self.rng.rand() * (n_d + self._K_alpha) < self._K_alpha:
            return self.rng.randint(self.K)
        else:
            return self.z[d][self.rng.randint(n_d)]

    def _sample_word_proposal(self, w):
        self._word_alias_samples[w] += 1
        if self._word_alias_samples[w] >= self._max_stale:
            self._rebuild_word_alias(w)
        return self._word_alias[w].sample(self.rng)

    def _mhw_sweep(self):
        alpha, beta = self.alpha, self.beta
        beta_V = beta * self.V
        rng = self.rng

        for d, doc in enumerate(self.docs):
            for i, w in enumerate(doc):
                old_topic = self.z[d][i]
                k = old_topic

                self.n_dk[d, k] -= 1
                self.n_kw[k, w] -= 1
                self.n_k[k] -= 1

                for _ in range(self.mh_steps):
                    # Doc proposal → accept/reject
                    # A = π(new)·q_d(old) / (π(old)·q_d(new))
                    # q_d samples from stale z[d], so only old_topic has +1 mass.
                    k_new = self._sample_doc_proposal(d)
                    if k_new != k:
                        n_new_d = self.n_dk[d, k_new] + alpha
                        n_old_d = self.n_dk[d, k] + alpha
                        n_new_w = self.n_kw[k_new, w] + beta
                        n_old_w = self.n_kw[k, w] + beta
                        n_new_sum = self.n_k[k_new] + beta_V
                        n_old_sum = self.n_k[k] + beta_V
                        doc_old_stale_count = 1 if k == old_topic else 0
                        doc_new_stale_count = 1 if k_new == old_topic else 0
                        proposal_old = n_old_d + doc_old_stale_count
                        proposal_new = n_new_d + doc_new_stale_count
                        numer = n_new_d * n_new_w * n_old_sum * proposal_old
                        denom = n_old_d * n_old_w * n_new_sum * proposal_new
                        if rng.rand() < numer / denom:
                            k = k_new

                    # Word proposal → accept/reject
                    # A = π(new)·q_w(old) / (π(old)·q_w(new))
                    # alias table built before decrement → only old_topic has +1 mass.
                    k_new = self._sample_word_proposal(w)
                    if k_new != k:
                        n_new_d = self.n_dk[d, k_new] + alpha
                        n_old_d = self.n_dk[d, k] + alpha
                        n_new_w = self.n_kw[k_new, w] + beta
                        n_old_w = self.n_kw[k, w] + beta
                        n_new_sum = self.n_k[k_new] + beta_V
                        n_old_sum = self.n_k[k] + beta_V
                        word_old_stale_count = 1 if k == old_topic else 0
                        word_new_stale_count = 1 if k_new == old_topic else 0
                        proposal_old = (n_old_w + word_old_stale_count) / (
                            n_old_sum + word_old_stale_count
                        )
                        proposal_new = (n_new_w + word_new_stale_count) / (
                            n_new_sum + word_new_stale_count
                        )
                        numer = n_new_d * n_new_w * n_old_sum * proposal_old
                        denom = n_old_d * n_old_w * n_new_sum * proposal_new
                        if rng.rand() < numer / denom:
                            k = k_new

                self.z[d][i] = k
                self.n_dk[d, k] += 1
                self.n_kw[k, w] += 1
                self.n_k[k] += 1

    def _compute_distributions(self):
        self.phi = (self.n_kw + self.beta) / (self.n_k[:, None] + self.V * self.beta)
        self.theta = (self.n_dk + self.alpha) / (
            self.n_dk.sum(axis=1, keepdims=True) + self.K * self.alpha
        )

    def _log_likelihood(self):
        ll = 0.0
        for d, doc in enumerate(self.docs):
            for i, w in enumerate(doc):
                k = self.z[d][i]
                pw = (self.n_kw[k, w] + self.beta) / (self.n_k[k] + self.V * self.beta)
                ll += np.log(pw + 1e-30)
        return ll

    def top_words(self, n=10):
        topics = []
        for k in range(self.K):
            top_ids = self.phi[k].argsort()[::-1][:n]
            topics.append(self.vocab.ids2words(top_ids))
        return topics

    def print_topics(self, n=10):
        for k, words in enumerate(self.top_words(n)):
            print(f"  Topic {k:2d}: {', '.join(words)}")

    def transform(self):
        return self.theta


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one topic model: lda, labeled_lda, or lightlda."
    )
    parser.add_argument(
        "model",
        choices=["lda", "labeled_lda", "labeled-lda", "lightlda"],
        help="Topic model to run.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Input corpus path. For lda/lightlda, supports .txt or .jsonl. "
            "For labeled_lda, expects JSONL with labels. Uses demo corpus if omitted."
        ),
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help="Optional vocabulary JSON path. If set, load and use it directly.",
    )
    parser.add_argument("--n-topics", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--top-words", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument(
        "--label-prior-strength",
        type=float,
        default=10.0,
        help=(
            "LabeledLDA only: sampling weight multiplier for preset label topics. "
            "Use 1.0 to make labels non-informative."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path to save model outputs.",
    )
    parser.add_argument(
        "--max-stale",
        type=int,
        default=50,
        help="LightLDA only: rebuild a word alias table after this many samples.",
    )
    parser.add_argument(
        "--mh-steps",
        type=int,
        default=2,
        help="LightLDA only: MH sub-steps per token.",
    )
    return parser


def load_unlabeled_documents(input_path):
    if input_path is None:
        return make_demo_corpus()
    return read_corpus(input_path)


def load_labeled_documents(input_path):
    if input_path is None:
        return make_labeled_demo_corpus()
    return read_labeled_corpus(input_path)


def load_or_build_vocab(documents, vocab_path):
    if vocab_path is not None:
        return Vocabulary.load(vocab_path)
    return Vocabulary.build(documents)


def run_lda(args):
    documents = load_unlabeled_documents(args.input)
    vocab = load_or_build_vocab(documents, args.vocab)
    n_topics = args.n_topics or len(DEMO_TOPIC_WORDS)

    model = LDA(
        n_topics=n_topics,
        alpha=args.alpha,
        beta=args.beta,
        random_state=args.seed,
    )
    model.fit(
        documents,
        vocab=vocab,
        n_iter=args.n_iter,
        verbose=not args.quiet,
        log_every=args.log_every,
    )
    return model, documents, vocab


def run_lightlda(args):
    documents = load_unlabeled_documents(args.input)
    vocab = load_or_build_vocab(documents, args.vocab)
    n_topics = args.n_topics or len(DEMO_TOPIC_WORDS)

    model = LightLDA(
        n_topics=n_topics,
        alpha=args.alpha,
        beta=args.beta,
        random_state=args.seed,
    )
    model.fit(
        documents,
        vocab=vocab,
        n_iter=args.n_iter,
        max_stale=args.max_stale,
        mh_steps=args.mh_steps,
        verbose=not args.quiet,
        log_every=args.log_every,
    )
    return model, documents, vocab


def run_labeled_lda(args):
    documents, labels = load_labeled_documents(args.input)
    vocab = load_or_build_vocab(documents, args.vocab)
    alpha = args.alpha if args.alpha is not None else 0.01

    model = LabeledLDA(
        n_topics=args.n_topics,
        alpha=alpha,
        beta=args.beta,
        label_prior_strength=args.label_prior_strength,
        random_state=args.seed,
    )
    model.fit(
        documents,
        labels,
        vocab=vocab,
        n_iter=args.n_iter,
        verbose=not args.quiet,
        log_every=args.log_every,
    )
    return model, documents, vocab


def print_run_summary(model_name, documents, vocab, model):
    print(f"Model: {model_name}")
    print(f"Corpus: {len(documents)} docs, {vocab}")
    print(f"Topics: {model.K}")


def build_save_payload(
    model_name,
    documents,
    vocab,
    model,
    top_words,
    theta_path,
    phi_path,
):
    payload = {
        "model": model_name,
        "n_docs": len(documents),
        "vocab_size": vocab.V,
        "n_topics": model.K,
        "vocabulary": vocab.vocab,
        "topics": model.top_words(n=top_words),
        "theta_path": str(theta_path),
        "phi_path": str(phi_path),
    }

    if model_name == "labeled_lda":
        payload["label_names"] = model.label_names
        payload["predicted_labels"] = model.predict_labels()

    return payload


def save_model_outputs(output_path, model_name, documents, vocab, model, top_words):
    output_json_path = Path(output_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    theta_path = output_json_path.with_suffix(".theta.npy")
    phi_path = output_json_path.with_suffix(".phi.npy")
    np.save(theta_path, model.theta)
    np.save(phi_path, model.phi)

    payload = build_save_payload(
        model_name,
        documents,
        vocab,
        model,
        top_words,
        theta_path,
        phi_path,
    )
    with open(output_json_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    print(f"\nSaved outputs to: {output_json_path}")
    print(f"Saved theta to: {theta_path}")
    print(f"Saved phi to: {phi_path}")


def main():
    args = build_parser().parse_args()
    model_name = args.model.replace("-", "_")

    if model_name == "lda":
        model, documents, vocab = run_lda(args)
    elif model_name == "lightlda":
        model, documents, vocab = run_lightlda(args)
    else:
        model, documents, vocab = run_labeled_lda(args)

    print_run_summary(model_name, documents, vocab, model)
    print("\nTop words:")
    model.print_topics(n=args.top_words)

    if args.output is not None:
        save_model_outputs(
            args.output,
            model_name,
            documents,
            vocab,
            model,
            args.top_words,
        )


if __name__ == "__main__":
    main()