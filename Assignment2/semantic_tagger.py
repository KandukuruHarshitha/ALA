import os
import string
import math
import numpy as np
import gensim.downloader as api


# ==============================================================================
# Vector Class Implementation (using tuple storage)
# ==============================================================================
class Vector:
    """Mathematical vector storing elements as a tuple for similarity computations."""

    def __init__(self, elements=None):
        if elements is None:
            self.elements = ()
        elif isinstance(elements, Vector):
            self.elements = elements.elements
        elif isinstance(elements, np.ndarray):
            self.elements = tuple(float(x) for x in elements.ravel())
        else:
            self.elements = tuple(float(x) for x in elements)

    def dot(self, other):
        """Compute the Euclidean dot product with another vector."""
        if len(self.elements) != len(other.elements):
            raise ValueError(f"Dimension mismatch: {len(self.elements)} vs {len(other.elements)}")
        return sum(a * b for a, b in zip(self.elements, other.elements))

    def norm(self):
        """Compute the Euclidean L2 norm of the vector."""
        return math.sqrt(sum(x * x for x in self.elements))

    def cosine_similarity(self, other):
        """Compute cosine similarity cos(u, v) = (u . v) / (||u|| * ||v||)."""
        norm_a = self.norm()
        norm_b = other.norm()
        if norm_a == 0.0 or norm_b == 0.0:
            raise ValueError("Cannot compute cosine similarity with a zero vector.")
        return self.dot(other) / (norm_a * norm_b)

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        if len(self.elements) <= 4:
            return f"Vector({self.elements})"
        return f"Vector({self.elements[:3]}...)"


# ==============================================================================
# Part A — 20 Tags Selection
# ==============================================================================
TAGS = [
    "academics", "discovery", "learning", "institute", "graduates",
    "professors", "library", "computing", "healthcare", "science",
    "pedagogy", "partnership", "journal", "experiment", "fellowship",
    "guidance", "placement", "incubation", "ranking", "higher education"
]


def build_tag_matrix(model, tags):
    """
    Construct the tag matrix T of shape (20, 50).
    For single-word tags, takes the 50-d GloVe vector directly.
    For multi-word tags, averages the individual word vectors.
    Raises ValueError if any component word is missing from the vocabulary.
    """
    rows = []
    for tag in tags:
        parts = tag.strip().lower().split()
        if not parts:
            raise ValueError("Tag name cannot be empty.")

        vec_list = []
        for word in parts:
            if word not in model.key_to_index:
                raise ValueError(f"Component word '{word}' in tag '{tag}' is not in model vocabulary.")
            vec_list.append(model[word])

        # Arithmetic mean for multi-word tag, or single vector for 1-word tag
        combined_vec = np.mean(vec_list, axis=0)
        rows.append(combined_vec)

    T = np.vstack(rows)
    return tags, T


# ==============================================================================
# Part B & C — Text Preprocessing Pipeline
# ==============================================================================
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to",
    "in", "on", "for", "is", "are", "was", "were",
    "with", "at", "by", "from",
    "its", "their", "our", "he", "them", "as", "they",
    "into", "through", "across", "such"
}


def read_text_file(filepath):
    """Read plain text from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def preprocess_text(raw_text):
    """
    Preprocess text strictly according to the 5-step order:
    1. convert to lowercase
    2. remove punctuation
    3. split into tokens
    4. remove stopwords
    5. remove tokens with length <= 2
    """
    # 1. Lowercase
    text = raw_text.lower()
    # 2. Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    # 3. Split into tokens
    tokens = text.split()
    # 4. Remove custom stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]
    # 5. Remove tokens with length <= 2
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


# ==============================================================================
# Part D — Build Text Matrix W
# ==============================================================================
def build_text_matrix(model, tokens):
    """
    Construct text matrix W of shape (n, 50).
    Preserves token order and repeated words.
    Separates tokens into in-vocabulary and out-of-vocabulary lists.
    Raises ValueError if n == 0.
    """
    in_vocab_tokens = []
    oov_tokens = []
    word_vectors = []

    for t in tokens:
        if t in model.key_to_index:
            in_vocab_tokens.append(t)
            word_vectors.append(model[t])
        else:
            oov_tokens.append(t)

    n = len(in_vocab_tokens)
    if n == 0:
        raise ValueError("Input text contains zero in-vocabulary words. Cannot construct matrix W.")

    W = np.vstack(word_vectors)
    return in_vocab_tokens, oov_tokens, W


# ==============================================================================
# Part E — Cosine Similarity Using Vector Class (Loops)
# ==============================================================================
def similarity_matrix_vector_class(W, T):
    """
    Compute similarity matrix S of shape (n, 20) using the custom Vector class.
    Uses nested Python loops where S[i, j] = Vector(W[i]).cosine_similarity(Vector(T[j])).
    """
    n_words = W.shape[0]
    n_tags = T.shape[0]
    S = np.zeros((n_words, n_tags), dtype=float)

    for i in range(n_words):
        for j in range(n_tags):
            S[i, j] = Vector(W[i]).cosine_similarity(Vector(T[j]))

    return S


# ==============================================================================
# Part F — Cosine Similarity Using NumPy Matrix Multiplication
# ==============================================================================
def similarity_matrix_numpy(W, T):
    """
    Compute similarity matrix S of shape (n, 20) using vectorized NumPy operations:
    1. Normalize rows of W and T.
    2. Check that norms are non-zero.
    3. Compute S = W_hat @ T_hat.T.
    """
    W_norms = np.linalg.norm(W, axis=1, keepdims=True)
    T_norms = np.linalg.norm(T, axis=1, keepdims=True)

    if np.any(W_norms == 0) or np.any(T_norms == 0):
        raise ValueError("Encountered a zero-norm vector during row normalization.")

    W_hat = W / W_norms
    T_hat = T / T_norms

    S = W_hat @ T_hat.T
    return S


# ==============================================================================
# Part H — Rank Tags Using Max Pooling
# ==============================================================================
def rank_tags(tag_names, S, in_vocab_tokens):
    """
    For each tag j, compute max-pool score r_j = max_i S[i, j].
    Find the text token that produced the maximum score.
    Returns tags sorted in descending order of score.
    """
    results = []
    num_tags = len(tag_names)

    for j in range(num_tags):
        col = S[:, j]
        best_i = int(np.argmax(col))
        best_score = float(col[best_i])
        best_token = in_vocab_tokens[best_i]
        exact_match = (tag_names[j].lower() == best_token.lower())

        results.append({
            "tag": tag_names[j],
            "score": best_score,
            "best_word": best_token,
            "is_exact": exact_match
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


# ==============================================================================
# Main Program Runner
# ==============================================================================
def main():
    print("=" * 65)
    print("  Applied Linear Algebra Lab - Assignment 2")
    print("  Semantic Tagging with GloVe and NumPy")
    print("=" * 65)

    # 1. Load Pretrained GloVe Model
    print("\n[1] Loading pretrained GloVe model (glove-wiki-gigaword-50)...")
    model = api.load("glove-wiki-gigaword-50")
    print("    Model successfully loaded.")

    # 2. Build Tag Matrix T
    print("\n[2] Building Tag Matrix T...")
    tag_names, T = build_tag_matrix(model, TAGS)
    print(f"    Total tags: {len(tag_names)}")
    print(f"    T.shape = {T.shape}")

    # 3. Read & Preprocess Text
    text_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manipal_text.txt")
    print(f"\n[3] Reading text file: {text_path}")
    raw_text = read_text_file(text_path)
    raw_words = raw_text.split()
    processed_tokens = preprocess_text(raw_text)

    # 4. Build Text Matrix W
    in_vocab_tokens, oov_tokens, W = build_text_matrix(model, processed_tokens)
    unique_oov = sorted(list(set(oov_tokens)))

    print("\n--- Token Statistics ---")
    print(f"Tokens before preprocessing : {len(raw_words)}")
    print(f"Tokens after preprocessing  : {len(processed_tokens)}")
    print(f"In-vocabulary tokens (n)    : {len(in_vocab_tokens)}")
    print(f"Out-of-vocabulary tokens    : {len(oov_tokens)}")
    print(f"Distinct OOV tokens ({len(unique_oov)}): {unique_oov}")

    print("\n--- Matrix Shapes Checkpoint ---")
    print(f"T.shape = {T.shape}")
    print(f"W.shape = {W.shape}")

    # 5. Compute S via Vector class
    print("\n[5] Computing S via Vector class (loops)...")
    S_vector = similarity_matrix_vector_class(W, T)
    print(f"    S_vector.shape = {S_vector.shape}")

    # 6. Compute S via NumPy
    print("\n[6] Computing S via NumPy matrix multiplication...")
    S_numpy = similarity_matrix_numpy(W, T)
    print(f"    S_numpy.shape  = {S_numpy.shape}")

    # 7. Numerical Verification
    print("\n[7] Numerical Verification:")
    is_close = np.allclose(S_vector, S_numpy, atol=1e-6)
    max_difference = float(np.max(np.abs(S_vector - S_numpy)))
    print(f"    np.allclose(S_vector, S_numpy) : {is_close}")
    print(f"    Max absolute numerical diff    : {max_difference:.2e}")

    # 8. Max Pooling Tag Ranking
    print("\n[8] Top 8 Ranked Tags (Max Pooling):")
    ranked_tags = rank_tags(tag_names, S_numpy, in_vocab_tokens)

    print("-" * 68)
    print(f"{'Rank':<6}{'Tag':<18}{'Score':<10}{'Best-Matching Word':<20}{'Match Type'}")
    print("-" * 68)
    for rank, entry in enumerate(ranked_tags[:8], start=1):
        match_type = "Exact Match" if entry["is_exact"] else "Semantic (Diff Word)"
        print(f"{rank:<6}{entry['tag']:<18}{entry['score']:<10.4f}{entry['best_word']:<20}{match_type}")
    print("-" * 68)


if __name__ == "__main__":
    main()
