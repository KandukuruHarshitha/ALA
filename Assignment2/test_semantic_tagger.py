import os
import pytest
import numpy as np
from semantic_tagger import (
    Vector,
    TAGS,
    build_tag_matrix,
    read_text_file,
    preprocess_text,
    build_text_matrix,
    similarity_matrix_vector_class,
    similarity_matrix_numpy,
)


@pytest.fixture(scope="module")
def setup_pipeline():
    import gensim.downloader as api
    model = api.load("glove-wiki-gigaword-50")
    tag_names, T = build_tag_matrix(model, TAGS)

    text_file = os.path.join(os.path.dirname(__file__), "manipal_text.txt")
    raw_text = read_text_file(text_file)
    clean_tokens = preprocess_text(raw_text)
    in_vocab, oov, W = build_text_matrix(model, clean_tokens)

    S_vec = similarity_matrix_vector_class(W, T)
    S_np = similarity_matrix_numpy(W, T)

    return {
        "model": model,
        "tag_names": tag_names,
        "T": T,
        "in_vocab": in_vocab,
        "oov": oov,
        "W": W,
        "S_vector": S_vec,
        "S_numpy": S_np,
    }


# ------------------------------------------------------------------------------
# Required Tests from Assignment 2 PDF
# ------------------------------------------------------------------------------

# Required Test 1: Verify T.shape == (20, 50)
def test_1_tag_matrix_shape(setup_pipeline):
    T = setup_pipeline["T"]
    assert T.shape == (20, 50)


# Required Test 2: Verify W.shape == (n, 50)
def test_2_text_matrix_shape(setup_pipeline):
    W = setup_pipeline["W"]
    n = len(setup_pipeline["in_vocab"])
    assert W.shape == (n, 50)


# Required Test 3: Verify S.shape == (n, 20)
def test_3_similarity_matrix_shape(setup_pipeline):
    S_numpy = setup_pipeline["S_numpy"]
    n = len(setup_pipeline["in_vocab"])
    assert S_numpy.shape == (n, 20)


# Required Test 4: Pairwise similarity verification for a chosen (word, tag) pair
def test_4_pairwise_vector_and_numpy_agreement(setup_pipeline):
    W = setup_pipeline["W"]
    T = setup_pipeline["T"]
    S_numpy = setup_pipeline["S_numpy"]

    # Test pair (first word, second tag)
    word_vec = Vector(W[0])
    tag_vec = Vector(T[1])
    scalar_sim = word_vec.cosine_similarity(tag_vec)

    assert abs(scalar_sim - S_numpy[0, 1]) < 1e-6


# Required Test 5: Verify np.allclose(S_vector, S_numpy)
def test_5_similarity_matrices_allclose(setup_pipeline):
    S_vec = setup_pipeline["S_vector"]
    S_np = setup_pipeline["S_numpy"]
    assert np.allclose(S_vec, S_np, atol=1e-6)


# Required Test 6: Verify OOV tokens are excluded from W and returned in oov_tokens
def test_6_oov_tokens_handling(setup_pipeline):
    model = setup_pipeline["model"]
    sample_tokens = ["learning", "nonexistentwordxyz12345", "computing"]
    in_vocab, oov, W = build_text_matrix(model, sample_tokens)

    assert "nonexistentwordxyz12345" in oov
    assert "nonexistentwordxyz12345" not in in_vocab
    assert W.shape == (2, 50)


# Required Test 7: Verify OOV tag raises ValueError in build_tag_matrix
def test_7_oov_tag_raises_error(setup_pipeline):
    model = setup_pipeline["model"]
    invalid_tags = ["academics", "definitely_not_a_valid_glove_word_999"]
    with pytest.raises(ValueError, match="is not in model vocabulary"):
        build_tag_matrix(model, invalid_tags)


# ------------------------------------------------------------------------------
# Additional Independently Designed Tests
# ------------------------------------------------------------------------------

# Additional Test 8: Verify multi-word tag is average of component vectors
def test_8_multi_word_tag_averaging(setup_pipeline):
    model = setup_pipeline["model"]
    tags, T = build_tag_matrix(model, ["higher education"])
    expected_vec = (model["higher"] + model["education"]) / 2.0
    assert np.allclose(T[0], expected_vec)


# Additional Test 9: Verify build_text_matrix raises ValueError when 0 in-vocab words
def test_9_empty_in_vocab_raises_error(setup_pipeline):
    model = setup_pipeline["model"]
    oov_only_tokens = ["fakeunrealword1", "fakeunrealword2"]
    with pytest.raises(ValueError, match="zero in-vocabulary words"):
        build_text_matrix(model, oov_only_tokens)


# Additional Test 10: Vector dimension mismatch and zero vector error handling
def test_10_vector_class_edge_cases():
    u = Vector([1.0, 2.0, 3.0])
    v_diff_dim = Vector([1.0, 2.0])
    with pytest.raises(ValueError, match="Dimension mismatch"):
        u.dot(v_diff_dim)

    zero_v = Vector([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="zero vector"):
        u.cosine_similarity(zero_v)
