import math
import pytest
from vec import Vec


# ------------------------------------------------------------------------------
# Tests for mean()
# ------------------------------------------------------------------------------

def test_mean_integers():
    v = Vec([3, 7, 5, 9])
    assert v.mean() == 6.0


def test_mean_with_negative_numbers():
    v = Vec([-10, 4, 6])
    assert v.mean() == 0.0


def test_mean_all_negative():
    v = Vec([-4.0, -8.0])
    assert v.mean() == -6.0


def test_mean_decimals():
    v = Vec([0.25, 0.75, 1.25, 1.75])
    assert v.mean() == pytest.approx(1.0)


def test_mean_constant_vector():
    v = Vec([8, 8, 8, 8])
    assert v.mean() == 8.0


def test_mean_single_element():
    v = Vec([15])
    assert v.mean() == 15.0


def test_mean_empty_vector_raises_error():
    v = Vec([])
    with pytest.raises(ValueError, match="empty vector"):
        v.mean()


# ------------------------------------------------------------------------------
# Tests for demean()
# ------------------------------------------------------------------------------

def test_demean_values():
    v = Vec([2, 6, 10, 14])  # mean is 8.0
    centered = v.demean()
    assert centered.elements == (-6.0, -2.0, 2.0, 6.0)


def test_demean_mean_equals_zero():
    v = Vec([1, 5, 12, 18])
    centered = v.demean()
    assert centered.mean() == pytest.approx(0.0)


def test_demean_preserves_length():
    v = Vec([10, 25, 40])
    assert len(v.demean()) == len(v)


def test_demean_constant_vector():
    v = Vec([7, 7, 7])
    assert v.demean().elements == (0.0, 0.0, 0.0)


def test_demean_does_not_modify_original():
    v = Vec([3, 6, 9])
    original_copy = v.elements
    v.demean()
    assert v.elements == original_copy


def test_demean_returns_new_instance():
    v = Vec([4, 8, 12])
    res = v.demean()
    assert isinstance(res, Vec)
    assert res is not v


# ------------------------------------------------------------------------------
# Tests for std()
# ------------------------------------------------------------------------------

def test_std_known_value():
    v = Vec([1, 3, 5, 7])  # mean=4, var=(9+1+1+9)/4 = 5
    assert v.std() == pytest.approx(math.sqrt(5.0))


def test_std_constant_vector_is_zero():
    v = Vec([9, 9, 9, 9])
    assert v.std() == 0.0


def test_std_single_element_is_zero():
    v = Vec([42])
    assert v.std() == 0.0


def test_std_is_non_negative():
    v = Vec([-15, 0, 25])
    assert v.std() >= 0.0


def test_std_shift_invariance():
    v1 = Vec([2, 5, 8, 11])
    v2 = Vec([102, 105, 108, 111])  # shifted by +100
    assert v1.std() == pytest.approx(v2.std())


def test_std_order_invariance():
    v1 = Vec([3, 1, 9, 7])
    v2 = Vec([9, 7, 3, 1])
    assert v1.std() == pytest.approx(v2.std())


def test_std_scaling():
    v = Vec([2, 4, 6])
    c = 3.0
    scaled = v.scalar_mult(c)
    assert scaled.std() == pytest.approx(abs(c) * v.std())


def test_std_empty_vector_raises_error():
    v = Vec([])
    with pytest.raises(ValueError, match="empty vector"):
        v.std()


def test_std_uses_demean(monkeypatch):
    v = Vec([2, 4, 6])
    called = False
    real_demean = v.demean

    def tracked_demean():
        nonlocal called
        called = True
        return real_demean()

    monkeypatch.setattr(v, "demean", tracked_demean)
    v.std()
    assert called is True

