import numpy as np

from stablewide.experiment import tie_aware_ranks, identifiable_topk_mask, query_invariance_pass


def test_tie_aware_ranks():
    r = tie_aware_ranks(np.array([3.0, 2.0, 0.0, 0.0, 0.0]))
    assert np.allclose(r, [1.0, 2.0, 4.0, 4.0, 4.0])


def test_identifiable_topk_rejects_tie_crossing_cutoff():
    s = np.array([3.0, 2.0, 0.0, 0.0, 0.0])
    m2, ok2 = identifiable_topk_mask(s, 2)
    m3, ok3 = identifiable_topk_mask(s, 3)
    assert ok2 and m2 is not None and m2.sum() == 2
    assert not ok3 and m3 is None


def test_query_invariance_control():
    preds = np.array([
        [[0.2, 0.8], [0.7, 0.3]],
        [[0.2, 0.8], [0.7, 0.3]],
    ])
    assert query_invariance_pass(preds, atol=1e-6, rtol=1e-5)
