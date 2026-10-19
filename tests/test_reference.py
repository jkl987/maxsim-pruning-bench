import unittest
import numpy as np
from maxsim_pruning import coarse_scores, independent_upper, audit


class ReferenceTests(unittest.TestCase):
    def test_random_negative_duplicates_and_budgets(self):
        rng = np.random.default_rng(19)
        for shift in (0, -2):
            lut = rng.normal(size=(73, 32))+shift
            docs = [rng.integers(0, 73, size=n) for n in (1, 7, 19, 64)]
            docs += [np.array([3, 3, 3]), np.arange(73)]
            truth = coarse_scores(lut, docs)
            for levels in (2, 4, 8):
                upper, packed = independent_upper(lut, docs, levels, 32)
                self.assertTrue(np.all(upper >= truth))
                self.assertTrue(all(len(x) == levels*4 for x in packed))
                self.assertEqual(audit(truth, upper, 3)['upper_violations'], 0)

    def test_equal_boundaries_and_ties(self):
        lut = np.array([[0, -1], [0, -1], [1, 0]], dtype=float)
        docs = [np.array([0]), np.array([1]), np.array([2])]
        scores = coarse_scores(lut, docs)
        upper, _ = independent_upper(lut, docs)
        result = audit(scores, upper, 2, np.array([2, 1]))
        self.assertEqual(result['strict_missing'], 0)
        self.assertEqual(result['selected_below_cutoff'], 0)

    def test_zero_fields_and_top_fields(self):
        lut = np.array([[-5.]*32, [0.]*32, [1.]*32])
        upper, packed = independent_upper(lut, [np.array([0]), np.array([2])], 4, 1)
        self.assertEqual(packed[0], bytes(16))
        self.assertEqual(packed[1], bytes([255])*16)
        self.assertTrue(np.all(upper >= np.array([-160., 32.])))

    def test_invalid_inputs(self):
        for lut, docs in [(np.array([[np.nan]]), [[0]]), (np.ones((2, 2)), [[2]]),
                          (np.ones((2, 2)), [[]]), (np.ones((2, 2)), [[0.5]])]:
            with self.assertRaises(ValueError):
                independent_upper(lut, docs)
        with self.assertRaises(ValueError):
            audit(np.ones(3), np.ones(3), 2, np.array([1, 1]))

    def test_arithmetic_contract_and_integer_parameters(self):
        for lut in (np.array([[1+2j]]),np.full((3,32),1e308)):
            with self.assertRaises(ValueError):independent_upper(lut,[[0]])
        for prefix in (True,1.5):
            with self.assertRaises(ValueError):independent_upper(np.ones((2,2)),[[0]],prefix=prefix)
        for budget in (True,1.5):
            with self.assertRaises(ValueError):audit(np.ones(3),np.ones(3),budget)

    def test_audit_detects_real_miss_and_false_survivor(self):
        z=audit(np.array([3.,2.,1.]),np.array([3.,3.,3.]),2,np.array([1,2]))
        self.assertEqual(z['strict_missing'],1)
        self.assertEqual(z['selected_below_cutoff'],1)
        self.assertEqual(z['false_survivors'],1)


if __name__ == '__main__':
    unittest.main()
