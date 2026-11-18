"""AGFL 层的六条结构性检查（对应第四轮指令 §4.1）。"""
import unittest

import torch

from deception.agfl import AGFLLayer, build_edge_mask, masked_topk_softmax


class AGFLLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_adjacency_rows_are_valid_probability_distributions(self):
        scores = torch.randn(4, 12, 12)
        mask = build_edge_mask(12)
        adjacency = masked_topk_softmax(scores, mask, kappa=5)
        self.assertTrue(torch.all(adjacency >= 0))
        row_sums = adjacency.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-6)
        self.assertLessEqual(int((adjacency > 0).sum(dim=-1).max()), 5)
        diagonal = torch.diagonal(adjacency, dim1=-2, dim2=-1)
        self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal)))

    def test_mask_is_applied_before_topk(self):
        scores = torch.zeros(1, 6, 6)
        mask = build_edge_mask(6)
        mask[0, 1] = False          # 1 不再是 0 的合法邻居
        scores[0, 0, 1] = 1e6       # 被屏蔽的边分数最高
        adjacency = masked_topk_softmax(scores, mask, kappa=5)
        self.assertEqual(float(adjacency[0, 0, 1]), 0.0)
        self.assertGreater(int((adjacency[0] > 0).sum(dim=-1).max()), 0)

    def test_block_mask_has_no_cross_modality_edges(self):
        mask = build_edge_mask(20, group_sizes=[16, 4])
        self.assertTrue(mask[:16, 16:].sum() == 0)
        self.assertTrue(mask[16:, :16].sum() == 0)
        self.assertTrue(mask[0, 15] and mask[16, 19])
        self.assertFalse(mask[0, 0])

    def test_node_without_eligible_neighbours_raises(self):
        mask = build_edge_mask(4)
        mask[2] = False
        with self.assertRaisesRegex(ValueError, "no eligible neighbour"):
            masked_topk_softmax(torch.randn(1, 4, 4), mask, kappa=2)

    def test_zero_hop_ignores_the_adjacency(self):
        layer = AGFLLayer(d=8, a=8, hops=0, kappa=3)
        layer.eval()
        h = torch.randn(3, 5, 8)
        first = layer(h, build_edge_mask(5))
        shifted = build_edge_mask(5)
        shifted[0, 1] = False
        second = layer(h, shifted)
        torch.testing.assert_close(first, second)

    def test_output_is_finite_and_norm_controlled(self):
        layer = AGFLLayer(d=8, a=8, hops=3, kappa=3)
        layer.eval()
        h = torch.randn(2, 6, 8)
        out, hops = layer(h, build_edge_mask(6), return_hops=True)
        self.assertEqual(out.shape, h.shape)
        self.assertEqual(out.dtype, h.dtype)
        self.assertTrue(torch.isfinite(out).all())
        reference = hops[0].norm(dim=-1)
        for value in hops[1:]:
            self.assertTrue(bool((value.norm(dim=-1) <= reference + 1e-5).all()))

    def test_same_seed_gives_identical_forward(self):
        torch.manual_seed(2026)
        layer = AGFLLayer(d=8, a=8, hops=2, kappa=3)
        h = torch.randn(2, 6, 8)
        mask = build_edge_mask(6)
        first = layer(h, mask).detach().clone()
        second = layer(h, mask).detach().clone()
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
