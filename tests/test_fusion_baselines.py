import unittest
import torch
from models.sequence_homology_fusion_baselines import (SequenceHomologyFixedFusionModel, SequenceHomologyIdentityFusionModel, identity_neural_weight)
from reliability_aware.utils.identity_fusion import build_identity_sidecar
from reliability_aware.utils.diamond_homology import DiamondSearchConfig
from pathlib import Path
import tempfile

class FusionBaselineTests(unittest.TestCase):
    def test_fixed_is_global_and_has_gradient(self):
        torch.manual_seed(1)
        m = SequenceHomologyFixedFusionModel(3, attn_hidden_dim=4, attn_dropout=0, head_hidden_dim=4, head_dropout=0)
        x = torch.randn(2, 3, 1280); mask = torch.ones(2, 3, dtype=torch.bool); h = torch.rand(2, 3)
        out = m(x, mask, h, gate_features=torch.randn(2,4))
        self.assertTrue(torch.equal(out["gate_weights"], torch.full((2,2), .5)))
        out["probs"].sum().backward()
        self.assertTrue(torch.isfinite(m.fusion_logit.grad) and m.fusion_logit.grad.abs() > 0)
        state = m.state_dict(); clone = SequenceHomologyFixedFusionModel(3, attn_hidden_dim=4, attn_dropout=0, head_hidden_dim=4, head_dropout=0); clone.load_state_dict(state)
        self.assertTrue(torch.allclose(out["probs"], clone(x, mask, h)["probs"]))

    def test_identity_formula_and_no_hit(self):
        s = torch.tensor([0., .5, 1.]); w = identity_neural_weight(s, .25, 5.)
        self.assertGreater(w[0], w[1]); self.assertGreater(w[1], w[2])
        self.assertTrue(torch.equal(identity_neural_weight(s, 1., 2.), torch.ones_like(s)))
        m = SequenceHomologyIdentityFusionModel(2, .25, 5., attn_hidden_dim=4, attn_dropout=0, head_hidden_dim=4, head_dropout=0)
        x = torch.randn(1, 2, 1280); mask = torch.ones(1,2,dtype=torch.bool); h = torch.rand(1,2)
        out = m(x, mask, h, identity_fraction=torch.tensor([0.]), has_retained_hit=torch.tensor([False]))
        self.assertTrue(torch.allclose(out["probs"], out["neural_probs"]))

    def test_top_five_weighted_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); hits = p / "h.tsv"; out = p / "sidecar.json"
            hits.write_text("q\th1\t1e-9\t100\t50\t100\t100\t50\t50\t50\nq\th2\t1e-9\t50\t50\t100\t100\t80\t80\t80\n")
            result = build_identity_sidecar(query_ids=["q"], hits_tsv=hits, output_path=out, config=DiamondSearchConfig(top_k=10))
            self.assertAlmostEqual(result["records"][0]["identity_fraction"], .6)

if __name__ == "__main__": unittest.main()
