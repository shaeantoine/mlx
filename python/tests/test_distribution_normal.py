import math
import mlx.distributions as dist

import mlx.core as mx
import mlx_tests


class TestDistributionNormal(mlx_tests.MLXTestCase):
    def test_basic_construction(self):
        # Test correct default behavior
        norm = dist.Normal()

        self.assertEqual(norm._loc.item(), 0.0)
        self.assertEqual(norm._scale.item(), 1.0)
        self.assertEqual(norm._loc.dtype, mx.float32)
        self.assertEqual(norm._scale.dtype, mx.float32)

        # Test correct scalar float parameter behavior
        norm = dist.Normal(loc=2.5, scale=1.5)
        self.assertEqual(norm._loc.item(), 2.5)
        self.assertEqual(norm._scale.item(), 1.5)
        
        # Test for correct array parameter behavior
        norm = dist.Normal(loc=mx.array(1.0), scale=mx.array(2.0))
        self.assertEqual(norm._loc.item(), 1.0)
        self.assertEqual(norm._scale.item(), 2.0)

    def test_parameter_validation(self):
        # Test protections for incorrect parameters
        dist.Normal(loc=0.0, scale=0.1)
        dist.Normal(loc=0.0, scale=1e-6)

        with self.assertRaises(ValueError):
            dist.Normal(loc=0.0, scale=0.0)
        
        with self.assertRaises(ValueError):
            dist.Normal(loc=0.0, scale=-1.0)
        
        with self.assertRaises(ValueError):
            dist.Normal(loc=0.0, scale=mx.array([-1.0, 1.0]))
            
        with self.assertRaises(ValueError):
            dist.Normal(loc=mx.array([1.0, 2.0]), scale=1.0)
        
        
if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
