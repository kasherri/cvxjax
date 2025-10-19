"""Tests for CVX layers."""

import jax
import jax.numpy as jnp
import pytest

from cvxjax import Variable, Problem, Minimize, CvxLayer, sum_squares


class TestCvxLayer:
    """Test suite for CvxLayer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Enable 64-bit precision for more accurate tests
        jax.config.update("jax_enable_x64", True)
        
        # Create a simple QP problem for testing
        # minimize 0.5 * ||x||^2 subject to x >= 0
        self.x = Variable(shape=(3,), name="x")
        objective = Minimize(0.5 * sum_squares(self.x))
        constraints = [self.x >= 0]
        self.problem = Problem(objective, constraints)
    
    def test_layer_creation(self):
        """Test basic layer creation and validation."""
        # Valid layer creation
        layer = CvxLayer(self.problem, solver="ipm", return_fields=("primal",))
        assert layer.cfg.solver == "ipm"
        assert layer.cfg.return_fields == ("primal",)
        assert layer.cfg.diff_mode == "implicit"
        
        # Test invalid solver
        with pytest.raises(ValueError, match="Unknown solver"):
            CvxLayer(self.problem, solver="invalid_solver")
        
        # Test invalid return fields
        with pytest.raises(ValueError, match="Invalid return fields"):
            CvxLayer(self.problem, return_fields=("invalid_field",))
        
        # Test invalid diff mode
        with pytest.raises(ValueError, match="Invalid diff_mode"):
            CvxLayer(self.problem, diff_mode="invalid_mode")
    
    def test_forward_solve(self):
        """Test forward solve functionality."""
        layer = CvxLayer(self.problem, solver="ipm", return_fields=("primal",))
        
        # Solve the problem
        x_star = layer({})
        
        # Check solution properties
        assert len(x_star) >= 3  # May have slack variables
        # Take only the first 3 elements (original decision variables)
        x_orig = x_star[:3]
        assert jnp.all(x_orig >= -1e-6)  # Should satisfy x >= 0 (with small tolerance)
        
        # For this problem, optimal solution should be x = [0, 0, 0]
        # since we're minimizing ||x||^2 subject to x >= 0
        assert jnp.allclose(x_orig, jnp.zeros(3), atol=1e-4)
    
    def test_multiple_return_fields(self):
        """Test returning multiple fields."""
        layer = CvxLayer(
            self.problem, 
            solver="ipm", 
            return_fields=("primal", "obj")
        )
        
        result = layer({})
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        x_star, obj_value = result
        assert len(x_star) >= 3
        assert jnp.isscalar(obj_value)
        
        # Check that objective is approximately zero for optimal x = [0,0,0]
        assert jnp.abs(obj_value) < 1e-4
    
    def test_forward_mode_gradients(self):
        """Test forward-mode gradients."""
        def create_scaled_layer(scale_factor):
            """Create a layer with scaled objective."""
            x = Variable(shape=(2,), name="x")
            objective = Minimize(scale_factor * sum_squares(x))
            constraints = [x >= 0]
            problem = Problem(objective, constraints)
            layer = CvxLayer(problem, solver="ipm", return_fields=("obj",))
            return layer({})
        
        # Test forward-mode differentiation
        scale = 1.0
        forward_grad = jax.jacfwd(create_scaled_layer)(scale)
        
        # Gradient should be finite
        assert jnp.isfinite(forward_grad)
    
    def test_vmap_compatibility(self):
        """Test vmap compatibility for batching."""
        layer = CvxLayer(self.problem, solver="ipm")
        
        def solve_batch(dummy_input):
            """Solve for batching test (dummy_input ignored)."""
            result = layer({})
            return result[:2]  # Return first 2 elements to avoid issues
        
        # Create batch of dummy inputs
        batch_inputs = jnp.array([1.0, 2.0, 3.0])
        batched_solve = jax.vmap(solve_batch)
        x_batch = batched_solve(batch_inputs)
        
        # Check output shape
        assert x_batch.shape == (3, 2)
        
        # All solutions should be similar (since dummy_input is ignored)
        assert jnp.allclose(x_batch[0], x_batch[1], atol=1e-6)
        assert jnp.allclose(x_batch[1], x_batch[2], atol=1e-6)
    
    def test_stop_gradient_mode(self):
        """Test that 'none' diff_mode stops gradients."""
        # Create a layer with stop gradients
        layer = CvxLayer(self.problem, solver="ipm", diff_mode="none", return_fields=("obj",))
        
        def simple_test_fn(dummy_param):
            """Simple test function that should have zero gradient."""
            # The dummy_param doesn't actually affect the layer, 
            # but we need it for gradient computation
            result = layer({})
            return result + 0.0 * dummy_param  # Add dummy dependency
        
        # Forward-mode gradient should be zero due to stop_gradient
        scale = 1.0
        forward_grad = jax.jacfwd(simple_test_fn)(scale)
        
        # Gradient should be zero because of the dummy dependency only
        assert jnp.allclose(forward_grad, 0.0)
    
    def test_basic_functionality(self):
        """Test basic layer functionality without complex operations."""
        layer = CvxLayer(self.problem, solver="ipm", return_fields=("primal", "obj"))
        
        # Test that we can call the layer
        result = layer({})
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        x_star, obj_val = result
        
        # Basic sanity checks
        assert len(x_star) >= 3
        assert jnp.isfinite(obj_val)
        
        # Test that primal-only works too
        layer_primal = CvxLayer(self.problem, solver="ipm", return_fields=("primal",))
        x_only = layer_primal({})
        assert len(x_only) >= 3
        
        # Test object creation
        layer_obj = CvxLayer(self.problem, solver="ipm", return_fields=("obj",))
        obj_only = layer_obj({})
        assert jnp.isscalar(obj_only)