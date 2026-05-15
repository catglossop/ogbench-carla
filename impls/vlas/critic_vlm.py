"""
VLM Critic network - make into a jax nnx module so we can correctly jit
"""

from flax import nnx
import ml_collections

class VLMCritic(nnx.Module):
    """
    TODO: Implement this: 
    1. Should be able to take the image observation 
    
    """
    
    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()
        self.config = config
        
        # Load the model 
    
    def __call__(self, observation: Observation, action: Action) -> jnp.ndarray:
        return self.forward(observation, action)
    
    def forward(self, observation: Observation, action: Action) -> jnp.ndarray:
        return self.model(observation, action)
    
    def sample_values(self, observation: Observation, action: Action) -> jnp.ndarray:
        return self.model.sample_values(observation, action)
    
    def compute_loss(self, observation: Observation, action: Action, value: jnp.ndarray) -> jnp.ndarray:
        pass