from .pendulum_model import PendulumModel
# from .baseline_network import BaselineNetwork
# from .mlp_128 import MLP128

MODEL_REGISTRY = {
    "PendulumModel": PendulumModel,
    # "BaselineNetwork": BaselineNetwork,
    # "MLP128": MLP128,
}