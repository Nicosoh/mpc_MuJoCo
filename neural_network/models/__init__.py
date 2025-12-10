from .pendulum_model import *
# from .baseline_network import BaselineNetwork
# from .mlp_128 import MLP128

MODEL_REGISTRY = {
    "PendulumModel": PendulumModel,
    # "PendulumModelTruncated": PendulumModelTruncated,
    # "BaselineNetwork": BaselineNetwork,
    # "MLP128": MLP128,
}