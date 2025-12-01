from .pendulum_dataset import PendulumDataset
# from .cartpole_dataset import CartPoleDataset
# from .mujoco_dataset import MuJoCoDataset

DATASET_REGISTRY = {
    "PendulumDataset": PendulumDataset,
    # "CartPoleDataset": CartPoleDataset,
    # "MuJoCoDataset": MuJoCoDataset,
}