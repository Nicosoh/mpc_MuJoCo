from .pendulum_dataset import PendulumDataset
# from .cartpole_dataset import CartPoleDataset
# from .mujoco_dataset import MuJoCoDataset
from .twodofarm_dataset import *
from .iiwa14_dataset import *

DATASET_REGISTRY = {
    "PendulumDataset": PendulumDataset,
    # "CartPoleDataset": CartPoleDataset,
    # "MuJoCoDataset": MuJoCoDataset,
    "TwoDofArmDataset": TwoDofArmDataset,
    "TwoDofArmDataset_eeTracker": TwoDofArmDataset_eeTracker,
    "iiwa14_eeTracker": iiwa14_eeTracker,
}