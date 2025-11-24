import numpy as np

collision_config = {
    "links": { # Frames from pinocchio model
        "link1": {"from": "base_link", "to": "j_1", "radius": 0.1,
        },
        "link2": {"from": "j_1", "to": "ee", "radius": 0.1,
        },
    },

    "obstacles": 
        {"obs1": {"from": np.array([1.1, 0.0, 0.5]), "to": np.array([1.1, 0.0, 0.8]), "radius": 0.1},
    },

    "collision_pairs": [("link2", "obs1"),
    ],
}