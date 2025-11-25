import numpy as np

collision_config = {
    "links": { # Frames from pinocchio model
        "link1": {"from": "base_link", "to": "j_1", "radius": 0.1,
        },
        "link2": {"from": "j_1", "to": "ee", "radius": 0.1,
        },
    },

    "obstacles": 
        {"obs1": {"from": np.array([0.0, 0.7, 0.35]), "to": np.array([0.0, 0.7, 0.65]), "radius": 0.05},
         "obs2": {"from": np.array([0.0, 0.5, 0.65]), "to": np.array([0.0, 0.5, 0.95]), "radius": 0.05}
         
    },

    "collision_pairs": [("link2", "obs1"),
    ],
}