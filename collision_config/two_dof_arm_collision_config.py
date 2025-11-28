import numpy as np

collision_config = {
    "links": { # Frames from pinocchio model # Print frames with: print([b for b in self.model.frames[:]])
        "link1": {"from": "universe", "to": "j_1", "radius": 0.1,
        },
        "link2": {"from": "j_1", "to": "attachment_site", "radius": 0.1,
        },
    },

    "obstacles": 
        {"obs1": {"from": np.array([0.8, 0.0, 0.25]), "to": np.array([0.8, 0.0, 0.55]), "radius": 0.05},
        #  "obs2": {"from": np.array([0.0, 0.5, 0.65]), "to": np.array([0.0, 0.5, 0.95]), "radius": 0.05}
         
    },

    "collision_pairs": [("link2", "obs1"),
                        # ("link1", "obs1")
    ],
}