import os
import torch
import l4casadi as l4c

from configparser import ConfigParser

from neural_network.models import MODEL_REGISTRY

def export_torch_model(config, worker_id):
    try:
        model_name = config["NN"]["model_name"]
        checkpoint_path = config["NN"]["checkpoint_path"]
    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")

    train_config_dir = os.path.dirname(checkpoint_path)
    train_config_path = os.path.join(train_config_dir, "train_config.ini")
    
    train_config = ConfigParser()
    train_config.read(train_config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    NNmodel = MODEL_REGISTRY[model_name](train_config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    NNmodel.load_state_dict(state_dict)
    NNmodel.eval()

    # Wrap for CasADi
    name = "l4casadi_f" + str(worker_id)

    l4c_model = l4c.L4CasADi(NNmodel, device=device, name=name)

    return l4c_model