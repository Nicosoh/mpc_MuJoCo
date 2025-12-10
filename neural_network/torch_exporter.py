import torch
import l4casadi as l4c

from neural_network.models import MODEL_REGISTRY

def export_torch_model(config):
    model_name = config["NN"]["model_name"]
    checkpoint_path = config["NN"]["checkpoint_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NNmodel = MODEL_REGISTRY[model_name]().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    NNmodel.load_state_dict(state_dict)
    NNmodel.eval()

    # Wrap for CasADi
    l4c_model = l4c.L4CasADi(NNmodel, device=device)

    return l4c_model