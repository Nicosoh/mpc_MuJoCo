from neural_network.models import MODEL_REGISTRY
from torchinfo import summary

ModelClass = MODEL_REGISTRY["PendulumModel"]
model = ModelClass(None).to("cpu")
summary(model, input_size=(1, 2))