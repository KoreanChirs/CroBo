from omegaconf import OmegaConf
from PIL import Image
import os
import hydra
import torch, torchvision.transforms as T
import numpy as np

def load_pretrained_model(load_path, input_type=np.ndarray, *args, **kwargs):
    """
    Load the pretrained model based on the config corresponding to the load_path
    """

    config_path = os.path.join(
        "/mnt/workspace/custom_models/yaml/", load_path + ".yaml"
    )
    print("Loading config path: %s" % config_path)
    config = OmegaConf.load(config_path)
    model, embedding_dim, transforms, metadata = hydra.utils.call(config)
    model = model.eval()  # model loading API is unreliable, call eval to be double sure

    def final_transforms(transforms):
        if input_type == np.ndarray:
            return lambda input: transforms(Image.fromarray(input)).unsqueeze(0)
        else:
            return transforms

    return model, embedding_dim, final_transforms(transforms), metadata