class VC1Enc(nn.Module):
    """
    Wrap vc_models loaded model so that forward(x) -> (B, D)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)

        # Common patterns: tensor / (tensor, ...) / dict
        if torch.is_tensor(out):
            emb = out
        elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
            emb = out[0]
        elif isinstance(out, dict):
            # try common keys
            for k in ["emb", "embedding", "feat", "features", "repr", "representation", "cls"]:
                if k in out and torch.is_tensor(out[k]):
                    emb = out[k]
                    break
            else:
                # fallback: first tensor value
                tensor_vals = [v for v in out.values() if torch.is_tensor(v)]
                if not tensor_vals:
                    raise RuntimeError(f"VC1 model output dict has no tensor values: keys={list(out.keys())}")
                emb = tensor_vals[0]
        else:
            raise RuntimeError(f"Unsupported VC1 model output type: {type(out)}")

        # If model returns token map (B, N, D), take CLS token by convention
        if emb.dim() == 3:
            emb = emb[:, 0, :]  # (B, D)

        if emb.dim() != 2:
            raise RuntimeError(f"Expected embedding shape (B,D), got {tuple(emb.shape)}")

        return emb