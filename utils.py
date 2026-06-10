import os, random, torch, numpy as np

def set_seed(seed: int = 42):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_sampler_generator(seed: int = 42):
    """Seeded generator for WeightedRandomSampler."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
