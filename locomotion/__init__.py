"""Deep learning for human locomotion analysis in lower-limb exoskeletons.

Reference implementation of:

    O. Coser, C. Tamantini, M. Tortora, L. Furia, R. Sicilia, L. Zollo, P. Soda,
    "Deep learning for human locomotion analysis in lower-limb exoskeletons:
    a comparative study", Frontiers in Computer Science, 2025.
    https://doi.org/10.3389/fcomp.2025.1597143
"""

from .config import ExperimentConfig

__version__ = "1.0.0"
__all__ = ["ExperimentConfig", "__version__"]
