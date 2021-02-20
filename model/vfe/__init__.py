from .PillarVFE import PillarVFE
from .GCNVFE import GCNVFE
from .PillarVFE_CYL import PillarVFE as PillarVFE_CYL
from pcdet.models.backbones_3d.vfe.mean_vfe import MeanVFE
__all__ = {
    'PillarVFE': PillarVFE,
    'GCNVFE': GCNVFE,
    'PillarVFE_CYL': PillarVFE_CYL,
    'MeanVFE': MeanVFE,
}
