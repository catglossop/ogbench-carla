from .crl import CRLAgent
from .dsrl import DSRLAgent
from .gcbc import GCBCAgent
from .gciql import GCIQLAgent
from .gcivl import GCIVLAgent
from .hiql import HIQLAgent
from .qrl import QRLAgent
from .sac import SACAgent
from .sac_residual import SACResidualAgent
from .best_of_n import BestOfNAgent

agents = dict(
    crl=CRLAgent,
    dsrl=DSRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    sac_residual=SACResidualAgent,
    best_of_n=BestOfNAgent,
)
