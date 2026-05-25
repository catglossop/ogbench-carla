from .crl import CRLAgent
from .dsrl import DSRLAgent
from .gcbc import GCBCAgent
from .gciql import GCIQLAgent
from .gcivl import GCIVLAgent
from .hiql import HIQLAgent
from .ogpo import OGPOAgent
from .qrl import QRLAgent
from .sac import SACAgent

agents = dict(
    crl=CRLAgent,
    dsrl=DSRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    ogpo=OGPOAgent,
    qrl=QRLAgent,
    sac=SACAgent,
)
