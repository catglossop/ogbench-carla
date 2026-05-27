from .crl import CRLAgent
from .dsrl import DSRLAgent
from .gcbc import GCBCAgent
from .gciql import GCIQLAgent
from .gcivl import GCIVLAgent
from .hiql import HIQLAgent
from .qrl import QRLAgent
from .sac import SACAgent
from .expo import EXPOAgent

agents = dict(
    crl=CRLAgent,
    dsrl=DSRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,g
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    expo=EXPOAgent,
)
