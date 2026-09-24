import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import nuke
from smartvector import nuke_node

nuke.menu("Nodes").addCommand("SEA-RAFT/SEA-RAFT SmartVector", nuke_node.create,
                               icon="SmartVector.png")
nuke.menu("Nodes").addCommand("SEA-RAFT/Upgrade Existing Nodes", nuke_node.upgrade_existing)
nuke.addOnScriptLoad(nuke_node.upgrade_existing)
