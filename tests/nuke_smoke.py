"""Run with Nuke -t to check graph creation and serialization."""
import sys
from pathlib import Path

import nuke

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from smartvector import nuke_node

nuke.scriptNew()
source = nuke.nodes.Constant()
source.setSelected(True)
group = nuke_node.create()
assert group.input(0) is source
assert group.Class() == "Group"
assert group["max_dimension"].value() == 1280
read, merge = nuke_node._graph(group)
assert read is not None and merge is not None
assert merge["disable"].value()
read["file"].setValue((root / "old_script" / "smartvectors.0001.exr").as_posix())
merge["disable"].setValue(False)
channels = set(group.channels())
assert "smartvector_f01_v01.n_u" in channels, sorted(channels)
assert "smartvector_f64_v01.p_v" in channels, sorted(channels)
assert "rgba.red" in channels, sorted(channels)
input_exr = root / "tests" / "smoke_input.0001.exr"
nuke_node._bake(group, source, 1, 1, str(root / "tests" / "smoke_input.####.exr"))
assert input_exr.is_file()
baked = nuke.nodes.Read(file=input_exr.as_posix())
assert "rgba.red" in baked.channels()
nuke.delete(baked)
input_exr.unlink()
output = root / "tests" / "smoke_output.nk"
nuke.scriptSaveAs(str(output), overwrite=1)
nuke.scriptOpen(str(output))
assert nuke.toNode(group.name()) is not None
output.unlink()
print("NUKE_SMOKE_OK")
