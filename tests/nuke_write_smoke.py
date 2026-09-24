"""Run with Nuke -t after the Windows installer; exercises Write render end to end."""
import sys
from pathlib import Path

import nuke

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from smartvector import nuke_node

nuke.scriptNew()
nuke.root()["first_frame"].setValue(1)
nuke.root()["last_frame"].setValue(2)
small = nuke.addFormat("128 128 1 sea_raft_smoke")
source = nuke.nodes.Constant()
source["format"].setValue(small)
source.setSelected(True)
group = nuke_node.create()
group["range_mode"].setValue("Custom")
group["first"].setValue(1)
group["last"].setValue(2)
group["max_dimension"].setValue(128)
group["work_id"].setValue("smoke")
group["overwrite"].setValue(True)
writes_before = set(nuke.allNodes("Write"))
group["exportWrite"].execute()
new_writes = set(nuke.allNodes("Write")) - writes_before
assert len(new_writes) == 1
write = new_writes.pop()
assert write["datatype"].value() == "32 bit float"
test_dir = root / ".runtime" / "nuke_write_smoke"
test_dir.mkdir(parents=True, exist_ok=True)
write["file"].setValue((test_dir / "final.####.exr").as_posix())
group["generateRender"].execute()
assert (test_dir / "final.0001.exr").is_file()
assert (test_dir / "final.0002.exr").is_file()
final = nuke.nodes.Read(file=(test_dir / "final.0001.exr").as_posix())
channels = set(final.channels())
assert "rgba.red" in channels
assert "smartvector_f01_v01.n_u" in channels
assert "smartvector_f64_v01.p_v" in channels
assert final.width() == 128 and final.height() == 128
print("NUKE_WRITE_SMOKE_OK")
