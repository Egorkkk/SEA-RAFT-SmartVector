"""Run with Nuke -t to check graph creation and serialization."""
import sys
import json
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
assert "exportWrite" in group.knobs()
assert "generateRender" in group.knobs()
assert "cache_path" not in group.knobs()
config_path = root / "smartvector" / "install_config.json"
if config_path.is_file():
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert group["python_path"].value() == config["python_path"]
    assert group["sea_raft_root"].value() == config["sea_raft_root"]
read, merge = nuke_node._graph(group)
assert read is not None and merge is not None
assert merge["disable"].value()
read["file"].setValue((root / "old_script" / "smartvectors.0001.exr").as_posix())
merge["disable"].setValue(False)
channels = set(group.channels())
assert "smartvector_f01_v01.n_u" in channels, sorted(channels)
assert "smartvector_f64_v01.p_v" in channels, sorted(channels)
assert "rgba.red" in channels, sorted(channels)
writes_before = set(nuke.allNodes("Write"))
group["exportWrite"].execute()
new_writes = set(nuke.allNodes("Write")) - writes_before
assert len(new_writes) == 1, f"Export Write button created {len(new_writes)} nodes"
write = new_writes.pop()
assert write.input(0) is group
assert write["channels"].value() == "all"
assert "verify_write_ready" in write["beforeRender"].value()
write["file"].setValue((root / ".runtime" / "smoke_export.####.exr").as_posix())
assert nuke_node._work_pattern(group, write).endswith("smartvector.####.exr")
try:
    nuke_node.verify_write_ready(write)
except RuntimeError:
    pass
else:
    raise AssertionError("Unprepared Write render should be rejected")
legacy = nuke.nodes.Group(name="Legacy_SEA_RAFT", inputs=[source])
legacy.begin()
try:
    legacy_input = nuke.nodes.Input()
    legacy_read = nuke.nodes.Read(name="SV_CacheRead")
    legacy_merge = nuke.nodes.Merge2(name="SV_Merge", inputs=[legacy_input, legacy_read])
    legacy_merge["disable"].setValue(True)
    nuke.nodes.Output(inputs=[legacy_merge])
finally:
    legacy.end()
legacy.addKnob(nuke.Tab_Knob("sea_raft", "SEA-RAFT SmartVector"))
legacy.addKnob(nuke.String_Knob("status", "Status"))
legacy.addKnob(nuke.Enumeration_Knob("range_mode", "Frame Range", ["Input", "Project", "Custom"]))
legacy.addKnob(nuke.File_Knob("cache_path", "Cache Directory"))
legacy["cache_path"].setValue((root / ".runtime" / "old_cache").as_posix())
legacy.addKnob(nuke.String_Knob("file_name", "Filename"))
legacy["file_name"].setValue("smartvector.####.exr")
assert nuke_node.upgrade_node(legacy)
assert "exportWrite" in legacy.knobs() and "cache_path" not in legacy.knobs()
assert "generateRender" in legacy.knobs()
assert "expandFilename" not in legacy["status"].value()
group.removeKnob(group["generateRender"])
assert nuke_node.upgrade_node(group)
assert "generateRender" in group.knobs()
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
