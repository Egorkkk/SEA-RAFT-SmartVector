"""Nuke-side UI and graph. No torch dependency is imported in Nuke."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import nuke

from .core import LEVELS, MODEL, frame_path, inference_size

_OLD_KNOBS = ("sea_raft", "status", "range_mode", "first", "last", "max_dimension", "levels",
              "cache_path", "file_name", "generate", "refresh", "advanced", "python_path",
              "sea_raft_root", "preprocess", "overwrite", "invert_v")


def _knob(node, name):
    return node[name].value()


def _install_config():
    path = Path(__file__).with_name("install_config.json")
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _cache_pattern(node):
    if "work_pattern" in node.knobs() and _knob(node, "work_pattern"):
        return _knob(node, "work_pattern")
    # Compatibility with Groups saved before the Export Write workflow.
    if "cache_path" in node.knobs():
        directory = Path(node["cache_path"].evaluate()).expanduser().resolve()
        file_name = _knob(node, "file_name")
        if Path(file_name).name != file_name or "####" not in file_name or not file_name.endswith(".exr"):
            raise ValueError("Filename must be a plain EXR name containing ####")
        return str(directory / file_name)
    return None


def _work_pattern(node, write):
    output = write["file"].evaluate()
    if not output or not output.lower().endswith(".exr"):
        raise ValueError("Set an EXR sequence path on the Export Write node first")
    if "work_directory" in node.knobs() and _knob(node, "work_directory"):
        base = Path(node["work_directory"].evaluate()).expanduser().resolve()
    else:
        base = Path(output).expanduser().resolve().parent / ".sea_raft_work"
    return str(base / _knob(node, "work_id") / "smartvector.####.exr")


def _range(node, source):
    mode = _knob(node, "range_mode")
    if mode == "Custom":
        first, last = int(_knob(node, "first")), int(_knob(node, "last"))
    elif mode == "Input" and source.Class() == "Read":
        first, last = int(source["first"].value()), int(source["last"].value())
    else:
        first, last = int(nuke.root()["first_frame"].value()), int(nuke.root()["last_frame"].value())
    if first > last:
        raise ValueError("First frame exceeds last frame")
    return first, last


def _levels(node):
    preset = _knob(node, "levels")
    return list(LEVELS if preset == "Full" else LEVELS[:6] if preset == "Medium" else LEVELS[:4])


def _graph(node):
    node.begin()
    try:
        return nuke.toNode("SV_CacheRead"), nuke.toNode("SV_Merge")
    finally:
        node.end()


def _set_status(node, message):
    node["status"].setValue(message)


def _activate(node, first, last):
    pattern = _cache_pattern(node)
    read, merge = _graph(node)
    if not pattern:
        merge["disable"].setValue(True)
        _set_status(node, "Export Write to generate vectors")
        return
    sidecar = Path(pattern).parent / "smartvectors.json"
    if not sidecar.is_file():
        merge["disable"].setValue(True)
        _set_status(node, "Needs preparation")
        return
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    settings = metadata.get("settings", {})
    source = node.input(0)
    expected = {"first": first, "last": last, "width": source.width(), "height": source.height(),
                "levels": _levels(node), "max_dimension": int(_knob(node, "max_dimension")),
                "inference_resolution": list(inference_size(source.width(), source.height(),
                                                             int(_knob(node, "max_dimension")))),
                "preprocess": _knob(node, "preprocess").lower(),
                "invert_v": bool(_knob(node, "invert_v")), "model": MODEL}
    if any(settings.get(key) != value for key, value in expected.items()):
        merge["disable"].setValue(True)
        _set_status(node, "Settings changed; regenerate")
        return
    if metadata.get("complete") and all(frame_path(pattern, frame).is_file() for frame in range(first, last + 1)):
        read["file"].setValue(Path(pattern.replace("####", "%04d")).as_posix())
        read["first"].setValue(first)
        read["last"].setValue(last)
        merge["disable"].setValue(False)
        _set_status(node, "Prepared")
    else:
        merge["disable"].setValue(True)
        _set_status(node, "Needs preparation")


def refresh(node=None):
    node = node or nuke.thisNode()
    source = node.input(0)
    if source is None:
        _set_status(node, "Connect an input")
        return
    first, last = _range(node, source)
    _activate(node, first, last)


def knob_changed():
    knob = nuke.thisKnob()
    if knob and knob.name() in {"cache_path", "file_name", "range_mode", "first", "last",
                                "max_dimension", "levels", "preprocess", "invert_v", "work_directory"}:
        try:
            refresh(nuke.thisNode())
        except Exception as exc:
            _set_status(nuke.thisNode(), f"Cache check failed: {exc}")


def _bake(node, source, first, last, pattern):
    """Render the actual connected upstream graph at its current format."""
    Path(pattern).parent.mkdir(parents=True, exist_ok=True)
    writer = nuke.nodes.Write(inputs=[source])
    try:
        writer["file"].setValue(Path(pattern.replace("####", "%04d")).as_posix())
        writer["file_type"].setValue("exr")
        writer["channels"].setValue("rgb")
        if "datatype" in writer.knobs():
            writer["datatype"].setValue("32 bit float")
        if "colorspace" in writer.knobs():
            options = writer["colorspace"].values()
            writer["colorspace"].setValue("raw" if "raw" in options else "linear")
        if "autocrop" in writer.knobs():
            writer["autocrop"].setValue(False)
        nuke.execute(writer, first, last)
    finally:
        nuke.delete(writer)


def _run_worker(node, job_path, python_path):
    root = str(Path(__file__).resolve().parent.parent)
    environment_values = os.environ.copy()
    environment_values["PYTHONPATH"] = root
    environment_values.pop("PYTHONHOME", None)
    for key in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        environment_values.pop(key, None)
    installed = _install_config()
    if installed.get("hf_home"):
        environment_values["HF_HOME"] = installed["hf_home"]
    if installed.get("torch_home"):
        environment_values["TORCH_HOME"] = installed["torch_home"]
    command = [python_path, "-m", "smartvector.worker", str(job_path)]
    if not nuke.GUI:
        result = subprocess.run(command, env=environment_values, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"SEA-RAFT worker failed (exit {result.returncode}):\n" + result.stdout[-4000:])
        return
    try:
        from PySide6 import QtCore, QtWidgets
    except ImportError:
        from PySide2 import QtCore, QtWidgets
    dialog = QtWidgets.QProgressDialog("Starting SEA-RAFT", "Cancel", 0, 100)
    dialog.setWindowTitle("SEA-RAFT SmartVector")
    dialog.setMinimumDuration(0)
    process = QtCore.QProcess(dialog)
    environment = QtCore.QProcessEnvironment()
    for key, value in environment_values.items():
        environment.insert(key, value)
    process.setProcessEnvironment(environment)
    process.setProgram(command[0])
    process.setArguments(command[1:])
    process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
    process.start()
    if not process.waitForStarted(10000):
        raise RuntimeError(f"Could not start Python: {python_path}")
    output = []
    buffer = ""
    try:
        while process.state() != QtCore.QProcess.NotRunning or process.bytesAvailable():
            process.waitForReadyRead(100)
            buffer += bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                output.append(line)
                if len(output) > 80:
                    output.pop(0)
                if line.startswith("FRAME "):
                    parts = line.split()
                    completed, total = map(int, parts[2].split("/"))
                    dialog.setValue(round(100 * completed / total))
                    dialog.setLabelText(f"Frame {parts[1]}: {completed}/{total}")
                elif line.startswith("PROGRESS "):
                    parts = line.split()
                    dialog.setLabelText(f"Frame {parts[1]} / {parts[2]}, level {parts[3]}")
            QtWidgets.QApplication.processEvents()
            if dialog.wasCanceled():
                process.terminate()
                if not process.waitForFinished(3000):
                    process.kill()
                    process.waitForFinished(3000)
                raise RuntimeError("Generation cancelled; finished frames remain on disk")
        if process.exitCode() != 0:
            raise RuntimeError("SEA-RAFT worker failed:\n" + "\n".join(output[-15:]))
    finally:
        dialog.close()


def generate(node=None, pattern=None):
    node = node or nuke.thisNode()
    source = node.input(0)
    if source is None:
        raise ValueError("Connect an upstream image")
    first, last = _range(node, source)
    pattern = pattern or _cache_pattern(node)
    if not pattern:
        raise ValueError("Create Export Write and set its EXR path first")
    width, height = source.width(), source.height()
    if width <= 0 or height <= 0:
        raise ValueError("Upstream image has no format")
    _graph(node)[1]["disable"].setValue(True)
    _set_status(node, "Baking input")
    input_pattern = str(Path(pattern).parent / ".source" / "source.####.exr")
    _bake(node, source, first, last, input_pattern)
    job = {"input": input_pattern, "output": pattern, "first": first, "last": last,
           "width": width, "height": height, "levels": _levels(node),
           "max_dimension": int(_knob(node, "max_dimension")),
           "device": "cuda", "preprocess": _knob(node, "preprocess").lower(),
           "invert_v": bool(_knob(node, "invert_v")), "overwrite": bool(_knob(node, "overwrite")),
           "model": MODEL, "sea_raft_root": _knob(node, "sea_raft_root")}
    job_path = Path(pattern).parent / "job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    _set_status(node, "Generating")
    _run_worker(node, job_path, _knob(node, "python_path"))
    if "work_pattern" in node.knobs():
        node["work_pattern"].setValue(pattern)
    _activate(node, first, last)


def export_write(node=None):
    """Match SmartVector's Export Write: create a connected EXR Write node."""
    node = node or nuke.thisNode()
    if node.input(0) is None:
        raise ValueError("Connect an upstream image first")
    write = nuke.nodes.Write(inputs=[node])
    write["file_type"].setValue("exr")
    write["channels"].setValue("all")
    if "datatype" in write.knobs():
        write["datatype"].setValue("32 bit float")
    if "colorspace" in write.knobs():
        options = write["colorspace"].values()
        write["colorspace"].setValue("raw" if "raw" in options else "linear")
    write.addKnob(nuke.Tab_Knob("sea_raft_write", "SEA-RAFT"))
    write.addKnob(nuke.PyScript_Knob("generate_render", "Generate + Render",
                                    "import smartvector.nuke_node as sv; sv.render_write(nuke.thisNode())"))
    write["beforeRender"].setValue(
        "import smartvector.nuke_node as sv; sv.verify_write_ready(nuke.thisNode())")
    write.setXYpos(node.xpos(), node.ypos() + 120)
    _set_status(node, "Set EXR path on Write, then Generate + Render")
    return write


def verify_write_ready(write=None):
    write = write or nuke.thisNode()
    node = write.input(0)
    if node is None or "sea_raft" not in node.knobs():
        raise RuntimeError("Export Write must be connected to a SEA-RAFT SmartVector node")
    if _cache_pattern(node) != _work_pattern(node, write):
        raise RuntimeError("Use Generate + Render on the Write node to prepare this output path")
    first, last = _range(node, node.input(0))
    _activate(node, first, last)
    if _knob(node, "status") != "Prepared":
        raise RuntimeError("SmartVectors are not ready. Use Generate + Render on the Write node")


def render_write(write=None):
    write = write or nuke.thisNode()
    node = write.input(0)
    if node is None or "sea_raft" not in node.knobs():
        raise ValueError("Export Write must be connected to a SEA-RAFT SmartVector node")
    pattern = _work_pattern(node, write)
    try:
        generate(node, pattern)
        first, last = _range(node, node.input(0))
        nuke.execute(write, first, last)
        _set_status(node, "Rendered through Write")
    except Exception as exc:
        _set_status(node, f"Error: {exc}")
        raise


def _gpu_name():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "CUDA GPU"


def _add_ui(group, saved=None):
    saved = saved or {}
    installed = _install_config()
    group.addKnob(nuke.Tab_Knob("sea_raft", "SEA-RAFT SmartVector"))
    group.addKnob(nuke.Text_Knob("gpu_info", "", "Local GPU: " + _gpu_name()))
    gpu = nuke.Boolean_Knob("use_gpu", "Use GPU (required)")
    group.addKnob(gpu)
    gpu.setValue(True)
    gpu.setEnabled(False)
    group.addKnob(nuke.Text_Knob("settings_heading", "Inference", ""))
    group.addKnob(nuke.Int_Knob("max_dimension", "Max Dimension (0=Full)"))
    group["max_dimension"].setValue(saved.get("max_dimension", 1280))
    group.addKnob(nuke.Enumeration_Knob("levels", "Temporal Levels", ["Full", "Medium", "Short"]))
    group["levels"].setValue(saved.get("levels", "Full"))
    group.addKnob(nuke.Text_Knob("export_heading", "Export", ""))
    group.addKnob(nuke.PyScript_Knob("exportWrite", "Export Write",
                                     "import smartvector.nuke_node as sv; sv.export_write(nuke.thisNode())"))
    status = nuke.String_Knob("status", "Status")
    status.setFlag(nuke.READ_ONLY)
    group.addKnob(status)
    group.addKnob(nuke.Tab_Knob("advanced", "Advanced", nuke.TABBEGINCLOSEDGROUP))
    group.addKnob(nuke.Enumeration_Knob("range_mode", "Frame Range", ["Input", "Project", "Custom"]))
    group["range_mode"].setValue(saved.get("range_mode", "Input"))
    group.addKnob(nuke.Int_Knob("first", "First"))
    group["first"].setValue(saved.get("first", int(nuke.root()["first_frame"].value())))
    group.addKnob(nuke.Int_Knob("last", "Last"))
    group["last"].setValue(saved.get("last", int(nuke.root()["last_frame"].value())))
    group.addKnob(nuke.File_Knob("work_directory", "Working Directory (optional)"))
    group["work_directory"].setValue(saved.get("work_directory", ""))
    group.addKnob(nuke.Enumeration_Knob("preprocess", "Input Preprocessing", ["Clamp", "Normalize", "Raw"]))
    group["preprocess"].setValue(saved.get("preprocess", "Clamp"))
    group.addKnob(nuke.Boolean_Knob("overwrite", "Overwrite Existing"))
    group["overwrite"].setValue(saved.get("overwrite", False))
    group.addKnob(nuke.PyScript_Knob("refresh", "Refresh Status",
                                     "import smartvector.nuke_node as sv; sv.refresh(nuke.thisNode())"))
    group.addKnob(nuke.File_Knob("python_path", "SEA-RAFT Python"))
    group["python_path"].setValue(saved.get("python_path", installed.get("python_path", sys.executable)))
    group.addKnob(nuke.File_Knob("sea_raft_root", "SEA-RAFT Repository"))
    group["sea_raft_root"].setValue(saved.get("sea_raft_root", installed.get("sea_raft_root", "")))
    group.addKnob(nuke.Boolean_Knob("invert_v", "Invert V"))
    group["invert_v"].setFlag(nuke.INVISIBLE)
    group["invert_v"].setValue(saved.get("invert_v", False))
    group.addKnob(nuke.String_Knob("work_id", "Work ID"))
    group["work_id"].setValue(saved.get("work_id", uuid.uuid4().hex[:12]))
    group["work_id"].setFlag(nuke.INVISIBLE)
    group.addKnob(nuke.String_Knob("work_pattern", "Work Pattern"))
    group["work_pattern"].setValue(saved.get("work_pattern", ""))
    group["work_pattern"].setFlag(nuke.INVISIBLE)
    group.addKnob(nuke.Tab_Knob("advanced_end", "", nuke.TABENDGROUP))
    group["knobChanged"].setValue("import smartvector.nuke_node as sv; sv.knob_changed()")
    _set_status(group, "Export Write to generate vectors")


def create():
    selected = nuke.selectedNodes()
    upstream = selected[0] if len(selected) == 1 else None
    group = nuke.nodes.Group(name="SEA_RAFT_SmartVector")
    if upstream is not None:
        group.setInput(0, upstream)
    group.begin()
    try:
        source = nuke.nodes.Input(name="SV_Input")
        read = nuke.nodes.Read(name="SV_CacheRead")
        merge = nuke.nodes.Merge2(name="SV_Merge", inputs=[source, read])
        merge["operation"].setValue("plus")
        if "channels" in merge.knobs():
            merge["channels"].setValue("all")
        if "also_merge" in merge.knobs():
            merge["also_merge"].setValue("all")
        merge["disable"].setValue(True)
        nuke.nodes.Output(inputs=[merge])
    finally:
        group.end()
    _add_ui(group)
    return group


def upgrade_node(node):
    """Update Groups saved with the previous cache-oriented controls in place."""
    if node.Class() != "Group" or "sea_raft" not in node.knobs() or "exportWrite" in node.knobs():
        return False
    old_pattern = _cache_pattern(node)
    saved = {name: _knob(node, name) for name in ("range_mode", "first", "last", "max_dimension",
                                                 "levels", "python_path", "sea_raft_root", "preprocess",
                                                 "overwrite", "invert_v") if name in node.knobs()}
    saved["work_pattern"] = old_pattern
    for name in reversed(_OLD_KNOBS):
        if name in node.knobs():
            node.removeKnob(node[name])
    _add_ui(node, saved)
    refresh(node)
    return True


def upgrade_existing():
    changed = 0
    for node in nuke.allNodes("Group"):
        changed += bool(upgrade_node(node))
    return changed
