"""Nuke-side UI and graph. No torch dependency is imported in Nuke."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import nuke

from .core import LEVELS, MODEL, frame_path, inference_size

_NODE_KNOBS = ("cache_path", "file_name", "range_mode", "first", "last", "max_dimension",
               "levels", "python_path", "sea_raft_root", "preprocess", "overwrite", "invert_v")


def _knob(node, name):
    return node[name].value()


def _cache_pattern(node):
    directory = Path(nuke.expandFilename(_knob(node, "cache_path"))).expanduser().resolve()
    file_name = _knob(node, "file_name")
    if Path(file_name).name != file_name or "####" not in file_name or not file_name.endswith(".exr"):
        raise ValueError("Filename must be a plain EXR name containing ####")
    return str(directory / file_name)


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
    sidecar = Path(pattern).parent / "smartvectors.json"
    if not sidecar.is_file():
        merge["disable"].setValue(True)
        _set_status(node, "Cache missing")
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
        _set_status(node, "Cache out of date")
        return
    if metadata.get("complete") and all(frame_path(pattern, frame).is_file() for frame in range(first, last + 1)):
        read["file"].setValue(Path(pattern.replace("####", "%04d")).as_posix())
        read["first"].setValue(first)
        read["last"].setValue(last)
        merge["disable"].setValue(False)
        _set_status(node, "Cached")
    else:
        merge["disable"].setValue(True)
        _set_status(node, "Cache missing")


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
                                "max_dimension", "levels", "preprocess", "invert_v"}:
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
        if "colorspace" in writer.knobs():
            options = writer["colorspace"].values()
            writer["colorspace"].setValue("raw" if "raw" in options else "linear")
        if "autocrop" in writer.knobs():
            writer["autocrop"].setValue(False)
        nuke.execute(writer, first, last)
    finally:
        nuke.delete(writer)


def _run_worker(node, job_path, python_path):
    try:
        from PySide6 import QtCore, QtWidgets
    except ImportError:
        from PySide2 import QtCore, QtWidgets
    dialog = QtWidgets.QProgressDialog("Starting SEA-RAFT", "Cancel", 0, 100)
    dialog.setWindowTitle("SEA-RAFT SmartVector")
    dialog.setMinimumDuration(0)
    process = QtCore.QProcess(dialog)
    environment = QtCore.QProcessEnvironment.systemEnvironment()
    root = str(Path(__file__).resolve().parent.parent)
    environment.insert("PYTHONPATH", root + os.pathsep + environment.value("PYTHONPATH"))
    process.setProcessEnvironment(environment)
    process.setProgram(python_path)
    process.setArguments(["-m", "smartvector.worker", str(job_path)])
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


def generate(node=None):
    node = node or nuke.thisNode()
    try:
        source = node.input(0)
        if source is None:
            raise ValueError("Connect an upstream image")
        first, last = _range(node, source)
        pattern = _cache_pattern(node)
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
        _activate(node, first, last)
    except Exception as exc:
        _set_status(node, f"Error: {exc}")
        nuke.message(str(exc))


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

    group.addKnob(nuke.Tab_Knob("sea_raft", "SEA-RAFT SmartVector"))
    status = nuke.String_Knob("status", "Status")
    status.setFlag(nuke.READ_ONLY)
    group.addKnob(status)
    group.addKnob(nuke.Enumeration_Knob("range_mode", "Frame Range", ["Input", "Project", "Custom"]))
    group.addKnob(nuke.Int_Knob("first", "First"))
    group.addKnob(nuke.Int_Knob("last", "Last"))
    group.addKnob(nuke.Int_Knob("max_dimension", "Inference Max Dimension (0=Full)"))
    group["max_dimension"].setValue(1280)
    group.addKnob(nuke.Enumeration_Knob("levels", "Temporal Levels", ["Full", "Medium", "Short"]))
    group.addKnob(nuke.File_Knob("cache_path", "Cache Directory"))
    script_path = Path(nuke.root().name())
    cache_base = script_path.parent if script_path.is_file() else Path.cwd()
    group["cache_path"].setValue(str(cache_base / "SMARTVECTORS" / group.name()))
    group.addKnob(nuke.String_Knob("file_name", "Filename"))
    group["file_name"].setValue("smartvector.####.exr")
    group.addKnob(nuke.PyScript_Knob("generate", "Generate SmartVectors",
                                     "import smartvector.nuke_node as sv; sv.generate(nuke.thisNode())"))
    group.addKnob(nuke.PyScript_Knob("refresh", "Refresh Cache",
                                     "import smartvector.nuke_node as sv; sv.refresh(nuke.thisNode())"))
    group.addKnob(nuke.Tab_Knob("advanced", "Advanced"))
    group.addKnob(nuke.File_Knob("python_path", "SEA-RAFT Python"))
    group["python_path"].setValue(sys.executable)
    group.addKnob(nuke.File_Knob("sea_raft_root", "SEA-RAFT Repository"))
    group.addKnob(nuke.Enumeration_Knob("preprocess", "Input Preprocessing", ["Clamp", "Normalize", "Raw"]))
    group.addKnob(nuke.Boolean_Knob("overwrite", "Overwrite Existing"))
    group.addKnob(nuke.Boolean_Knob("invert_v", "Invert V"))
    group["invert_v"].setFlag(nuke.INVISIBLE)
    group["first"].setValue(int(nuke.root()["first_frame"].value()))
    group["last"].setValue(int(nuke.root()["last_frame"].value()))
    group["knobChanged"].setValue("import smartvector.nuke_node as sv; sv.knob_changed()")
    _set_status(group, "Ready")
    return group
