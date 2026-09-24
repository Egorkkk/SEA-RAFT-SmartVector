import sys
sys.path.append("core")

import os
import re
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import OpenEXR
import Imath

from config.parser import parse_args
from raft import RAFT


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def frame_number_and_padding(path):
    """
    Returns:
        frame_num (int or None)
        padding   (int, default 4)
    """
    basename = os.path.basename(path)
    matches = list(re.finditer(r"(\d+)", basename))

    if not matches:
        return None, 4

    m = matches[-1]
    num_str = m.group(1)
    return int(num_str), len(num_str)


def sorted_image_list(input_dir):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not files:
        raise RuntimeError(f"No input images found in: {input_dir}")

    def sort_key(path):
        num, _ = frame_number_and_padding(path)
        if num is None:
            return (0, path)
        return (1, num)

    files = sorted(files, key=sort_key)
    return files


def parse_levels(levels_str):
    levels = []
    for x in levels_str.split(","):
        x = x.strip()
        if not x:
            continue
        levels.append(int(x))
    levels = sorted(set(levels))
    return levels


def load_image(path, device):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img.to(device)


@torch.no_grad()
def calc_flow(args, model, image1, image2):
    """
    Calculates optical flow image1 -> image2
    Returns H x W x 2 float32 numpy array in pixel units
    for the resolution of image1/image2.
    """
    img1 = F.interpolate(
        image1,
        scale_factor=2 ** args.scale,
        mode="bilinear",
        align_corners=False
    )

    img2 = F.interpolate(
        image2,
        scale_factor=2 ** args.scale,
        mode="bilinear",
        align_corners=False
    )

    output = model(
        img1,
        img2,
        iters=args.iters,
        test_mode=True
    )

    flow = output["flow"][-1]

    flow = F.interpolate(
        flow,
        scale_factor=0.5 ** args.scale,
        mode="bilinear",
        align_corners=False
    ) * (0.5 ** args.scale)

    flow = flow[0].permute(1, 2, 0)
    return flow.cpu().numpy().astype(np.float32)


def resize_flow(flow, out_w, out_h):
    """
    Resize flow field to another resolution and scale vector magnitude.
    """
    in_h, in_w, _ = flow.shape

    if in_w == out_w and in_h == out_h:
        return flow

    scale_x = float(out_w) / float(in_w)
    scale_y = float(out_h) / float(in_h)

    u = flow[..., 0]
    v = flow[..., 1]

    u_resized = cv2.resize(u, (out_w, out_h), interpolation=cv2.INTER_LINEAR) * scale_x
    v_resized = cv2.resize(v, (out_w, out_h), interpolation=cv2.INTER_LINEAR) * scale_y

    out = np.stack([u_resized, v_resized], axis=-1).astype(np.float32)
    return out


def zeros_flow(h, w):
    return np.zeros((h, w), dtype=np.float32)


def smartvector_channel_names(level):
    tag = f"smartvector_f{level:02d}_v01"
    return {
        "p_u": f"{tag}.p_u",
        "p_v": f"{tag}.p_v",
        "n_u": f"{tag}.n_u",
        "n_v": f"{tag}.n_v",
    }


def make_zero_channel_dict(levels, h, w):
    channels = {}
    for level in levels:
        names = smartvector_channel_names(level)
        channels[names["p_u"]] = zeros_flow(h, w)
        channels[names["p_v"]] = zeros_flow(h, w)
        channels[names["n_u"]] = zeros_flow(h, w)
        channels[names["n_v"]] = zeros_flow(h, w)
    return channels


def write_multichannel_exr(path, channels_dict, width, height):
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    header = OpenEXR.Header(width, height)

    header["compression"] = Imath.Compression(
        Imath.Compression.ZIP_COMPRESSION
    )

    header["channels"] = {
        name: Imath.Channel(FLOAT)
        for name in channels_dict.keys()
    }

    out = OpenEXR.OutputFile(path, header)

    out.writePixels({
        name: arr.astype(np.float32).tobytes()
        for name, arr in channels_dict.items()
    })

    out.close()

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        required=True,
        help="SEA-RAFT config, e.g. config/eval/spring-M.json"
    )

    parser.add_argument(
        "--url",
        default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
        help="HuggingFace model name"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input directory with image sequence"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for SmartVector EXRs"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda or cpu"
    )

    parser.add_argument(
        "--levels",
        default="1,2,4,8,16,32,64",
        help="Comma-separated smartvector levels"
    )

    parser.add_argument(
        "--out-width",
        type=int,
        default=0,
        help="Optional output width. 0 = same as input sequence"
    )

    parser.add_argument(
        "--out-height",
        type=int,
        default=0,
        help="Optional output height. 0 = same as input sequence"
    )

    parser.add_argument(
        "--flip-v",
        action="store_true",
        help="Flip sign of V when writing. "
             "Useful if vertical direction appears inverted in Nuke."
    )

    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    args = parse_args(parser)

    levels = parse_levels(args.levels)
    if not levels:
        raise RuntimeError("No levels specified")

    device = torch.device(args.device)

    print("Loading SEA-RAFT...")
    model = RAFT.from_pretrained(args.url, args=args)
    model = model.to(device)
    model.eval()

    files = sorted_image_list(args.input)
    os.makedirs(args.output, exist_ok=True)

    # inspect first frame for dimensions and numbering
    first_img_cpu = cv2.imread(files[0], cv2.IMREAD_COLOR)
    if first_img_cpu is None:
        raise RuntimeError(f"Cannot read first frame: {files[0]}")

    in_h, in_w = first_img_cpu.shape[:2]

    out_w = args.out_width if args.out_width > 0 else in_w
    out_h = args.out_height if args.out_height > 0 else in_h

    first_num, pad = frame_number_and_padding(files[0])
    if first_num is None:
        first_num = 1

    print(f"Found {len(files)} frames")
    print(f"Input resolution:  {in_w} x {in_h}")
    print(f"Output resolution: {out_w} x {out_h}")
    print(f"Levels: {levels}")
    print(f"First frame number (for naming): {first_num}")
    print()

    # optional tiny CPU-side image cache
    image_cache = {}

    def get_image(idx):
        path = files[idx]
        if path not in image_cache:
            image_cache[path] = load_image(path, device)
            # keep cache from growing too much
            if len(image_cache) > 20:
                # pop oldest inserted item
                oldest_key = next(iter(image_cache.keys()))
                del image_cache[oldest_key]
        return image_cache[path]

    num_frames = len(files)

    for i in range(num_frames):
        curr_path = files[i]
        curr_num, curr_pad = frame_number_and_padding(curr_path)
        if curr_num is None:
            curr_num = first_num + i
            curr_pad = pad

        if args.verbose:
            print(f"Frame index {i+1}/{num_frames}  file={curr_path}")

        channels = make_zero_channel_dict(levels, out_h, out_w)

        # Sparse smartvector logic:
        # level L exists only when (i % L == 0), with i relative to first frame.
        for level in levels:
            if (i % level) != 0:
                continue

            names = smartvector_channel_names(level)

            prev_idx = i - level
            next_idx = i + level

            # backward / p
            if prev_idx >= 0:
                curr_img = get_image(i)
                prev_img = get_image(prev_idx)

                backward = calc_flow(args, model, curr_img, prev_img)
                backward = resize_flow(backward, out_w, out_h)

                channels[names["p_u"]] = backward[..., 0].astype(np.float32)
                channels[names["p_v"]] = (
                    -backward[..., 1] if args.flip_v else backward[..., 1]
                ).astype(np.float32)

                if args.verbose:
                    pu = channels[names["p_u"]]
                    pv = channels[names["p_v"]]
                    print(
                        f"  f{level:02d} p: idx {i} -> {prev_idx}   "
                        f"u[{pu.min():.3f},{pu.max():.3f}] "
                        f"v[{pv.min():.3f},{pv.max():.3f}]"
                    )

            # forward / n
            if next_idx < num_frames:
                curr_img = get_image(i)
                next_img = get_image(next_idx)

                forward = calc_flow(args, model, curr_img, next_img)
                forward = resize_flow(forward, out_w, out_h)

                channels[names["n_u"]] = forward[..., 0].astype(np.float32)
                channels[names["n_v"]] = (
                    -forward[..., 1] if args.flip_v else forward[..., 1]
                ).astype(np.float32)

                if args.verbose:
                    nu = channels[names["n_u"]]
                    nv = channels[names["n_v"]]
                    print(
                        f"  f{level:02d} n: idx {i} -> {next_idx}   "
                        f"u[{nu.min():.3f},{nu.max():.3f}] "
                        f"v[{nv.min():.3f},{nv.max():.3f}]"
                    )

        out_name = f"smartvectors.{curr_num:0{curr_pad}d}.exr"
        out_path = os.path.join(args.output, out_name)

        write_multichannel_exr(out_path, channels, out_w, out_h)
        print(f"written: {out_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
