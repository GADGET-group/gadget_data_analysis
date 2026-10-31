#!/usr/bin/env python3
"""Flag candidate multi-particle topologies in GADGET GUI event PNGs.

Save this file as flag_double_particle_events.py in your framework.
Requires your trained Faster R-CNN ResNet50-FPN best_model.pth checkpoint,
PyTorch, matching torchvision, and Pillow. No ROOT/H5 files or annotations CSV
are required for inference. Source images are never modified.

Example (run from the framework directory):
    python flag_double_particle_events.py --images /path/to/cut_images \
        --weights ./Models/best_model.pth --output ./double_particle_flags

The input directory is searched recursively for run<number>_event<number>.png.
PNG pixels are converted to RGB without cropping, resizing, flipping, or adding
normalization before the model. Faster R-CNN retains its standard internal
resize/normalization, as in the supplied notebook. Expected GUI size is 224x224.

Default decision reproduces the supplied notebook's geometry rule: score >=0.4;
largest box is the presumed track; smaller contained boxes are peak candidates;
remove containing/nested candidates and any overlapping lower-score candidates;
apply the original size-dependent separation test; flag >=2 remaining peaks.
This default ignores predicted class labels, exactly as the supplied notebook.
Use --track-label N --bragg-label M ONLY if the training label mapping is known.
Flags are review candidates, not a calibrated two-particle probability.

Outputs in a NEW directory (existing directories are refused):
  results.csv              one row per examined event image, including errors
  flagged_events.csv       flagged rows, with original run/event identifiers
  flagged_events.txt       selected_events = [(run, event), ...] for rve.py
  flagged_images.txt       absolute paths of flagged original images
  detections.jsonl         raw detections and retained peaks for each image
  annotated/...            enlarged previews with boxes; not CNN input images
  summary.json             configuration, counts, completion/failure status

--limit 20 is a quick first check; --device cpu or cuda:1 selects a device.
--copy-flagged also copies unchanged flagged PNGs into flagged_originals/.
Source-run/date subdirectories are retained in the output paths.
Exit codes: 0 complete; 1 failed/interrupted; 2 complete with image-read errors.
"""
import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

EVENT_NAME = re.compile(r"run(\d+)_event(\d+)\.png", re.IGNORECASE)
FIELDS = ["run", "event", "image_path", "status", "peak_count",
          "peak_scores", "annotated_path", "error"]


def is_inside(inner, outer):
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def is_overlapping(a, b):
    return (min(a[2], b[2]) > max(a[0], b[0])
            and min(a[3], b[3]) > max(a[1], b[1]))


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def select_peaks(prediction, threshold=0.6, track_label=None, bragg_label=None):
    """Return presumed track and surviving peak detections, using notebook rule."""
    if not (len(prediction["boxes"]) == len(prediction["scores"])
            == len(prediction["labels"])):
        raise ValueError("Detection boxes, scores, and labels have different lengths.")
    detections = []
    for idx, (box, score, label) in enumerate(zip(
            prediction["boxes"], prediction["scores"], prediction["labels"])):
        box, score = list(map(float, box)), float(score)
        if (len(box) != 4 or not all(math.isfinite(v) for v in box + [score])
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Model returned an invalid detection box/score.")
        if score >= threshold:
            detections.append(dict(index=idx, box=box, score=score, label=int(label)))
    tracks = [d for d in detections if track_label is None or d["label"] == track_label]
    if not tracks:
        return None, []
    track = max(tracks, key=lambda d: box_area(d["box"]))
    contained = [d for d in detections if d["index"] != track["index"]
                 and (bragg_label is None or d["label"] == bragg_label)
                 and is_inside(d["box"], track["box"])]
    # Preserve the notebook rule: discard a candidate containing another one.
    valid = [d for d in contained if not any(
        other["index"] != d["index"] and is_inside(other["box"], d["box"])
        for other in contained)]
    peaks = []
    while valid:
        best = max(valid, key=lambda d: d["score"])
        peaks.append(best)
        valid = [d for d in valid if not is_overlapping(d["box"], best["box"])]
    width = track["box"][2] - track["box"][0]
    height = track["box"][3] - track["box"][1]
    separation = 0 if width * height < 2700 else min(width, height) * 0.3
    remove = set()
    for i, first in enumerate(peaks):
        for j in range(i + 1, len(peaks)):
            second = peaks[j]
            a, b = first["box"], second["box"]
            distance = math.hypot((a[0]+a[2]-b[0]-b[2])/2,
                                  (a[1]+a[3]-b[1]-b[3])/2)
            if distance < separation:
                remove.add(i if first["score"] < second["score"] else j)
    return track, [d for i, d in enumerate(peaks) if i not in remove]


def find_images(root, output):
    """Strict GUI filenames exclude JPG previews and annotated PNGs."""
    paths = []
    for parent, dirs, files in __import__("os").walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if Path(parent, d).resolve() != output)
        for name in sorted(files):
            path = Path(parent, name)
            if EVENT_NAME.fullmatch(name) and not path.is_symlink():
                paths.append(path)
    return sorted(paths)


def load_detector(weights, device_name, threshold):
    """Load full trained state without downloading generic pretrained weights."""
    try:
        import torch
        import torchvision
        from torchvision.ops.misc import FrozenBatchNorm2d
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Install/use an environment with compatible torch and torchvision, "
                           "plus Pillow. Original import error: " + str(exc)) from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Use --device auto, cpu, cuda, or cuda:N.")
    if device.type == "cuda" and (not torch.cuda.is_available()
            or (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ValueError(f"Requested GPU {device} is unavailable.")
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a saved state_dict or checkpoint dictionary.")
    state = checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state = checkpoint[key]
            break
    if state and all(isinstance(k, str) and k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    key = "roi_heads.box_predictor.cls_score.weight"
    if key not in state:
        raise ValueError("Missing Faster R-CNN classifier weights in checkpoint.")
    num_classes = int(state[key].shape[0])
    if num_classes < 2:
        raise ValueError("Checkpoint needs background plus at least one foreground class.")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=num_classes,
        box_score_thresh=min(0.05, threshold))
    # pretrained=True in the supplied notebook uses FrozenBatchNorm2d (COCO,
    # eps=0). Simply constructing a completely untrained model would otherwise
    # choose ordinary BatchNorm2d, with different epsilon/state semantics.
    frozen = not any(k.startswith("backbone.") and k.endswith("num_batches_tracked") for k in state)
    if frozen:
        def replace_norm(parent):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.BatchNorm2d):
                    setattr(parent, name, FrozenBatchNorm2d(child.num_features, eps=0.0))
                else:
                    replace_norm(child)
        replace_norm(model.backbone)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("Checkpoint does not match Faster R-CNN ResNet50-FPN v1. "
                         "Use the original model architecture/environment. " + str(exc)) from exc
    model.to(device).eval()
    metadata = dict(device=str(device), num_classes_including_background=num_classes,
                    torch_version=torch.__version__, torchvision_version=torchvision.__version__,
                    backbone_normalization="FrozenBatchNorm2d_eps0" if frozen else "BatchNorm2d")
    return model, device, metadata


def infer(model, device, images):
    import torch
    from torchvision.transforms.functional import to_tensor
    with torch.inference_mode():
        outputs = model([to_tensor(image).to(device) for image in images])
    return [{key: out[key].detach().cpu().tolist() for key in ("boxes", "scores", "labels")}
            for out in outputs]


def save_preview(image, track, peaks, path, run, event):
    from PIL import ImageDraw
    # Enlarging is for review output only; model sees unchanged source pixels.
    scale = 3
    preview = image.resize((image.width * scale, image.height * scale))
    draw = ImageDraw.Draw(preview)
    for detection, color, text in [(track, "red", "Presumed track")] + [
            (d, "lime", f"Peak {i+1}: {d['score']:.2f} (label {d['label']})")
            for i, d in enumerate(peaks)]:
        box = [v * scale for v in detection["box"]]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], max(0, box[1]-13)), text, fill=color, stroke_width=1, stroke_fill="black")
    draw.text((5, 5), f"run{run}_event{event}: {len(peaks)} candidate peaks",
              fill="white", stroke_width=1, stroke_fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path)


def write_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def run_scan(args):
    from PIL import Image, UnidentifiedImageError
    root = Path(args.images).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    output = (Path(args.output).expanduser().resolve() if args.output else
              root.parent / (root.name + "_Threshold_0.3_Seperation_0.3"))
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    if not weights.is_file():
        raise FileNotFoundError(f"Trained weights not found: {weights}. Obtain best_model.pth from the model author.")
    if output == root or output in root.parents:
        raise ValueError("Output must be a new directory, separate from or below the input directory.")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output folder.")
    paths = find_images(root, output)
    total_found = len(paths)
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise ValueError("No run<number>_event<number>.png images found. Select a GUI export folder or cut_images parent.")
    model, device, metadata = load_detector(weights, args.device, args.score_threshold)
    for label in (args.track_label, args.bragg_label):
        if label is not None and label >= metadata["num_classes_including_background"]:
            raise ValueError(f"Class {label} is not present in this checkpoint.")
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status="in_progress", input=str(root), output=str(output), weights=str(weights),
                   images_found=total_found, images_scheduled=len(paths), examined=0,
                   flagged=0, image_errors=0, unexpected_image_sizes=0, limit=args.limit,
                   score_threshold=args.score_threshold, track_label=args.track_label,
                   bragg_label=args.bragg_label, batch_size=args.batch_size, **metadata)
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    print(f"Device: {device}; scanning {len(paths)} images; score >= {args.score_threshold}")
    print("Rule: " + ("class-aware contained peaks" if args.track_label is not None else
                      "original geometry heuristic (predicted labels are not used to select boxes)"))
    print(f"Output: {output}")
    pairs = set()
    try:
        with (output/"results.csv").open("w", newline="", encoding="utf-8") as all_file, \
             (output/"flagged_events.csv").open("w", newline="", encoding="utf-8") as flagged_file, \
             (output/"flagged_images.txt").open("w", encoding="utf-8") as names_file, \
             (output/"detections.jsonl").open("w", encoding="utf-8") as detections_file:
            all_writer, flag_writer = (csv.DictWriter(f, fieldnames=FIELDS) for f in (all_file, flagged_file))
            all_writer.writeheader()
            flag_writer.writeheader()
            for start in range(0, len(paths), args.batch_size):
                batch = []
                for path in paths[start:start+args.batch_size]:
                    match = EVENT_NAME.fullmatch(path.name)
                    run, event = map(int, match.groups())
                    try:
                        with Image.open(path) as opened:
                            image = opened.convert("RGB")
                            image.load()
                    except (OSError, UnidentifiedImageError, ValueError) as exc:
                        all_writer.writerow(dict(run=run, event=event, image_path=str(path),
                                                 status="error", error=str(exc)))
                        summary["image_errors"] += 1
                        summary["examined"] += 1
                        continue
                    if image.size != (224, 224):
                        summary["unexpected_image_sizes"] += 1
                        if summary["unexpected_image_sizes"] == 1:
                            print(f"NOTE: {path} is {image.size}, not GUI 224x224. "
                                  "No resize is applied before the model; review image/training compatibility.")
                    batch.append((path, run, event, image))
                predictions = infer(model, device, [item[3] for item in batch]) if batch else []
                if len(predictions) != len(batch):
                    raise ValueError("Model did not return one prediction per image.")
                for (path, run, event, image), prediction in zip(batch, predictions):
                    track, peaks = select_peaks(prediction, args.score_threshold, args.track_label, args.bragg_label)
                    flagged = len(peaks) >= 2
                    relative = path.relative_to(root)
                    preview_path = output / "annotated" / relative.parent / (path.stem + "_with_boxes.png")
                    if flagged:
                        save_preview(image, track, peaks, preview_path, run, event)
                        if args.copy_flagged:
                            destination = output / "flagged_originals" / relative
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(path, destination)
                    row = dict(run=run, event=event, image_path=str(path),
                               status="flagged" if flagged else "not_flagged", peak_count=len(peaks),
                               peak_scores=json.dumps([d["score"] for d in peaks]),
                               annotated_path=str(preview_path) if flagged else "", error="")
                    all_writer.writerow(row)
                    detections_file.write(json.dumps(dict(run=run, event=event, image_path=str(path),
                        flagged=flagged, track=track, retained_peaks=peaks, predictions=prediction), allow_nan=False)+"\n")
                    summary["examined"] += 1
                    if flagged:
                        flag_writer.writerow(row)
                        names_file.write(str(path) + "\n")
                        pairs.add((run, event))
                        summary["flagged"] += 1
                for handle in (all_file, flagged_file, names_file, detections_file):
                    handle.flush()
                # Keep usable partial results if interrupted during a later batch.
                (output/"flagged_events.txt").write_text(
                    "selected_events = " + repr(sorted(pairs)) + "\n", encoding="utf-8")
                write_json(summary_path, summary)
                print(f"Examined {summary['examined']}/{len(paths)}; "
                      f"flagged {summary['flagged']}; image errors {summary['image_errors']}", flush=True)
        summary["status"] = "complete_with_image_errors" if summary["image_errors"] else "complete"
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error"] = str(exc) or type(exc).__name__
        write_json(summary_path, summary)
        raise
    write_json(summary_path, summary)
    print(f"Finished. Review {output / 'flagged_events.csv'} and annotated/.")
    return 2 if summary["image_errors"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default="/egr/research-tpc/shared/proc_runs/e25058/new_h5/cut_images/October31", help="Directory containing the run/date image folders",)
    parser.add_argument("--weights",default="/egr/research-tpc/shared/ObjDet/best_model.pth", help="Trained Faster R-CNN state_dict checkpoint",)
    parser.add_argument("--output", help="New results directory; default is a sibling of --images")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Process only the first N images as a smoke check")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--track-label", type=int, help="Known training class ID for the whole track (optional)")
    parser.add_argument("--bragg-label", type=int, help="Known training class ID for a Bragg peak (optional)")
    parser.add_argument("--copy-flagged", action="store_true", help="Also copy flagged original PNGs, unchanged")
    args = parser.parse_args(argv)
    if not math.isfinite(args.score_threshold) or not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")
    if (args.track_label is None) != (args.bragg_label is None):
        parser.error("Supply --track-label and --bragg-label together, or neither")
    if args.track_label is not None and (min(args.track_label, args.bragg_label) < 1
                                       or args.track_label == args.bragg_label):
        parser.error("Track and Bragg labels must be different positive foreground class IDs")
    try:
        return run_scan(args)
    except KeyboardInterrupt:
        print("Interrupted. Consult summary.json for partial scan status.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
