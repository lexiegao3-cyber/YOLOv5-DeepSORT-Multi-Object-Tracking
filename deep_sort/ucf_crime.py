"""Helpers for selecting videos from the official UCF-Crime layout."""

from __future__ import annotations

from pathlib import Path


UCF_ANOMALY_CATEGORIES = (
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
)


def _normalise_entry(value):
    return value.strip().replace("\\", "/").lstrip("./")


def find_ucf_videos_root(root):
    """Find the extracted UCF-Crime ``Videos`` directory."""
    root = Path(root).expanduser().resolve()
    candidates = (
        root / "Videos",
        root / "UCF_Crimes" / "Videos",
        root,
    )
    for candidate in candidates:
        if any((candidate / category).is_dir() for category in UCF_ANOMALY_CATEGORIES):
            return candidate
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "UCF-Crime Videos directory not found. Expected one of: " + expected
    )


def _find_split_file(videos_root, split):
    split_name = f"Anomaly_{split.capitalize()}.txt"
    candidates = (
        videos_root.parent / "Anomaly_Detection_splits" / split_name,
        videos_root.parent.parent / "Anomaly_Detection_splits" / split_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _split_entries(videos_root, split):
    split_file = _find_split_file(videos_root, split)
    if split_file is None:
        raise FileNotFoundError(
            f"UCF-Crime {split} split file was not found. "
            "Expected Anomaly_Detection_splits/Anomaly_Train.txt or "
            "Anomaly_Test.txt."
        )
    entries = set()
    for line in split_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(_normalise_entry(line.split()[0]))
    return entries


def _is_in_split(video_path, videos_root, entries):
    try:
        relative = _normalise_entry(video_path.relative_to(videos_root).as_posix())
    except ValueError as error:
        raise ValueError(
            f"UCF-Crime video must be inside the selected Videos directory: {video_path}"
        ) from error
    return relative in entries or video_path.name in entries or any(
        entry.endswith("/" + relative) for entry in entries
    )


def resolve_ucf_source(root, category="", video="", split="all"):
    """Resolve one safe UCF-Crime source for the tracker.

    A single video is required for train/test selection because DeepSORT state
    must not carry an identity from one independent UCF video into the next.
    A category directory can be selected when ``split`` is ``all``.
    """
    videos_root = find_ucf_videos_root(root)
    category = category.strip()
    video = video.strip()
    split = split.lower()

    if category:
        category_path = next(
            (
                path
                for path in videos_root.iterdir()
                if path.is_dir() and path.name.lower() == category.lower()
            ),
            None,
        )
        if category_path is None:
            available = sorted(path.name for path in videos_root.iterdir() if path.is_dir())
            raise FileNotFoundError(
                f"UCF-Crime category not found: {category}. Available: {available}"
            )
    else:
        category_path = videos_root

    if not video:
        if split != "all":
            raise ValueError(
                "--ucf-video is required when --ucf-split is train or test, "
                "so tracker IDs reset between independent videos."
            )
        return str(category_path)

    requested = Path(video).expanduser()
    if requested.is_absolute() and requested.is_file():
        video_path = requested.resolve()
    else:
        direct = category_path / video
        if direct.is_file():
            video_path = direct.resolve()
        else:
            matches = list(category_path.rglob(video))
            if not matches and not requested.suffix:
                matches = list(category_path.rglob(requested.name + ".mp4"))
            if not matches:
                raise FileNotFoundError(
                    f"UCF-Crime video not found under {category_path}: {video}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"UCF-Crime video name is ambiguous: {video}. "
                    "Pass category and the relative video path."
                )
            video_path = matches[0].resolve()

    if category and video_path.parent.name.lower() != category.lower():
        raise ValueError(
            f"Video {video_path.name} is not inside requested category {category}."
        )
    if split != "all":
        entries = _split_entries(videos_root, split)
        if not _is_in_split(video_path, videos_root, entries):
            raise ValueError(
                f"UCF-Crime video is not listed in the {split} split: {video_path.name}"
            )
    return str(video_path)
