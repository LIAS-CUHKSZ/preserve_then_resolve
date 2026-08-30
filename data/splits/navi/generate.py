#!/usr/bin/env python3
"""Sample image-disjoint NAVI-Wild splits with original image dimensions.

The generator first partitions every object's images between estimation and
correspondence. That one image assignment is shared by all angular bins. It
then selects equally sized pair sets with matched five-degree angle histograms
and low endpoint reuse. Every sampled CSV row receives image height and width
read directly from the source images and checked against NAVI annotations. No
descriptor, mask, or evaluation result is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Mapping, Sequence

from PIL import Image


SPLIT_VERSION = "navi-wild-joint"
SEED = 20260808
PAIR_COUNT = 500
ANGLE_SUBBIN_DEGREES = 5
PARTITION_TRIALS = 4096
SELECTION_TRIALS = 192
CAPACITY_SLACK = 4
MAX_BIN_ENDPOINT_REUSE = 4
MAX_TOTAL_ENDPOINT_REUSE = 6
FAMILIES = ("estimation", "correspondence")
BINS: dict[str, tuple[float, float]] = {
    "0-40": (0.0, 40.0),
    "40-80": (40.0, 80.0),
    "80-120": (80.0, 120.0),
}
PAIR_HEADER = (
    "image_1",
    "image_2",
    "angular_distance_degrees",
    "image_1_height",
    "image_1_width",
    "image_2_height",
    "image_2_width",
)


@dataclass(frozen=True, order=True)
class Edge:
    left: str
    right: str
    distance: float
    distance_text: str

    @property
    def object_name(self) -> str:
        left_object = self.left.split("/", 1)[0]
        right_object = self.right.split("/", 1)[0]
        if left_object != right_object:
            raise ValueError(f"Cross-object edge: {self.left} {self.right}")
        return left_object

    @property
    def endpoints(self) -> tuple[str, str]:
        return self.left, self.right

    def reversed(self) -> "Edge":
        return Edge(self.right, self.left, self.distance, self.distance_text)


@dataclass(frozen=True)
class ImageMetadata:
    height: int
    width: int
    official_split: str
    camera_model: str
    occluded: bool


@dataclass(frozen=True)
class SelectionCandidate:
    """One quota-complete family selection considered by the joint optimizer."""

    selected: tuple[tuple[str, Edge], ...]
    degree_score: tuple[int, int, int, int, int]
    signature: int
    same_camera_by_bin: tuple[int, ...]
    camera_pair_counts: Mapping[tuple[str, str, str], int]


class SourceGraph(dict[str, dict[str, list[Edge]]]):
    """Binned edge graph plus every image observed in the source file."""

    def __init__(
        self,
        bins: Mapping[str, dict[str, list[Edge]]],
        all_images: Mapping[str, set[str]],
    ) -> None:
        super().__init__(bins)
        self.all_images = {name: set(images) for name, images in all_images.items()}


def _stable_int(*tokens: object) -> int:
    payload = ":".join((str(SEED), *(str(token) for token in tokens)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_identity(value: str) -> dict[str, int | str]:
    encoded = value.encode("utf-8")
    return {
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _bin_name(distance: float) -> str | None:
    for name, (lower, upper) in BINS.items():
        if lower <= distance < upper:
            return name
    return None


def _subbin_index(distance: float, bin_name: str) -> int:
    lower, upper = BINS[bin_name]
    if not lower <= distance < upper:
        raise ValueError(f"Angle {distance} is outside {bin_name}")
    return int((distance - lower) // ANGLE_SUBBIN_DEGREES)


def read_source(path: Path) -> SourceGraph:
    """Read and validate the NAVI source graph for the experiment bins."""
    by_object: dict[str, dict[str, list[Edge]]] = defaultdict(
        lambda: {name: [] for name in BINS}
    )
    all_images: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: expected image image angle")
        left, right, distance_text = fields
        if left == right:
            raise ValueError(f"{path}:{line_number}: self edge")
        distance = float(distance_text)
        if not math.isfinite(distance) or not 0.0 <= distance <= 180.0:
            raise ValueError(f"{path}:{line_number}: invalid angular distance {distance_text}")
        edge = Edge(left, right, distance, distance_text)
        object_name = edge.object_name
        all_images[object_name].update(edge.endpoints)
        _ = by_object[object_name]
        key = tuple(sorted((left, right)))
        if key in seen:
            raise ValueError(f"{path}:{line_number}: duplicate undirected edge {key}")
        seen.add(key)
        bin_name = _bin_name(distance)
        if bin_name is not None:
            by_object[object_name][bin_name].append(edge)
    if not by_object:
        raise ValueError(f"No edges in [0, 120) degrees: {path}")
    for bins in by_object.values():
        for edges in bins.values():
            edges.sort()
    return SourceGraph(dict(by_object), all_images)


def _object_images(bins: Mapping[str, Sequence[Edge]]) -> list[str]:
    return sorted({image for edges in bins.values() for edge in edges for image in edge.endpoints})


def _all_object_images(
    graph: Mapping[str, Mapping[str, Sequence[Edge]]], object_name: str
) -> list[str]:
    if isinstance(graph, SourceGraph):
        return sorted(graph.all_images[object_name])
    return _object_images(graph[object_name])


def load_image_metadata(
    image_root: Path,
    graph: Mapping[str, Mapping[str, Sequence[Edge]]],
) -> dict[str, ImageMetadata]:
    """Load the nuisance strata and declared sizes for every graph image."""
    metadata: dict[str, ImageMetadata] = {}
    for object_name in sorted(graph):
        annotation_path = image_root / object_name / "wild_set" / "annotations.json"
        rows = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected an annotation list: {annotation_path}")
        by_filename = {str(row["filename"]): row for row in rows}
        if len(by_filename) != len(rows):
            raise ValueError(f"Duplicate annotation filename: {annotation_path}")
        for relative in _all_object_images(graph, object_name):
            token = Path(relative)
            if token.parts[:3] != (object_name, "wild_set", "images"):
                raise ValueError(f"Unexpected NAVI image path: {relative}")
            try:
                row = by_filename[token.name]
            except KeyError as exc:
                raise KeyError(f"No annotation for {relative}") from exc
            height, width = (int(value) for value in row["image_size"])
            if height <= 0 or width <= 0:
                raise ValueError(f"Invalid annotated image size for {relative}")
            camera = row.get("camera", {})
            metadata[relative] = ImageMetadata(
                height=height,
                width=width,
                official_split=str(row.get("split", "")),
                camera_model=str(camera.get("camera_model", "")),
                occluded=bool(row.get("occluded", False)),
            )
            if metadata[relative].occluded:
                raise ValueError(f"Source graph unexpectedly contains occluded image {relative}")
    return metadata


def _metadata_sha256(metadata: Mapping[str, ImageMetadata]) -> str:
    digest = hashlib.sha256()
    for image in sorted(metadata):
        value = metadata[image]
        digest.update(
            (
                f"{image}\t{value.height}\t{value.width}\t{value.official_split}\t"
                f"{value.camera_model}\t{int(value.occluded)}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def make_quotas(objects: Sequence[str]) -> dict[str, dict[str, int]]:
    """Return fixed, family-independent 14/15 pair quotas.

    The ten extra rows in each bin are assigned to disjoint object groups when
    possible, spreading the unavoidable 30 extras over the object set.
    """
    base, extras = divmod(PAIR_COUNT, len(objects))
    if base <= 0:
        raise ValueError("More objects than requested pairs")
    order = sorted(objects, key=lambda name: (_stable_int("quota", name), name))
    quotas = {name: {bin_name: base for bin_name in BINS} for name in objects}
    cursor = 0
    for bin_name in BINS:
        for offset in range(extras):
            quotas[order[(cursor + offset) % len(order)]][bin_name] += 1
        cursor += extras
    return quotas


def _estimation_sizes(
    graph: Mapping[str, Mapping[str, Sequence[Edge]]]
) -> dict[str, int]:
    objects = sorted(graph)
    sizes = {name: len(_all_object_images(graph, name)) for name in objects}
    total = sum(sizes.values())
    target = total // 2
    floor_total = sum(size // 2 for size in sizes.values())
    ceil_count = target - floor_total
    odd = sorted(
        (name for name, size in sizes.items() if size % 2),
        key=lambda name: (_stable_int("larger-side", name), name),
    )
    if not 0 <= ceil_count <= len(odd):
        raise RuntimeError("Cannot balance the global image partition")
    larger = set(odd[:ceil_count])
    return {name: sizes[name] // 2 + int(name in larger) for name in objects}


def _induced_edges(edges: Sequence[Edge], images: set[str]) -> list[Edge]:
    return [edge for edge in edges if edge.left in images and edge.right in images]


def _partition_score(
    graph: Mapping[str, Sequence[Edge]],
    estimation: set[str],
    all_images: set[str],
    quotas: Mapping[str, int],
    image_metadata: Mapping[str, ImageMetadata],
) -> tuple[int, ...]:
    correspondence = all_images - estimation
    cells: list[tuple[str, int, int, int, int]] = []
    for bin_name, edges in graph.items():
        induced_e = _induced_edges(edges, estimation)
        induced_c = _induced_edges(edges, correspondence)
        subbins_e = Counter(_subbin_index(edge.distance, bin_name) for edge in induced_e)
        subbins_c = Counter(_subbin_index(edge.distance, bin_name) for edge in induced_c)
        common_angle_capacity = sum(
            min(subbins_e[index], subbins_c[index]) for index in range(8)
        )
        cells.append(
            (bin_name, len(induced_e), len(induced_c), len(edges), common_angle_capacity)
        )
    deficit = sum(
        max(0, quotas[bin_name] + CAPACITY_SLACK - count)
        for bin_name, count_e, count_c, _, _ in cells
        for count in (count_e, count_c)
    )
    angle_support_deficit = sum(
        max(0, quotas[bin_name] - common_angle_capacity)
        for bin_name, _, _, _, common_angle_capacity in cells
    )
    capped_slack_shortfall = sum(
        max(0, quotas[bin_name] + CAPACITY_SLACK + 1 - count)
        for bin_name, count_e, count_c, _, _ in cells
        for count in (count_e, count_c)
    )

    def stratum_imbalance(attribute: str) -> int:
        estimation_counts = Counter(
            getattr(image_metadata[image], attribute) for image in estimation
        )
        correspondence_counts = Counter(
            getattr(image_metadata[image], attribute) for image in correspondence
        )
        return sum(
            abs(estimation_counts[value] - correspondence_counts[value])
            for value in estimation_counts.keys() | correspondence_counts.keys()
        )

    official_split_imbalance = stratum_imbalance("official_split")
    resolution_imbalance = sum(
        abs(
            sum(
                image_metadata[image].height == height
                and image_metadata[image].width == width
                for image in estimation
            )
            - sum(
                image_metadata[image].height == height
                and image_metadata[image].width == width
                for image in correspondence
            )
        )
        for height, width in {
            (image_metadata[image].height, image_metadata[image].width)
            for image in all_images
        }
    )
    camera_model_imbalance = stratum_imbalance("camera_model")
    imbalance = sum(
        abs(count_e - count_c) * 1_000_000 // max(total, 1)
        for _, count_e, count_c, total, _ in cells
    )
    random_deviation = sum(
        (abs(4 * count_e - total) + abs(4 * count_c - total)) * 1_000_000
        // max(total, 1)
        for _, count_e, count_c, total, _ in cells
    )
    return (
        deficit + angle_support_deficit,
        angle_support_deficit,
        capped_slack_shortfall,
        official_split_imbalance,
        resolution_imbalance,
        camera_model_imbalance,
        imbalance,
        random_deviation,
    )


def partition_images(
    graph: Mapping[str, Mapping[str, Sequence[Edge]]],
    quotas: Mapping[str, Mapping[str, int]],
    image_metadata: Mapping[str, ImageMetadata],
) -> dict[str, str]:
    """Create a deterministic, balanced image assignment shared by all bins."""
    estimation_sizes = _estimation_sizes(graph)
    assignment: dict[str, str] = {}
    for object_name in sorted(graph):
        images = _all_object_images(graph, object_name)
        all_images = set(images)
        count = estimation_sizes[object_name]
        rng = random.Random(_stable_int("partition", object_name))
        best_score: tuple[int, ...] | None = None
        best: tuple[str, ...] | None = None
        for trial in range(PARTITION_TRIALS):
            if trial == 0:
                candidate = tuple(images[:count])
            else:
                shuffled = images.copy()
                rng.shuffle(shuffled)
                candidate = tuple(sorted(shuffled[:count]))
            score = _partition_score(
                graph[object_name],
                set(candidate),
                all_images,
                quotas[object_name],
                image_metadata,
            )
            candidate_key = _stable_int("partition-tie", object_name, *candidate)
            ranked_score = (*score, candidate_key)
            if best_score is None or ranked_score < best_score:
                best_score = ranked_score
                best = candidate
        if best is None or best_score is None or best_score[0] != 0:
            raise RuntimeError(
                f"Could not find a feasible image partition for {object_name}; "
                f"best score={best_score}"
            )
        estimation = set(best)
        for image in images:
            assignment[image] = "estimation" if image in estimation else "correspondence"
    return assignment


def _matched_subbin_quotas(
    raw_edges: Sequence[Edge],
    estimation_edges: Sequence[Edge],
    correspondence_edges: Sequence[Edge],
    bin_name: str,
    target: int,
) -> tuple[int, ...]:
    count = int((BINS[bin_name][1] - BINS[bin_name][0]) / ANGLE_SUBBIN_DEGREES)
    raw = Counter(_subbin_index(edge.distance, bin_name) for edge in raw_edges)
    available_e = Counter(_subbin_index(edge.distance, bin_name) for edge in estimation_edges)
    available_c = Counter(_subbin_index(edge.distance, bin_name) for edge in correspondence_edges)
    capacity = [min(available_e[index], available_c[index]) for index in range(count)]
    if sum(capacity) < target:
        raise RuntimeError(
            f"Only {sum(capacity)} edges share angle support in {bin_name}; need {target}"
        )
    allocation = [0] * count
    raw_total = sum(raw.values())
    for _ in range(target):
        candidates = [index for index in range(count) if allocation[index] < capacity[index]]
        if not candidates:
            raise RuntimeError(f"Sub-bin allocation exhausted in {bin_name}")
        chosen = min(
            candidates,
            key=lambda index: (
                allocation[index] * raw_total - target * raw[index],
                index,
            ),
        )
        allocation[chosen] += 1
    return tuple(allocation)


def _selection_score(
    selected: Sequence[tuple[str, Edge]], all_images: Sequence[str]
) -> tuple[int, int, int, int, int]:
    bin_degree: Counter[tuple[str, str]] = Counter()
    total_degree: Counter[str] = Counter()
    for bin_name, edge in selected:
        for image in edge.endpoints:
            bin_degree[(bin_name, image)] += 1
            total_degree[image] += 1
    return (
        max(bin_degree.values(), default=0),
        max(total_degree.values(), default=0),
        sum(value * value for value in bin_degree.values()),
        sum(value * value for value in total_degree.values()),
        sum(1 for image in all_images if total_degree[image] == 0),
    )


def _camera_pair(
    edge: Edge, image_metadata: Mapping[str, ImageMetadata]
) -> tuple[str, str]:
    return tuple(
        sorted(
            (
                image_metadata[edge.left].camera_model,
                image_metadata[edge.right].camera_model,
            )
        )
    )


def _selection_camera_profile(
    selected: Sequence[tuple[str, Edge]],
    image_metadata: Mapping[str, ImageMetadata],
) -> tuple[tuple[int, ...], dict[tuple[str, str, str], int]]:
    same_camera: Counter[str] = Counter()
    camera_pairs: Counter[tuple[str, str, str]] = Counter()
    for bin_name, edge in selected:
        first, second = _camera_pair(edge, image_metadata)
        same_camera[bin_name] += int(first == second)
        camera_pairs[(bin_name, first, second)] += 1
    return (
        tuple(same_camera[bin_name] for bin_name in BINS),
        dict(camera_pairs),
    )


def _family_edge_candidates(
    object_name: str,
    family: str,
    eligible: Mapping[str, Sequence[Edge]],
    subbin_quotas: Mapping[str, Sequence[int]],
    all_images: Sequence[str],
    image_metadata: Mapping[str, ImageMetadata],
) -> list[SelectionCandidate]:
    groups: list[tuple[str, int, int, list[Edge]]] = []
    for bin_name in BINS:
        for subbin, target in enumerate(subbin_quotas[bin_name]):
            if target == 0:
                continue
            edges = [
                edge
                for edge in eligible[bin_name]
                if _subbin_index(edge.distance, bin_name) == subbin
            ]
            if len(edges) < target:
                raise RuntimeError(
                    f"{object_name}/{family}/{bin_name}/{subbin}: "
                    f"{len(edges)} candidates for quota {target}"
                )
            groups.append((bin_name, subbin, target, edges))

    candidates: dict[tuple[tuple[str, Edge], ...], SelectionCandidate] = {}
    for trial in range(SELECTION_TRIALS):
        rng = random.Random(_stable_int("selection", object_name, family, trial))
        ordered = sorted(
            groups,
            key=lambda item: (
                len(item[3]) / item[2],
                rng.random(),
                item[0],
                item[1],
            ),
        )
        chosen: list[tuple[str, Edge]] = []
        bin_degree: Counter[tuple[str, str]] = Counter()
        total_degree: Counter[str] = Counter()
        for bin_name, _, target, edges in ordered:
            remaining = list(edges)
            for _ in range(target):
                ranked: list[tuple[tuple[object, ...], Edge]] = []
                for edge in remaining:
                    projected_bin = [bin_degree[(bin_name, image)] + 1 for image in edge.endpoints]
                    projected_total = [total_degree[image] + 1 for image in edge.endpoints]
                    score = (
                        max(projected_bin),
                        max(projected_total),
                        sum(value * value for value in projected_bin),
                        sum(value * value for value in projected_total),
                        rng.random(),
                        edge,
                    )
                    ranked.append((score, edge))
                edge = min(ranked, key=lambda item: item[0])[1]
                chosen.append((bin_name, edge))
                for image in edge.endpoints:
                    bin_degree[(bin_name, image)] += 1
                    total_degree[image] += 1
                remaining.remove(edge)
        score = _selection_score(chosen, all_images)
        signature = _stable_int(
            "selection-tie",
            object_name,
            family,
            *(f"{name}:{edge.left}:{edge.right}" for name, edge in sorted(chosen)),
        )
        selection_key = tuple(sorted(chosen))
        same_camera, camera_pairs = _selection_camera_profile(
            selection_key, image_metadata
        )
        candidates.setdefault(
            selection_key,
            SelectionCandidate(
                selected=selection_key,
                degree_score=score,
                signature=signature,
                same_camera_by_bin=same_camera,
                camera_pair_counts=camera_pairs,
            ),
        )
    if not candidates:
        raise RuntimeError(f"No edge selection for {object_name}/{family}")
    feasible = [
        candidate
        for candidate in candidates.values()
        if candidate.degree_score[0] <= MAX_BIN_ENDPOINT_REUSE
        and candidate.degree_score[1] <= MAX_TOTAL_ENDPOINT_REUSE
    ]
    if not feasible:
        best = min(
            candidates.values(),
            key=lambda candidate: (*candidate.degree_score, candidate.signature),
        )
        raise RuntimeError(
            f"No endpoint-reuse-feasible selection for {object_name}/{family}; "
            f"best degree score={best.degree_score}"
        )
    return feasible


def _camera_mismatch(
    estimation: SelectionCandidate, correspondence: SelectionCandidate
) -> tuple[int, int, int, int]:
    same_differences = [
        abs(left - right)
        for left, right in zip(
            estimation.same_camera_by_bin,
            correspondence.same_camera_by_bin,
        )
    ]
    pair_l1_by_bin = []
    for bin_name in BINS:
        keys = {
            key
            for key in (
                estimation.camera_pair_counts.keys()
                | correspondence.camera_pair_counts.keys()
            )
            if key[0] == bin_name
        }
        pair_l1_by_bin.append(
            sum(
                abs(
                    estimation.camera_pair_counts.get(key, 0)
                    - correspondence.camera_pair_counts.get(key, 0)
                )
                for key in keys
            )
        )
    return (
        max(same_differences, default=0),
        sum(same_differences),
        max(pair_l1_by_bin, default=0),
        sum(pair_l1_by_bin),
    )


def _select_joint_candidates(
    object_name: str,
    estimation: Sequence[SelectionCandidate],
    correspondence: Sequence[SelectionCandidate],
) -> tuple[SelectionCandidate, SelectionCandidate]:
    best: tuple[SelectionCandidate, SelectionCandidate] | None = None
    best_rank: tuple[int, ...] | None = None
    for candidate_e in estimation:
        for candidate_c in correspondence:
            score_e = candidate_e.degree_score
            score_c = candidate_c.degree_score
            rank = (
                *_camera_mismatch(candidate_e, candidate_c),
                max(score_e[0], score_c[0]),
                max(score_e[1], score_c[1]),
                score_e[0] + score_c[0],
                score_e[1] + score_c[1],
                score_e[2] + score_c[2],
                score_e[3] + score_c[3],
                score_e[4] + score_c[4],
                _stable_int(
                    "joint-selection-tie",
                    object_name,
                    candidate_e.signature,
                    candidate_c.signature,
                ),
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = candidate_e, candidate_c
    if best is None:
        raise RuntimeError(f"No joint edge selection for {object_name}")
    return best


def _candidate_result(candidate: SelectionCandidate) -> dict[str, list[Edge]]:
    result = {bin_name: [] for bin_name in BINS}
    for bin_name, edge in candidate.selected:
        result[bin_name].append(edge)
    for edges in result.values():
        edges.sort()
    return result


def select_pairs(
    graph: Mapping[str, Mapping[str, Sequence[Edge]]],
    assignment: Mapping[str, str],
    quotas: Mapping[str, Mapping[str, int]],
    image_metadata: Mapping[str, ImageMetadata],
) -> tuple[dict[str, dict[str, list[Edge]]], list[dict[str, object]]]:
    selected = {family: {bin_name: [] for bin_name in BINS} for family in FAMILIES}
    capacities: list[dict[str, object]] = []
    for object_name in sorted(graph):
        images = _all_object_images(graph, object_name)
        image_sets = {
            family: {image for image in images if assignment[image] == family}
            for family in FAMILIES
        }
        eligible = {
            family: {
                bin_name: _induced_edges(graph[object_name][bin_name], image_sets[family])
                for bin_name in BINS
            }
            for family in FAMILIES
        }
        angle_quotas: dict[str, tuple[int, ...]] = {}
        for bin_name in BINS:
            target = quotas[object_name][bin_name]
            angle_quotas[bin_name] = _matched_subbin_quotas(
                graph[object_name][bin_name],
                eligible["estimation"][bin_name],
                eligible["correspondence"][bin_name],
                bin_name,
                target,
            )
            capacities.append(
                {
                    "object_name": object_name,
                    "angular_bin": bin_name,
                    "pair_quota": target,
                    "raw_edge_count": len(graph[object_name][bin_name]),
                    "estimation_edge_capacity": len(eligible["estimation"][bin_name]),
                    "correspondence_edge_capacity": len(eligible["correspondence"][bin_name]),
                    "estimation_image_count": len(image_sets["estimation"]),
                    "correspondence_image_count": len(image_sets["correspondence"]),
                    "five_degree_pair_quotas": ";".join(str(value) for value in angle_quotas[bin_name]),
                }
            )
        candidates = {
            family: _family_edge_candidates(
                object_name,
                family,
                eligible[family],
                angle_quotas,
                sorted(image_sets[family]),
                image_metadata,
            )
            for family in FAMILIES
        }
        chosen_e, chosen_c = _select_joint_candidates(
            object_name,
            candidates["estimation"],
            candidates["correspondence"],
        )
        family_selections = {
            "estimation": _candidate_result(chosen_e),
            "correspondence": _candidate_result(chosen_c),
        }
        for family in FAMILIES:
            for bin_name in BINS:
                selected[family][bin_name].extend(family_selections[family][bin_name])
    return selected, capacities


def balance_pair_orientation(
    selected: Mapping[str, Mapping[str, Sequence[Edge]]]
) -> dict[str, dict[str, list[Edge]]]:
    """Flip exactly half of each family/bin, stratified by object.

    The source graph orders every edge by capture/image ID. Balancing this role
    prevents image-1/image-2 from becoming a proxy for acquisition order.
    """
    balanced = {family: {bin_name: [] for bin_name in BINS} for family in FAMILIES}
    for family in FAMILIES:
        for bin_name in BINS:
            by_object: dict[str, list[Edge]] = defaultdict(list)
            for edge in selected[family][bin_name]:
                by_object[edge.object_name].append(edge)
            odd_objects = [name for name, edges in by_object.items() if len(edges) % 2]
            if len(odd_objects) % 2:
                raise RuntimeError(f"Cannot balance odd object quotas in {family}/{bin_name}")
            round_up = set(
                sorted(
                    odd_objects,
                    key=lambda name: (_stable_int("orientation-round", bin_name, name), name),
                )[: len(odd_objects) // 2]
            )
            reversed_count = 0
            for object_name in sorted(by_object):
                edges = sorted(
                    by_object[object_name],
                    key=lambda edge: (
                        _stable_int(
                            "orientation-edge",
                            family,
                            bin_name,
                            object_name,
                            edge.left,
                            edge.right,
                        ),
                        edge,
                    ),
                )
                flip_count = len(edges) // 2 + int(object_name in round_up)
                reversed_count += flip_count
                balanced[family][bin_name].extend(
                    edge.reversed() if index < flip_count else edge
                    for index, edge in enumerate(edges)
                )
            balanced[family][bin_name].sort()
            if reversed_count != PAIR_COUNT // 2:
                raise AssertionError(
                    f"{family}/{bin_name}: {reversed_count} reversed rows, "
                    f"expected {PAIR_COUNT // 2}"
                )
    return balanced


def _image_hw(
    image_root: Path,
    relative: str,
    cache: dict[str, tuple[int, int]],
    metadata: Mapping[str, ImageMetadata],
) -> tuple[int, int]:
    if relative not in cache:
        token = Path(relative)
        if token.is_absolute() or ".." in token.parts:
            raise ValueError(f"Image path escapes NAVI root: {relative}")
        with Image.open(image_root / token) as image:
            width, height = image.size
        expected = metadata[relative]
        if (height, width) != (expected.height, expected.width):
            raise ValueError(
                f"Image and annotation dimensions disagree for {relative}: "
                f"{height}x{width} versus {expected.height}x{expected.width}"
            )
        cache[relative] = height, width
    return cache[relative]


def _csv_text(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _write_text(path: Path, value: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value.encode("utf-8"))
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _degree_statistics(edges: Sequence[Edge]) -> dict[str, int]:
    degree = Counter(image for edge in edges for image in edge.endpoints)
    return {
        "unique_images": len(degree),
        "max_endpoint_reuse": max(degree.values(), default=0),
    }


def _camera_pair_balance_audit(
    selected: Mapping[str, Mapping[str, Sequence[Edge]]],
    image_metadata: Mapping[str, ImageMetadata],
) -> dict[str, object]:
    per_bin: dict[str, dict[str, int | str]] = {}
    global_max_same = 0
    global_max_pair_l1 = 0
    global_sum_same = 0
    global_sum_pair_l1 = 0
    for bin_name in BINS:
        by_family_object: dict[str, dict[str, list[Edge]]] = {
            family: defaultdict(list) for family in FAMILIES
        }
        for family in FAMILIES:
            for edge in selected[family][bin_name]:
                by_family_object[family][edge.object_name].append(edge)

        same_totals = {family: 0 for family in FAMILIES}
        same_difference_sum = 0
        camera_pair_l1_sum = 0
        worst_same = (0, "")
        worst_pair_l1 = (0, "")
        object_names = sorted(
            by_family_object["estimation"].keys()
            | by_family_object["correspondence"].keys()
        )
        for object_name in object_names:
            histograms: dict[str, Counter[tuple[str, str]]] = {}
            same_counts: dict[str, int] = {}
            for family in FAMILIES:
                histogram = Counter(
                    _camera_pair(edge, image_metadata)
                    for edge in by_family_object[family][object_name]
                )
                histograms[family] = histogram
                same_counts[family] = sum(
                    count for (first, second), count in histogram.items() if first == second
                )
                same_totals[family] += same_counts[family]
            same_difference = abs(
                same_counts["estimation"] - same_counts["correspondence"]
            )
            pair_keys = (
                histograms["estimation"].keys()
                | histograms["correspondence"].keys()
            )
            pair_l1 = sum(
                abs(
                    histograms["estimation"][key]
                    - histograms["correspondence"][key]
                )
                for key in pair_keys
            )
            same_difference_sum += same_difference
            camera_pair_l1_sum += pair_l1
            worst_same = max(worst_same, (same_difference, object_name))
            worst_pair_l1 = max(worst_pair_l1, (pair_l1, object_name))

        global_max_same = max(global_max_same, worst_same[0])
        global_max_pair_l1 = max(global_max_pair_l1, worst_pair_l1[0])
        global_sum_same += same_difference_sum
        global_sum_pair_l1 += camera_pair_l1_sum
        per_bin[bin_name] = {
            "estimation_same_camera_pairs": same_totals["estimation"],
            "correspondence_same_camera_pairs": same_totals["correspondence"],
            "sum_object_same_camera_absolute_difference": same_difference_sum,
            "max_object_same_camera_absolute_difference": worst_same[0],
            "max_object_same_camera_difference_object": worst_same[1],
            "sum_object_unordered_camera_pair_l1": camera_pair_l1_sum,
            "max_object_unordered_camera_pair_l1": worst_pair_l1[0],
            "max_object_unordered_camera_pair_l1_object": worst_pair_l1[1],
        }
    return {
        "definition": "unordered camera-model pair, compared per object and angular bin",
        "max_object_bin_same_camera_absolute_difference": global_max_same,
        "sum_object_bin_same_camera_absolute_difference": global_sum_same,
        "max_object_bin_unordered_camera_pair_l1": global_max_pair_l1,
        "sum_object_bin_unordered_camera_pair_l1": global_sum_pair_l1,
        "per_bin": per_bin,
    }


def validate(
    graph: Mapping[str, Mapping[str, Sequence[Edge]]],
    assignment: Mapping[str, str],
    quotas: Mapping[str, Mapping[str, int]],
    selected: Mapping[str, Mapping[str, Sequence[Edge]]],
    image_metadata: Mapping[str, ImageMetadata],
) -> dict[str, object]:
    image_sets = {
        family: {image for image, owner in assignment.items() if owner == family}
        for family in FAMILIES
    }
    if image_sets["estimation"] & image_sets["correspondence"]:
        raise AssertionError("Image partition is not disjoint")
    source_oriented = {
        (edge.left, edge.right, edge.distance_text)
        for bins in graph.values()
        for edges in bins.values()
        for edge in edges
    }
    source_undirected = {
        (tuple(sorted(edge.endpoints)), edge.distance_text)
        for bins in graph.values()
        for edges in bins.values()
        for edge in edges
    }
    report: dict[str, object] = {
        "global_image_disjoint": True,
        "image_counts": {family: len(images) for family, images in image_sets.items()},
        "families": {},
    }
    for family in FAMILIES:
        family_report: dict[str, object] = {}
        union: set[str] = set()
        cumulative_degree: Counter[str] = Counter()
        for bin_name, edges in selected[family].items():
            if len(edges) != PAIR_COUNT:
                raise AssertionError(f"{family}/{bin_name}: {len(edges)} != {PAIR_COUNT}")
            counts = Counter(edge.object_name for edge in edges)
            expected = {name: quotas[name][bin_name] for name in graph}
            if dict(counts) != expected:
                raise AssertionError(f"{family}/{bin_name}: object quotas differ")
            if len({(edge.left, edge.right) for edge in edges}) != len(edges):
                raise AssertionError(f"{family}/{bin_name}: duplicate edge")
            reversed_rows = 0
            for edge in edges:
                if (tuple(sorted(edge.endpoints)), edge.distance_text) not in source_undirected:
                    raise AssertionError(f"Selected edge is absent from source: {edge}")
                reversed_rows += int(
                    (edge.left, edge.right, edge.distance_text) not in source_oriented
                )
                if _bin_name(edge.distance) != bin_name:
                    raise AssertionError(f"Selected edge is in the wrong bin: {edge}")
                if any(assignment[image] != family for image in edge.endpoints):
                    raise AssertionError(f"Selected edge crosses the image partition: {edge}")
                union.update(edge.endpoints)
                cumulative_degree.update(edge.endpoints)
            statistics = _degree_statistics(edges)
            statistics["rows_reversed_from_source"] = reversed_rows
            statistics["same_camera_pairs"] = sum(
                _camera_pair(edge, image_metadata)[0]
                == _camera_pair(edge, image_metadata)[1]
                for edge in edges
            )
            if reversed_rows != PAIR_COUNT // 2:
                raise AssertionError(
                    f"{family}/{bin_name}: source/target roles are not balanced"
                )
            if statistics["max_endpoint_reuse"] > MAX_BIN_ENDPOINT_REUSE:
                raise AssertionError(
                    f"{family}/{bin_name}: endpoint reuse exceeds "
                    f"{MAX_BIN_ENDPOINT_REUSE}"
                )
            family_report[bin_name] = statistics
        family_report["union_unique_images"] = len(union)
        maximum_total_reuse = max(cumulative_degree.values(), default=0)
        if maximum_total_reuse > MAX_TOTAL_ENDPOINT_REUSE:
            raise AssertionError(
                f"{family}: endpoint reuse across bins exceeds "
                f"{MAX_TOTAL_ENDPOINT_REUSE}"
            )
        family_report["max_endpoint_reuse_across_bins"] = maximum_total_reuse
        report["families"][family] = family_report
    for object_name in graph:
        for bin_name in BINS:
            histograms = []
            for family in FAMILIES:
                histograms.append(
                    Counter(
                        _subbin_index(edge.distance, bin_name)
                        for edge in selected[family][bin_name]
                        if edge.object_name == object_name
                    )
                )
            if histograms[0] != histograms[1]:
                raise AssertionError(f"Angle histogram mismatch: {object_name}/{bin_name}")
    report["camera_pair_balance"] = _camera_pair_balance_audit(
        selected, image_metadata
    )
    return report


def generate(
    source: Path,
    image_root: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    planned = [
        *(output_dir / family / f"pairs_wildset_{bin_name}.csv" for family in FAMILIES for bin_name in BINS),
        output_dir / "image_partition.csv",
        output_dir / "partition_capacity.csv",
        output_dir / "manifest.json",
        output_dir / "SHA256SUMS",
    ]
    existing = [path for path in planned if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path.relative_to(output_dir)) for path in existing)
        raise FileExistsError(f"Refusing to overwrite generated split files: {names}")
    source_sha256 = _sha256(source)
    graph = read_source(source)
    image_metadata = load_image_metadata(image_root, graph)
    objects = sorted(graph)
    quotas = make_quotas(objects)
    assignment = partition_images(graph, quotas, image_metadata)
    selected, capacities = select_pairs(
        graph, assignment, quotas, image_metadata
    )
    selected = balance_pair_orientation(selected)
    audit = validate(graph, assignment, quotas, selected, image_metadata)

    payloads: dict[Path, str] = {}
    size_cache: dict[str, tuple[int, int]] = {}
    for family in FAMILIES:
        for bin_name in BINS:
            edges = list(selected[family][bin_name])
            random.Random(_stable_int("row-order", family, bin_name)).shuffle(edges)
            rows = []
            for edge in edges:
                h1, w1 = _image_hw(image_root, edge.left, size_cache, image_metadata)
                h2, w2 = _image_hw(image_root, edge.right, size_cache, image_metadata)
                rows.append((edge.left, edge.right, edge.distance_text, h1, w1, h2, w2))
            path = output_dir / family / f"pairs_wildset_{bin_name}.csv"
            payloads[path] = _csv_text(PAIR_HEADER, rows)

    partition_path = output_dir / "image_partition.csv"
    partition_rows = (
        (
            image.split("/", 1)[0],
            image,
            assignment[image],
            image_metadata[image].official_split,
            image_metadata[image].height,
            image_metadata[image].width,
            image_metadata[image].camera_model,
            int(image_metadata[image].occluded),
        )
        for image in sorted(assignment)
    )
    payloads[partition_path] = _csv_text(
        (
            "object_name",
            "image_path",
            "split",
            "navi_official_split",
            "image_height",
            "image_width",
            "camera_model",
            "occluded",
        ),
        partition_rows,
    )

    capacity_path = output_dir / "partition_capacity.csv"
    capacity_header = tuple(capacities[0])
    payloads[capacity_path] = _csv_text(
        capacity_header,
        ([row[key] for key in capacity_header] for row in capacities),
    )

    file_records = {
        str(path.relative_to(output_dir)): {
            **_text_identity(value),
        }
        for path, value in sorted(payloads.items())
    }
    if _sha256(source) != source_sha256:
        raise RuntimeError(f"Source pair graph changed during generation: {source}")
    manifest = {
        "schema_version": 1,
        "split_version": SPLIT_VERSION,
        "seed": SEED,
        "source": source.name,
        "source_sha256": source_sha256,
        "generator_sha256": _sha256(Path(__file__)),
        "image_metadata_sha256": _metadata_sha256(image_metadata),
        "algorithm": {
            "partition_trials_per_object": PARTITION_TRIALS,
            "selection_trials_per_object_family": SELECTION_TRIALS,
            "capacity_slack": CAPACITY_SLACK,
            "angle_subbin_degrees": ANGLE_SUBBIN_DEGREES,
            "max_bin_endpoint_reuse": MAX_BIN_ENDPOINT_REUSE,
            "max_total_endpoint_reuse": MAX_TOTAL_ENDPOINT_REUSE,
            "nuisance_strata_in_partition_objective": [
                "navi_official_split",
                "image_resolution",
                "camera_model",
            ],
            "joint_selection_soft_objectives": [
                "same_vs_cross_camera_pair_balance",
                "unordered_camera_pair_histogram_balance",
            ],
            "balanced_pair_orientation": True,
            "selection_uses_descriptor_or_evaluation_results": False,
            "pair_csv_image_sizes": (
                "read-from-source-image-and-checked-against-navi-annotation"
            ),
        },
        "counts": {
            "objects": len(objects),
            "source_images": len(assignment),
            "pairs_per_family_bin": PAIR_COUNT,
        },
        "bins_degrees": {name: list(bounds) for name, bounds in BINS.items()},
        "audit": audit,
        "files": file_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_text = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    checksum_path = output_dir / "SHA256SUMS"
    checksum_records = {
        **{relative: str(record["sha256"]) for relative, record in file_records.items()},
        "manifest.json": str(_text_identity(manifest_text)["sha256"]),
    }
    checksums = "".join(
        f"{digest}  {relative}\n" for relative, digest in sorted(checksum_records.items())
    )

    # All source parsing, optimization, image reads, and serialization finish
    # before any canonical file is replaced. Manifest/checksums are published
    # last and therefore act as integrity markers for interrupted overwrites.
    for path, value in sorted(payloads.items()):
        _write_text(path, value, overwrite)
    _write_text(manifest_path, manifest_text, overwrite)
    _write_text(checksum_path, checksums, overwrite)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--source", type=Path, default=repo_root / "data/navi/pairs-wild_set.txt")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=repo_root / "data/navi",
        help=(
            "NAVI root used to read and validate image_1/2 height and width "
            "while sampling"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate(args.source, args.image_root, args.output_dir, args.overwrite)
    print(json.dumps(manifest["audit"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
