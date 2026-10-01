#!/usr/bin/env python3
"""Generate deterministic GRR-003 spatial/program geometry datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


SPLIT_SEEDS = {
    "train": 71001,
    "validation": 82001,
    "test": 93001,
}

SPLIT_PREFIXES = {
    "train": "TR",
    "validation": "VA",
    "test": "TE",
}

RELATION_LABELS = {
    "point_identity": ["same_point", "different_point"],
    "line_membership": ["on_line", "off_line", "on_segment", "off_segment"],
    "containment": ["contains", "does_not_contain"],
    "distance_comparison": ["closer_to_first", "closer_to_second", "same_distance"],
    "horizontal_order": ["left_of", "right_of", "same_x"],
    "vertical_order": ["below", "above", "same_y"],
    "inside": ["inside", "boundary", "outside"],
    "between": ["between", "not_between"],
}

RELATION_FAMILIES = list(RELATION_LABELS)

POINT_RE = re.compile(r"^point ([A-Za-z][A-Za-z0-9_]*) = \(([^,]+), ([^)]+)\)$")
LINE_RE = re.compile(
    r"^(line|segment) ([A-Za-z][A-Za-z0-9_]*) = (?:line|segment)\(([A-Za-z][A-Za-z0-9_]*), ([A-Za-z][A-Za-z0-9_]*)\)$"
)
BOX_RE = re.compile(
    r"^box ([A-Za-z][A-Za-z0-9_]*) = box\(([^,]+), ([^,]+), ([^,]+), ([^)]+)\)$"
)


Point = tuple[Fraction, Fraction]
Box = tuple[Fraction, Fraction, Fraction, Fraction]


@dataclass(frozen=True)
class Scene:
    points: dict[str, Point]
    lines: dict[str, tuple[str, str]]
    segments: dict[str, tuple[str, str]]
    boxes: dict[str, Box]


def parse_number(raw: str) -> Fraction:
    return Fraction(raw.strip())


def format_number(value: Fraction | int) -> str:
    number = value if isinstance(value, Fraction) else Fraction(value)
    if number.denominator == 1:
        return str(number.numerator)
    return f"{number.numerator}/{number.denominator}"


def point_statement(name: str, x: Fraction | int, y: Fraction | int) -> str:
    return f"point {name} = ({format_number(x)}, {format_number(y)})"


def line_statement(name: str, left: str, right: str) -> str:
    return f"line {name} = line({left}, {right})"


def segment_statement(name: str, left: str, right: str) -> str:
    return f"segment {name} = segment({left}, {right})"


def box_statement(
    name: str,
    x_min: Fraction | int,
    y_min: Fraction | int,
    x_max: Fraction | int,
    y_max: Fraction | int,
) -> str:
    return (
        f"box {name} = box({format_number(x_min)}, {format_number(y_min)}, "
        f"{format_number(x_max)}, {format_number(y_max)})"
    )


def parse_program(program: list[str]) -> Scene:
    points: dict[str, Point] = {}
    lines: dict[str, tuple[str, str]] = {}
    segments: dict[str, tuple[str, str]] = {}
    boxes: dict[str, Box] = {}

    for statement in program:
        if match := POINT_RE.match(statement):
            name, x_raw, y_raw = match.groups()
            points[name] = (parse_number(x_raw), parse_number(y_raw))
            continue
        if match := LINE_RE.match(statement):
            kind, name, left, right = match.groups()
            if kind == "line":
                lines[name] = (left, right)
            else:
                segments[name] = (left, right)
            continue
        if match := BOX_RE.match(statement):
            name, x_min, y_min, x_max, y_max = match.groups()
            box = (
                parse_number(x_min),
                parse_number(y_min),
                parse_number(x_max),
                parse_number(y_max),
            )
            if box[0] > box[2] or box[1] > box[3]:
                raise ValueError(f"{name}: invalid box bounds")
            boxes[name] = box
            continue
        raise ValueError(f"unsupported program statement: {statement}")

    return Scene(points=points, lines=lines, segments=segments, boxes=boxes)


def cross(a: Point, b: Point, c: Point) -> Fraction:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def is_collinear(a: Point, b: Point, c: Point) -> bool:
    return cross(a, b, c) == 0


def point_on_segment(point: Point, left: Point, right: Point) -> bool:
    if not is_collinear(left, right, point):
        return False
    return (
        min(left[0], right[0]) <= point[0] <= max(left[0], right[0])
        and min(left[1], right[1]) <= point[1] <= max(left[1], right[1])
    )


def dist2(left: Point, right: Point) -> Fraction:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def classify_inside(point: Point, box: Box) -> str:
    x_min, y_min, x_max, y_max = box
    x, y = point
    if x_min < x < x_max and y_min < y < y_max:
        return "inside"
    if x_min <= x <= x_max and y_min <= y <= y_max:
        return "boundary"
    return "outside"


def box_contains_point(box: Box, point: Point) -> bool:
    return classify_inside(point, box) in {"inside", "boundary"}


def box_contains_box(outer: Box, inner: Box) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def evaluate_relation(
    program: list[str], relation_family: str, query_objects: list[str]
) -> tuple[str, str]:
    scene = parse_program(program)

    if relation_family == "point_identity":
        left, right = query_objects
        label = "same_point" if scene.points[left] == scene.points[right] else "different_point"
        return label, f"{left}={scene.points[left]}; {right}={scene.points[right]}"

    if relation_family == "line_membership":
        point_name, target_name = query_objects
        point = scene.points[point_name]
        if target_name in scene.lines:
            left_name, right_name = scene.lines[target_name]
            on_line = is_collinear(scene.points[left_name], scene.points[right_name], point)
            label = "on_line" if on_line else "off_line"
            return label, f"cross({left_name},{right_name},{point_name})={cross(scene.points[left_name], scene.points[right_name], point)}"
        left_name, right_name = scene.segments[target_name]
        on_segment = point_on_segment(point, scene.points[left_name], scene.points[right_name])
        label = "on_segment" if on_segment else "off_segment"
        return label, f"{point_name} on segment {target_name}={on_segment}"

    if relation_family == "containment":
        outer_name, inner_name = query_objects
        outer = scene.boxes[outer_name]
        if inner_name in scene.points:
            contains = box_contains_point(outer, scene.points[inner_name])
        else:
            contains = box_contains_box(outer, scene.boxes[inner_name])
        label = "contains" if contains else "does_not_contain"
        return label, f"{outer_name} contains {inner_name}={contains}"

    if relation_family == "distance_comparison":
        anchor_name, first_name, second_name = query_objects
        first = dist2(scene.points[anchor_name], scene.points[first_name])
        second = dist2(scene.points[anchor_name], scene.points[second_name])
        if first < second:
            label = "closer_to_first"
        elif second < first:
            label = "closer_to_second"
        else:
            label = "same_distance"
        return label, f"dist2({anchor_name},{first_name})={first}; dist2({anchor_name},{second_name})={second}"

    if relation_family == "horizontal_order":
        left_name, right_name = query_objects
        left_x = scene.points[left_name][0]
        right_x = scene.points[right_name][0]
        if left_x < right_x:
            label = "left_of"
        elif left_x > right_x:
            label = "right_of"
        else:
            label = "same_x"
        return label, f"x({left_name})={left_x}; x({right_name})={right_x}"

    if relation_family == "vertical_order":
        lower_name, upper_name = query_objects
        lower_y = scene.points[lower_name][1]
        upper_y = scene.points[upper_name][1]
        if lower_y < upper_y:
            label = "below"
        elif lower_y > upper_y:
            label = "above"
        else:
            label = "same_y"
        return label, f"y({lower_name})={lower_y}; y({upper_name})={upper_y}"

    if relation_family == "inside":
        point_name, box_name = query_objects
        label = classify_inside(scene.points[point_name], scene.boxes[box_name])
        return label, f"{point_name} in {box_name}={label}"

    if relation_family == "between":
        point_name, left_name, right_name = query_objects
        between = point_on_segment(
            scene.points[point_name], scene.points[left_name], scene.points[right_name]
        )
        label = "between" if between else "not_between"
        return label, f"{point_name} between {left_name},{right_name}={between}"

    raise ValueError(f"unknown relation family: {relation_family}")


def difficulty_for_index(index: int) -> int:
    return (index % 8) + 1


def distractor_count_for_difficulty(rng: random.Random, difficulty: int) -> int:
    if difficulty == 1:
        return 0
    if difficulty == 2:
        return rng.randint(1, 2)
    if difficulty in {3, 4}:
        return 1
    if difficulty in {5, 6}:
        return 2
    if difficulty == 7:
        return rng.randint(3, 5)
    return rng.randint(6, 8)


def family_for_index(index: int) -> str:
    return RELATION_FAMILIES[index % len(RELATION_FAMILIES)]


def name_factory(split: str, index: int) -> dict[str, str]:
    prefix = SPLIT_PREFIXES[split]
    base = f"{prefix}{index + 1:04d}"
    return {
        "a": f"P{base}A",
        "b": f"P{base}B",
        "c": f"P{base}C",
        "d": f"P{base}D",
        "line": f"L{base}",
        "segment": f"S{base}",
        "box": f"B{base}",
        "box2": f"C{base}",
    }


def make_item(
    program: list[str],
    relation_family: str,
    query_objects: list[str],
) -> dict[str, Any]:
    answer, _ = evaluate_relation(program, relation_family, query_objects)
    return {
        "program": program,
        "query": f"What is the relation for {relation_family}({', '.join(query_objects)})?",
        "allowed_labels": RELATION_LABELS[relation_family],
        "answer": answer,
    }


def add_distractors(
    program: list[str],
    rng: random.Random,
    names: dict[str, str],
    count: int,
) -> list[str]:
    output = list(program)
    for offset in range(count):
        if offset % 3 == 0:
            output.append(
                point_statement(
                    f"D{names['a']}{offset}",
                    rng.randint(-4, 8),
                    rng.randint(-4, 8),
                )
            )
        elif offset % 3 == 1:
            left = f"D{names['b']}{offset}"
            right = f"D{names['c']}{offset}"
            output.extend(
                [
                    point_statement(left, rng.randint(-4, 8), rng.randint(-4, 8)),
                    point_statement(right, rng.randint(-4, 8), rng.randint(-4, 8)),
                    line_statement(f"D{names['line']}{offset}", left, right),
                ]
            )
        else:
            x_min = rng.randint(-3, 3)
            y_min = rng.randint(-3, 3)
            output.append(
                box_statement(
                    f"D{names['box']}{offset}",
                    x_min,
                    y_min,
                    x_min + rng.randint(1, 4),
                    y_min + rng.randint(1, 4),
                )
            )
    return output


def base_pair_for_family(
    relation_family: str, names: dict[str, str], variant: int
) -> tuple[list[str], list[str], list[str], int]:
    a = names["a"]
    b = names["b"]
    c = names["c"]
    line = names["line"]
    segment = names["segment"]
    box = names["box"]
    box2 = names["box2"]

    if relation_family == "point_identity":
        canonical = [point_statement(a, 0, 0), point_statement(b, 0, 0)]
        perturbed = [canonical[0], point_statement(b, 1, 0)]
        return canonical, perturbed, [a, b], 1

    if relation_family == "line_membership" and variant % 2 == 0:
        canonical = [
            point_statement(a, 0, 0),
            point_statement(b, 4, 0),
            point_statement(c, 2, 0),
            line_statement(line, a, b),
        ]
        perturbed = [canonical[0], canonical[1], point_statement(c, 2, 1), canonical[3]]
        return canonical, perturbed, [c, line], 2

    if relation_family == "line_membership":
        canonical = [
            point_statement(a, 0, 0),
            point_statement(b, 4, 0),
            point_statement(c, 2, 0),
            segment_statement(segment, a, b),
        ]
        perturbed = [canonical[0], canonical[1], point_statement(c, 5, 0), canonical[3]]
        return canonical, perturbed, [c, segment], 2

    if relation_family == "containment" and variant % 2 == 0:
        canonical = [box_statement(box, 0, 0, 4, 4), point_statement(a, 2, 2)]
        perturbed = [canonical[0], point_statement(a, 5, 2)]
        return canonical, perturbed, [box, a], 1

    if relation_family == "containment":
        canonical = [box_statement(box, 0, 0, 5, 5), box_statement(box2, 1, 1, 3, 3)]
        perturbed = [canonical[0], box_statement(box2, 1, 1, 6, 3)]
        return canonical, perturbed, [box, box2], 1

    if relation_family == "distance_comparison" and variant % 3 == 0:
        canonical = [point_statement(a, 0, 0), point_statement(b, 1, 0), point_statement(c, 4, 0)]
        perturbed = [canonical[0], point_statement(b, 5, 0), canonical[2]]
        return canonical, perturbed, [a, b, c], 1

    if relation_family == "distance_comparison" and variant % 3 == 1:
        canonical = [point_statement(a, 0, 0), point_statement(b, 5, 0), point_statement(c, 2, 0)]
        perturbed = [canonical[0], point_statement(b, 1, 0), canonical[2]]
        return canonical, perturbed, [a, b, c], 1

    if relation_family == "distance_comparison":
        canonical = [point_statement(a, 0, 0), point_statement(b, 2, 0), point_statement(c, 0, 2)]
        perturbed = [canonical[0], canonical[1], point_statement(c, 5, 0)]
        return canonical, perturbed, [a, b, c], 2

    if relation_family == "horizontal_order" and variant % 3 == 0:
        canonical = [point_statement(a, 0, 0), point_statement(b, 2, 0)]
        perturbed = [point_statement(a, 3, 0), canonical[1]]
        return canonical, perturbed, [a, b], 0

    if relation_family == "horizontal_order" and variant % 3 == 1:
        canonical = [point_statement(a, 3, 0), point_statement(b, 2, 0)]
        perturbed = [point_statement(a, 1, 0), canonical[1]]
        return canonical, perturbed, [a, b], 0

    if relation_family == "horizontal_order":
        canonical = [point_statement(a, 2, 0), point_statement(b, 2, 3)]
        perturbed = [point_statement(a, 0, 0), canonical[1]]
        return canonical, perturbed, [a, b], 0

    if relation_family == "vertical_order" and variant % 3 == 0:
        canonical = [point_statement(a, 0, 0), point_statement(b, 0, 2)]
        perturbed = [point_statement(a, 0, 3), canonical[1]]
        return canonical, perturbed, [a, b], 0

    if relation_family == "vertical_order" and variant % 3 == 1:
        canonical = [point_statement(a, 0, 3), point_statement(b, 0, 2)]
        perturbed = [point_statement(a, 0, 1), canonical[1]]
        return canonical, perturbed, [a, b], 0

    if relation_family == "vertical_order":
        canonical = [point_statement(a, 0, 2), point_statement(b, 3, 2)]
        perturbed = [point_statement(a, 0, 0), canonical[1]]
        return canonical, perturbed, [a, b], 0

    if relation_family == "inside" and variant % 3 == 0:
        canonical = [box_statement(box, 0, 0, 4, 4), point_statement(a, 2, 2)]
        perturbed = [canonical[0], point_statement(a, 5, 5)]
        return canonical, perturbed, [a, box], 1

    if relation_family == "inside" and variant % 3 == 1:
        canonical = [box_statement(box, 0, 0, 4, 4), point_statement(a, 0, 2)]
        perturbed = [canonical[0], point_statement(a, 2, 2)]
        return canonical, perturbed, [a, box], 1

    if relation_family == "inside":
        canonical = [box_statement(box, 0, 0, 4, 4), point_statement(a, 5, 5)]
        perturbed = [canonical[0], point_statement(a, 4, 2)]
        return canonical, perturbed, [a, box], 1

    if relation_family == "between" and variant % 3 == 0:
        canonical = [point_statement(a, 0, 0), point_statement(b, 4, 0), point_statement(c, 2, 0)]
        perturbed = [canonical[0], canonical[1], point_statement(c, 5, 0)]
        return canonical, perturbed, [c, a, b], 2

    if relation_family == "between" and variant % 3 == 1:
        canonical = [point_statement(a, 0, 0), point_statement(b, 0, 4), point_statement(c, 0, 2)]
        perturbed = [canonical[0], canonical[1], point_statement(c, 1, 2)]
        return canonical, perturbed, [c, a, b], 2

    canonical = [point_statement(a, 0, 0), point_statement(b, 4, 4), point_statement(c, 2, 2)]
    perturbed = [canonical[0], canonical[1], point_statement(c, 3, 2)]
    return canonical, perturbed, [c, a, b], 2


def generate_record(split: str, index: int) -> dict[str, Any]:
    seed = SPLIT_SEEDS[split] + index
    rng = random.Random(seed)
    difficulty = difficulty_for_index(index)
    relation_family = family_for_index(index)
    names = name_factory(split, index)
    variant = index // len(RELATION_FAMILIES)
    canonical_program, perturbed_program, query_objects, changed_statement_index = base_pair_for_family(
        relation_family, names, variant
    )
    distractor_count = distractor_count_for_difficulty(rng, difficulty)
    distractors = add_distractors([], rng, names, distractor_count)
    canonical_program = [*canonical_program, *distractors]
    perturbed_program = [*perturbed_program, *distractors]

    canonical = make_item(canonical_program, relation_family, query_objects)
    perturbed = make_item(perturbed_program, relation_family, query_objects)
    _, canonical_trace = evaluate_relation(canonical_program, relation_family, query_objects)
    _, perturbed_trace = evaluate_relation(perturbed_program, relation_family, query_objects)

    return {
        "rail_id": "GRR-003",
        "record_id": f"grr003-{split}-{index + 1:06d}",
        "split": split,
        "difficulty": difficulty,
        "relation_family": relation_family,
        "pair_kind": "spatial_label_flip",
        "expected_relation": "label_changes",
        "changed_statement_index": changed_statement_index,
        "distractor_count": distractor_count,
        "canonical": canonical,
        "perturbed": perturbed,
        "query_objects": query_objects,
        "verifier_trace": {
            "canonical": canonical_trace,
            "perturbed": perturbed_trace,
        },
        "generation_seed": seed,
    }


def generate_split(split: str, count: int) -> list[dict[str, Any]]:
    return [generate_record(split, index) for index in range(count)]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            digest.update(line.encode("utf-8"))
            handle.write(line)
    return digest.hexdigest()


def program_signature(record: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(record["canonical"]["program"]), tuple(record["perturbed"]["program"])


def object_names(program: list[str]) -> set[str]:
    scene = parse_program(program)
    return set(scene.points) | set(scene.lines) | set(scene.segments) | set(scene.boxes)


def validate_splits(generated: dict[str, list[dict[str, Any]]]) -> list[str]:
    errors: list[str] = []
    split_objects: dict[str, set[str]] = {}
    signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    for split, records in generated.items():
        objects: set[str] = set()
        family_counts = {family: 0 for family in RELATION_FAMILIES}
        for record in records:
            if record["split"] != split:
                errors.append(f"{record['record_id']}: wrong split {record['split']}")
            family = record["relation_family"]
            family_counts[family] += 1
            canonical_label, _ = evaluate_relation(
                record["canonical"]["program"], family, record["query_objects"]
            )
            perturbed_label, _ = evaluate_relation(
                record["perturbed"]["program"], family, record["query_objects"]
            )
            if canonical_label != record["canonical"]["answer"]:
                errors.append(f"{record['record_id']}: canonical checker disagreement")
            if perturbed_label != record["perturbed"]["answer"]:
                errors.append(f"{record['record_id']}: perturbed checker disagreement")
            if canonical_label == perturbed_label:
                errors.append(f"{record['record_id']}: labels do not change")
            if record["canonical"]["allowed_labels"] != RELATION_LABELS[family]:
                errors.append(f"{record['record_id']}: canonical allowed-label mismatch")
            if record["perturbed"]["allowed_labels"] != RELATION_LABELS[family]:
                errors.append(f"{record['record_id']}: perturbed allowed-label mismatch")
            signature = program_signature(record)
            if signature in signatures:
                errors.append(f"{record['record_id']}: duplicate canonical/perturbed program pair")
            signatures.add(signature)
            objects.update(object_names(record["canonical"]["program"]))
            objects.update(object_names(record["perturbed"]["program"]))

        if records:
            counts = list(family_counts.values())
            if max(counts) - min(counts) > 1:
                errors.append(f"{split}: relation families are not balanced: {family_counts}")
        split_objects[split] = objects

    splits = sorted(generated)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            overlap = split_objects[left] & split_objects[right]
            if overlap:
                errors.append(f"{left}/{right}: object vocabulary overlap: {sorted(overlap)[:3]}")
    return errors


def answer_label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for side in ("canonical", "perturbed"):
            label = record[side]["answer"]
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def relation_family_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        family = record["relation_family"]
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/grr003")
    parser.add_argument("--train-count", type=int, default=112)
    parser.add_argument("--validation-count", type=int, default=24)
    parser.add_argument("--test-count", type=int, default=24)
    args = parser.parse_args()

    split_counts = {
        "train": args.train_count,
        "validation": args.validation_count,
        "test": args.test_count,
    }
    generated = {
        split: generate_split(split, count) for split, count in split_counts.items()
    }
    errors = validate_splits(generated)
    if errors:
        for validation_error in errors:
            print(validation_error)
        return 1

    output_dir = Path(args.output_dir)
    manifest: dict[str, Any] = {
        "rail_id": "GRR-003",
        "generator": "src/generate_grr003.py",
        "split_seeds": SPLIT_SEEDS,
        "split_counts": split_counts,
        "relation_families": RELATION_FAMILIES,
        "relation_labels": RELATION_LABELS,
        "expected_relation": "label_changes",
        "files": {},
    }
    for split, records in generated.items():
        path = output_dir / f"grr003-{split}.jsonl"
        digest = write_jsonl(path, records)
        manifest["files"][split] = {
            "path": str(path),
            "records": len(records),
            "sha256": digest,
            "relation_family_counts": relation_family_counts(records),
            "answer_label_counts": answer_label_counts(records),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
