from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

import compute_associativity


GEO_AUGMENTATION_THRESHOLD = 0.7


@dataclass(frozen=True)
class GeoAugmentationCandidate:
    source_table: str
    source_attr: str
    target_table: str
    target_attr: str
    geo_attr: str
    source_score: float
    target_join_score: float
    target_score: float
    threshold: float
    dropped_rows: int

    @property
    def eligible(self) -> bool:
        return self.target_score >= self.threshold


def evaluate_geo_augmentation(
    *,
    source_table: str,
    source_df: pd.DataFrame,
    source_attr: str,
    target_table: str,
    target_df: pd.DataFrame,
    target_attr: str,
    geo_columns: set[str] | list[str],
    source_geo_association_details: dict[str, dict[str, object]] | None = None,
    threshold: float = GEO_AUGMENTATION_THRESHOLD,
) -> list[GeoAugmentationCandidate]:
    candidates: list[GeoAugmentationCandidate] = []
    unique_geo_columns = sorted(dict.fromkeys(str(column) for column in geo_columns))
    if not source_attr or not target_attr:
        return candidates

    target_primary_key = compute_associativity.infer_primary_key(target_df)
    for geo_attr in unique_geo_columns:
        source_score = compute_associativity.geo_associativity_score(
            source_df,
            source_attr,
            geo_attr,
            geo_association_details=source_geo_association_details,
        )
        target_join_score = source_score
        dropped_rows = 0
        if target_primary_key and target_primary_key == target_attr:
            target_score = target_join_score
        else:
            entropy_result = compute_associativity.normalized_entropy_score(
                target_df,
                target_primary_key or "",
                target_attr,
            )
            dropped_rows = entropy_result.dropped_rows
            target_score = round(target_join_score * entropy_result.score, 3)
        candidates.append(
            GeoAugmentationCandidate(
                source_table=source_table,
                source_attr=source_attr,
                target_table=target_table,
                target_attr=target_attr,
                geo_attr=geo_attr,
                source_score=source_score,
                target_join_score=target_join_score,
                target_score=target_score,
                threshold=threshold,
                dropped_rows=dropped_rows,
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.target_score,
            candidate.source_score,
            candidate.geo_attr,
        ),
        reverse=True,
    )


def best_geo_augmentation(
    **kwargs,
) -> GeoAugmentationCandidate | None:
    candidates = evaluate_geo_augmentation(**kwargs)
    return candidates[0] if candidates else None


def serialize_candidate(candidate: GeoAugmentationCandidate) -> dict[str, object]:
    payload = asdict(candidate)
    payload["eligible"] = candidate.eligible
    return payload
