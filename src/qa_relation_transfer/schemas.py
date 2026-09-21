from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Split(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class PassageRole(StrEnum):
    TARGET = "target"
    DONOR = "donor"
    CONTROL = "control"


class PatchCondition(StrEnum):
    BASELINE = "baseline"
    SELF_PATCH = "self_patch"
    SAME_ENTITY_DONOR = "same_entity_donor"
    SAME_ENTITY_CONTROL = "same_entity_control"
    DIFFERENT_ENTITY_DONOR = "different_entity_donor"


class RelationPassage(BaseModel):
    id: str
    text: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    evidence_relation_id: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    role: PassageRole
    source_id: str | None = None
    source_text: str | None = None
    evidence_mode: str = "raw"


class RelationExample(BaseModel):
    id: str
    entity_id: str = Field(min_length=1)
    entity_label: str = Field(min_length=1)
    relation_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    relation_span: tuple[int, int]
    answer: str = Field(min_length=1)
    target_passage_id: str
    donor_passage_id: str
    control_passage_id: str
    paired_example_ids: dict[str, str] = Field(min_length=2)
    passages: list[RelationPassage] = Field(min_length=6, max_length=6)
    split: Split

    @model_validator(mode="after")
    def validate_example(self):
        start, end = self.relation_span
        if not 0 <= start < end <= len(self.question):
            raise ValueError("relation_span must delimit text in question")
        ids = {passage.id for passage in self.passages}
        required = {self.target_passage_id, self.donor_passage_id, self.control_passage_id}
        if not required.issubset(ids):
            raise ValueError("target, donor, and control passages must be candidates")
        target = next(passage for passage in self.passages if passage.id == self.target_passage_id)
        donor = next(passage for passage in self.passages if passage.id == self.donor_passage_id)
        control = next(passage for passage in self.passages if passage.id == self.control_passage_id)
        if target.role != PassageRole.TARGET or target.evidence_relation_id != self.relation_id:
            raise ValueError("target passage must support the target relation")
        if donor.role != PassageRole.DONOR or donor.entity_id != self.entity_id:
            raise ValueError("donor passage must be a same-entity relation alternative")
        if control.role != PassageRole.CONTROL or control.entity_id != self.entity_id:
            raise ValueError("control passage must be a same-entity relation alternative")
        return self


class RankedRelationPassage(RelationPassage):
    score: float
    rank: int = Field(ge=1)


class RelationRetrieval(BaseModel):
    example_id: str
    condition: PatchCondition
    layer: int | None = None
    donor_example_id: str | None = None
    donor_relation_id: str | None = None
    patch_token_indices: list[int] = Field(default_factory=list)
    patch_mode: str = "none"
    passages: list[RankedRelationPassage]

    @property
    def score_by_id(self) -> dict[str, float]:
        return {passage.id: passage.score for passage in self.passages}
