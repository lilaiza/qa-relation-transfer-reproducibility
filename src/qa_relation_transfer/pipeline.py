from __future__ import annotations

from .agents import DenseRelationRetriever, EvidenceVerifier, ExtractiveReader
from .schemas import PatchCondition, RelationExample


class RelationQAPipeline:
    def __init__(self, retriever: DenseRelationRetriever, reader: ExtractiveReader | None = None, verifier: EvidenceVerifier | None = None):
        self.retriever = retriever
        self.reader = reader
        self.verifier = verifier

    def run(self, target: RelationExample, *, condition: PatchCondition, layer: int | None, donor: RelationExample | None) -> dict:
        retrieval = self.retriever.retrieve(target, condition=condition, layer=layer, donor=donor)
        top_passage = retrieval.passages[0]
        reader_answer, reader_confidence = (None, None)
        if self.reader is not None:
            reader_answer, reader_confidence = self.reader.answer(target.question, top_passage.text)
        verifier_probability = None
        if self.verifier is not None:
            verifier_probability = self.verifier.support_probability(target.question, top_passage.text)
        return {
            "example_id": target.id,
            "entity_id": target.entity_id,
            "relation_id": target.relation_id,
            "target_passage_id": target.target_passage_id,
            "donor_passage_id": target.donor_passage_id,
            "control_passage_id": target.control_passage_id,
            "condition": condition.value,
            "layer": layer,
            "donor_example_id": donor.id if donor else None,
            "retrieval": retrieval.model_dump(mode="json"),
            "top_passage_id": top_passage.id,
            "top_passage_relation_id": top_passage.evidence_relation_id,
            "reader_answer": reader_answer,
            "reader_confidence": reader_confidence,
            "verifier_support_probability": verifier_probability,
        }
