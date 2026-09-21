from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer

from .dataset import stable_seed
from .hf_config import hf_token
from .schemas import PatchCondition, RankedRelationPassage, RelationExample, RelationRetrieval


RETRIEVER_MODEL = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
READER_MODEL = "deepset/roberta-base-squad2"
VERIFIER_MODEL = "cross-encoder/nli-deberta-v3-base"


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def relation_token_indices(offset_mapping: torch.Tensor, relation_span: tuple[int, int]) -> list[int]:
    """Return non-special tokens overlapping the declared character span."""
    start, end = relation_span
    indices = []
    for index, (token_start, token_end) in enumerate(offset_mapping.tolist()):
        if token_start == token_end == 0:
            continue
        if token_start < end and token_end > start:
            indices.append(index)
    if not indices:
        raise ValueError(f"no tokenizer offsets overlap relation span {relation_span}")
    return indices


def patch_relation_span(
    original: torch.Tensor,
    target_indices: Sequence[int],
    donor_state: torch.Tensor,
    donor_indices: Sequence[int],
    *,
    self_patch: bool,
) -> tuple[torch.Tensor, str]:
    """Replace only target relation-token states and preserve all other tokens."""
    patched = original.clone()
    if self_patch:
        replacement = original[0, list(target_indices)].clone()
        mode = "tokenwise-self"
    elif len(target_indices) == len(donor_indices):
        replacement = donor_state
        mode = "tokenwise-donor"
    else:
        replacement = donor_state.mean(dim=0, keepdim=True).expand(len(target_indices), -1)
        mode = "pooled-donor"
    patched[0, list(target_indices)] = replacement.to(patched)
    return patched, mode


def donor_for_condition(
    target: RelationExample,
    examples_by_id: dict[str, RelationExample],
    split_examples: Sequence[RelationExample],
    condition: PatchCondition,
) -> RelationExample | None:
    if condition == PatchCondition.BASELINE:
        return None
    if condition == PatchCondition.SELF_PATCH:
        return target
    if condition == PatchCondition.SAME_ENTITY_DONOR:
        donor_relation = next(
            passage.evidence_relation_id for passage in target.passages if passage.id == target.donor_passage_id
        )
        return examples_by_id[target.paired_example_ids[donor_relation]]
    if condition == PatchCondition.SAME_ENTITY_CONTROL:
        control_relation = next(
            passage.evidence_relation_id for passage in target.passages if passage.id == target.control_passage_id
        )
        return examples_by_id[target.paired_example_ids[control_relation]]
    if condition == PatchCondition.DIFFERENT_ENTITY_DONOR:
        donor_relation = next(
            passage.evidence_relation_id for passage in target.passages if passage.id == target.donor_passage_id
        )
        candidates = sorted(
            (
                example
                for example in split_examples
                if example.entity_id != target.entity_id and example.relation_id == donor_relation
            ),
            key=lambda example: example.id,
        )
        if not candidates:
            raise ValueError(f"no different-entity donor for {target.id}")
        return candidates[stable_seed(target.id, condition.value) % len(candidates)]
    raise ValueError(f"unsupported condition {condition}")


class DenseRelationRetriever:
    """Question-side relation-span patching with untouched document encodings."""

    def __init__(self, model_name: str = RETRIEVER_MODEL, *, device: str | None = None, cache_size: int = 20_000):
        token = hf_token()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=True)
        if not self.tokenizer.is_fast:
            raise TypeError("relation-span patching requires a fast tokenizer with offset mappings")
        self.model = AutoModel.from_pretrained(model_name, token=token)
        if device:
            self.model.to(device)
        self.model.eval()
        self.cache_size = cache_size
        self._document_cache: dict[str, torch.Tensor] = {}

    def available_layers(self) -> int:
        base = getattr(self.model, "base_model", self.model)
        layers = getattr(getattr(base, "encoder", None), "layer", None)
        if layers is None:
            raise TypeError("retriever does not expose encoder.layer")
        return len(layers)

    def _layer_modules(self):
        base = getattr(self.model, "base_model", self.model)
        layers = getattr(getattr(base, "encoder", None), "layer", None)
        if layers is None:
            raise TypeError("retriever does not expose encoder.layer")
        return layers

    def _tokenize(self, texts: Sequence[str], *, offsets: bool = False):
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=256,
            return_offsets_mapping=offsets,
            return_tensors="pt",
        )
        offset_mapping = inputs.pop("offset_mapping", None)
        return inputs.to(_model_device(self.model)), offset_mapping

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weighted = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        return weighted.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def _encode_documents(self, passages: Iterable) -> torch.Tensor:
        passages = list(passages)
        missing = [passage for passage in passages if passage.id not in self._document_cache]
        if missing:
            inputs, _ = self._tokenize([passage.text for passage in missing])
            with torch.inference_mode():
                outputs = self.model(**inputs)
            vectors = functional.normalize(self._mean_pool(outputs.last_hidden_state, inputs.attention_mask), p=2, dim=1)
            for passage, vector in zip(missing, vectors.detach().cpu(), strict=True):
                self._document_cache[passage.id] = vector
            while len(self._document_cache) > self.cache_size:
                self._document_cache.pop(next(iter(self._document_cache)))
        return torch.stack([self._document_cache[passage.id] for passage in passages]).to(_model_device(self.model))

    def _span_state(self, example: RelationExample, layer: int) -> tuple[torch.Tensor, list[int]]:
        inputs, offsets = self._tokenize([example.question], offsets=True)
        if offsets is None:
            raise RuntimeError("fast tokenizer failed to return offset mappings")
        indices = relation_token_indices(offsets[0], example.relation_span)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer][0, indices].detach(), indices

    def _query_vector(
        self,
        target: RelationExample,
        *,
        condition: PatchCondition,
        layer: int | None,
        donor: RelationExample | None,
    ) -> tuple[torch.Tensor, list[int], str]:
        if condition == PatchCondition.BASELINE:
            inputs, _ = self._tokenize([target.question])
            with torch.inference_mode():
                outputs = self.model(**inputs)
            return functional.normalize(self._mean_pool(outputs.last_hidden_state, inputs.attention_mask), p=2, dim=1), [], "none"
        if layer is None or not 1 <= layer <= self.available_layers():
            raise ValueError(f"layer must be between 1 and {self.available_layers()}")
        if donor is None:
            raise ValueError("patch condition requires a donor example")

        donor_state, donor_indices = self._span_state(donor, layer)
        inputs, offsets = self._tokenize([target.question], offsets=True)
        if offsets is None:
            raise RuntimeError("fast tokenizer failed to return offset mappings")
        target_indices = relation_token_indices(offsets[0], target.relation_span)
        modules = self._layer_modules()

        def hook(_module, _inputs, output):
            original = output[0] if isinstance(output, tuple) else output
            patched, mode = patch_relation_span(
                original,
                target_indices,
                donor_state,
                donor_indices,
                self_patch=condition == PatchCondition.SELF_PATCH,
            )
            hook.mode = mode
            return (patched, *output[1:]) if isinstance(output, tuple) else patched

        hook.mode = "unknown"
        handle = modules[layer - 1].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                outputs = self.model(**inputs)
        finally:
            handle.remove()
        vector = functional.normalize(self._mean_pool(outputs.last_hidden_state, inputs.attention_mask), p=2, dim=1)
        return vector, target_indices, hook.mode

    def retrieve(
        self,
        target: RelationExample,
        *,
        condition: PatchCondition = PatchCondition.BASELINE,
        layer: int | None = None,
        donor: RelationExample | None = None,
    ) -> RelationRetrieval:
        query, token_indices, patch_mode = self._query_vector(target, condition=condition, layer=layer, donor=donor)
        documents = self._encode_documents(target.passages)
        scores = (query @ documents.T)[0].detach().cpu().tolist()
        ranked = sorted(zip(target.passages, scores, strict=True), key=lambda item: (-item[1], item[0].id))
        passages = [
            RankedRelationPassage(**passage.model_dump(), score=float(score), rank=index)
            for index, (passage, score) in enumerate(ranked, start=1)
        ]
        return RelationRetrieval(
            example_id=target.id,
            condition=condition,
            layer=layer,
            donor_example_id=donor.id if donor else None,
            donor_relation_id=donor.relation_id if donor else None,
            patch_token_indices=token_indices,
            patch_mode=patch_mode,
            passages=passages,
        )


class ExtractiveReader:
    def __init__(self, model_name: str = READER_MODEL, *, device: str | None = None):
        token = hf_token()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, token=token)
        if device:
            self.model.to(device)
        self.model.eval()

    def answer(self, question: str, passage: str) -> tuple[str, float]:
        inputs = self.tokenizer(question, passage, return_tensors="pt", truncation=True, max_length=384).to(_model_device(self.model))
        with torch.inference_mode():
            outputs = self.model(**inputs)
        start = int(outputs.start_logits.argmax(dim=-1).item())
        end = int(outputs.end_logits.argmax(dim=-1).item())
        if end < start:
            end = start
        token_ids = inputs.input_ids[0, start : end + 1]
        answer = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        confidence = float((outputs.start_logits[0, start] + outputs.end_logits[0, end]).sigmoid().item() / 2)
        return answer or "NO_ANSWER", confidence


class EvidenceVerifier:
    def __init__(self, model_path: str, *, device: str | None = None):
        token = hf_token()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, token=token)
        if device:
            self.model.to(device)
        self.model.eval()

    def support_probability(self, question: str, passage: str) -> float:
        inputs = self.tokenizer(question, passage, return_tensors="pt", truncation=True, max_length=384).to(_model_device(self.model))
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        probabilities = logits.softmax(dim=-1)[0]
        return float(probabilities[1].item())
