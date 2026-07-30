"""Alignment header compatible with the commonly used pysam surface."""

from __future__ import annotations

from copy import deepcopy


def _coerce_header_value(tag: str, value: str):
    if tag in {"LN", "PI"}:
        try:
            return int(value)
        except ValueError:
            pass
    return value


class AlignmentHeader:
    def __init__(self, text: str = "", references=(), lengths=()):
        self._text = text
        self._references = tuple(references)
        self._lengths = tuple(int(value) for value in lengths)
        if text and not self._references:
            parsed = self.to_dict()
            sq = parsed.get("SQ", [])
            self._references = tuple(entry["SN"] for entry in sq)
            self._lengths = tuple(int(entry["LN"]) for entry in sq)

    @classmethod
    def from_text(cls, text: str) -> "AlignmentHeader":
        if text and not text.endswith("\n"):
            text += "\n"
        return cls(text)

    @classmethod
    def from_references(cls, reference_names, reference_lengths) -> "AlignmentHeader":
        refs = tuple(reference_names)
        lengths = tuple(reference_lengths)
        if len(refs) != len(lengths):
            raise ValueError("reference_names and reference_lengths must have equal length")
        text = "".join(f"@SQ\tSN:{name}\tLN:{length}\n" for name, length in zip(refs, lengths))
        return cls(text, refs, lengths)

    @classmethod
    def from_dict(cls, header: dict) -> "AlignmentHeader":
        lines: list[str] = []
        for record_type, records in header.items():
            if record_type == "CO":
                for comment in records:
                    lines.append(f"@CO\t{comment}")
                continue
            if isinstance(records, dict):
                records = [records]
            for record in records:
                fields = "\t".join(f"{key}:{value}" for key, value in record.items())
                lines.append(f"@{record_type}\t{fields}")
        return cls.from_text("\n".join(lines))

    @property
    def references(self) -> tuple[str, ...]:
        return self._references

    @property
    def lengths(self) -> tuple[int, ...]:
        return self._lengths

    @property
    def nreferences(self) -> int:
        return len(self._references)

    def get_reference_name(self, tid: int) -> str | None:
        return self._references[tid] if 0 <= tid < len(self._references) else None

    def get_tid(self, reference: str) -> int:
        try:
            return self._references.index(reference)
        except ValueError:
            return -1

    def to_dict(self) -> dict:
        result: dict = {}
        for line in self._text.splitlines():
            if not line.startswith("@") or len(line) < 3:
                continue
            record_type = line[1:3]
            fields = line.split("\t")[1:]
            if record_type == "CO":
                result.setdefault("CO", []).append("\t".join(fields))
                continue
            record = {}
            for field in fields:
                if ":" in field:
                    key, value = field.split(":", 1)
                    record[key] = _coerce_header_value(key, value)
            if record_type == "HD":
                result["HD"] = record
            else:
                result.setdefault(record_type, []).append(record)
        return deepcopy(result)

    def to_string(self) -> str:
        return self._text

    def copy(self) -> "AlignmentHeader":
        return AlignmentHeader(self._text, self._references, self._lengths)

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"<mojopysam.AlignmentHeader with {self.nreferences} references>"

