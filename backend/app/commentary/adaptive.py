from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


COMMENTARY_LEVELS = ("Beginner", "Intermediate", "Expert")
COMMENTARY_VERBOSITY = ("low", "medium", "high")
COMMENTARY_STYLES = ("neutral", "friendly", "analytical", "coach")

TACTICAL_GLOSSARY = {
    "mid block": "a medium defensive line that protects space without pressing too high",
    "low block": "a deep defensive line close to goal",
    "high press": "pressure high up the pitch to force mistakes quickly",
    "wide structure": "players spread the pitch to stretch the defence",
    "central overload": "extra players gather in the middle to outnumber opponents",
    "vertical support structure": "teammates position themselves in forward passing lanes",
    "balanced structure": "the team keeps support on both sides of the ball",
    "compact shape": "players stay close together to reduce space",
    "stretched shape": "the team is spread out over a longer distance",
    "vertical shape": "players are stacked in deeper and higher lanes",
    "balanced shape": "the team keeps even spacing across the pitch",
}
TACTICAL_JARGON = {
    "overload",
    "press",
    "pressing",
    "compact",
    "block",
    "rest defense",
    "trigger",
    "width",
    "vertical",
    "transition",
    "lane",
    "shape",
    "overload",
}


def normalize_commentary_level(value: str | None, default: str = "Intermediate") -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        for candidate in COMMENTARY_LEVELS:
            if lowered == candidate.lower():
                return candidate
    return default


def normalize_commentary_verbosity(value: str | None, default: str = "medium") -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in COMMENTARY_VERBOSITY:
            return lowered
    return default


def normalize_commentary_style(value: str | None, default: str = "neutral") -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in COMMENTARY_STYLES:
            return lowered
    return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


@dataclass(slots=True)
class AudienceProfile:
    level: str = "Intermediate"
    verbosity: str = "medium"
    educational_mode: bool = False
    style: str = "neutral"
    source: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CommentaryAdaptationPolicy:
    level: str
    verbosity: str
    max_sentences: int
    target_sentence_style: str
    tactical_depth: str
    assume_knowledge: str
    explain_terms: bool
    include_educational_hints: bool
    allow_dense_jargon: bool
    concise_analytical_phrasing: bool
    include_extra_context: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_audience_level(signals: Mapping[str, Any] | None = None) -> tuple[str, str]:
    signals = dict(signals or {})
    educational_mode = _coerce_bool(signals.get("educational_mode"), default=False)
    verbosity = normalize_commentary_verbosity(signals.get("verbosity"))
    style = normalize_commentary_style(signals.get("style"))
    football_knowledge = str(signals.get("football_knowledge", "") or "").strip().lower()

    if educational_mode:
        return "Beginner", "educational_mode"
    if football_knowledge in {"expert", "advanced", "high"}:
        return "Expert", "football_knowledge"
    if style in {"analytical", "coach"} and verbosity in {"low", "medium"}:
        return "Expert", "style_and_verbosity"
    if football_knowledge in {"beginner", "novice", "low"}:
        return "Beginner", "football_knowledge"
    if style == "friendly" and verbosity == "high":
        return "Beginner", "style_and_verbosity"
    return "Intermediate", "default"


def resolve_audience_profile(
    *,
    level: str | None = None,
    profile: AudienceProfile | Mapping[str, Any] | None = None,
    signals: Mapping[str, Any] | None = None,
) -> AudienceProfile:
    if isinstance(profile, AudienceProfile):
        base = profile.to_dict()
    else:
        base = dict(profile or {})

    explicit_level = level or base.get("level") or base.get("preferred_commentary_level") or base.get("commentary_level")
    if explicit_level:
        resolved_level = normalize_commentary_level(str(explicit_level))
        source = "explicit"
    else:
        inference_level, source = infer_audience_level({**base, **dict(signals or {})})
        resolved_level = inference_level

    verbosity = normalize_commentary_verbosity(base.get("verbosity") or base.get("commentary_verbosity"))
    educational_mode = _coerce_bool(base.get("educational_mode"), default=False)
    style = normalize_commentary_style(base.get("style") or base.get("commentary_style"))

    if signals:
        verbosity = normalize_commentary_verbosity(signals.get("verbosity"), default=verbosity)
        educational_mode = _coerce_bool(signals.get("educational_mode"), default=educational_mode)
        style = normalize_commentary_style(signals.get("style"), default=style)

    return AudienceProfile(
        level=resolved_level,
        verbosity=verbosity,
        educational_mode=educational_mode,
        style=style,
        source=source,
    )


def build_adaptation_policy(profile: AudienceProfile) -> CommentaryAdaptationPolicy:
    sentence_count = {"low": 2, "medium": 3, "high": 4}[profile.verbosity]
    if profile.level == "Beginner":
        return CommentaryAdaptationPolicy(
            level=profile.level,
            verbosity=profile.verbosity,
            max_sentences=max(sentence_count, 3),
            target_sentence_style="short and clear",
            tactical_depth="light",
            assume_knowledge="low",
            explain_terms=True,
            include_educational_hints=True or profile.educational_mode,
            allow_dense_jargon=False,
            concise_analytical_phrasing=False,
            include_extra_context=True,
        )
    if profile.level == "Expert":
        return CommentaryAdaptationPolicy(
            level=profile.level,
            verbosity=profile.verbosity,
            max_sentences=min(sentence_count, 3),
            target_sentence_style="compact and analytical",
            tactical_depth="high",
            assume_knowledge="high",
            explain_terms=False,
            include_educational_hints=False,
            allow_dense_jargon=True,
            concise_analytical_phrasing=True,
            include_extra_context=profile.verbosity != "low",
        )
    return CommentaryAdaptationPolicy(
        level=profile.level,
        verbosity=profile.verbosity,
        max_sentences=sentence_count,
        target_sentence_style="balanced",
        tactical_depth="medium",
        assume_knowledge="moderate",
        explain_terms=profile.educational_mode,
        include_educational_hints=profile.educational_mode,
        allow_dense_jargon=False,
        concise_analytical_phrasing=False,
        include_extra_context=True,
    )


def build_llm_adaptation_instructions(profile: AudienceProfile, policy: CommentaryAdaptationPolicy) -> str:
    shared = [
        f"Audience level: {profile.level}",
        f"Sentence style: {policy.target_sentence_style}",
        f"Tactical depth: {policy.tactical_depth}",
        f"Assumed football knowledge: {policy.assume_knowledge}",
    ]
    if profile.level == "Beginner":
        shared.extend(
            [
                "Explain tactical ideas in simple language.",
                "If you use a tactical term, immediately explain what it means in plain words.",
                "Focus on what is happening and why it matters.",
            ]
        )
    elif profile.level == "Expert":
        shared.extend(
            [
                "Be concise and analysis-heavy.",
                "Use formation names, compactness, overloads, pressing language, and transition terms naturally.",
                "Avoid over-explaining basic football concepts.",
            ]
        )
    else:
        shared.extend(
            [
                "Balance clarity with tactical detail.",
                "Use some football terms, but keep the explanation readable.",
            ]
        )
    if profile.educational_mode:
        shared.append("Educational mode is on: include one helpful teaching-style explanation when it fits naturally.")
    return "\n".join(f"- {line}" for line in shared)


def _normalize_label(label: Any, fallback: str) -> str:
    text = str(label or "").strip()
    return text if text else fallback


def _formation_clause(formation: str) -> str:
    lowered = formation.lower()
    if lowered in {"unknown", "unclear"}:
        return "without a clearly defined base shape"
    article = "an" if lowered[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"in {article} {lowered}"


def _attacking_clause(attacking_structure: str, level: str) -> str:
    mapping = {
        "wide structure": "using width in the attack",
        "central overload": "through a central overload",
        "vertical support structure": "through vertical support",
        "balanced structure": "through a balanced attacking pattern",
    }
    beginner_mapping = {
        "wide structure": "with players spread wide",
        "central overload": "with extra support through the middle",
        "vertical support structure": "with forward passing options ahead of the ball",
        "balanced structure": "with support on both sides of the ball",
    }
    lowered = attacking_structure.lower()
    if lowered in {"unknown", "unclear"}:
        return "without a clearly defined attacking pattern"
    if level == "Beginner":
        return beginner_mapping.get(lowered, f"with a clear attacking pattern")
    return mapping.get(lowered, f"through {lowered}")


def _shape_clause(team_shape: str) -> str:
    lowered = team_shape.lower()
    if lowered in {"unknown", "unclear"}:
        return "an unclear overall shape"
    article = "an" if lowered[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {lowered}"


def _defensive_clause(defensive_shape: str) -> str:
    lowered = defensive_shape.lower()
    if lowered in {"unknown", "unclear"}:
        return "with an unclear defensive shape"
    article = "an" if lowered[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"in {article} {lowered}"


def _beginner_explanation(tactical_labels: Mapping[str, Any]) -> str | None:
    attacking_structure = _normalize_label(tactical_labels.get("attacking_structure"), "Unknown").lower()
    defensive_block = _normalize_label(tactical_labels.get("defensive_block"), "Unknown").lower()
    if attacking_structure in TACTICAL_GLOSSARY:
        return f"In simple terms, that means {TACTICAL_GLOSSARY[attacking_structure]}."
    if defensive_block in TACTICAL_GLOSSARY:
        return f"In simple terms, that means {TACTICAL_GLOSSARY[defensive_block]}."
    return None


def build_adaptive_commentary_fallback(
    *,
    team_name: str | None,
    tactical_labels: Mapping[str, Any] | None,
    audience_profile: AudienceProfile,
    opposition_effect: str | None = None,
    support_context: str | None = None,
    sequence_summary: str | None = None,
) -> str:
    labels = dict(tactical_labels or {})
    policy = build_adaptation_policy(audience_profile)

    team_name = team_name or "The possession side"
    formation = _normalize_label(labels.get("formation_approx") or labels.get("formation"), "Unclear")
    team_shape = _normalize_label(labels.get("team_shape"), "Unknown")
    attacking_structure = _normalize_label(labels.get("attacking_structure"), "Unknown")
    defensive_shape = _normalize_label(labels.get("defensive_shape"), "Unknown")

    formation_clause = _formation_clause(formation)
    attacking_clause = _attacking_clause(attacking_structure, audience_profile.level)
    shape_clause = _shape_clause(team_shape)
    defensive_clause = _defensive_clause(defensive_shape)

    sentences: list[str] = []
    if sequence_summary and policy.include_extra_context:
        sentences.append(sequence_summary)

    if audience_profile.level == "Beginner":
        sentences.append(
            f"{team_name} are set {formation_clause}, {attacking_clause}, and keep {shape_clause}."
        )
        explanation = _beginner_explanation(labels)
        if explanation:
            sentences.append(explanation)
        sentences.append(f"The opposition defend {defensive_clause}.")
    elif audience_profile.level == "Expert":
        sentences.append(
            f"{team_name} settle {formation_clause}, {attacking_clause}, and maintain {shape_clause}."
        )
        sentences.append(f"The opposition defend {defensive_clause}.")
    else:
        sentences.append(
            f"{team_name} are set {formation_clause}, {attacking_clause}, and keep {shape_clause}."
        )
        sentences.append(f"The opposition defend {defensive_clause}.")

    if opposition_effect:
        sentences.append(opposition_effect)
    if support_context and policy.include_extra_context:
        sentences.append(support_context)

    return " ".join(sentences[: policy.max_sentences + 2]).strip()


def commentary_metrics(text: str) -> dict[str, int | float]:
    stripped = (text or "").strip()
    words = re.findall(r"[A-Za-z0-9'-]+", stripped)
    sentences = [segment.strip() for segment in re.split(r"[.!?]+", stripped) if segment.strip()]
    lowered = stripped.lower()
    jargon_count = sum(lowered.count(term) for term in TACTICAL_JARGON)
    explanation_count = lowered.count("that means") + lowered.count("in simple terms")
    avg_sentence_length = (len(words) / len(sentences)) if sentences else 0.0
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "jargon_count": jargon_count,
        "explanation_count": explanation_count,
    }


def build_level_comparison(
    *,
    team_name: str | None,
    tactical_labels: Mapping[str, Any] | None,
    opposition_effect: str | None = None,
    support_context: str | None = None,
    sequence_summary: str | None = None,
    base_profile: AudienceProfile | Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    for level in COMMENTARY_LEVELS:
        profile = resolve_audience_profile(level=level, profile=base_profile)
        text = build_adaptive_commentary_fallback(
            team_name=team_name,
            tactical_labels=tactical_labels,
            audience_profile=profile,
            opposition_effect=opposition_effect,
            support_context=support_context,
            sequence_summary=sequence_summary,
        )
        comparison[level] = {
            "profile": profile.to_dict(),
            "policy": build_adaptation_policy(profile).to_dict(),
            "text": text,
            "metrics": commentary_metrics(text),
        }
    return comparison


def validate_level_progression(comparison: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    beginner = comparison.get("Beginner", {})
    intermediate = comparison.get("Intermediate", {})
    expert = comparison.get("Expert", {})

    beginner_metrics = beginner.get("metrics", {})
    expert_metrics = expert.get("metrics", {})
    intermediate_metrics = intermediate.get("metrics", {})

    checks = {
        "distinct_texts": len({comparison[level]["text"] for level in COMMENTARY_LEVELS if level in comparison}) == 3,
        "expert_not_longer_than_beginner": expert_metrics.get("word_count", 0) <= beginner_metrics.get("word_count", 0),
        "beginner_has_more_explanations": beginner_metrics.get("explanation_count", 0) >= expert_metrics.get("explanation_count", 0),
        "expert_has_at_least_as_much_jargon": expert_metrics.get("jargon_count", 0) >= beginner_metrics.get("jargon_count", 0),
        "intermediate_between_levels": intermediate_metrics.get("word_count", 0) <= beginner_metrics.get("word_count", 0) + 10,
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def comparison_to_json(comparison: Mapping[str, Mapping[str, Any]]) -> str:
    return json.dumps(comparison, indent=2)
