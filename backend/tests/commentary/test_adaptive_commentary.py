from app.commentary.adaptive import (
    build_adaptation_policy,
    build_level_comparison,
    commentary_metrics,
    resolve_audience_profile,
    validate_level_progression,
)


TACTICAL_LABELS = {
    "formation": "4-3-3",
    "formation_approx": "4-3-3",
    "team_shape": "Balanced Shape",
    "attacking_structure": "Balanced Structure",
    "defensive_block": "Mid Block",
    "defensive_shape": "Compact Balanced Mid Block",
}


def test_resolve_audience_profile_prefers_explicit_level():
    profile = resolve_audience_profile(
        level="Expert",
        profile={"verbosity": "low", "educational_mode": True, "style": "coach"},
    )

    assert profile.level == "Expert"
    assert profile.verbosity == "low"
    assert profile.educational_mode is True
    assert profile.style == "coach"
    assert profile.source == "explicit"


def test_resolve_audience_profile_can_infer_beginner_from_educational_mode():
    profile = resolve_audience_profile(profile={"educational_mode": True})
    assert profile.level == "Beginner"
    assert profile.source == "educational_mode"


def test_resolve_audience_profile_can_infer_expert_from_style_and_verbosity():
    profile = resolve_audience_profile(signals={"style": "analytical", "verbosity": "low"})
    assert profile.level == "Expert"
    assert profile.source == "style_and_verbosity"


def test_build_adaptation_policy_changes_by_level():
    beginner_policy = build_adaptation_policy(resolve_audience_profile(level="Beginner"))
    expert_policy = build_adaptation_policy(resolve_audience_profile(level="Expert", profile={"verbosity": "low"}))

    assert beginner_policy.explain_terms is True
    assert beginner_policy.include_educational_hints is True
    assert expert_policy.allow_dense_jargon is True
    assert expert_policy.concise_analytical_phrasing is True
    assert expert_policy.max_sentences <= beginner_policy.max_sentences


def test_build_level_comparison_produces_meaningfully_different_outputs():
    comparison = build_level_comparison(
        team_name="Blue FC",
        tactical_labels=TACTICAL_LABELS,
        opposition_effect="The opposition protect the middle.",
        support_context="A nearby passing option is available.",
        sequence_summary="Event type: Pass",
    )

    assert comparison["Beginner"]["text"] != comparison["Intermediate"]["text"]
    assert comparison["Intermediate"]["text"] != comparison["Expert"]["text"]
    assert comparison["Beginner"]["metrics"]["word_count"] >= comparison["Expert"]["metrics"]["word_count"]
    assert comparison["Expert"]["metrics"]["jargon_count"] >= comparison["Beginner"]["metrics"]["jargon_count"]


def test_validate_level_progression_passes_for_default_comparison():
    comparison = build_level_comparison(
        team_name="Blue FC",
        tactical_labels=TACTICAL_LABELS,
        opposition_effect="The opposition protect the middle.",
        support_context="A nearby passing option is available.",
        sequence_summary="Event type: Pass",
    )

    validation = validate_level_progression(comparison)
    assert validation["all_passed"] is True


def test_commentary_metrics_capture_explanations():
    metrics = commentary_metrics(
        "Blue FC are in a balanced shape. In simple terms, that means the team keeps support on both sides."
    )
    assert metrics["sentence_count"] == 2
    assert metrics["explanation_count"] >= 1
