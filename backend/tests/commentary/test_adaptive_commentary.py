from app.commentary.adaptive import (
    AUDIENCE_MODEL_FEATURES,
    build_adaptation_policy,
    build_audience_feature_vector,
    build_level_comparison,
    commentary_metrics,
    infer_audience_level,
    infer_audience_level_rules,
    predict_audience_level_from_model,
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
    assert profile.source in {"educational_mode", "learned_model"}


def test_resolve_audience_profile_can_infer_expert_from_style_and_verbosity():
    profile = resolve_audience_profile(signals={"style": "analytical", "verbosity": "low"})
    assert profile.level == "Expert"
    assert profile.source in {"style_and_verbosity", "learned_model"}


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


def test_build_audience_feature_vector_encodes_expected_dimensions():
    vector, normalized = build_audience_feature_vector(
        {"educational_mode": True, "verbosity": "high", "style": "friendly", "football_knowledge": "beginner"}
    )
    assert len(vector) == len(AUDIENCE_MODEL_FEATURES)
    assert normalized["educational_mode"] is True
    assert normalized["verbosity"] == "high"
    assert normalized["style"] == "friendly"
    assert normalized["football_knowledge"] == "low"


def test_predict_audience_level_from_model_uses_bundle_coefficients():
    feature_count = len(AUDIENCE_MODEL_FEATURES)
    bundle = {
        "feature_names": list(AUDIENCE_MODEL_FEATURES),
        "classes": ["Beginner", "Intermediate", "Expert"],
        "coefficients": [
            [4.0 if name in {"educational_mode", "verbosity_high", "style_friendly", "knowledge_low"} else 0.0 for name in AUDIENCE_MODEL_FEATURES],
            [0.1] * feature_count,
            [4.0 if name in {"verbosity_low", "style_analytical", "knowledge_high"} else 0.0 for name in AUDIENCE_MODEL_FEATURES],
        ],
        "intercepts": [0.0, 0.0, 0.0],
        "min_confidence": 0.5,
    }

    beginner_prediction = predict_audience_level_from_model(
        {"educational_mode": True, "verbosity": "high", "style": "friendly", "football_knowledge": "beginner"},
        model_bundle=bundle,
    )
    expert_prediction = predict_audience_level_from_model(
        {"verbosity": "low", "style": "analytical", "football_knowledge": "expert"},
        model_bundle=bundle,
    )

    assert beginner_prediction is not None
    assert expert_prediction is not None
    assert beginner_prediction["level"] == "Beginner"
    assert expert_prediction["level"] == "Expert"
    assert beginner_prediction["accepted"] is True
    assert expert_prediction["accepted"] is True


def test_infer_audience_level_falls_back_to_rules_when_model_is_uncertain():
    feature_count = len(AUDIENCE_MODEL_FEATURES)
    uncertain_bundle = {
        "feature_names": list(AUDIENCE_MODEL_FEATURES),
        "classes": ["Beginner", "Intermediate", "Expert"],
        "coefficients": [[0.0] * feature_count, [0.0] * feature_count, [0.0] * feature_count],
        "intercepts": [0.0, 0.0, 0.0],
        "min_confidence": 0.95,
    }

    learned_level, learned_source = infer_audience_level(
        {"educational_mode": True},
        model_bundle=uncertain_bundle,
    )
    rules_level, _rules_source = infer_audience_level_rules({"educational_mode": True})

    assert learned_level == rules_level == "Beginner"
    assert learned_source == "educational_mode"
