"""OpenAI client for AI expert analysis generation."""

import json
from typing import Optional
from app.config.settings import settings


async def generate_expert_analysis(analytics_data: dict) -> dict:
    """Generate expert football analysis using OpenAI API.

    Args:
        analytics_data: Full match analytics payload (from GET /match/{id}/analytics)

    Returns:
        Dictionary with analysis sections.
    """
    if not settings.OPENAI_API_KEY:
        return {
            "error": "OpenAI API key not configured",
            "fallback_analysis": _generate_fallback_analysis(analytics_data),
        }

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    system_prompt = _build_system_prompt(analytics_data)

    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate a comprehensive expert analysis of this football match."},
        ],
        temperature=0.7,
        max_tokens=3000,
        response_format={"type": "json_object"},
    )

    try:
        analysis = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        analysis = {"raw_analysis": response.choices[0].message.content}

    return analysis


def _build_system_prompt(analytics_data: dict) -> str:
    """Build a detailed system prompt with match analytics."""
    home_team = analytics_data.get("home_team", "Home")
    away_team = analytics_data.get("away_team", "Away")
    home_stats = analytics_data.get("home_team_stats", {})
    away_stats = analytics_data.get("away_team_stats", {})
    player_stats = analytics_data.get("player_stats", [])

    # Top performers by rating
    top_players = sorted(player_stats, key=lambda p: p.get("rating", 0), reverse=True)[:5]

    prompt = f"""You are an elite football analyst. Analyze this match between {home_team} and {away_team}.

## Match Analytics Data

### Team Stats
**{home_team}**: xG={home_stats.get('total_xg', 0):.3f}, xT={home_stats.get('total_xt', 0):.3f}, VAEP={home_stats.get('total_vaep', 0):.3f}, Passes={home_stats.get('total_passes', 0)}, Pass Accuracy={home_stats.get('avg_pass_accuracy', 0):.1f}%, Shots={home_stats.get('total_shots', 0)}

**{away_team}**: xG={away_stats.get('total_xg', 0):.3f}, xT={away_stats.get('total_xt', 0):.3f}, VAEP={away_stats.get('total_vaep', 0):.3f}, Passes={away_stats.get('total_passes', 0)}, Pass Accuracy={away_stats.get('avg_pass_accuracy', 0):.1f}%, Shots={away_stats.get('total_shots', 0)}

### Top Performers (by rating)
"""
    for p in top_players:
        prompt += f"- {p.get('player_name', '?')} ({p.get('team', '?')}): Rating={p.get('rating', 0):.1f}, xG={p.get('xg', 0):.3f}, xT={p.get('xt', 0):.3f}, VAEP={p.get('vaep', 0):.3f}, Passes={p.get('passes', 0)}, Pass Acc={p.get('pass_accuracy', 0):.1f}%\n"

    prompt += f"""
### All Player Stats
{json.dumps(player_stats, indent=2)}

## Instructions
Provide your analysis as JSON with these keys:
- "match_overview": Brief overall match summary (2-3 sentences)
- "tactical_analysis": Tactical breakdown of how both teams played
- "key_moments": Key tactical moments and turning points
- "player_of_the_match": Name and justification for POTM
- "top_performers": Analysis of the top 3 performers
- "team_comparison": Comparative analysis of both teams' performances
- "areas_to_improve": For each team, what they could improve
- "xg_analysis": Analysis of expected goals vs actual performance
- "possession_quality": Assessment of possession quality using xT and VAEP data
"""
    return prompt


def _generate_fallback_analysis(analytics_data: dict) -> dict:
    """Generate a simple rule-based analysis when OpenAI is unavailable."""
    home = analytics_data.get("home_team", "Home")
    away = analytics_data.get("away_team", "Away")
    hs = analytics_data.get("home_team_stats", {})
    aws = analytics_data.get("away_team_stats", {})
    players = analytics_data.get("player_stats", [])

    top = sorted(players, key=lambda p: p.get("rating", 0), reverse=True)

    dominant = home if hs.get("total_vaep", 0) > aws.get("total_vaep", 0) else away

    return {
        "match_overview": f"{dominant} were the dominant side based on VAEP metrics.",
        "player_of_the_match": top[0].get("player_name", "Unknown") if top else "N/A",
        "home_xg": hs.get("total_xg", 0),
        "away_xg": aws.get("total_xg", 0),
        "note": "Full analysis requires OpenAI API key. Set OPENAI_API_KEY in .env",
    }
