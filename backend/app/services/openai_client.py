"""OpenAI client for AI expert analysis generation."""

import json

import httpx

from app.config.settings import settings

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


async def generate_expert_analysis(analytics_data: dict) -> dict:
    """Generate expert football analysis using the OpenAI REST API."""
    if not settings.OPENAI_API_KEY:
        return _build_error_response("OpenAI API key not configured", analytics_data)

    system_prompt = _build_system_prompt(analytics_data)
    payload = {
        "model": settings.OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate a comprehensive expert analysis of this football match."},
        ],
        "temperature": 0.7,
        "max_tokens": 3000,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENAI_CHAT_COMPLETIONS_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return _build_error_response(
            f"OpenAI API request failed ({exc.response.status_code}): {_extract_error_message(exc.response)}",
            analytics_data,
        )
    except httpx.HTTPError as exc:
        return _build_error_response(f"OpenAI API connection failed: {exc}", analytics_data)

    try:
        response_payload = response.json()
    except ValueError:
        return _build_error_response("OpenAI API returned invalid JSON", analytics_data)

    content = _extract_message_content(response_payload)
    if not content:
        return _build_error_response("OpenAI API returned an empty analysis", analytics_data)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_analysis": content}


def _extract_message_content(response_payload: dict) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        return ""

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    return ""


def _extract_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:300] or "Unknown error"

    error = payload.get("error", {})
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return response.text.strip()[:300] or "Unknown error"


def _build_error_response(error_message: str, analytics_data: dict) -> dict:
    return {
        "error": error_message,
        "fallback_analysis": _generate_fallback_analysis(analytics_data, reason=error_message),
    }


def _build_system_prompt(analytics_data: dict) -> str:
    """Build a detailed system prompt with match analytics."""
    home_team = analytics_data.get("home_team", "Home")
    away_team = analytics_data.get("away_team", "Away")
    home_stats = analytics_data.get("home_team_stats", {})
    away_stats = analytics_data.get("away_team_stats", {})
    player_stats = analytics_data.get("player_stats", [])

    top_players = sorted(player_stats, key=lambda p: p.get("rating", 0), reverse=True)[:5]

    prompt = f"""You are an elite football analyst. Analyze this match between {home_team} and {away_team}.

## Match Analytics Data

### Team Stats
**{home_team}**: xG={home_stats.get('total_xg', 0):.3f}, xT={home_stats.get('total_xt', 0):.3f}, VAEP={home_stats.get('total_vaep', 0):.3f}, Passes={home_stats.get('total_passes', 0)}, Pass Accuracy={home_stats.get('avg_pass_accuracy', 0):.1f}%, Shots={home_stats.get('total_shots', 0)}

**{away_team}**: xG={away_stats.get('total_xg', 0):.3f}, xT={away_stats.get('total_xt', 0):.3f}, VAEP={away_stats.get('total_vaep', 0):.3f}, Passes={away_stats.get('total_passes', 0)}, Pass Accuracy={away_stats.get('avg_pass_accuracy', 0):.1f}%, Shots={away_stats.get('total_shots', 0)}

### Top Performers (by rating)
"""
    for player in top_players:
        prompt += (
            f"- {player.get('player_name', '?')} ({player.get('team', '?')}): "
            f"Rating={player.get('rating', 0):.1f}, "
            f"xG={player.get('xg', 0):.3f}, "
            f"xT={player.get('xt', 0):.3f}, "
            f"VAEP={player.get('vaep', 0):.3f}, "
            f"Passes={player.get('passes', 0)}, "
            f"Pass Acc={player.get('pass_accuracy', 0):.1f}%\n"
        )

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


def _generate_fallback_analysis(analytics_data: dict, reason: str | None = None) -> dict:
    """Generate a simple rule-based analysis when OpenAI is unavailable."""
    home = analytics_data.get("home_team", "Home")
    away = analytics_data.get("away_team", "Away")
    hs = analytics_data.get("home_team_stats", {})
    aws = analytics_data.get("away_team_stats", {})
    players = analytics_data.get("player_stats", [])

    top = sorted(players, key=lambda p: p.get("rating", 0), reverse=True)
    dominant = home if hs.get("total_vaep", 0) > aws.get("total_vaep", 0) else away

    note = "Full analysis is unavailable."
    if reason:
        note = f"{note} Reason: {reason}"

    return {
        "match_overview": f"{dominant} were the dominant side based on VAEP metrics.",
        "player_of_the_match": top[0].get("player_name", "Unknown") if top else "N/A",
        "home_xg": hs.get("total_xg", 0),
        "away_xg": aws.get("total_xg", 0),
        "note": note,
    }
