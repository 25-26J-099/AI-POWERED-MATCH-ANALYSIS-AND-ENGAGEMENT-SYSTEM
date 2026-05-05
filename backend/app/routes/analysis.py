"""AI Expert Analysis endpoint — uses OpenAI to generate match analysis."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.routes.matches import build_match_analytics
from app.services.openai_client import generate_expert_analysis
from app.services.security import require_enterprise_api_key

router = APIRouter()


@router.post("/match/{match_id}/ai-analysis", dependencies=[Depends(require_enterprise_api_key)])
async def ai_analysis(match_id: int, db: AsyncSession = Depends(get_db)):
    """Generate AI expert analysis using match analytics data."""
    analytics_data = await build_match_analytics(match_id, db)

    try:
        analysis = await generate_expert_analysis(analytics_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis generation failed: {str(e)}")

    return {
        "match_id": match_id,
        "analysis": analysis,
    }
