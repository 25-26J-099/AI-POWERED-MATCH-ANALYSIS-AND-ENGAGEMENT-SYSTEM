"""Singleton service dependencies for FastAPI routes."""
from app.config.settings import settings
from app.services.analysis_service import AnalysisService
from app.services.artifact_service import ArtifactService
from app.services.job_service import JobService

analysis_service = AnalysisService()
artifact_service = ArtifactService()
job_service = JobService(
    analysis_service=analysis_service,
    max_workers=max(1, int(settings.ANALYSIS_MAX_WORKERS)),
)


def get_analysis_service() -> AnalysisService:
    return analysis_service


def get_artifact_service() -> ArtifactService:
    return artifact_service


def get_job_service() -> JobService:
    return job_service
