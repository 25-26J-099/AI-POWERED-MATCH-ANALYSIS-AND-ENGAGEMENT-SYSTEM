"""Singleton service dependencies for FastAPI routes."""
from fastapi_app.services.analysis_service import AnalysisService
from fastapi_app.services.artifact_service import ArtifactService
from fastapi_app.services.job_service import JobService

analysis_service = AnalysisService()
artifact_service = ArtifactService()
job_service = JobService(analysis_service=analysis_service)


def get_analysis_service() -> AnalysisService:
    return analysis_service


def get_artifact_service() -> ArtifactService:
    return artifact_service


def get_job_service() -> JobService:
    return job_service

