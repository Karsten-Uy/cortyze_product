"""Campaigns CRUD — Stage 3.

Each authenticated user has their own private list of ad-campaigns. The
sidebar groups runs under their campaign_id so creators can keep
"Holiday 2026" runs separate from "Summer Drop" runs.

All routes require auth. RLS guarantees cross-user isolation at the
database layer; the API also sanity-checks ownership before responding.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.schemas import Campaign, CampaignSummary
from services.persistence.campaigns import get_store as get_campaigns

from ..auth import require_user

router = APIRouter()


class CampaignCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)


class CampaignUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)


def _store():
    s = get_campaigns()
    if s is None:
        raise HTTPException(
            status_code=501,
            detail="Campaigns persistence not configured. Set DATABASE_URL env var.",
        )
    return s


@router.get("/campaigns", response_model=list[CampaignSummary])
def list_campaigns(user_id: str = Depends(require_user)) -> list[CampaignSummary]:
    return _store().list_for_user(user_id)


@router.post("/campaigns", response_model=Campaign, status_code=201)
def create_campaign(
    body: CampaignCreate, user_id: str = Depends(require_user)
) -> Campaign:
    return _store().create(
        user_id=user_id, name=body.name, description=body.description
    )


@router.patch("/campaigns/{campaign_id}", response_model=Campaign)
def update_campaign(
    campaign_id: str,
    body: CampaignUpdate,
    user_id: str = Depends(require_user),
) -> Campaign:
    updated = _store().update(
        campaign_id, user_id=user_id, name=body.name, description=body.description
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"campaign {campaign_id} not found")
    return updated


@router.delete("/campaigns/{campaign_id}", status_code=204)
def delete_campaign(campaign_id: str, user_id: str = Depends(require_user)) -> None:
    if not _store().delete(campaign_id, user_id=user_id):
        raise HTTPException(status_code=404, detail=f"campaign {campaign_id} not found")
    # 204 No Content
    return None
