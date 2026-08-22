"""Aggregate dashboard workflow routers."""

from fastapi import APIRouter

from . import (
    augmentation,
    pipeline_configuration,
    pipeline_inference,
    segmentation,
    working_image,
)

router = APIRouter()
router.include_router(pipeline_configuration.router)
router.include_router(pipeline_inference.router)
router.include_router(working_image.router)
router.include_router(segmentation.router)
router.include_router(augmentation.router)
