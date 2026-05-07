import sys
from loguru import logger
from src.config import settings


def setup_logger() -> None:
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    )
    if settings.environment == "production":
        # Structured JSON logs in production (plays nicely with CloudWatch)
        logger.add(
            sys.stdout,
            level=settings.log_level.upper(),
            serialize=True,
        )
    else:
        logger.add(sys.stdout, level=settings.log_level.upper(), format=fmt, colorize=True)

    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="14 days",
        level="DEBUG",
        serialize=True,
        catch=True,
    )


setup_logger()

__all__ = ["logger"]
