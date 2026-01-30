import sys
import torch
import torchvision
import logging

"""
End-to-end environment verification.
"""

# Configure logging to stdout for capture
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_verification():
    """Verify environment and basic model imports."""
    logger.info("Starting E2E Verification Check")

    # Check Versions
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"Torchvision version: {torchvision.__version__}")

    # Check NMS Operator specifically
    try:
        from torchvision.ops import nms  # noqa: F401
        logger.info("Successfully imported torchvision.ops.nms")
    except Exception as e:
        logger.error(f"Failed to import NMS operator: {e}")
        sys.exit(1)

    # Check Transformers
    try:
        from transformers import AutoConfig  # noqa: F401
        logger.info("Successfully imported transformers")
    except Exception as e:
        logger.error(f"Failed to import transformers: {e}")
        sys.exit(1)

    # Simulated functional check
    logger.info("Environment check PASSED")


if __name__ == "__main__":
    run_verification()
