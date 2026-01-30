import logging
import sys
import torch
import torch.nn as nn
import numpy as np

"""
Inference pipeline for the model.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class InferencePipeline(nn.Module):
    """
    Simple inference pipeline using a Linear layer.
    """
    def __init__(self):
        """Initialize the pipeline."""
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        """Forward pass."""
        return self.fc(x)

    def run_inference(self):
        """
        Run a simulated inference.
        """
        logger.info("Starting inference...")
        input_data = torch.randn(1, 10)
        with torch.no_grad():
            output = self.forward(input_data)

        # Simulated use of numpy to avoid unused import
        logger.info(f"Mean output: {np.mean(output.numpy())}")
        logger.info("Inference complete.")


if __name__ == "__main__":
    pipeline = InferencePipeline()
    pipeline.run_inference()
