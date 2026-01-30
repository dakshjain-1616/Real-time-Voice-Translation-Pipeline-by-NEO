import sys
import logging
import os

"""
Benchmark performance of the pipeline.
"""

# Append project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """
    Benchmark class for measuring performance.
    """
    def __init__(self):
        """Initialize the benchmark."""
        pass

    def run(self):
        """
        Execute benchmark.
        """
        logger.info("Running performance benchmark...")
        # Get project root to show path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logger.info(f"Project root: {project_root}")


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    benchmark.run()
