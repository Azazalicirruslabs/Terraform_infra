import logging


def setup_logging():
    """Set up logging configuration for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console
            # logging.FileHandler("api.log"),  # Uncomment for file logging
        ],
    )
