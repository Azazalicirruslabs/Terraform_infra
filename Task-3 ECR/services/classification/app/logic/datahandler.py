import pandas as pd


class DataHandler:
    """
    Handles a pandas DataFrame for analysis (assumed loaded from S3).

    Args:
        df: The dataset to be analyzed (assumed loaded from S3).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.target_column = None

    def preview(self, n: int = 5) -> dict:
        """
        Returns top N rows of dataset as dictionary for frontend preview.
        """
        return self.df.head(n).to_dict(orient="records")

    def list_columns(self) -> list:
        """
        Lists all column names for target selection UI.
        """
        return self.df.columns.tolist()

    def set_target_column(self, column: str):
        """
        Stores chosen target column and validates its existence.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")
        self.target_column = column
        return self.target_column

    def get_target_column(self) -> str:
        """
        Returns selected target column.
        """
        return self.target_column

    def get_dataset(self) -> pd.DataFrame:
        """
        Returns the full dataset, useful for other modules.
        """
        return self.df
