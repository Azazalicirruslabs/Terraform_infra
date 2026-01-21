"""
Feature Interaction Generator
Generates interaction features from user-selected protected columns
Supports both pairwise interactions and custom interaction functions
"""

import logging
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureInteractionGenerator:
    """
    Generate and manage feature interactions for fairness analysis
    Supports both automatic pairwise interactions and custom interaction definitions
    """

    def __init__(self):
        self.interaction_columns = {}
        self.interaction_metadata = {}

    def create_interaction_features(
        self, df: pd.DataFrame, interaction_specs: List[Dict[str, Any]], interaction_type: str = "concatenate"
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Create interaction features based on specifications

        Args:
            df: DataFrame to add interactions to
            interaction_specs: List of interaction specifications
                Each spec should have: {"columns": ["col1", "col2"], "name": "optional_name", "type": "optional_type"}
            interaction_type: Default interaction type ("concatenate", "multiply", "add")

        Returns:
            Tuple of (modified DataFrame, interaction metadata)
        """
        df_with_interactions = df.copy()
        interaction_metadata = {}

        for spec in interaction_specs:
            try:
                columns = spec.get("columns", [])
                custom_name = spec.get("name")
                custom_type = spec.get("type", interaction_type)

                if len(columns) < 2:
                    logger.warning(f"Interaction spec requires at least 2 columns: {spec}")
                    continue

                # Generate interaction feature
                interaction_col, metadata = self._generate_interaction(
                    df_with_interactions, columns, custom_name, custom_type
                )

                # Add to dataframe
                interaction_name = custom_name or metadata["name"]
                df_with_interactions[interaction_name] = interaction_col
                interaction_metadata[interaction_name] = metadata

                logger.info(f"Created interaction feature: {interaction_name}")

            except Exception as e:
                logger.error(f"Failed to create interaction for {spec}: {str(e)}")
                continue

        self.interaction_columns = interaction_metadata
        return df_with_interactions, interaction_metadata

    def _generate_interaction(
        self, df: pd.DataFrame, columns: List[str], custom_name: Optional[str], interaction_type: str
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate a single interaction feature

        Supported interaction types:
        - concatenate: Combine string representations (e.g., "gender_race")
        - multiply: Multiply numerical values
        - add: Add numerical values
        - and: Logical AND for binary features
        - or: Logical OR for binary features
        """

        # Validate columns exist
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in dataframe")

        # Determine data types
        [self._determine_column_type(df[col]) for col in columns]

        # Generate interaction based on type
        if interaction_type == "concatenate":
            interaction_col = self._concatenate_interaction(df, columns)
            interaction_data_type = "categorical"

        elif interaction_type == "multiply":
            interaction_col = self._multiply_interaction(df, columns)
            interaction_data_type = "numerical"

        elif interaction_type == "add":
            interaction_col = self._add_interaction(df, columns)
            interaction_data_type = "numerical"

        elif interaction_type == "and":
            interaction_col = self._logical_and_interaction(df, columns)
            interaction_data_type = "binary"

        elif interaction_type == "or":
            interaction_col = self._logical_or_interaction(df, columns)
            interaction_data_type = "binary"

        else:
            # Default to concatenate
            logger.warning(f"Unknown interaction type {interaction_type}, using concatenate")
            interaction_col = self._concatenate_interaction(df, columns)
            interaction_data_type = "categorical"

        # Generate name if not provided
        if not custom_name:
            custom_name = f"{'_x_'.join(columns)}"

        # Create metadata
        metadata = {
            "name": custom_name,
            "source_columns": columns,
            "interaction_type": interaction_type,
            "data_type": interaction_data_type,
            "unique_groups": len(interaction_col.unique()),
            "null_count": interaction_col.isnull().sum(),
        }

        return interaction_col, metadata

    def _determine_column_type(self, series: pd.Series) -> str:
        """Determine if column is numerical, categorical, or binary"""
        if series.dtype in ["object", "category", "string"]:
            return "categorical"
        elif series.nunique() == 2:
            return "binary"
        elif pd.api.types.is_numeric_dtype(series):
            return "numerical"
        else:
            return "categorical"

    def _concatenate_interaction(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Concatenate column values with separator"""
        # Convert all columns to string and concatenate
        interaction = df[columns[0]].astype(str)
        for col in columns[1:]:
            interaction = interaction + "_" + df[col].astype(str)
        return interaction

    def _multiply_interaction(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Multiply numerical columns"""
        interaction = df[columns[0]].astype(float)
        for col in columns[1:]:
            interaction = interaction * df[col].astype(float)
        return interaction

    def _add_interaction(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Add numerical columns"""
        interaction = df[columns[0]].astype(float)
        for col in columns[1:]:
            interaction = interaction + df[col].astype(float)
        return interaction

    def _logical_and_interaction(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Logical AND across columns"""
        interaction = df[columns[0]].astype(bool)
        for col in columns[1:]:
            interaction = interaction & df[col].astype(bool)
        return interaction.astype(int)

    def _logical_or_interaction(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Logical OR across columns"""
        interaction = df[columns[0]].astype(bool)
        for col in columns[1:]:
            interaction = interaction | df[col].astype(bool)
        return interaction.astype(int)

    def generate_all_pairwise_interactions(
        self, df: pd.DataFrame, columns: List[str], interaction_type: str = "concatenate"
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Generate all pairwise interactions for the specified columns

        Args:
            df: DataFrame to add interactions to
            columns: List of columns to create pairwise interactions from
            interaction_type: Type of interaction to create

        Returns:
            Tuple of (modified DataFrame, interaction metadata)
        """
        if len(columns) < 2:
            logger.warning("Need at least 2 columns for pairwise interactions")
            return df, {}

        # Generate all pairwise combinations
        pairwise_specs = []
        for col1, col2 in combinations(columns, 2):
            pairwise_specs.append({"columns": [col1, col2], "type": interaction_type})

        logger.info(f"Generating {len(pairwise_specs)} pairwise interactions")
        return self.create_interaction_features(df, pairwise_specs, interaction_type)

    def get_interaction_groups(self, df: pd.DataFrame, interaction_name: str) -> List[str]:
        """
        Get unique groups for an interaction feature

        Args:
            df: DataFrame containing the interaction
            interaction_name: Name of the interaction column

        Returns:
            List of unique group values
        """
        if interaction_name not in df.columns:
            raise ValueError(f"Interaction {interaction_name} not found in dataframe")

        unique_values = df[interaction_name].dropna().unique()

        # Limit to reasonable number for display
        if len(unique_values) <= 20:
            return [str(val) for val in sorted(unique_values)]
        else:
            # Return top 20 most frequent
            top_values = df[interaction_name].value_counts().head(20).index
            return [str(val) for val in top_values] + [f"... and {len(unique_values) - 20} more"]

    def validate_interaction_spec(
        self, df: pd.DataFrame, interaction_spec: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate an interaction specification

        Returns:
            Tuple of (is_valid, error_message)
        """
        columns = interaction_spec.get("columns", [])

        if not columns or len(columns) < 2:
            return False, "Interaction requires at least 2 columns"

        # Check columns exist
        for col in columns:
            if col not in df.columns:
                return False, f"Column {col} not found in dataset"

        # Check interaction type
        interaction_type = interaction_spec.get("type", "concatenate")
        valid_types = ["concatenate", "multiply", "add", "and", "or"]
        if interaction_type not in valid_types:
            return False, f"Invalid interaction type. Must be one of: {valid_types}"

        return True, None

    def remove_interaction(self, df: pd.DataFrame, interaction_name: str) -> pd.DataFrame:
        """
        Remove an interaction column from the dataframe

        Args:
            df: DataFrame to remove interaction from
            interaction_name: Name of the interaction to remove

        Returns:
            Modified DataFrame
        """
        if interaction_name in df.columns:
            df_modified = df.drop(columns=[interaction_name])
            if interaction_name in self.interaction_columns:
                del self.interaction_columns[interaction_name]
            logger.info(f"Removed interaction: {interaction_name}")
            return df_modified
        else:
            logger.warning(f"Interaction {interaction_name} not found in dataframe")
            return df

    def get_interaction_statistics(self, df: pd.DataFrame, interaction_name: str) -> Dict[str, Any]:
        """
        Get statistics about an interaction feature

        Args:
            df: DataFrame containing the interaction
            interaction_name: Name of the interaction column

        Returns:
            Dictionary of statistics
        """
        if interaction_name not in df.columns:
            raise ValueError(f"Interaction {interaction_name} not found in dataframe")

        col = df[interaction_name]

        stats = {
            "name": interaction_name,
            "total_records": len(col),
            "unique_groups": col.nunique(),
            "null_count": col.isnull().sum(),
            "null_percentage": round(col.isnull().sum() / len(col) * 100, 2),
            "most_common_group": col.value_counts().index[0] if len(col) > 0 else None,
            "most_common_count": col.value_counts().iloc[0] if len(col) > 0 else 0,
            "group_distribution": col.value_counts().to_dict(),
        }

        # Add metadata if available
        if interaction_name in self.interaction_columns:
            stats.update(self.interaction_columns[interaction_name])

        return stats

    def auto_detect_best_interaction(self, df: pd.DataFrame, columns: List[str]) -> str:
        """
        Automatically detect the best interaction type based on column types

        Returns the recommended interaction type
        """
        col_types = [self._determine_column_type(df[col]) for col in columns]

        # All categorical -> concatenate
        if all(ct == "categorical" for ct in col_types):
            return "concatenate"

        # All binary -> and/or (prefer 'and' for intersectional)
        elif all(ct == "binary" for ct in col_types):
            return "and"

        # All numerical -> multiply (captures interaction effects better)
        elif all(ct == "numerical" for ct in col_types):
            return "multiply"

        # Mixed types -> concatenate (most flexible)
        else:
            return "concatenate"

    def create_smart_interaction(
        self, df: pd.DataFrame, columns: List[str], custom_name: Optional[str] = None, force_type: Optional[str] = None
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Create interaction with automatic type detection

        Args:
            df: DataFrame
            columns: Columns to interact
            custom_name: Optional custom name
            force_type: Force specific interaction type, otherwise auto-detect
        """
        # Auto-detect if not forced
        if force_type:
            interaction_type = force_type
        else:
            interaction_type = self.auto_detect_best_interaction(df, columns)

        return self._generate_interaction(df, columns, custom_name, interaction_type)

    def generate_n_way_interactions(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n: int,
        interaction_type: Optional[str] = None,
        auto_detect: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Generate all n-way interactions

        Args:
            df: DataFrame
            columns: Columns to create interactions from
            n: Size of combinations (2=pairwise, 3=three-way, etc.)
            interaction_type: Fixed type or None for auto-detection
            auto_detect: Whether to auto-detect best type per combination
        """
        if len(columns) < n:
            logger.warning(f"Need at least {n} columns for {n}-way interactions")
            return df, {}

        interaction_specs = []
        for combo in combinations(columns, n):
            spec = {"columns": list(combo)}

            # Auto-detect or use fixed type
            if auto_detect and interaction_type is None:
                detected_type = self.auto_detect_best_interaction(df, list(combo))
                spec["type"] = detected_type
            else:
                spec["type"] = interaction_type or "concatenate"

            interaction_specs.append(spec)

        logger.info(f"Generating {len(interaction_specs)} {n}-way interactions")
        return self.create_interaction_features(df, interaction_specs)
