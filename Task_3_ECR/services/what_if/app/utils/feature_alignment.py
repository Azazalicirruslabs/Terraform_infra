def _ensure_feature_alignment(X_processed_df):
    """Ensure feature alignment between model and data by removing preprocessing prefixes"""
    if hasattr(X_processed_df, "columns"):
        # Create a mapping to clean column names
        new_columns = []
        for col in X_processed_df.columns:
            # Remove common preprocessing prefixes
            clean_col = col
            if "__" in col:
                clean_col = col.split("__", 1)[1]  # Remove prefix like 'num__', 'cat__'
            new_columns.append(clean_col)

        # Update the DataFrame columns
        X_processed_df.columns = new_columns

    return X_processed_df
