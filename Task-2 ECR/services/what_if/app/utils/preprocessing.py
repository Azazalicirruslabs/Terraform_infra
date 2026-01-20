import pandas as pd
from category_encoders import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_preprocessing_pipeline(df: pd.DataFrame, target: str, drop_thresh: float = 0.95):
    """Create automated preprocessing pipeline"""

    # Remove target from features
    X = df.drop(columns=[target])
    df[target]

    # Drop high-cardinality columns
    nunique_ratio = X.nunique() / len(X)
    keep_cols = nunique_ratio[nunique_ratio < drop_thresh].index
    X = X[keep_cols]

    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Create transformers
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    # Handle categorical features
    low_cardinality_cats = []
    high_cardinality_cats = []

    for col in categorical_cols:
        if X[col].nunique() <= 30:
            low_cardinality_cats.append(col)
        else:
            high_cardinality_cats.append(col)

    transformers = []

    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))

    if low_cardinality_cats:
        cat_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", sparse_output=False)),
            ]
        )
        transformers.append(("cat_low", cat_transformer, low_cardinality_cats))

    if high_cardinality_cats:
        target_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("target", TargetEncoder(min_samples_leaf=20, smoothing=10.0)),
            ]
        )
        transformers.append(("cat_high", target_transformer, high_cardinality_cats))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return preprocessor, keep_cols.tolist()
