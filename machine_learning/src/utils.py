# ==================
# IMPORTS
# ==================
import pandas as pd
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.
    NOTE: return hint should be -> 'pd.DataFrame', not 'pd.DataFrame()'
    (harmless typo, just technically wrong)
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")

    return pd.read_csv(path)


def fill_missing(data: pd.DataFrame, target: str, feature: str):
    """
    Fill NaNs in 'target' with the median per group of 'feature',
    not a global median.
    GOTCHA: if a whole group is NaN, it's median is NaN too -
    value stays unfilled. Pair this with drop_unfillable().
    Mutates 'data' is place.
    """
    data[target] = data.groupby(feature)[target].transform(
        lambda x: x.fillna(x.median())
    )
    return data


def drop_unfillable(data: pd.DataFrame, target: str):
    """
    Drop rows where 'target' is still NaN (couldn't be filled).
    GOTCHA: in a loop over columns, row loses compounds -
    each call shrinks the data the next one sees.
    """
    before = len(data)
    data = data[data[target].notna()]

    dropped = before - len(data)
    print(f"{target}: dropped {dropped} rows")

    return data


def clip_outliers(data: pd.DataFrame, target: str, upper_quantile: float=0.99, lower_quantile: float=0.01):
    """
    Cap extreme values to given quantile bounds (no rows dropped).
    GOTCHA: quantiles depend on current state of 'data' -
    call order in the pipeline affects results.
    Don't use on a target variable if extreme carry real signal.
    """
    upper = data[target].quantile(upper_quantile)
    lower = data[target].quantile(lower_quantile)
    data[target] = data[target].clip(lower=lower, upper=upper)
    return data