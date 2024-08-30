import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from ensure import ensure_annotations


@ensure_annotations
def iqr_filter(
    data: pd.DataFrame,
    features: dict[str],
    is_submission_col: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filters the input DataFrame by removing rows that have
    values outside the interquartile range (IQR) for specified
    features, while keeping rows with NaNs.

    Args:
        data (pd.DataFrame):
            The input DataFrame to filter.
        features (dict[str]):
            Dictionary of feature names and their corresponding
            threshold values for the IQR filter.
        verbose (bool):
            Whether to print the number of rows removed
            and the percentage of rows removed.
            Defaults to True.

    Returns:
        pd.DataFrame:
            The input DataFrame without outliers.
    """
    len_old = len(data)
    mask = pd.Series(True, index=data.index)
    for feature, threshold in features.items():
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        if is_submission_col:
            mask &= (
                ((data[feature] >= lower) & (data[feature] <= upper))
                | (data[feature].isna())
                | (data[is_submission_col] == True)
            )
            data.loc[data[is_submission_col], feature] = data.loc[
                data[is_submission_col], feature
            ].clip(lower, upper)
        else:
            mask &= ((data[feature] >= lower) & (data[feature] <= upper)) | data[
                feature
            ].isna()
    data = data[mask]
    if verbose:
        print(
            f"Removed {len_old - len(data)} outliers, ratio - "
            f"{round(100 * (len_old - len(data)) / len_old, 2)}%"
        )
    return data


@ensure_annotations
def get_bins(x: int) -> int:
    """
    Calculates the appropriate number of bins for the histogram
    according to the number of the observations

    Args:
        x (int):
            Number of the observations

    Returns:
        int:
            Number of bins
    """
    if x > 0:
        n_bins = max(int(1 + 3.2 * np.log(x)), int(1.72 * x ** (1 / 3)))
    else:
        message = (
            "An invalid input value passed. Expected a positive "
            + "integer, but got {x}"
        )
        raise ValueError(message)
    return n_bins


@ensure_annotations
def save_predictions(df_submission: pd.DataFrame, filename: str):
    """
    Saves the predicted item counts for the test set to a CSV file.

    Args:
        df_submission (pd.DataFrame):
            A DataFrame containing the predicted item counts
            for each shop and item.
        filename (str):
            The name of the csv file to save the predictions to
    """
    submission = pd.read_csv("../data/raw/sample_submission.csv").drop(columns="sales")
    submission = submission.merge(
        df_submission[["sales"]].reset_index(),
        on="id",
        how="left",
    )
    submission[["id", "sales"]].to_csv(f"../data/submissions/{filename}", index=False)
    print(f"csv file saved at: ../data/predictions/{filename}")


@ensure_annotations
def reduce_size(df: pd.DataFrame):
    """
    Reduces the size of the DataFrame by converting integer
    and float columns to smaller data types.

    This function iterates through each column in the DataFrame and
    checks the minimum and maximum values. It then converts the column
    to a smaller data type if possible, such as `uint8`, `uint16`,
    `int8`, or `int16`, to reduce the memory footprint of the DataFrame.

    Args:
        df (pd.DataFrame):
            The DataFrame to be reduced in size.
    """
    for col in tqdm(df.columns):
        if "int" in df[col].dtype.name:
            if df[col].min() >= 0:
                if df[col].max() <= 255:
                    df[col] = df[col].astype("uint8")
                elif df[col].max() <= 65535:
                    df[col] = df[col].astype("uint16")
                else:
                    df[col] = df[col].astype("uint32")
            else:
                if max(abs(df[col].min()), df[col].max()) <= 127:
                    df[col] = df[col].astype("int8")
                elif max(abs(df[col].min()), df[col].max()) <= 32767:
                    df[col] = df[col].astype("int16")
                else:
                    df[col] = df[col].astype("int32")
        elif "float" in df[col].dtype.name:
            df[col] = df[col].astype("float32")


@ensure_annotations
def save_pkl(model: object, path: Path):
    """
    Saves a model object to a file using joblib.

    Args:
        model (object):
            The model object to be saved.
        path (Path):
            The path to the file where the model will be saved.

    Raises:
        ValueError:
            If the file does not have a .pkl extension.
        FileNotFoundError:
            If the directory to save the model does not exist.
        IOError:
            If an I/O error occurs during the saving process.
        Exception:
            If an unexpected error occurs during the saving process.
    """

    if path.suffix != ".pkl":
        raise ValueError(f"The file {path} is not a pkl file")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise Exception(f"An error occurred while creating the directory: {e}") from e

    try:
        with open(path, "wb") as f:
            joblib.dump(model, f)
        print(f"Model file saved at: {path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Directory '{path.parent}' does not exist: {e}") from e
    except IOError as e:
        raise IOError(f"An I/O error occurred while saving the model: {e}") from e
    except Exception as e:
        raise Exception(
            f"An unexpected error occurred while saving the model: {e}"
        ) from e


@ensure_annotations
def read_pkl(path: Path) -> object:
    """
    Reads a model object from a file using joblib.

    Args:
        path (Path):
            The path to the file with the model to load.

    Returns:
        object:
            The loaded model object.

    Raises:
        ValueError:
            If the file does not have a .pkl extension.
        FileNotFoundError:
            If the file does not exist.
        IOError:
            If an I/O error occurs during the loading process.
        Exception:
            If an unexpected error occurs while loading the model.
    """

    if path.suffix != ".pkl":
        raise ValueError(f"The file {path} is not a pkl file")

    try:
        with open(path, "rb") as f:
            model = joblib.load(f)
        return model
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File '{path}' does not exist: {e}") from e
    except IOError as e:
        raise IOError(f"An I/O error occurred while loading the model: {e}") from e
    except Exception as e:
        raise Exception(
            f"An unexpected error occurred while loading the model: {e}"
        ) from e


@ensure_annotations
def save_dict_as_json(data: dict, path: Path):
    """
    Saves a dictionary object to a file in JSON format.

    Args:
        data (dict):
            The dictionary object to be saved.
        path (Path):
            The path to the file where the dictionary will be saved.

    Raises:
        FileNotFoundError:
            If the directory to save the dictionary does not exist.
        IOError:
            If an I/O error occurs during the saving process.
        Exception:
            If an unexpected error occurs during the saving process.
    """

    if path.suffix != ".json":
        raise ValueError(f"The file {path} is not a json file")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise Exception(
            f"An error occurred while creating the directory {path.parent}: {e}"
        ) from e

    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"JSON file saved at: {path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Directory '{path.parent}' does not exist: {e}") from e
    except IOError as e:
        raise IOError(f"An I/O error occurred while saving the JSON file: {e}") from e
    except Exception as e:
        raise Exception(
            f"An unexpected error occurred while saving the JSON file: {e}"
        ) from e
