"""
This file contains functions needed for appropriately constructing the
dataset that will be used to train the neural network.
"""
import os
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import TypeVar
from scipy.interpolate import CubicSpline

DataFrameStr = TypeVar("pandas.core.frame.DataFrame(str)")

def compute_spline(
    x: ArrayLike, 
    y: ArrayLike, 
    N: int, 
    grid: int
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Function used to apply cublic spline interpolation to dataset.

    Parameters
    ----------
    x: array
        x values used for interpolation.
    y: array
        y values used for interpolation.
    N: int
        Specifies value of Nmax.
    grid: int
        Interpolation step size.

    Returns
    -------
    new_x: array
        Interpolated dataset of x values.
    new_y: array
        Interpolated dataset of y values.
    N_vals: array
        Interpolated dataset of N values.
    """
    # Creates values for hw from x[0] to x[-1], with points=grid in between
    new_x = np.linspace(x[0], x[-1], grid)
    CS = CubicSpline(x, y)
    new_y = CS(new_x)
    
    # Fills an array with Nmax, length of interpolated values
    N_vals = np.full(len(new_y), N)
     
    # Gets rid of first and last values since they appear twice
    index = [0, len(new_x) - 1] # First and last value
    new_x = np.delete(new_x, index)
    new_y = np.delete(new_y, index)
    N_vals = np.delete(N_vals, index)
    
    return new_x, new_y, N_vals

def get_training_data(
    param: str, 
    df: DataFrameStr, 
    step: int, 
    spline: bool
) -> tuple[DataFrameStr, list[DataFrameStr]]:
    """
    Function used to properly configure dataset.
    If spline==True, a cubic spline interpolation is applied to the dataset.

    Parameters
    ----------
    param: str
        Parameter we want to predict.
    df: DataFrame
        DataFrame with dataset.
    step: int
        Interpolation grid step size.
    spline: bool
        Dictates whether spline is applied.
        If True, apply spline.
        If False, do not apply spline.

    Returns
    -------
    all_Nmax: DataFrame
        Original DataFrame.
    all_Nmax_spline: list[DataFrame]
        DataFrame with splined dataset in a list with increasing Nmax.
    """
    all_Nmax = separate_by_N(df)
    all_Nmax_spline = []
    
    if spline:
        tot_Nmax = df["Nmax"].unique()
        
        for i in range(tot_Nmax.shape[0]):
            all_Nmax[i] = remove_unneeded_data(all_Nmax[i], 10)
            new_x, new_y, new_N = compute_spline(sorted(all_Nmax[i]["hw"].values), 
                                                 all_Nmax[i][param].values, tot_Nmax[i], step)
            
            df = pd.DataFrame(columns=["hw", "Nmax", param])
            df["hw"], df["Nmax"], df[param] = new_x, new_N, new_y
            
            Nmax_spline = pd.concat([all_Nmax[i], df], ignore_index=True, sort=False)
            Nmax_spline.sort_values(by=["hw"], inplace=True)
            all_Nmax_spline.append(Nmax_spline)
            
        for i in range(len(all_Nmax)):
            all_Nmax_spline[i] = df_col_difference(all_Nmax_spline[i], 
                                                   all_Nmax_spline[-1])
    
    return all_Nmax, all_Nmax_spline

def construct_training_sets(
    Nmax: DataFrameStr, 
    dims: int,
    pred_param: str
) -> tuple[ArrayLike, ArrayLike]:
    """
    Function used to correctly construct the new DataFrame that 
    will be used to train the neural network.

    Parameters
    ----------
    Nmax: DataFrame
        DataFrame with original data.
    dims: int
        Dimension of neural network input.
    pred_param: str
        Parameter we want to predict.

    Returns
    -------
    X: array
        X values of dataset separated by Nmax.
    y: array
        y values of dataset separated by Nmax.
    """
    df6 = [Nmax[2]]
    df8 = [Nmax[2], Nmax[3]]
    df10 = [Nmax[2], Nmax[3], Nmax[4]]
    df12 = [Nmax[2], Nmax[3], Nmax[4], Nmax[5]]

    Nmax06  = pd.concat(df6)
    Nmax08  = pd.concat(df8)
    Nmax10  = pd.concat(df10)
    Nmax12  = pd.concat(df12)
    
    Nmax_df = [Nmax06, Nmax08, Nmax10, Nmax12]
    
    X = np.zeros((len(Nmax_df),), dtype=object)
    y = np.zeros((len(Nmax_df),), dtype=object)
    
    
    if dims == 3:
        cols = ["hw", "Nmax", "Ediff"]
    else:
        cols = ["hw", "Nmax"]
    
    for i in range(X.shape[0]):
#         Nmax_df[i] = Nmax_df[i][~(Nmax_df[i]['hw'] < 20)].reset_index(drop=True)
        X[i] = Nmax_df[i][cols]
        y[i] = Nmax_df[i].loc[:, pred_param].values
    
    return X, y

def remove_unneeded_data(
    df: DataFrameStr, 
    num: int
) -> DataFrameStr:
    """
    Function used to remove unneeded information from dataset.

    Parameters
    ----------
    df: DataFrame
        DataFrame with data needed for neural network.
    num: int
        All hw values below this num are removed.

    Returns
    -------
    df: DataFrame
        New DataFrame with values removed.
    """
    df = df.drop(df.index[df['hw'] < num].tolist(), axis=0)
    df = (df.reset_index()).drop(['index'], axis=1)
    return df

def separate_by_N(
    df: DataFrameStr
) -> list[DataFrameStr]:
    """
    Function used to correctly separate original DataFrame by increasing Nmax.

    Parameters
    ----------
    df: DataFrame
        Original DataFrame with imported dataset.

    Returns
    -------
    df: list[DataFrame]
        A list of DataFrames with increasing Nmax.
    """
    Nmax02 = pd.DataFrame()
    Nmax04 = pd.DataFrame()
    Nmax06 = pd.DataFrame()
    Nmax08 = pd.DataFrame()
    Nmax10 = pd.DataFrame()
    Nmax12 = pd.DataFrame()

    Nmax = []

    for m in range(len(df["Nmax"].unique())):
        Nmax.append(2 * m + 2)

    for i in range(len(df["Nmax"])):
        if int(df.iloc[i][1]) == Nmax[0]:
            Nmax02 = df.loc[0:i]

        elif int(df.iloc[i][1]) == Nmax[1]:
            Nmax04 = df.loc[len(Nmax02):i]

        elif int(df.iloc[i][1]) == Nmax[2]:
            Nmax06 = df.loc[len(Nmax02) + \
                            len(Nmax04):i]

        elif int(df.iloc[i][1]) == Nmax[3]:
            Nmax08 = df.loc[len(Nmax02) + \
                            len(Nmax04) + \
                            len(Nmax06):i]

        elif int(df.iloc[i][1]) == Nmax[4]:
            Nmax10 = df.loc[len(Nmax02) + \
                            len(Nmax04) + \
                            len(Nmax06) + \
                            len(Nmax08):i]

        elif int(df.iloc[i][1]) == Nmax[5]:
            Nmax12 = df.loc[len(Nmax02) + \
                            len(Nmax04) + \
                            len(Nmax06) + \
                            len(Nmax08) + \
                            len(Nmax10):i]
        else:
            raise ValueError("Wrong input!")
    
    return [Nmax02, Nmax04, Nmax06, Nmax08, Nmax10, Nmax12]

def df_col_difference(
    df1: DataFrameStr, 
    df2: DataFrameStr
) -> DataFrameStr:
    """
    Function used to calculate the energy difference between values of Nmax.

    Parameters
    ----------
    df1: DataFrame
        First DataFrame used for calculating difference.
    df2: DataFrame
        Second DataFrame used for calculating difference.

    Returns
    -------
    df: DataFrame
        New DataFrame with column containing energy differences.
    """
    df_new = df2.iloc[:, 2].values - df1.iloc[:, 2]
    df_new.name = "Ediff"
    
    return pd.concat([df1, df_new.to_frame()], axis=1)

def check_model_exists(
    model: str, 
    best_model: str
) -> None:
    """
    Function used to check whether the models exist in the directory.

    Parameters
    ----------
    model: str
        Path for model.
    best_model: str
        Path for best_model.

    Returns
    -------
    None
    """
    if os.path.exists(model):
        os.remove(model)
        print("Deleted file: model")
    else:
        print("File model does not exist")

    if os.path.exists(best_model):
        os.remove(best_model)
        print("Deleted file: best_model")
    else:
        print("File best_model does not exist")
    
    return None


