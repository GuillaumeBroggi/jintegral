import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import cKDTree


def compute_interp_step(x, y, agg="mean"):
    # compute x and y spacing approximate for a grid interpolation
    # TODO implement a all-nearest-neigbors algorithm : https://stackoverflow.com/questions/34812372/interpolate-unstructured-x-y-z-data-on-best-grid-based-on-nearest-neighbour-dist

    xy = np.c_[x, y]  # X,Y data converted for use with KDTree
    tree = cKDTree(xy)  # Create KDtree for X,Y coordinates.

    # Calculate step
    distances, _ = tree.query(xy, k=2)  # Query distances for X,Y points
    distances = distances[:, 1:].reshape(-1)  # Remove k=1 zero distances
    if agg == "mean":
        step = np.mean(distances)
    elif agg == "min":
        sample = int(distances.shape[0] * 0.1)
        idx = np.argpartition(distances, sample)[:sample]
        print(idx.shape)
        step = np.mean(distances[idx])
    elif agg == "max":
        sample = int(distances.shape[0] * 0.1)
        idx = np.argpartition(distances, -sample)[-sample:]
        step = np.mean(distances[idx])
    else:
        raise ValueError
    return step


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def load_mts_data(path):
    # MTS files contains intermediate headers - it's an issue
    # Load the file as bulk text
    # Drop duplicates to drop intermediate headers - fist one is kept
    # Drop NA to remove blank lines
    # Drop first reminaing lines which are metadata
    df = (
        pd.read_csv(path, skiprows=3, header=None).drop_duplicates()
        # .dropna()
        .drop(index=[0, 1, 2])
    )

    # Some intermediate headers are timeds so drop duplicates ignores them : drop them manally by string query
    df = df[~df[0].str.match("Data Header:")]
    # Split ans expand the text
    df = df[0].str.split("\s+", expand=True)
    # print(df[0])
    # print(df[0].str.split("\s+", expand=True))

    # Use first row as headers
    df.columns = [str(i).lower() for i in df.iloc[0]]
    df = df[["time", "displacement", "force", "strain", "torque"]]
    df.rename(columns=dict(strain="cod"), inplace=True)

    # Extract units
    units = df.iloc[1].to_dict()

    # Remove header and unit rows
    df = df.drop(df.index[[0, 1]])

    df = df.reset_index(drop=True)

    df = df.astype("float64")

    # Correct by clip gage calibration
    df["cod"] /= 0.9679

    # shift

    df["time"] -= df["time"].min()

    return df


# griddata.py - 2010-07-11 ccampo
def binning2d(x, y, z, binsize=0.01, retbin=True, retloc=True):
    """
    Place unevenly spaced 2D data on a grid by 2D binning (nearest
    neighbor interpolation).

    Parameters
    ----------
    x : ndarray (1D)
        The idependent data x-axis of the grid.
    y : ndarray (1D)
        The idependent data y-axis of the grid.
    z : ndarray (1D)
        The dependent data in the form z = f(x,y).
    binsize : scalar, optional
        The full width and height of each bin on the grid.  If each
        bin is a cube, then this is the x and y dimension.  This is
        the step in both directions, x and y. Defaults to 0.01.
    retbin : boolean, optional
        Function returns `bins` variable (see below for description)
        if set to True.  Defaults to True.
    retloc : boolean, optional
        Function returns `wherebins` variable (see below for description)
        if set to True.  Defaults to True.

    Returns
    -------
    grid : ndarray (2D)
        The evenly gridded data.  The value of each cell is the median
        value of the contents of the bin.
    bins : ndarray (2D)
        A grid the same shape as `grid`, except the value of each cell
        is the number of points in that bin.  Returns only if
        `retbin` is set to True.
    wherebin : list (2D)
        A 2D list the same shape as `grid` and `bins` where each cell
        contains the indicies of `z` which contain the values stored
        in the particular bin.

    Revisions
    ---------
    2010-07-11  ccampo  Initial version
    """
    # get extrema values.
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # make coordinate arrays.
    xi = np.arange(xmin, xmax + binsize, binsize)
    yi = np.arange(ymin, ymax + binsize, binsize)
    xi, yi = np.meshgrid(xi, yi)

    # make the grid.
    grid = np.zeros(xi.shape, dtype=x.dtype)
    nrow, ncol = grid.shape
    if retbin:
        bins = np.copy(grid)

    # create list in same shape as grid to store indices
    if retloc:
        wherebin = np.copy(grid)
        wherebin = wherebin.tolist()

    # fill in the grid.
    for row in range(nrow):
        for col in range(ncol):
            xc = xi[row, col]  # x coordinate.
            yc = yi[row, col]  # y coordinate.

            # find the position that xc and yc correspond to.
            posx = np.abs(x - xc)
            posy = np.abs(y - yc)
            ibin = np.logical_and(posx < binsize / 2.0, posy < binsize / 2.0)
            ind = np.where(ibin == True)[0]

            # fill the bin.
            bin = z[ibin]
            if retloc:
                wherebin[row][col] = ind
            if retbin:
                bins[row, col] = bin.size
            if bin.size != 0:
                binval = np.median(bin)
                grid[row, col] = binval
            else:
                grid[row, col] = np.nan  # fill empty bins with nans.

    xi = np.arange(xmin, xmax + binsize, binsize)
    yi = np.arange(ymin, ymax + binsize, binsize)

    # return the grid
    if retbin:
        if retloc:
            return grid, xi, yi, bins, wherebin
        else:
            return grid, xi, yi, bins
    else:
        if retloc:
            return grid, xi, yi, wherebin
        else:
            return grid, xi, yi


def compute_lag(load_df_1, load_df_2, method="max_force", df1_frame_at_max_load=None):
    """
    Assume both df starts at t=0s and positive load. Returns the lag of the first df
    """
    if method not in ("max_force", "cross_corr", "manual_max_force"):
        raise ValueError(
            "Method should be either max_force, cross_corr or manual_max_force."
        )

    load_df_1 = load_df_1.copy()
    load_df_2 = load_df_2.copy()

    if method == "cross_corr":
        sampling_period_1 = compute_sampling_period(load_df_1)
        sampling_period_2 = compute_sampling_period(load_df_2)

        if sampling_period_1 > sampling_period_2:
            sampling_period = sampling_period_1
            load_df_2 = resample(load_df_2, sampling_period)

        else:
            sampling_period = sampling_period_2
            load_df_1 = resample(load_df_1, sampling_period)

        sig1 = load_df_1["force"] - load_df_1["force"].mean()

        sig2 = load_df_2["force"] - load_df_2["force"].mean()

        corr = np.correlate(a=sig1, v=sig2, mode="full")

        lag = corr.argmax() - (len(sig2) - 1)

        lag *= sampling_period

    elif method == "max_force":
        time1_at_max_load = load_df_1.loc[load_df_1["force"].abs().idxmax()]["time"]
        time2_at_max_load = load_df_2.loc[load_df_2["force"].abs().idxmax()]["time"]

        lag = time1_at_max_load - time2_at_max_load

    elif method == "manual_max_force":
        time1_at_max_load = load_df_1.query("frame == @df1_frame_at_max_load")[
            "time"
        ].to_numpy()
        time2_at_max_load = load_df_2.loc[load_df_2["force"].abs().idxmax()]["time"]
        lag = time1_at_max_load - time2_at_max_load

    # fig = go.Figure()
    # fig.add_scatter(x=load_df_1.time, y=load_df_1.force)
    # fig.add_scatter(x=load_df_2.time + lag, y=load_df_2.force)
    # fig.show()

    return lag


def compute_sampling_period(load_df):
    sampling_period = load_df["time"].sub(load_df["time"].shift()).mean(skipna=True)

    return sampling_period


def resample(load_df, sampling_period):
    # sampling period in s
    load_df.set_index("time", inplace=True)
    load_df.index = pd.to_datetime(load_df.index, unit="s")

    # express sampling period in microseconds (U)
    load_df = load_df.resample("{}U".format(int(sampling_period * 1000000))).nearest()

    # Datetime is time (int) in nanosecond
    load_df["time"] = load_df.index.astype("int64") / 10**9

    load_df.reset_index(drop=True, inplace=True)

    return load_df


# def identify_propagation_points(load_df):

#     load_df["dforce"] = np.gradient(load_df["force"], edge_order=2)

#     load_df["dforce"] = pywt.threshold(
#         data=load_df["dforce"],
#         value=ut.threshold_value(coeffs=load_df["dforce"], method="sqtwolog"),
#         mode="hard",
#     )

#     idx = load_df[
#         (load_df["dforce"].shift(1).gt(load_df["dforce"]))
#         & (load_df["dforce"].shift(-1).gt(load_df["dforce"]))
#     ].index

#     n = 2
#     idx_max = load_df.shape[0] - 1

#     load_df["prepropagation"] = False
#     load_df["postpropagation"] = False

#     for i in idx:
#         load_df.loc[
#             load_df.loc[np.arange(max(i - n, 0), min(i + n + 1, idx_max))][
#                 "force"
#             ].idxmax(),
#             "prepropagation",
#         ] = True
#         load_df.loc[
#             load_df.loc[np.arange(max(i - n, 0), min(i + n + 1, idx_max))][
#                 "force"
#             ].idxmin(),
#             "postpropagation",
#         ] = True

# fig = px.line(data_frame=load_df, x="time", y="dforce")
# fig.add_scatter(x=load_df["time"], y=load_df["force"], mode="lines")
# fig.add_scatter(
#     x=load_df.query("prepropagation == True")["time"],
#     y=load_df.query("prepropagation == True")["force"],
#     mode="markers",
# )
# fig.add_scatter(
#     x=load_df.query("postpropagation == True")["time"],
#     y=load_df.query("postpropagation == True")["force"],
#     mode="markers",
# )
# fig.show()


def load_daq_data(path, usecols=("Count", "Displacement", "Force", "Time_0")):
    df = pd.read_csv(
        path,
        usecols=usecols,
    )
    df.rename(
        columns=dict(
            Count="frame", Displacement="displacement", Force="force", Time_0="time"
        ),
        inplace=True,
    )

    df["time"] -= df["time"].min()

    # Depending of the wire polarity, the force may have been recorded as negative values
    try:
        if df["force"].mean() < 0:
            df["force"] *= -1
    except KeyError:
        pass

    return df


def synchronize_data(self, method="max_force", frame_at_max_load=None):
    self.daq_df["time"] -= compute_lag(
        load_df_1=self.daq_df,
        load_df_2=self.mts_df,
        method=method,
        df1_frame_at_max_load=frame_at_max_load,
    )


def remove_V_outliers(
    df, e1_min=None, e1_max=None, e2_min=None, e2_max=None, plot=False
):
    # grad helps identifying outliers when using an appropriate filter
    df["grad"] = np.gradient(df["V"])

    th_value = threshold_value(coeffs=df["grad"].abs())

    if not e1_min:
        e1_min = df["X"].min()
    if not e1_max:
        e1_max = df["X"].max()
    if not e2_min:
        e2_min = df["Y"].min()
    if not e2_max:
        e2_max = df["Y"].max()

    keeped = df.query(
        "(abs(grad)<@th_value) | (X<@e1_min) | (X>@e1_max) | (Y<@e2_min) | (Y>@e2_max)"
    )

    if plot:
        # Plot for presentation purposes
        removed = df.query(
            "~((abs(grad)<@th_value) | (X<@e1_min) | (X>@e1_max) | (Y<@e2_min) | (Y>@e2_max))"
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=keeped["X"],
                y=keeped["Y"],
                z=keeped["V"],
                mode="markers",
                name="keeped",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=removed["X"],
                y=removed["Y"],
                z=removed["V"],
                mode="markers",
                name="removed",
            )
        )
        fig.show()

    return keeped


def compute_mad(coeffs):
    """
    Mean Absolute Deviation (MAD)
    """
    abs_coeffs = np.abs(coeffs)
    mad = np.median(abs_coeffs) / 0.6745

    return mad


def threshold_value(coeffs, method="sqtwolog"):
    """
    Source : pyawt library
    """
    coeffs = coeffs.copy()
    mad = compute_mad(coeffs=coeffs)

    coeffs /= mad

    n = len(coeffs)

    if method == "sqtwolog":
        th_value = np.sqrt(2 * np.log(n))

    elif method == "rigrsure":
        th_value = sure_treshold(coeffs=coeffs)

    elif method == "heursure":
        dyad_length = compute_dyad_length(coeffs=coeffs)
        magic = np.sqrt(2 * np.log(n))
        eta = (np.linalg.norm(coeffs) ** 2 - n) / n
        crit = dyad_length ** (1.5) / np.sqrt(n)
        if eta < crit:
            th_value = magic
        else:
            th_value = np.min((sure_treshold(coeffs), magic))
            mad

    elif method == "minimaxi":
        lamlist = [
            0,
            0,
            0,
            0,
            0,
            1.27,
            1.474,
            1.669,
            1.860,
            2.048,
            2.232,
            2.414,
            2.594,
            2.773,
            2.952,
            3.131,
            3.310,
            3.49,
            3.67,
            3.85,
            4.03,
            4.21,
        ]
        dyad_length = compute_dyad_length(coeffs=coeffs)
        if dyad_length <= len(lamlist):
            th_value = lamlist[dyad_length - 1]
        else:
            th_value = 4.21 + (dyad_length - len(lamlist)) * 0.18

    else:
        print("Warning: no threshold methode, defaulted to 0")
        th_value = 0

    th_value *= mad
    # print(th_value)
    return th_value


def sure_treshold(coeffs):
    """
    Source : pyawt library
    """
    n = np.size(coeffs)

    a = np.sort(np.abs(coeffs)) ** 2

    c = np.linspace(n - 1, 0, n)
    s = np.cumsum(a) + c * a
    risk = (n - (2 * np.arange(n)) + s) / n
    ibest = np.argmin(risk)
    th_value = np.sqrt(a[ibest])
    return th_value


def compute_dyad_length(coeffs):
    n = len(coeffs)
    dyad_length = np.ceil(np.log(n) / np.log(2)).astype(np.int)
    return dyad_length
