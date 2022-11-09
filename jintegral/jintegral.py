from cmath import nan
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from jintegral.utils import remove_V_outliers
from scipy.interpolate import (
    SmoothBivariateSpline,
    griddata,
)
from csaps import csaps
import pandas as pd
import pickle
from jintegral.fig_template import fig_template, color_discrete_sequence
import logging


class Jintegral:
    """
    A Jintegral computing unit representation.
    Expose the methods to prepare and achieve a Jintegral calculation.
    The init arguments control how the data are prepared and should not vary during a given experiments.
    Additional parameters (such as offset) are expected to vary and are defined at calculation time when calling the appropriate method.

    Args:
        fit_method: method used to fit the discrete data with a differentiable representation. Chose between "sbs" and "csaps". The later was found to produce the best results. See Notes for more details.
        pre_fit_method: method used to fit the discrete data over a regular grid as required by "csaps"
        spline_degree: spline degree used when using the "sbs" method
        csaps_smooth: csaps smoothin parameters. See csaps documentation


    Returns:
        fem_df: a dataframe containing the fem data

    """

    def __init__(
        self,
        fit_method: str,
        pre_fit_method: str,
        spline_degree: int,
        gd_method,
        missing_values_method,
        step_size,
        crack_tip_method,
        csaps_smooth=None,
        fitting_offset_post_crack=0.5,
        fitting_offset_upper_crack=0,
        fitting_offset_lower_crack=0,
    ) -> None:

        self.spline_degree = spline_degree
        self.fit_method = fit_method
        self.pre_fit_method = pre_fit_method
        self.gd_method = gd_method
        self.missing_values_method = missing_values_method
        self.step_size = step_size
        self.csaps_smooth = csaps_smooth
        self.fit_normalization = {}
        self.stress_df = pd.DataFrame()
        self.fitting_offset_post_crack = fitting_offset_post_crack
        self.fitting_offset_upper_crack = fitting_offset_upper_crack
        self.fitting_offset_lower_crack = fitting_offset_lower_crack
        self.crack_tip_method = crack_tip_method

        self.fields = [
            dict(
                name="u1_fit_upper_crack",
                query="Y >= @upper_field_e2_limit + @self.fitting_offset_upper_crack",
                field="U",
            ),
            dict(
                name="u2_fit_upper_crack",
                query="Y >= @upper_field_e2_limit + @self.fitting_offset_upper_crack",
                field="V",
            ),
            dict(
                name="u1_fit_lower_crack",
                query="Y <= @lower_field_e2_limit - @self.fitting_offset_lower_crack",
                field="U",
            ),
            dict(
                name="u2_fit_lower_crack",
                query="Y <= @lower_field_e2_limit - @self.fitting_offset_lower_crack",
                field="V",
            ),
            dict(
                name="u1_fit_post_crack",
                query="(X >= (@crack_tip['e1'] + @self.fitting_offset_post_crack))",
                field="U",
            ),
            dict(
                name="u2_fit_post_crack",
                query="(X >= (@crack_tip['e1'] +  @self.fitting_offset_post_crack))",
                field="V",
            ),
            dict(
                name="error_fit",
                query="all",
                field="error",
            ),
            # TODO : csaps fitting is not required for error (no derivative calculation): assess performances
        ]

    def dump_displacement_field_fit(self, frame, dump_path):

        dump_path.mkdir(parents=True, exist_ok=True)
        for field in self.fields:
            with open(
                dump_path / f"frame-{frame.id:0>4d}_{field['name']}.pickle",
                "wb",
            ) as file:
                pickle.dump(getattr(self, field["name"]), file)

    def load_displacement_field_fit(self, frame, dump_path):
        """
        Load a displacement field previously fitted and saved as a pickle object on disk to speed up running time when the fitting conditions are not changed"""
        for field in self.fields:
            with open(
                dump_path / f"frame-{frame.id:0>4d}_{field['name']}.pickle",
                "rb",
            ) as file:
                fit = pickle.load(file)
            setattr(self, field["name"], fit)

    def fit_displacement_field(self, frame, crack):

        # TODO algorithm for outlier
        # TODO generalization for not straight and horizontal crack

        # Compute utility fields to select the fititng area
        crack_tip = crack.get_crack_tip(frame_id=frame.id, method=self.crack_tip_method)
        crack_data = crack.crack_data[frame.id]

        # Upper field limit is defined with respect to the crack tip, looking for a e2 position with enough data points and coherent displacement

        upper_field_e2_limit, _, _ = crack._identify_cod_profile_e2_position(
            frame=frame,
            e1_crack_tip=crack_tip["e1"],
            e2_crack_tip=crack_tip["e2"],
            offset_multiplier=1,
            allowable_missing_points_prop=0.2,
            number_of_point_per_bin=1,
            e2_bins=1,
        )

        # Idem for lower field limit

        lower_field_e2_limit, _, _ = crack._identify_cod_profile_e2_position(
            frame=frame,
            e1_crack_tip=crack_tip["e1"],
            e2_crack_tip=crack_tip["e2"],
            offset_multiplier=-1,
            allowable_missing_points_prop=0.2,
            number_of_point_per_bin=1,
            e2_bins=1,
        )

        for fit in self.fields:

            logging.info(f"Frame {frame.id}: Starting {fit['name']} fitting.")

            if fit["query"] == "all":
                df = frame.displacement_field_df.copy()

            else:

                df = frame.displacement_field_df.query(fit["query"]).copy()

            # # TODO Normalization

            # x_max = df["X"].abs().max()
            # df["X"] /= x_max

            # y_max = df["Y"].abs().max()
            # df["Y"] /= y_max

            # field_max = df[fit["field"]].abs().max()
            # df[fit["field"]] /= field_max

            # self.fit_normalization.update(
            #     {fit["name"]: dict(x_max=x_max, y_max=y_max, field_max=field_max)}
            # )

            if self.fit_method == "sbs":

                setattr(
                    self,
                    fit["name"],
                    SmoothBivariateSpline(
                        x=df["X"],
                        y=df["Y"],
                        z=df[fit["field"]],
                        kx=self.spline_degree,
                        ky=self.spline_degree,
                    ),
                )
            elif self.fit_method == "csaps":

                # fit over a grid as csaps requires structured data

                if self.pre_fit_method == "gd":

                    offset_multiplier = dict(
                        e1_start=0, e2_start=0, e1_stop=0, e2_stop=0
                    )

                    step = frame.interp_step
                    # print(df["X"].min(), df["X"].max(), frame.interp_step)
                    e1 = np.arange(
                        start=df["X"].min(),
                        stop=df["X"].max() + frame.interp_step,
                        step=frame.interp_step,
                    )
                    e2 = np.arange(
                        start=df["Y"].min(),
                        stop=df["Y"].max() + frame.interp_step,
                        step=frame.interp_step,
                    )

                    e1_grid, e2_grid = np.meshgrid(e1, e2, indexing="ij")

                    field_over_grid = griddata(
                        points=df[["X", "Y"]].to_numpy(),
                        values=df[fit["field"]].to_numpy(),
                        xi=(e1_grid, e2_grid),
                        method="cubic",
                    )

                    nan_iterp_data = np.isnan(field_over_grid)

                    offset_multiplier = dict(
                        e1_start=0, e2_start=0, e1_stop=0, e2_stop=0
                    )

                    if nan_iterp_data.any and self.missing_values_method == "reduce":

                        while nan_iterp_data.any():

                            # brute force to detect where point should be removed

                            nan_values = dict(
                                e1_start=np.sum(
                                    np.isnan(
                                        field_over_grid[offset_multiplier["e1_start"]]
                                    )
                                ),
                                e2_start=np.sum(
                                    np.isnan(
                                        field_over_grid[
                                            :, offset_multiplier["e2_start"]
                                        ]
                                    )
                                ),
                                e1_stop=np.sum(
                                    np.isnan(
                                        field_over_grid[
                                            -1 - 1 * offset_multiplier["e1_stop"]
                                        ]
                                    )
                                ),
                                e2_stop=np.sum(
                                    np.isnan(
                                        field_over_grid[
                                            :, -1 - 1 * offset_multiplier["e2_stop"]
                                        ]
                                    )
                                ),
                            )

                            # get side where nan_values is the greater

                            side = sorted(nan_values, key=nan_values.get)[-1]

                            # Increcment corresponding offset
                            offset_multiplier[side] += 1

                            # Update
                            nan_iterp_data = np.isnan(
                                field_over_grid[
                                    slice(
                                        offset_multiplier["e1_start"],
                                        field_over_grid.shape[0]
                                        - 1 * offset_multiplier["e1_stop"],
                                    ),
                                    slice(
                                        offset_multiplier["e2_start"],
                                        field_over_grid.shape[1]
                                        - 1 * offset_multiplier["e2_stop"],
                                    ),
                                ]
                            )

                        # Remove nan
                        field_over_grid = field_over_grid[
                            slice(
                                offset_multiplier["e1_start"],
                                field_over_grid.shape[0]
                                - 1 * offset_multiplier["e1_stop"],
                            ),
                            slice(
                                offset_multiplier["e2_start"],
                                field_over_grid.shape[1]
                                - 1 * offset_multiplier["e2_stop"],
                            ),
                        ]

                        e1_grid = e1_grid[
                            slice(
                                offset_multiplier["e1_start"],
                                e1_grid.shape[0] - 1 * offset_multiplier["e1_stop"],
                            ),
                            slice(
                                offset_multiplier["e2_start"],
                                e1_grid.shape[1] - 1 * offset_multiplier["e2_stop"],
                            ),
                        ]

                        e2_grid = e2_grid[
                            slice(
                                offset_multiplier["e1_start"],
                                e2_grid.shape[0] - 1 * offset_multiplier["e1_stop"],
                            ),
                            slice(
                                offset_multiplier["e2_start"],
                                e2_grid.shape[1] - 1 * offset_multiplier["e2_stop"],
                            ),
                        ]

                    elif (
                        nan_iterp_data.any() and self.missing_values_method == "nearest"
                    ):
                        # Replace nan by nearest point.
                        nearest_field_over_grid = griddata(
                            points=df[["X", "Y"]].to_numpy(),
                            values=df[fit["field"]].to_numpy(),
                            xi=(e1_grid, e2_grid),
                            method="nearest",
                        )
                        mask = np.isnan(field_over_grid)
                        field_over_grid[mask] = nearest_field_over_grid[mask]
                        nan_iterp_data = np.isnan(field_over_grid)

                elif self.pre_fit_method == "sbs":

                    sbs_fit = SmoothBivariateSpline(
                        x=df["X"], y=df["Y"], z=df[fit["field"]]
                    )

                    e1 = np.arange(
                        start=df["X"].min(),
                        stop=df["X"].max(),
                        step=step,
                    )
                    e2 = np.arange(
                        start=df["Y"].min(),
                        stop=df["Y"].max(),
                        step=step,
                    )

                    e1_grid, e2_grid = np.meshgrid(e1, e2, indexing="ij")

                    field_over_grid = sbs_fit(x=e1_grid, y=e2_grid, grid=False)

                fit["fit"] = csaps(
                    xdata=(e1_grid[:, 0], e2_grid[0]),
                    ydata=field_over_grid,
                    smooth=self.csaps_smooth,
                    normalizedsmooth=True,
                )
                setattr(self, fit["name"], fit["fit"])

    def define_contour_coordinates(
        self, frame, crack_tip, offset, offset_mode="tip", closed_path=False
    ):

        """
        Define the contour as a list of dict containing the contour coordinates. Assuming a rectangular like contour.
        If tip, the offset is defined with respect to the crack tip
        """

        A_e2 = crack_tip["e2"] - offset["A"]  # safety margin from crack surface
        F_e2 = crack_tip["e2"] + offset["A"]

        if offset_mode == "tip":
            A_e1 = crack_tip["e1"] - offset["AB"]

            F_e1 = crack_tip["e1"] - offset["AB"]

            B_e2 = crack_tip["e2"] - offset["BC"]

            C_e1 = crack_tip["e1"] + offset["CD"]

            D_e2 = crack_tip["e2"] + offset["BC"]

            self.contour = [
                {
                    "name": "A",
                    "e1": A_e1,
                    "e2": A_e2,
                },
                {
                    "name": "B",
                    "e1": A_e1,
                    "e2": B_e2,
                },
                {
                    "name": "C",
                    "e1": C_e1,
                    "e2": B_e2,
                },
                {
                    "name": "D",
                    "e1": C_e1,
                    "e2": D_e2,
                },
                {
                    "name": "E",
                    "e1": F_e1,
                    "e2": D_e2,
                },
                {
                    "name": "F",
                    "e1": F_e1,
                    "e2": F_e2,
                },
            ]

            if closed_path:
                raise NotImplementedError

        elif offset_mode == "edge":

            e1_min = frame.displacement_field_df["X"].min()
            e1_max = frame.displacement_field_df["X"].max()
            e2_min = frame.displacement_field_df["Y"].min()
            e2_max = frame.displacement_field_df["Y"].max()

            self.contour = [
                {
                    "name": "A",
                    "e1": e1_min + offset,
                    "e2": self.A_e2,
                },
                {
                    "name": "B",
                    "e1": e1_min + offset,
                    "e2": e2_min + offset,
                },
                {"name": "C", "e1": e1_max - offset, "e2": e2_min + offset},
                {"name": "D", "e1": e1_max - offset, "e2": e2_max - offset},
                {
                    "name": "E",
                    "e1": e1_min + offset,
                    "e2": e2_max - offset,
                },
                {
                    "name": "F",
                    "e1": e1_min + offset,
                    "e2": self.F_e2,
                },
            ]

            if closed_path:
                self.contour += [
                    # # Validate that a closed path without the crack tip shows a null Jintegral
                    {
                        "name": "F2",
                        "e1": e1_min + min(2, offset) + 2,
                        "e2": self.F_e2,
                    },
                    {
                        "name": "E2",
                        "e1": e1_min + min(2, offset) + 2,
                        "e2": e2_max - offset - 2,
                    },
                    {
                        "name": "D2",
                        "e1": e1_max - offset - 2,
                        "e2": e2_max - offset - 2,
                    },
                    {
                        "name": "C2",
                        "e1": e1_max - offset - 2,
                        "e2": e2_min + offset + 2,
                    },
                    {
                        "name": "B2",
                        "e1": e1_min + min(2, offset) + 2,
                        "e2": e2_min + offset + 2,
                    },
                    {
                        "name": "A2",
                        "e1": e1_min + min(2, offset) + 2,
                        "e2": self.A_e2,
                    },
                    {
                        "name": "A",
                        "e1": e1_min + min(2, offset),
                        "e2": self.A_e2,
                    },
                ]

    def define_area_coordinates(
        self, frame, crack_tip, offset, width, offset_mode="tip", closed_path=False
    ):

        """
        Define the area contour as a list of dict containing the contour coordinates. Assuming a rectangular like contour.
        If tip, the offset is defined with respect to the crack tip
        """

        A_e2 = crack_tip["e2"] - offset["A"]  # safety margin from crack surface
        F_e2 = crack_tip["e2"] + offset["A"]

        if offset_mode == "tip":
            A_e1 = crack_tip["e1"] - offset["AB"]

            B_e2 = crack_tip["e2"] - offset["BC"]

            C_e1 = crack_tip["e1"] + offset["CD"]

            D_e2 = crack_tip["e2"] + offset["BC"]

            F_e1 = crack_tip["e1"] - offset["AB"]

            self.area = [
                {
                    "name": "A1",
                    "e1": [A_e1 - width, A_e1, A_e1, A_e1 - width],
                    "e2": [A_e2, A_e2, B_e2, B_e2],
                },
                {
                    "name": "A2",
                    "e1": [A_e1 - width, A_e1, A_e1, A_e1 - width],
                    "e2": [B_e2, B_e2, B_e2 - width, B_e2 - width],
                },
                {
                    "name": "A3",
                    "e1": [A_e1, C_e1, C_e1, A_e1],
                    "e2": [B_e2, B_e2, B_e2 - width, B_e2 - width],
                },
                {
                    "name": "A4",
                    "e1": [C_e1, C_e1 + width, C_e1 + width, C_e1],
                    "e2": [B_e2, B_e2, B_e2 - width, B_e2 - width],
                },
                {
                    "name": "A5",
                    "e1": [C_e1, C_e1 + width, C_e1 + width, C_e1],
                    "e2": [D_e2, D_e2, B_e2, B_e2],
                },
                {
                    "name": "A6",
                    "e1": [C_e1, C_e1 + width, C_e1 + width, C_e1],
                    "e2": [D_e2 + width, D_e2 + width, D_e2, D_e2],
                },
                {
                    "name": "A7",
                    "e1": [A_e1, C_e1, C_e1, A_e1],
                    "e2": [D_e2 + width, D_e2 + width, D_e2, D_e2],
                },
                {
                    "name": "A8",
                    "e1": [A_e1 - width, A_e1, A_e1, A_e1 - width],
                    "e2": [D_e2 + width, D_e2 + width, D_e2, D_e2],
                },
                {
                    "name": "A9",
                    "e1": [A_e1 - width, A_e1, A_e1, A_e1 - width],
                    "e2": [D_e2, D_e2, F_e2, F_e2],
                },
            ]

            if closed_path:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def construct_jintegral_path(self):
        # create local aliases for readability
        e1 = (1, 0)
        e2 = (0, 1)

        self.jintegral_path = []

        for k, v in enumerate(self.contour[:-1]):
            # Loop over the contour to construct path vector (except last point)
            path = (
                self.contour[k + 1]["e1"] - self.contour[k]["e1"],
                self.contour[k + 1]["e2"] - self.contour[k]["e2"],
            )
            # print(self.contour[k], self.contour[k])
            # print(path)
            if not any(path):
                raise ValueError("Path can't be null")
            # Normalized path vector is used to determine path direction and orientation
            path_norm = path / np.linalg.norm(path)
            if (path_norm != e1).all() and (path_norm != e2).all():
                raise ValueError(
                    "Path vector is {v} but it should be horizontal or vertical.".format(
                        v=str(path_norm)
                    )
                )

            n1 = np.cross((e1), path_norm)
            n2 = np.cross((e2), path_norm)

            e1_grid, e2_grid, ds = self.derive_path_evaluation_grids(
                path=self.contour[k : k + 2],
                path_norm=path_norm,
                step_size=self.step_size,
            )

            path_name = self.contour[k]["name"] + self.contour[k + 1]["name"]

            self.jintegral_path.append(
                dict(
                    e1_grid=e1_grid,
                    e2_grid=e2_grid,
                    ds=ds,
                    n1=n1,
                    n2=n2,
                    path_norm=path_norm,
                    name=path_name,
                )
            )

    def construct_jintegral_area(self, width):

        self.jintegral_area = []

        for area in self.area:

            e1_grid, e2_grid = self.derive_area_evaluation_grids(
                area=area, step_size=self.step_size
            )
            q = self.calculate_q_over_grid(
                area=area, width=width, e1_grid=e1_grid, e2_grid=e2_grid
            )
            # print(q.shape)
            try:
                q_fit = csaps(
                    xdata=(e1_grid, e2_grid), ydata=q, smooth=1, normalizedsmooth=True
                )
            except ValueError as e:
                print(e1_grid, e2_grid)
                print(e1_grid.shape, e2_grid.shape)
                print(e)
                raise e

            self.jintegral_area.append(
                dict(
                    name=area["name"],
                    e1_grid=e1_grid,
                    e2_grid=e2_grid,
                    q=q,
                    q_fit=q_fit,
                    # ds=ds,
                    # n1=n1,
                    # n2=n2,
                    # path_norm=path_norm,
                    # name=path_name,
                )
            )

    def fit_path(self, path, crack_tip, field, dx=0, dy=0):

        if self.fit_method == "sbs":
            kwargs = dict(
                x=path["e1_grid"], y=path["e2_grid"], dx=dx, dy=dy, grid=False
            )

        elif self.fit_method == "csaps":
            # assume straigth paths
            # TODO interpolate arbitrary path?
            if np.all(path["e1_grid"] == path["e1_grid"][0]):
                # Means that e1 coordinates does not vary

                # csaps contructs a meshgrid with x vectors so one of the vectors should have one coordinate only

                kwargs = dict(
                    x=(path["e1_grid"][0], path["e2_grid"]),
                    nu=(dx, dy),
                )

            else:
                kwargs = dict(
                    x=(path["e1_grid"], path["e2_grid"][0]),
                    nu=(dx, dy),
                )

        if field == "error":
            fit = self.error_fit(**kwargs)

        elif (path["e2_grid"][0] <= crack_tip["e2"] + 0.1) and (
            path["e2_grid"][-1] <= crack_tip["e2"] + 0.1
        ):

            if field == "u1":

                fit = self.u1_fit_lower_crack(**kwargs)
            elif field == "u2":

                fit = self.u2_fit_lower_crack(**kwargs)
        elif (path["e2_grid"][0] >= crack_tip["e2"] - 0.1) and (
            path["e2_grid"][-1] >= crack_tip["e2"] - 0.1
        ):
            if field == "u1":

                fit = self.u1_fit_upper_crack(**kwargs)
            elif field == "u2":

                fit = self.u2_fit_upper_crack(**kwargs)
        elif (path["e1_grid"][0] >= crack_tip["e1"]) and (
            path["e1_grid"][-1] >= crack_tip["e1"]
        ):
            if field == "u1":

                u1_fit = self.u1_fit_post_crack(**kwargs).reshape(-1)

                fit = u1_fit

            elif field == "u2":

                fit = self.u2_fit_post_crack(**kwargs)

        return fit.reshape(-1)

    def fit_area(self, area, crack_tip, field, dx=0, dy=0):

        if self.fit_method == "sbs":
            raise NotImplementedError
            # kwargs = dict(
            #     x=path["e1_grid"], y=path["e2_grid"], dx=dx, dy=dy, grid=False
            # )

        elif self.fit_method == "csaps":

            kwargs = dict(
                x=(area["e1_grid"], area["e2_grid"]),
                nu=(dx, dy),
            )

        if field == "error":
            fit = self.error_fit(**kwargs)

        elif np.max(area["e2_grid"]) <= crack_tip["e2"] + 0.1:

            if field == "u1":

                fit = self.u1_fit_lower_crack(**kwargs)
            elif field == "u2":

                fit = self.u2_fit_lower_crack(**kwargs)
        elif np.min(area["e2_grid"]) >= crack_tip["e2"] - 0.1:
            if field == "u1":

                fit = self.u1_fit_upper_crack(**kwargs)
            elif field == "u2":

                fit = self.u2_fit_upper_crack(**kwargs)
        elif np.min(area["e1_grid"]) >= crack_tip["e1"]:
            if field == "u1":

                u1_fit = self.u1_fit_post_crack(**kwargs)

                fit = u1_fit

            elif field == "u2":

                fit = self.u2_fit_post_crack(**kwargs)
        # print(fit.shape)
        return fit.reshape(-1)

    @staticmethod
    def derive_path_evaluation_grids(path, path_norm, step_size=0.1):
        if np.abs(path_norm[0]) == 1:
            e1_grid = (
                np.arange(
                    min(path[0]["e1"], path[1]["e1"]) * 10,
                    max(path[0]["e1"], path[1]["e1"]) * 10 + 0.001,
                    step_size * 10,
                )
                / 10
            )
            e2_grid = np.full(len(e1_grid), path[0]["e2"])
            # Apply sign of contour to ds
            ds = e1_grid * path_norm[0]
        elif np.abs(path_norm[1]) == 1:
            e2_grid = (
                np.arange(
                    min(path[0]["e2"], path[1]["e2"]) * 10,
                    max(path[0]["e2"], path[1]["e2"]) * 10 + 0.001,
                    step_size * 10,
                )
                / 10
            )
            e1_grid = np.full(len(e2_grid), path[0]["e1"])
            # Apply sign of contour to ds
            ds = e2_grid * path_norm[1]
        else:
            raise ValueError
        return (e1_grid, e2_grid, ds)

    @staticmethod
    def derive_area_evaluation_grids(area, step_size=0.1):

        e1_grid = (
            np.arange(
                min(area["e1"]) * 10,
                max(area["e1"]) * 10 + 0.001,
                step_size * 10,
            )
            / 10
        )
        e2_grid = (
            np.arange(
                min(area["e2"]) * 10,
                max(area["e2"]) * 10 + 0.001,
                step_size * 10,
            )
            / 10
        )
        # ds = e1_grid * path_norm[0]

        return (e1_grid, e2_grid)

    @staticmethod
    def calculate_q_over_grid(area, width, e1_grid, e2_grid):

        e1_grid, e2_grid = np.meshgrid(e1_grid, e2_grid, indexing="ij")

        if area["name"] in ("A1", "A9"):

            q = 1 * (e1_grid - np.min(e1_grid)) / (np.max(e1_grid) - np.min(e1_grid))

        elif area["name"] in ("A3",):

            q = 1 * (e2_grid - np.min(e2_grid)) / (np.max(e2_grid) - np.min(e2_grid))

        elif area["name"] in ("A5",):

            q = 1 * (e1_grid - np.max(e1_grid)) / (np.min(e1_grid) - np.max(e1_grid))

        elif area["name"] in ("A7",):

            q = 1 * (e2_grid - np.max(e2_grid)) / (np.min(e2_grid) - np.max(e2_grid))

        elif area["name"] in ("A7",):

            q = 1 * (e2_grid - np.max(e2_grid)) / (np.min(e2_grid) - np.max(e2_grid))

        elif area["name"] in ("A2",):
            f = lambda x, y: np.sqrt((x - np.max(x)) ** 2 + (y - np.max(y)) ** 2)

            distance = f(e1_grid, e2_grid)

            f = lambda x: np.maximum(
                0, 1 * (width - x) / (np.max(distance) - np.min(distance))
            )

            q = f(distance)

        elif area["name"] in ("A4",):
            f = lambda x, y: np.sqrt((x - np.min(x)) ** 2 + (y - np.max(y)) ** 2)

            distance = f(e1_grid, e2_grid)

            f = lambda x: np.maximum(
                0, 1 * (width - x) / (np.max(distance) - np.min(distance))
            )

            q = f(distance)

        elif area["name"] in ("A6",):
            f = lambda x, y: np.sqrt((x - np.min(x)) ** 2 + (y - np.min(y)) ** 2)

            distance = f(e1_grid, e2_grid)

            f = lambda x: np.maximum(
                0, 1 * (width - x) / (np.max(distance) - np.min(distance))
            )

            q = f(distance)

        elif area["name"] in ("A8",):
            f = lambda x, y: np.sqrt((x - np.max(x)) ** 2 + (y - np.min(y)) ** 2)

            distance = f(e1_grid, e2_grid)

            f = lambda x: np.maximum(
                0, 1 * (width - x) / (np.max(distance) - np.min(distance))
            )

            q = f(distance)

        else:
            raise ValueError

        return q

    def compute_contour_jintegral(self, material, crack_tip, output_path_data=False):
        J_dict = {}
        self.stress_df = []
        error_tot = []
        error_mean = []

        for i, path in enumerate(self.jintegral_path):

            # print(path)

            strain_1 = self.fit_path(path=path, crack_tip=crack_tip, field="u1", dx=1)
            strain_2 = self.fit_path(path=path, crack_tip=crack_tip, field="u2", dy=1)
            strain_3 = self.fit_path(
                path=path, crack_tip=crack_tip, field="u1", dy=1
            ) + self.fit_path(path=path, crack_tip=crack_tip, field="u2", dx=1)
            strain = np.array([strain_1, strain_2, strain_3])
            stress = material.C.dot(strain)

            strain_energy = 1 / 2 * strain.transpose().dot(stress).diagonal()

            J_strain = (
                np.trapz(y=strain_energy, x=path["e2_grid"]) * path["path_norm"][1]
            )

            # sum required because norm is a vector
            J_t_11 = np.sum(
                path["path_norm"]
                * path["n1"]
                * np.trapz(
                    y=self.fit_path(path=path, crack_tip=crack_tip, field="u1", dx=1)
                    * stress[0],
                    x=path["ds"],
                ),
            )

            J_t_22 = np.sum(
                path["path_norm"]
                * path["n2"]
                * np.trapz(
                    y=self.fit_path(path=path, crack_tip=crack_tip, field="u2", dx=1)
                    * stress[1],
                    x=path["ds"],
                )
            )
            J_t_12 = np.sum(
                path["path_norm"]
                * path["n2"]
                * np.trapz(
                    y=self.fit_path(path=path, crack_tip=crack_tip, field="u1", dx=1)
                    * stress[2],
                    x=path["ds"],
                )
            )
            J_t_21 = np.sum(
                path["path_norm"]
                * path["n1"]
                * np.trapz(
                    y=self.fit_path(path=path, crack_tip=crack_tip, field="u2", dx=1)
                    * stress[2],
                    x=path["ds"],
                )
            )
            J_traction = -np.sum((J_t_11, J_t_22, J_t_12, J_t_21))

            if path["name"] == "BC" or path["name"] == "DE":
                J_simplified = -np.sum((J_t_11, J_t_12, J_t_21))

            elif path["name"] == "CD":
                J_simplified = -np.sum((J_t_22, J_t_12, J_t_21))
            else:
                J_simplified = J_traction

            J = J_strain + J_traction

            error = self.fit_path(path=path, crack_tip=crack_tip, field="error")

            error_tot.append(np.sum(error))
            error_mean.append(np.mean(error))

            J_dict.update(
                {
                    "J_" + path["name"]: J,
                    "J_traction_" + path["name"]: J_traction,
                    "J_strain_" + path["name"]: J_strain,
                    "J_simplified_" + path["name"]: J_simplified,
                    # "J_t_u1_" + path["name"]: J_t_u1,
                    # "J_t_u2_" + path["name"]: J_t_u2,
                    "J_t_11_" + path["name"]: J_t_11,
                    "J_t_22_" + path["name"]: J_t_22,
                    "J_t_12_" + path["name"]: J_t_12,
                    "J_t_21_" + path["name"]: J_t_21,
                    "error_tot" + path["name"]: error_tot[-1],
                    "error_mean" + path["name"]: error_mean[-1],
                }
            )

            if output_path_data:
                stress_dict = {
                    "stress_11": stress[0],
                    "stress_22": stress[1],
                    "stress_12": stress[2],
                    "fit_11": self.fit_path(
                        path=path, crack_tip=crack_tip, field="u1", dx=1
                    )
                    * stress[0],
                    "fit_22": self.fit_path(
                        path=path, crack_tip=crack_tip, field="u2", dx=1
                    )
                    * stress[1],
                    "fit_21": self.fit_path(
                        path=path, crack_tip=crack_tip, field="u2", dx=1
                    )
                    * stress[2],
                    "strain_energy": strain_energy,
                    "du1_de1": self.fit_path(
                        path=path, crack_tip=crack_tip, field="u1", dx=1
                    ),
                    "du2_de1": self.fit_path(
                        path=path, crack_tip=crack_tip, field="u2", dx=1
                    ),
                    "e1_grid": path["e1_grid"],
                    "e2_grid": path["e2_grid"],
                    "contour": path["name"],
                }

                self.stress_df.append(stress_dict)

        if output_path_data:

            self.stress_df = pd.DataFrame(self.stress_df)

        # Sum J for each path
        J_tot = sum(
            [v for k, v in J_dict.items() if ("J_traction" in k) or ("J_strain" in k)]
        )
        J_simplified = sum(
            [v for k, v in J_dict.items() if ("J_simplified" in k) or ("J_strain" in k)]
        )

        error_tot = sum(error_tot)
        error_mean = np.mean(error_mean)

        J_dict.update(dict(J=J_tot, error_tot=error_tot, error_mean=error_mean))
        J_dict.update(dict(J_simplified=J_simplified))
        # J_dict.update(
        #     dict(
        #         offset=offset,
        #         fit_method=self.fit_method,
        #         spline_degree=self.spline_degree,
        #         pre_fit_method=self.pre_fit_method,
        #         gd_method=self.gd_method,
        #         missing_values_method=self.missing_values_method,
        #         csaps_smooth=self.csaps_smooth,
        #     )
        # )

        self.J_dict = J_dict

    def compute_area_jintegral(self, material, crack_tip):
        J_tot = []
        error_tot = []
        error_mean = []
        J_dict = {}
        self.stress_df = []

        for i, area in enumerate(self.jintegral_area):

            # print(area["e1_grid"].shape)

            strain_1 = self.fit_area(area=area, crack_tip=crack_tip, field="u1", dx=1)
            strain_2 = self.fit_area(area=area, crack_tip=crack_tip, field="u2", dy=1)
            strain_3 = self.fit_area(
                area=area, crack_tip=crack_tip, field="u1", dy=1
            ) + self.fit_area(area=area, crack_tip=crack_tip, field="u2", dx=1)
            strain = np.array([strain_1, strain_2, strain_3])
            # print(strain.shape)
            stress = material.C.dot(strain)

            strain_energy = 1 / 2 * strain.transpose().dot(stress).diagonal()

            strain_energy = strain_energy.reshape(
                area["e1_grid"].shape[0], area["e2_grid"].shape[0]
            )

            # The J strain is only evaluated for 11, the kronecker delta is  null otherwise
            dx = 1
            dy = 0
            x = (area["e1_grid"], area["e2_grid"])
            J_strain = np.trapz(
                y=np.trapz(
                    y=strain_energy * area["q_fit"](x=x, nu=(dx, dy)),
                    x=area["e2_grid"],
                ),
                x=area["e1_grid"],
            )

            J_t_11 = np.trapz(
                y=np.trapz(
                    y=self.fit_area(
                        area=area, crack_tip=crack_tip, field="u1", dx=1
                    ).reshape(area["e1_grid"].shape[0], area["e2_grid"].shape[0])
                    * stress[0].reshape(
                        area["e1_grid"].shape[0], area["e2_grid"].shape[0]
                    )
                    * area["q_fit"](x=x, nu=(dx, dy)),
                    x=area["e2_grid"],
                ),
                x=area["e1_grid"],
            )

            dx = 0
            dy = 1
            J_t_22 = np.trapz(
                y=np.trapz(
                    y=self.fit_area(
                        area=area, crack_tip=crack_tip, field="u2", dx=1
                    ).reshape(area["e1_grid"].shape[0], area["e2_grid"].shape[0])
                    * stress[1].reshape(
                        area["e1_grid"].shape[0], area["e2_grid"].shape[0]
                    )
                    * area["q_fit"](x=x, nu=(dx, dy)),
                    x=area["e2_grid"],
                ),
                x=area["e1_grid"],
            )

            dx = 0
            dy = 1
            J_t_12 = np.trapz(
                y=np.trapz(
                    y=self.fit_area(
                        area=area, crack_tip=crack_tip, field="u1", dx=1
                    ).reshape(area["e1_grid"].shape[0], area["e2_grid"].shape[0])
                    * stress[2].reshape(
                        area["e1_grid"].shape[0], area["e2_grid"].shape[0]
                    )
                    * area["q_fit"](x=x, nu=(dx, dy)),
                    x=area["e2_grid"],
                ),
                x=area["e1_grid"],
            )

            dx = 1
            dy = 0
            J_t_21 = np.trapz(
                y=np.trapz(
                    y=self.fit_area(
                        area=area, crack_tip=crack_tip, field="u2", dx=1
                    ).reshape(area["e1_grid"].shape[0], area["e2_grid"].shape[0])
                    * stress[2].reshape(
                        area["e1_grid"].shape[0], area["e2_grid"].shape[0]
                    )
                    * area["q_fit"](x=x, nu=(dx, dy)),
                    x=area["e2_grid"],
                ),
                x=area["e1_grid"],
            )

            J_traction = np.sum((J_t_11, J_t_22, J_t_12, J_t_21))

            J = J_traction - J_strain
            J_tot.append(J)

            error = self.fit_area(area=area, crack_tip=crack_tip, field="error")

            error_tot.append(np.sum(error))
            error_mean.append(np.mean(error))

            J_dict.update(
                {
                    "J_" + area["name"]: J,
                    "J_traction_" + area["name"]: J_traction,
                    "J_strain_" + area["name"]: J_strain,
                    "J_t_11_" + area["name"]: J_t_11,
                    "J_t_22_" + area["name"]: J_t_22,
                    "J_t_12_" + area["name"]: J_t_12,
                    "J_t_21_" + area["name"]: J_t_21,
                    "error_tot" + area["name"]: error_tot[-1],
                    "error_mean" + area["name"]: error_mean[-1],
                }
            )

        J_dict.update(
            dict(J=sum(J_tot), error_tot=sum(error_tot), error_mean=np.mean(error_mean))
        )

        self.J_dict = J_dict

    def construct_fit_plots(self, frame, crack_tip):

        self.u1_fig = go.Figure()

        for i in [
            dict(
                name="u1_fit_upper_crack",
                fit=self.u1_fit_upper_crack,
                e1_min=frame.e1_min,
                e1_max=frame.e1_max,
                e2_min=crack_tip["e2"],
                e2_max=frame.e2_max,
            ),
            dict(
                name="u1_fit_lower_crack",
                fit=self.u1_fit_lower_crack,
                e1_min=frame.e1_min,
                e1_max=frame.e1_max,
                e2_min=frame.e2_min,
                e2_max=crack_tip["e2"],
            ),
            dict(
                name="u1_fit_post_crack",
                fit=self.u1_fit_post_crack,
                e1_min=crack_tip["e1"],
                e1_max=frame.e1_max,
                e2_min=frame.e2_min,
                e2_max=frame.e2_max,
            ),
        ]:

            e1 = np.arange(i["e1_min"] * 10, i["e1_max"] * 10 + 5, 5) / 10
            e2 = np.arange(i["e2_min"] * 10, i["e2_max"] * 10 + 5, 5) / 10

            plot_grid_e1, plot_grid_e2 = np.meshgrid(e1, e2, indexing="ij")
            plot_grid_e1 = plot_grid_e1.reshape(-1, 1).squeeze()
            plot_grid_e2 = plot_grid_e2.reshape(-1, 1).squeeze()
            dx = 0
            dy = 0

            if self.fit_method == "sbs":
                kwargs = dict(x=e1, y=e2, grid=True, dx=dx, dy=dy)
            elif self.fit_method == "csaps":
                kwargs = dict(x=(e1, e2), nu=(dx, dy))
            self.u1_fig.add_heatmap(
                x=plot_grid_e1,
                y=plot_grid_e2,
                z=i["fit"](**kwargs).reshape(-1, 1).squeeze(),
                name=i["name"],
                # opacity=0.9,
                coloraxis="coloraxis",
                colorscale="Plasma",
            )

        if hasattr(self, "jintegral_path"):
            for path in self.jintegral_path:
                self.u1_fig.add_scatter(
                    x=path["e1_grid"],
                    y=path["e2_grid"],
                    # z=self.fit_path(path=path, field="u1"),
                    name=path["name"],
                    mode="lines",
                    line_width=5,
                )

                if path["name"] == "CD":
                    self.CD_fig = go.Figure(
                        go.Scatter(
                            x=path["e2_grid"],
                            y=self.fit_path(path=path, field="u1"),
                        )
                    )

        self.u1_fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        self.u1_fig.update_layout(
            template=fig_template,
            xaxis_title="<b>x [mm]</b>",
            yaxis_title="<b>y [mm]</b>",
            # move colorbar
            coloraxis_colorbar=dict(
                title="<b>u [mm]</b>",
                yanchor="top",
                y=1,
                x=-0.1,
            ),
            coloraxis_colorscale="Plasma",
        )

        self.u2_fig = go.Figure()

        for i in [
            dict(
                name="u2_fit_upper_crack",
                fit=self.u1_fit_upper_crack,
                e1_min=frame.e1_min,
                e1_max=frame.e1_max,
                e2_min=crack_tip["e2"],
                e2_max=frame.e2_max,
            ),
            dict(
                name="u2_fit_lower_crack",
                fit=self.u1_fit_lower_crack,
                e1_min=frame.e1_min,
                e1_max=frame.e1_max,
                e2_min=frame.e2_min,
                e2_max=crack_tip["e2"],
            ),
            dict(
                name="u2_fit_post_crack",
                fit=self.u1_fit_post_crack,
                e1_min=crack_tip["e1"],
                e1_max=frame.e1_max,
                e2_min=frame.e2_min,
                e2_max=frame.e2_max,
            ),
        ]:

            e1 = np.arange(i["e1_min"] * 10, i["e1_max"] * 10 + 5, 5) / 10
            e2 = np.arange(i["e2_min"] * 10, i["e2_max"] * 10 + 5, 5) / 10

            plot_grid_e1, plot_grid_e2 = np.meshgrid(e1, e2, indexing="ij")
            plot_grid_e1 = plot_grid_e1.reshape(-1, 1).squeeze()
            plot_grid_e2 = plot_grid_e2.reshape(-1, 1).squeeze()
            dx = 0
            dy = 0

            if self.fit_method == "sbs":
                kwargs = dict(x=e1, y=e2, grid=True, dx=dx, dy=dy)
            elif self.fit_method == "csaps":
                kwargs = dict(x=(e1, e2), nu=(dx, dy))
            self.u2_fig.add_heatmap(
                x=plot_grid_e1,
                y=plot_grid_e2,
                z=i["fit"](**kwargs).reshape(-1, 1).squeeze(),
                name=i["name"],
                # opacity=0.9,
                coloraxis="coloraxis",
                colorscale="Plasma",
            )

        if hasattr(self, "jintegral_path"):
            for path in self.jintegral_path:
                self.u1_fig.add_scatter(
                    x=path["e1_grid"],
                    y=path["e2_grid"],
                    # z=self.fit_path(path=path, field="u1"),
                    name=path["name"],
                    mode="lines",
                    line_width=5,
                )

                if path["name"] == "CD":
                    self.CD_fig = go.Figure(
                        go.Scatter(
                            x=path["e2_grid"],
                            y=self.fit_path(path=path, field="u1"),
                        )
                    )

        self.u2_fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        self.u2_fig.update_layout(
            template=fig_template,
            xaxis_title="<b>x [mm]</b>",
            yaxis_title="<b>y [mm]</b>",
            # move colorbar
            coloraxis_colorbar=dict(
                title="<b>v [mm]</b>",
                yanchor="top",
                y=1,
                x=-0.1,
            ),
            coloraxis_colorscale="Plasma",
        )

    def construct_fit_3d_scatter(self, frame, crack):
        min_e1 = frame.displacement_field_df["X"].min()
        max_e1 = frame.displacement_field_df["X"].max()
        min_e2 = frame.displacement_field_df["Y"].min()
        max_e2 = frame.displacement_field_df["Y"].max()

        crack_tip = crack.get_crack_tip(frame_id=frame.id, method=self.crack_tip_method)
        # lower_crack_edge = crack.crack_data[frame.id]["lower_crack_edge"]
        # upper_crack_edge = crack.crack_data[frame.id]["upper_crack_edge"]

        self.u1_fig = px.scatter_3d(
            data_frame=frame.displacement_field_df.sample(n=1000),
            # data_frame=frame.displacement_field_df,
            x="X",
            y="Y",
            z="U",
            labels=dict(X="<b>e1 [mm]</b>", Y="<b>e2 [mm]</b>", U="<b>u1 [mm]</b>"),
        )
        self.u1_fig.update_traces(marker_size=2, marker_color="black")

        # self.u1_fig.add_scatter3d(
        #     x=upper_crack_edge["X"],
        #     y=upper_crack_edge["Y"],
        #     z=upper_crack_edge["U"],
        #     mode="lines",
        #     showlegend=False,
        # )

        # self.u1_fig.add_scatter3d(
        #     x=lower_crack_edge["X"],
        #     y=lower_crack_edge["Y"],
        #     z=lower_crack_edge["U"],
        #     mode="lines",
        #     showlegend=False,
        # )

        for i in [
            dict(
                name="u1_fit_upper_crack",
                fit=self.u1_fit_upper_crack,
                e1_min=min_e1,
                e1_max=max_e1,
                e2_min=crack_tip["e2"],
                e2_max=max_e2,
            ),
            dict(
                name="u1_fit_lower_crack",
                fit=self.u1_fit_lower_crack,
                e1_min=min_e1,
                e1_max=max_e1,
                e2_min=min_e2,
                e2_max=crack_tip["e2"],
            ),
            dict(
                name="u1_fit_post_crack",
                fit=self.u1_fit_post_crack,
                e1_min=crack_tip["e1"] + self.fitting_offset_post_crack,
                e1_max=max_e1,
                e2_min=min_e2,
                e2_max=max_e2,
            ),
        ]:

            e1 = np.arange(i["e1_min"], i["e1_max"], 0.5)
            e2 = np.arange(i["e2_min"], i["e2_max"], 0.5)

            plot_grid_e1, plot_grid_e2 = np.meshgrid(e1, e2, indexing="ij")

            dx = 0
            dy = 0

            if self.fit_method == "sbs":
                kwargs = dict(x=e1, y=e2, grid=True, dx=dx, dy=dy)
            elif self.fit_method == "csaps":
                kwargs = dict(x=(e1, e2), nu=(dx, dy))
            self.u1_fig.add_surface(
                x=plot_grid_e1,
                y=plot_grid_e2,
                z=i["fit"](**kwargs),
                name=i["name"],
                opacity=0.9,
                coloraxis="coloraxis",
            )

        if hasattr(self, "jintegral_path"):
            for path in self.jintegral_path:
                self.u1_fig.add_scatter3d(
                    x=path["e1_grid"],
                    y=path["e2_grid"],
                    z=self.fit_path(path=path, field="u1", crack_tip=crack_tip),
                    name=path["name"],
                )

                if path["name"] == "CD":
                    self.CD_fig = go.Figure(
                        go.Scatter(
                            x=path["e2_grid"],
                            y=self.fit_path(path=path, field="u1", crack_tip=crack_tip),
                        )
                    )

        elif hasattr(self, "jintegral_area"):
            for area in self.jintegral_area:
                e1_grid, e2_grid = np.meshgrid(
                    area["e1_grid"], area["e2_grid"], indexing="ij"
                )
                self.u1_fig.add_scatter3d(
                    x=e1_grid.reshape(-1, 1).squeeze(),
                    y=e2_grid.reshape(-1, 1).squeeze(),
                    z=self.fit_area(area=area, field="u1", crack_tip=crack_tip),
                    name=area["name"],
                )

        self.u1_fig.update_layout(template=fig_template)
        self.u1_fig.update_coloraxes(
            colorscale="Plasma",
            colorbar_title="<b>u<sub>1</sub> [mm]</b>",
        )

        self.u2_fig = px.scatter_3d(
            data_frame=frame.displacement_field_df.sample(n=1000),
            # data_frame=frame.displacement_field_df,
            x="X",
            y="Y",
            z="V",
            labels=dict(
                X="<b>e<sub>1</sub> [mm]</b>",
                Y="<b>e<sub>2</sub> [mm]</b>",
                V="<b>u<sub>2</sub> [mm]</b>",
            ),
        )
        self.u2_fig.update_traces(marker_size=2, marker_color="black")

        # self.u2_fig.add_scatter3d(
        #     x=upper_crack_edge["X"],
        #     y=upper_crack_edge["Y"],
        #     z=upper_crack_edge["V"],
        #     mode="lines",
        #     showlegend=False,
        # )

        # self.u2_fig.add_scatter3d(
        #     x=lower_crack_edge["X"],
        #     y=lower_crack_edge["Y"],
        #     z=lower_crack_edge["V"],
        #     mode="lines",
        #     showlegend=False,
        # )

        for i in [
            dict(
                name="u2_fit_upper_crack",
                fit=self.u2_fit_upper_crack,
                e1_min=min_e1,
                e1_max=max_e1,
                e2_min=crack_tip["e2"],
                e2_max=max_e2,
            ),
            dict(
                name="u2_fit_lower_crack",
                fit=self.u2_fit_lower_crack,
                e1_min=min_e1,
                e1_max=max_e1,
                e2_min=min_e2,
                e2_max=crack_tip["e2"],
            ),
            dict(
                name="u2_fit_post_crack",
                fit=self.u2_fit_post_crack,
                e1_min=crack_tip["e1"] + self.fitting_offset_post_crack,
                e1_max=max_e1,
                e2_min=min_e2,
                e2_max=max_e2,
            ),
        ]:

            e1 = np.arange(i["e1_min"], i["e1_max"], 0.5)
            e2 = np.arange(i["e2_min"], i["e2_max"], 0.5)

            plot_grid_e1, plot_grid_e2 = np.meshgrid(e1, e2, indexing="ij")

            dx = 0
            dy = 0

            if self.fit_method == "sbs":
                kwargs = dict(x=e1, y=e2, grid=True, dx=dx, dy=dy)
            elif self.fit_method == "csaps":
                kwargs = dict(x=(e1, e2), nu=(dx, dy))

            self.u2_fig.add_surface(
                x=plot_grid_e1,
                y=plot_grid_e2,
                z=i["fit"](**kwargs),
                name=i["name"],
                opacity=0.9,
                coloraxis="coloraxis",
            )

        if hasattr(self, "jintegral_path"):
            for path in self.jintegral_path:
                self.u2_fig.add_scatter3d(
                    x=path["e1_grid"],
                    y=path["e2_grid"],
                    z=self.fit_path(path=path, field="u2", crack_tip=crack_tip),
                    name=path["name"],
                )

        elif hasattr(self, "jintegral_area"):
            for area in self.jintegral_area:
                e1_grid, e2_grid = np.meshgrid(
                    area["e1_grid"], area["e2_grid"], indexing="ij"
                )
                self.u2_fig.add_scatter3d(
                    x=e1_grid.reshape(-1, 1).squeeze(),
                    y=e2_grid.reshape(-1, 1).squeeze(),
                    z=self.fit_area(area=area, field="u2", crack_tip=crack_tip),
                    name=area["name"],
                )

        self.u2_fig.update_layout(template=fig_template)
        self.u2_fig.update_coloraxes(
            colorscale="Plasma",
            colorbar_title="<b>u<sub>2</sub> [mm]</b>",
        )
