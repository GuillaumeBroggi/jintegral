from cmath import isnan
from importlib.metadata import metadata
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn import linear_model
from jintegral import utils as ut
import pickle
from plotly.subplots import make_subplots
from jintegral.fig_template import fig_template, color_discrete_sequence
import logging
import warnings
import math
from scipy.interpolate import griddata
from csaps import CubicSmoothingSpline


# from utils import remove_V_outliers


class Crack:
    """
    Represents a crack computing unit.
    """

    def __init__(self) -> None:
        # Crack data is a convenience dict to store results, with the frame id as the key.
        self.crack_data = dict()

    def get_crack_tip(self, frame_id, method):
        """Method may be one of intersec, u2_inflex, u1u2_inflex"""

        crack_data = self.crack_data[frame_id]
        return dict(
            e1=crack_data[f"{method}_e1_crack_tip"],
            e2=crack_data["e2_crack_tip"],
            u2=crack_data["u2_crack_tip"],
        )

    @staticmethod
    def _identify_cod_profile_e2_position(
        frame,
        e1_crack_tip,
        e2_crack_tip,
        offset_multiplier=1,
        allowable_missing_points_prop=0.2,
        number_of_point_per_bin=2,
        e2_bins=2,
    ):

        """
        Identify the cod profiles over the sample length, starting from the crack plane (Y=e2_crack_tip) plus an offset, then looking for an horizontal profile (in the sample reference) where there more than

        The cod profiles are very similar when the distance from the crack plane increases, see validation details.
        """
        # First check that the selected profile contain enough data taking advantage of pandas value_counts
        # To to do so the data from the begining of the data to the crack tip is binned with a bin size corresponding to the best fitting step

        e1_bins = math.floor((frame.e1_max - frame.e1_min) / frame.interp_step)
        e1_bins_to_crack_tip = math.floor(
            (e1_crack_tip - frame.e1_min) / frame.interp_step
        )
        significative_displacement = 0.02

        logging.info(
            f"Start cod profile identification with offset multiplier {offset_multiplier} and significant displacement {significative_displacement}."
        )

        if e1_bins_to_crack_tip <= 10:
            # Relax the constraint
            allowable_missing_points_prop = 0.5

        i = 0
        while True:
            logging.info(
                f"Cod profile identification iteration {i}, offset multiplier {offset_multiplier}"
            )
            i += 1

            # Compute the new e2_profile
            e2_profile = e2_crack_tip + offset_multiplier * frame.interp_step

            if (
                e2_profile >= frame.e2_max - e2_bins
                or e2_profile <= frame.e2_min + e2_bins
            ):

                logging.info(
                    f"Reached end of dataset without identifyng cod profile, constraint probably too strict, or not enough bins before the crack tip ({e1_bins_to_crack_tip} bin(s))."
                )
                significative_displacement += 0.01
                if significative_displacement > 0.2:
                    warnings.warn("Issue with dataset.")
                if significative_displacement > 1:
                    raise ValueError
                offset_multiplier = math.copysign(1, offset_multiplier)
                logging.info(
                    f"Relaxing constraints on significative displacement : {significative_displacement}"
                )

            profile_df = frame.displacement_field_df.query(
                "abs(Y - @e2_profile)< @e2_bins / 2 * @frame.interp_step"
            )

            # Compute the number of bins without data before crack tip

            try:
                bins_without_data = (
                    profile_df[profile_df["X"] <= e1_crack_tip]["X"].value_counts(
                        bins=e1_bins_to_crack_tip
                    )
                    < number_of_point_per_bin * e2_bins
                ).sum()
            except ValueError:
                # Handle case where the query is empty and cut raise an error
                bins_without_data = profile_df.shape[0]

            # Check that all points with a significative e2 displacement have the same direction before crack tip
            # u2_sign is calculated before the crack tip
            u2_sign = np.sign(profile_df[profile_df["X"] <= e1_crack_tip]["V"].mean())
            data_points_with_inverse_displacement = (
                np.sign(
                    profile_df[
                        (profile_df["V"].abs() > significative_displacement)
                        & (profile_df["X"] <= e1_crack_tip)
                    ]["V"]
                )
                != u2_sign
            ).any()

            logging.debug(
                f"Bins wihtout data: {bins_without_data} - Bins to crack tip: {e1_bins_to_crack_tip}"
            )

            if (
                bins_without_data / e1_bins_to_crack_tip
                <= allowable_missing_points_prop
                and not data_points_with_inverse_displacement
            ):
                # if there is less bins without data than allowable proportion

                break

            else:
                logging.info(
                    f"Too much bins without data: {bins_without_data / e1_bins_to_crack_tip*100} %"
                )
                # increment the multiplier, either in the positive or negative direction
                offset_multiplier = math.copysign(
                    abs(offset_multiplier) + 1, offset_multiplier
                )

        logging.debug(
            f"COD profile identified at e2={e2_profile} mm with an offset multiplier of {offset_multiplier}."
        )

        return e2_profile, offset_multiplier, u2_sign

    @staticmethod
    def _extract_cod_profile(frame, e2_profile, e2_bins):
        profile_df = frame.displacement_field_df.query(
            "abs(Y - @e2_profile)< @e2_bins / 2 * @frame.interp_step"
        ).dropna()

        # do not start at min_e1 to avoid edge effect
        e1_profile = np.arange(
            start=frame.e1_min + frame.interp_step,
            stop=frame.e1_max - frame.interp_step,
            step=frame.interp_step,
        )

        e1_grid, e2_grid = np.meshgrid(e1_profile, e2_profile)
        u2_profile = griddata(
            points=(profile_df["X"].to_numpy(), profile_df["Y"].to_numpy()),
            values=profile_df["V"].to_numpy(),
            xi=(e1_grid, e2_grid),
            method="cubic",
            rescale=True,
        )[0]

        # if np.isnan(u2_profile).any():
        #     print(u2_profile)
        #     print(profile_df)
        #     print(profile_df.isnull().sum().sum())
        #     print(e2_profile)

        u1_profile = griddata(
            points=(profile_df["X"].to_numpy(), profile_df["Y"].to_numpy()),
            values=profile_df["U"].to_numpy(),
            xi=(e1_grid, e2_grid),
            method="cubic",
            rescale=True,
        )[0]

        cod_profile = dict(e1=e1_profile, e2=e2_profile, u1=u1_profile, u2=u2_profile)

        return cod_profile

    @staticmethod
    def _fit_cod_profile(frame, cod_profile, e1_crack_tip, method, field, offset=None):
        """Model the upper and lower cod profiles as linear up to the crack tip minus an offset to avoid issues at the crack tip position"""

        # If offset is not provided, intitialize to a multiple of interp step
        if not offset:
            offset = 2 * frame.interp_step

        if method == "linear":

            # construct a mask to ignore data after crack tip and nan
            mask = (
                ~np.isnan(cod_profile["e1"])
                & ~np.isnan(cod_profile[field])
                & (cod_profile["e1"] <= e1_crack_tip - offset)
            )

            # fit is fed with data in (x, 1) shape, thus need to reshape (see documentation)
            try:
                cod_profile_fit = linear_model.HuberRegressor()
                cod_profile_fit.fit(
                    cod_profile["e1"][mask].reshape(-1, 1),
                    cod_profile[field][mask],
                )
            except ValueError as e:
                print(cod_profile)
                raise e

        elif method == "csaps":

            # construct a mask to ignore data after crack tip and nan
            mask = ~np.isnan(cod_profile["e1"]) & ~np.isnan(cod_profile[field])

            cod_profile_fit = CubicSmoothingSpline(
                cod_profile["e1"][mask],
                cod_profile[field][mask],
                smooth=0.9,
                normalizedsmooth=True,
            )
        else:
            raise ValueError

        return cod_profile_fit

    @staticmethod
    def _compute_e1_crack_tip(frame, upper_cod_profile_fit, lower_cod_profile_fit):
        """
        The crack tip e1 coordinate is defined as the intersection between the nearly linear cod profiles
        """
        if upper_cod_profile_fit.coef_ == lower_cod_profile_fit.coef_:
            raise Exception("Displacement lines are either parallel or identical")

        # e1 coordinate of the crack tip (intersection of two lines given the lines equations a*x+c=b*x+d)
        e1_crack_tip = (
            lower_cod_profile_fit.intercept_ - upper_cod_profile_fit.intercept_
        ) / (upper_cod_profile_fit.coef_ - lower_cod_profile_fit.coef_).item()

        # Validation of crack tip
        if e1_crack_tip <= frame.e1_min:
            # Reset initial crack tip, could also be random
            e1_crack_tip = 5
        if e1_crack_tip >= frame.e1_max:
            # TODO warning
            # Crack tip cannot be grater that the sample size minus a safety margin(test is not valid with such a small distance between crack tip anf the backend)
            e1_crack_tip = frame.e1_max - 5

        return e1_crack_tip

    @staticmethod
    def _compute_u2_crack_tip(e1_crack_tip, upper_cod_profile_fit):
        u2_crack_tip = (
            upper_cod_profile_fit.coef_ * e1_crack_tip
            + upper_cod_profile_fit.intercept_
        )
        u2_crack_tip = u2_crack_tip.item()

        return u2_crack_tip

    @staticmethod
    def _identify_e2_crack_tip(frame):

        # As a first estimate, the crack tip is approximately at the middle of a compact tension specimen
        e2_crack_tip = (frame.e2_min + frame.e2_max) / 2

        # Refine e2_crack_tip by identifying the center of rotation of each arm

        frame.displacement_field_df["u_plane"] = np.sqrt(
            frame.displacement_field_df["U"] ** 2
            + frame.displacement_field_df["V"] ** 2
        )

        e2_center_rotation_upper_arm = frame.displacement_field_df.loc[
            frame.displacement_field_df.query("Y > @e2_crack_tip")["u_plane"].idxmin()
        ]["Y"]

        e2_center_rotation_lower_arm = frame.displacement_field_df.loc[
            frame.displacement_field_df.query("Y < @e2_crack_tip")["u_plane"].idxmin()
        ]["Y"]

        e2_crack_tip = (e2_center_rotation_upper_arm + e2_center_rotation_lower_arm) / 2

        return e2_crack_tip

    def compute_new_crack_tip_coordinates(
        self,
        frame,
        penultimate_e1_crack_tip,
        e2_crack_tip,
        e2_upper_profile=None,
        e2_lower_profile=None,
    ):
        # if profile position is not forced
        if not e2_upper_profile or not e2_lower_profile:
            e2_upper_offset_multiplier = 1
            e2_lower_offset_multiplier = -1
            while True:
                # Upper cod profile identification, request 3 bins in e2 direction to ensure that the fit will be stable (the bins are centered around the profile)
                (
                    e2_upper_profile,
                    e2_upper_offset_multiplier,
                    u2_upper_sign,
                ) = self._identify_cod_profile_e2_position(
                    frame=frame,
                    e1_crack_tip=penultimate_e1_crack_tip,
                    e2_crack_tip=e2_crack_tip,
                    offset_multiplier=e2_upper_offset_multiplier,
                    number_of_point_per_bin=1,
                    e2_bins=3,
                )

                upper_cod_profile = self._extract_cod_profile(
                    frame=frame, e2_profile=e2_upper_profile, e2_bins=3
                )

                # Lower cod profile identification
                (
                    e2_lower_profile,
                    e2_lower_offset_multiplier,
                    u2_lower_sign,
                ) = self._identify_cod_profile_e2_position(
                    frame=frame,
                    e1_crack_tip=penultimate_e1_crack_tip,
                    e2_crack_tip=e2_crack_tip,
                    offset_multiplier=e2_lower_offset_multiplier,  # negative multiplier for lower cod profile
                    number_of_point_per_bin=1,
                    e2_bins=3,
                )

                lower_cod_profile = self._extract_cod_profile(
                    frame=frame, e2_profile=e2_lower_profile, e2_bins=3
                )

                if u2_lower_sign != u2_upper_sign:
                    break
                else:
                    if abs(e2_upper_offset_multiplier) < abs(
                        e2_lower_offset_multiplier
                    ):
                        e2_upper_offset_multiplier += 1
                    else:
                        e2_lower_offset_multiplier -= 1
                    logging.info(
                        f"Lower and upper cod have the same sign, try a new pass with lower multiplier {e2_lower_offset_multiplier} and upper multiplier {e2_upper_offset_multiplier}."
                    )

        else:
            upper_cod_profile = self._extract_cod_profile(
                frame=frame, e2_profile=e2_upper_profile, e2_bins=3
            )
            lower_cod_profile = self._extract_cod_profile(
                frame=frame, e2_profile=e2_lower_profile, e2_bins=3
            )

        # Fitting

        upper_cod_profile_u2_linear_fit = self._fit_cod_profile(
            frame=frame,
            cod_profile=upper_cod_profile,
            e1_crack_tip=penultimate_e1_crack_tip,
            method="linear",
            field="u2",
        )

        # Fitting

        lower_cod_profile_u2_linear_fit = self._fit_cod_profile(
            frame=frame,
            cod_profile=lower_cod_profile,
            e1_crack_tip=penultimate_e1_crack_tip,
            method="linear",
            field="u2",
        )

        # Crack tip may be defined as the intersection of the linear fits

        new_e1_crack_tip = self._compute_e1_crack_tip(
            frame=frame,
            upper_cod_profile_fit=upper_cod_profile_u2_linear_fit,
            lower_cod_profile_fit=lower_cod_profile_u2_linear_fit,
        )

        new_u2_crack_tip = self._compute_u2_crack_tip(
            e1_crack_tip=new_e1_crack_tip,
            upper_cod_profile_fit=upper_cod_profile_u2_linear_fit,
        )

        self.crack_data[frame.id] = dict(
            intersec_e1_crack_tip=new_e1_crack_tip,
            e2_crack_tip=e2_crack_tip,
            u2_crack_tip=new_u2_crack_tip,
            lower_cod_profile_u2_linear_fit=lower_cod_profile_u2_linear_fit,
            upper_cod_profile_u2_linear_fit=upper_cod_profile_u2_linear_fit,
            lower_cod_profile=lower_cod_profile,
            upper_cod_profile=upper_cod_profile,
        )

        return new_e1_crack_tip

    def iterative_crack_extraction(
        self,
        frame,
        e1_crack_tip_guess=None,
        threshold=0.001,
        e2_upper_profile=None,
        e2_lower_profile=None,
    ):
        """
        Look iteratively for the crack tip position until convergence by finding the intersection between the displacement of upper and lower crack edges (assuming a vertical displacement linearly function of the distance with respect to the crack tip).
        If convergence can not be achieved, check if the results is oscillating between the penultimate and the antepenultimate value.
        Convergence is speedup by using the penultimate crack tip  as the crack tip can only move forward
        """

        # Initialize utils variables
        antepenultimate_e1_crack_tip = 0

        # The previous crack tip position is initilized with the guess
        # If e1_crack_tip_guess is not provided, intitialize to a multiple of best interp step
        if not e1_crack_tip_guess:
            try:

                if not np.isnan(frame.metadata["optical_crack_tip_e1"]):
                    e1_crack_tip_guess = frame.metadata["optical_crack_tip_e1"] - 1
                else:
                    raise KeyError
            except KeyError:
                e1_crack_tip_guess = 10 * frame.interp_step

        if e1_crack_tip_guess > 7 * frame.interp_step:

            penultimate_e1_crack_tip = e1_crack_tip_guess

        else:
            penultimate_e1_crack_tip = 7 * frame.interp_step
            warnings.warn(
                f"Crack tip guess for iterative extraction was overwritten to {penultimate_e1_crack_tip}. Choose a greater value for better performance."
            )
        if not e2_lower_profile and not e2_upper_profile:
            # Identify the crack plane, which is e2_crack_tip assuming a straight crack propagation along e1 direction
            e2_crack_tip = self._identify_e2_crack_tip(frame=frame)
        else:
            e2_crack_tip = None

        logging.info(
            f"Frame {frame.id}: Starting iterative crack extraction with a guess of {penultimate_e1_crack_tip} mm and a convergence threshold of {threshold} mm."
        )

        i = 0
        while True:
            # if i % 10 == 0:
            logging.info(f"Crack identification: iteration {i}...")
            i += 1

            # Identify the crack cod profile up to the penultimate crack tip
            new_e1_crack_tip = self.compute_new_crack_tip_coordinates(
                frame=frame,
                penultimate_e1_crack_tip=penultimate_e1_crack_tip,
                e2_crack_tip=e2_crack_tip,
                e2_upper_profile=e2_upper_profile,
                e2_lower_profile=e2_lower_profile,
            )

            logging.debug(
                f"Frame {frame.id}: New crak tip found at ({new_e1_crack_tip}, {e2_crack_tip})"
            )

            # Check the convergence of the crack tip position with respect to the threshold
            if abs(new_e1_crack_tip - penultimate_e1_crack_tip) < threshold:
                break
            elif abs(antepenultimate_e1_crack_tip - new_e1_crack_tip) < threshold:
                new_e1_crack_tip = (new_e1_crack_tip + penultimate_e1_crack_tip) / 2
                break
            else:
                antepenultimate_e1_crack_tip = penultimate_e1_crack_tip
                penultimate_e1_crack_tip = new_e1_crack_tip
                if i % 200 == 0:
                    logging.warning(
                        f"No crack convergence. Relaxing convergence threshold from {threshold} to {threshold*10} mm"
                    )
                    threshold *= 10
                logging.info("Waiting for convergence...")

        logging.info(f"Crack identification convergence reached in {i} iteration(s).")
        logging.info(
            f"Frame {frame.id}: Optical crack tip at {frame.metadata['optical_crack_tip_e1']} ({frame.X_offset} offset from frame registration)."
        )
        logging.info(
            f"Frame {frame.id}: New crak tip converged at ({new_e1_crack_tip}, {e2_crack_tip}) with interception method."
        )

        # Correct crack tip position
        if frame.metadata["init"]:
            self.init_intersec_error = (
                new_e1_crack_tip - frame.metadata["optical_crack_tip_e1"]
            )
            self.init_J = frame.metadata["J_guess"]

        if hasattr(self, "init_intersec_error"):
            if self.init_J <= 0 or frame.metadata["J_guess"] <= 0:
                logging.info(
                    f"J cant be negative, no crack tip correction. Uncorrected crack tip is reported instead."
                )
                # Assign value for compatibility
                u2_intersec_corrected_e1_crack_tip = new_e1_crack_tip
            else:
                u2_intersec_corrected_e1_crack_tip = (
                    new_e1_crack_tip
                    - self.init_intersec_error
                    * np.sqrt(self.init_J)
                    / np.sqrt(frame.metadata["J_guess"])
                )

            logging.info(
                f"Frame {frame.id}: New crak tip converged at ({u2_intersec_corrected_e1_crack_tip}, {e2_crack_tip}) with corrected interception method."
            )

        else:
            u2_intersec_corrected_e1_crack_tip = None

            logging.info(
                f"Frame {frame.id}: No intersection correction possible as no initial frame present."
            )

        # Additional crack evaluation methods, as an inflexion point in displacemnrt field

        # Spline fitting

        upper_cod_profile = self.crack_data[frame.id]["upper_cod_profile"]

        upper_cod_profile["u1u2"] = np.sqrt(
            upper_cod_profile["u1"] ** 2 + upper_cod_profile["u2"] ** 2
        )

        upper_cod_profile_u1u2_csaps_fit = self._fit_cod_profile(
            frame=frame,
            cod_profile=upper_cod_profile,
            e1_crack_tip=penultimate_e1_crack_tip,
            method="csaps",
            field="u1u2",
        )

        fine_e1_range = np.arange(
            np.min(upper_cod_profile["e1"]),
            np.max(upper_cod_profile["e1"]),
            0.01,
        )

        u1u2_inflex_e1_crack_tip = fine_e1_range[
            upper_cod_profile_u1u2_csaps_fit(
                fine_e1_range,
                nu=2,
            ).argmax()
        ]
        logging.info(
            f"Frame {frame.id}: Crak tip found at ({u1u2_inflex_e1_crack_tip}, {e2_crack_tip}) with u1u2 inflexion method."
        )

        upper_cod_profile_u2_csaps_fit = self._fit_cod_profile(
            frame=frame,
            cod_profile=upper_cod_profile,
            e1_crack_tip=penultimate_e1_crack_tip,
            method="csaps",
            field="u2",
        )

        u2_inflex_e1_crack_tip = fine_e1_range[
            upper_cod_profile_u2_csaps_fit(
                fine_e1_range,
                nu=2,
            ).argmax()
        ]

        logging.info(
            f"Frame {frame.id}: Crak tip found at ({u2_inflex_e1_crack_tip}, {e2_crack_tip}) with u2 inflexion method."
        )

        self.crack_data[frame.id] = dict(
            self.crack_data[frame.id],
            upper_cod_profile_u1u2_csaps_fit=upper_cod_profile_u1u2_csaps_fit,
            upper_cod_profile_u2_csaps_fit=upper_cod_profile_u2_csaps_fit,
            u2_inflex_e1_crack_tip=u2_inflex_e1_crack_tip,
            u1u2_inflex_e1_crack_tip=u1u2_inflex_e1_crack_tip,
            u2_intersec_corrected_e1_crack_tip=u2_intersec_corrected_e1_crack_tip,
        )

    def identify_crack_tip(
        self, frame, method="iterative", e1_crack_tip_guess=None, threshold=0.001
    ):

        if method == "iterative":
            self.iterative_crack_extraction(
                frame, e1_crack_tip_guess=e1_crack_tip_guess, threshold=threshold
            )
        elif method == "u2_inflex":
            # TODO
            pass
        else:
            raise ValueError

    def dump_crack_data(self, dump_path):

        dump_path.mkdir(parents=True, exist_ok=True)

        with open(
            dump_path / f"crack_data.pickle",
            "wb",
        ) as file:
            pickle.dump(self.crack_data, file)

    def load_crack_data(self, dump_path):
        """
        Load the crack data previously computed and save as a pickle object on disk to speed up running time when the running conditions are not changed"""
        with open(
            dump_path / f"crack_data.pickle",
            "rb",
        ) as file:
            self.crack_data = pickle.load(file)

    def plot_cod_profile(self, frame):

        fig = go.Figure()
        fig.add_scatter(
            x=self.crack_data[frame.id]["upper_cod_profile"]["e1"],
            y=self.crack_data[frame.id]["upper_cod_profile"]["u2"],
            mode="lines",
            name="Upper cod profile",
        )
        fig.add_scatter(
            x=self.crack_data[frame.id]["lower_cod_profile"]["e1"],
            y=self.crack_data[frame.id]["lower_cod_profile"]["u2"],
            mode="lines",
            name="Upper cod profile",
        )
        fig.add_scatter(
            x=self.crack_data[frame.id]["upper_cod_profile"]["e1"],
            y=self.crack_data[frame.id]["upper_cod_profile_u2_linear_fit"].predict(
                self.crack_data[frame.id]["upper_cod_profile"]["e1"].reshape(-1, 1)
            ),
            mode="lines",
            name="Upper cod linear fit",
        )
        fig.add_scatter(
            x=self.crack_data[frame.id]["lower_cod_profile"]["e1"],
            y=self.crack_data[frame.id]["lower_cod_profile_u2_linear_fit"].predict(
                self.crack_data[frame.id]["lower_cod_profile"]["e1"].reshape(-1, 1)
            ),
            mode="lines",
            name="Lower cod linear fit",
        )

        fig.add_vline(
            x=self.crack_data[frame.id]["intersec_e1_crack_tip"],
            line_color="black",
            line_dash="dash",
        )
        fig.update_layout(
            template=fig_template,
            # xaxis_range=[
            #     min(self.crack_data[frame.id]["lower_cod_profile"]["e1"]),
            #     self.crack_data[frame.id]["intersec_e1_crack_tip"] + 2,
            # ],
            yaxis_range=[
                min(
                    *self.crack_data[frame.id]["upper_cod_profile"]["u2"],
                    *self.crack_data[frame.id]["lower_cod_profile"]["u2"],
                )
                * 1.1,
                max(
                    *self.crack_data[frame.id]["upper_cod_profile"]["u2"],
                    *self.crack_data[frame.id]["lower_cod_profile"]["u2"],
                )
                * 1.1,
            ],
            xaxis_title="<b>e<sub>1</sub> [mm]</b>",
            yaxis_title="<b>u<sub>2</sub> [mm]</b>",
            legend=dict(
                yanchor="top", y=0.99, xanchor="right", x=0.99, title_text=None
            ),
            colorway=color_discrete_sequence,
        )

        self.u2_interception_fig = fig

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_scatter(
            x=self.crack_data[frame.id]["upper_cod_profile"]["e1"],
            y=self.crack_data[frame.id]["upper_cod_profile_u2_csaps_fit"](
                self.crack_data[frame.id]["upper_cod_profile"]["e1"], nu=0
            ),
            mode="lines",
            name="Upper cod profile",
        )

        fig.add_scatter(
            x=self.crack_data[frame.id]["upper_cod_profile"]["e1"],
            y=self.crack_data[frame.id]["upper_cod_profile_u2_csaps_fit"](
                self.crack_data[frame.id]["upper_cod_profile"]["e1"], nu=2
            ),
            mode="lines",
            name="Second derivative",
            secondary_y=True,
        )

        fig.add_vline(
            x=self.crack_data[frame.id]["u2_inflex_e1_crack_tip"],
            line_color="black",
            line_dash="dash",
        )

        fig.update_layout(
            template=fig_template,
            xaxis_title="<b>e<sub>1</sub> [mm]</b>",
            yaxis_title="<b>u<sub>2</sub> [mm]</b>",
            xaxis_mirror="ticks",
            yaxis_mirror=False,
            legend=dict(
                yanchor="top", y=0.99, xanchor="right", x=0.92, title_text=None
            ),
            colorway=color_discrete_sequence,
        )
        fig.update_yaxes(title="<b>u<sub>2</sub>'' [mm]</b>", secondary_y=True)

        self.u2_inflexion_fig = fig

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_scatter(
            x=self.crack_data[frame.id]["upper_cod_profile"]["e1"],
            y=self.crack_data[frame.id]["upper_cod_profile_u1u2_csaps_fit"](
                self.crack_data[frame.id]["upper_cod_profile"]["e1"], nu=0
            ),
            mode="lines",
            name="Upper cod profile",
        )

        fig.add_scatter(
            x=self.crack_data[frame.id]["upper_cod_profile"]["e1"],
            y=self.crack_data[frame.id]["upper_cod_profile_u1u2_csaps_fit"](
                self.crack_data[frame.id]["upper_cod_profile"]["e1"], nu=2
            ),
            mode="lines",
            name="Second derivative",
            secondary_y=True,
        )

        fig.add_vline(
            x=self.crack_data[frame.id]["u1u2_inflex_e1_crack_tip"],
            line_color="black",
            line_dash="dash",
        )

        fig.update_layout(
            template=fig_template,
            xaxis_title="<b>e<sub>1</sub> [mm]</b>",
            yaxis_title="<b>u<sub>1</sub>u<sub>2</sub> [mm]</b>",
            xaxis_mirror="ticks",
            yaxis_mirror=False,
            legend=dict(
                yanchor="top", y=0.99, xanchor="right", x=0.92, title_text=None
            ),
            colorway=color_discrete_sequence,
        )
        fig.update_yaxes(
            title="<b>u<sub>1</sub>u<sub>2</sub>'' [mm]</b>", secondary_y=True
        )

        self.u1u2_inflexion_fig = fig
