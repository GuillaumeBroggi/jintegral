from random import sample
from re import template
from jintegral import crack, frame, sample as sp
from jintegral import frame as fr
from jintegral import material as mt
from jintegral import jintegral as jt
import pandas as pd
import plotly.express as px
import numpy as np
from jintegral.fig_template import fig_template
import logging


class AbaqusSample(sp.Sample):
    def __init__(
        self, frame_data_path, material_data_path, material_name, name="abaqus_sample"
    ) -> None:
        super().__init__(frame_data_path, name)
        self._load_material(name=material_name, material_data_path=material_data_path)

    def _construct_frame_df(self, frame_data_path):
        """
        Overwrite frame_df method from parent class Sample.


        frame_data_path: Path object pointing to the frame folder
        """

        frame_dict = {}
        for i, path in enumerate(sorted(frame_data_path.rglob("*/*.csv"))):
            frame_id = int(path.stem.split("-")[-1])
            frame_dict[frame_id] = dict(path=path)

        frame_df = pd.DataFrame.from_dict(frame_dict, orient="index")

        frame_info = pd.read_csv(frame_data_path / "frames_info.csv", index_col="id")

        frame_df = frame_df.merge(
            right=frame_info, how="left", left_index=True, right_index=True
        )

        self.frame_df = frame_df

    def _load_frame(self, frame_id, crop_data=None):
        self.frame = fr.Frame(
            id=frame_id,
            metadata=self.frame_df.loc[frame_id].copy(),
        )
        self.frame._load_abaqus_displacement_field_df(crop_data=crop_data)

    def _analyze_crack(self):
        """
        Values selected for Abaqus results
        Load the proper frame first.
        """
        e1_crack_tip_guess = 3
        threshold = 0.001
        self.crack.iterative_crack_extraction(
            frame=self.frame, e1_crack_tip_guess=e1_crack_tip_guess, threshold=threshold
        )

    def prepare_data(
        self,
        frame_id=0,
        load_crack_data=False,
        dump_crack_data=True,
        dump_field_fits=True,
        load_field_fits=False,
        sigma_error=None,
        random_seed=None,
        plot_error=False,
        crop_data=None,
        overwrite_interp_step=None,
    ):
        # Step 1: load frame data
        self._load_frame(frame_id=frame_id, crop_data=crop_data)
        if overwrite_interp_step:
            self.frame.interp_step = overwrite_interp_step
            logging.info(f"Overwrite interpolation step to {self.frame.interp_step}")

        # Step 1bis: add error if required
        if sigma_error:
            self.simulate_error(
                sigma=sigma_error, plot=plot_error, random_seed=random_seed
            )

        # Step 2: derive crack information from frame data or load a previous derivation
        if load_crack_data:
            self.crack.load_crack_data(dump_path=self.result_path / "crack_data")
        else:
            self._analyze_crack()
            if dump_crack_data:
                self.crack.dump_crack_data(dump_path=self.result_path / "crack_data")

        # Store crack results at sample level: update the entry if existing or create it
        crack_values = [
            "intersec_e1_crack_tip",
            "u2_inflex_e1_crack_tip",
            "u1u2_inflex_e1_crack_tip",
            "e2_crack_tip",
            "u2_crack_tip",
        ]
        self.crack_tip_df.loc[frame_id, crack_values] = [
            self.crack.crack_data[frame_id][k] for k in crack_values
        ]

        # Step 3: fit displacement fields or load a previous fit
        if load_field_fits:
            self.jintegral.load_displacement_field_fit(
                frame=self.frame, dump_path=self.result_path / "field_fits"
            )

        else:
            self.jintegral.fit_displacement_field(
                frame=self.frame,
                crack=self.crack,
            )
            if dump_field_fits:
                dump_path = self.result_path / "field_fits"
                self.jintegral.dump_displacement_field_fit(
                    frame=self.frame, dump_path=dump_path
                )

    def save_results(self):
        self.result_path.mkdir(parents=True, exist_ok=True)

        self.crack_tip_df.to_csv(
            self.result_path / "{n}_crack_tip.csv".format(n=self.name)
        )
        self.err_df.to_csv(self.result_path / "{n}_err.csv".format(n=self.name))

        self.err_detailed_df.to_csv(
            self.result_path / "{n}_err_detailed.csv".format(n=self.name)
        )

    def plot2(self, sample_size=None, moved=True):
        if moved:
            # Moved points
            if sample_size:
                fig = px.scatter(
                    self.frame.displacement_field_df.sample(int(sample_size)),
                    "x_u",
                    "y_v",
                    color="V",
                )
            else:
                fig = px.scatter(
                    self.frame.displacement_field_df, "x_u", "y_v", color="V"
                )
            fig.add_scatter(
                x=self.crack.upper_crack_edge["X"] + self.crack.upper_crack_edge["U"],
                y=self.crack.upper_crack_edge["Y"] + self.crack.upper_crack_edge["V"],
                mode="lines",
                name="Upper edge",
            )
            fig.add_scatter(
                x=self.crack.lower_crack_edge["X"] + self.crack.lower_crack_edge["U"],
                y=self.crack.lower_crack_edge["Y"] + self.crack.lower_crack_edge["V"],
                mode="lines",
                name="Lower edge",
            )

            x = np.arange(self.crack.first_e1, self.crack.e1_crack_tip + 1, 1)

            fig.add_scatter(
                x=x,
                y=self.crack.u2_upper_crack_fit.predict(x.reshape(-1, 1)) + 30,
                mode="lines",
                name="Upper fit",
            )
            fig.add_scatter(
                x=x,
                y=self.crack.u2_lower_crack_fit.predict(x.reshape(-1, 1)) + 30,
                mode="lines",
                name="Lower fit",
            )

        else:
            if sample_size:
                fig = px.scatter(
                    self.frame.displacement_field_df.sample(int(sample_size)),
                    "x_u",
                    "y_v",
                    color="V",
                )
            else:
                fig = px.scatter(self.frame.displacement_field_df, "X", "Y", color="V")
            fig.add_scatter(
                x=self.crack.upper_crack_edge["X"],
                y=self.crack.upper_crack_edge["Y"],
                mode="lines",
            )
            fig.add_scatter(
                x=self.crack.lower_crack_edge["X"],
                y=self.crack.lower_crack_edge["Y"],
                mode="lines",
            )

            x = np.arange(self.crack.first_e1, self.crack.e1_crack_tip + 1, 1)

            fig.add_scatter(
                x=x,
                y=self.crack.u2_upper_crack_fit.predict(x.reshape(-1, 1)) + 30,
                mode="lines",
                name="Upper fit",
            )
            fig.add_scatter(
                x=x,
                y=self.crack.u2_lower_crack_fit.predict(x.reshape(-1, 1)) + 30,
                mode="lines",
                name="Lower fit",
            )
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        fig.show()

    def simulate_error(self, sigma=2.2e-4, plot=False, random_seed=None):
        """Simulate an error roughly equivalent to projection errors reported by DIC software.
        For instance Vic#D reports sigma in pixels (1 standard deviation according to documentation).
        Thus displacement fields are noised with an additive white gaussian noise of std=sigma*mm/pixel.
        The noise is rough and overpredicted because sigma is reported as a global error and not componet wise.
        """
        self.frame._add_noise_to_displacement_field(
            fields=["U", "V"],
            noise_type="awgn",
            noise_std=sigma,
            random_seed=random_seed,
        )

        if plot:
            plot_path = self.result_path / "noise"
            plot_path.mkdir(parents=True, exist_ok=True)

            for field in ("U", "V"):
                fig = px.histogram(
                    data_frame=self.frame.displacement_field_df[f"{field}_noise"],
                    nbins=100,
                    histnorm="probability",
                    color_discrete_sequence=px.colors.qualitative.Prism,
                )
                fig.update_layout(
                    template=fig_template,
                    xaxis_title="Displacement noise [mm]",
                    yaxis_title="Probability",
                    showlegend=False,
                )
                fig.write_image(
                    plot_path / f"{field}_awgn_distribution_frame-{self.frame.id}.svg"
                )
