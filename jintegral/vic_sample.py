from jintegral import crack, frame, sample as sp
from jintegral import frame as fr
from jintegral import material as mt
from jintegral import jintegral as jt
import pandas as pd
import plotly.express as px
import numpy as np
import logging


class VicSample(sp.Sample):
    def __init__(self, frame_data_path, material_data_path, name="vic_sample") -> None:
        super().__init__(frame_data_path, name)
        self.id = int(frame_data_path.stem.split("-")[-1])
        samples_df = pd.read_csv(material_data_path, index_col="sample")
        material_name = samples_df.loc[self.id]["material"]
        self._load_material(name=material_name)

        logging.info(
            f"Sample {name} initialized with material constants: E1 = {self.material.E1} MPa,  E2 = {self.material.E2} MPa,  G12 = {self.material.G12} MPa, nu12 = {self.material.nu12}, nu21 = {self.material.nu21}"
        )

        logging.info(f"Compliance tensor: {self.material.C}")

    def _construct_frame_df(self, frame_data_path):
        """
        Overwrite frame_df method from parent class Sample.


        frame_data_path: Path object pointing to the frame folder
        """

        frame_dict = {}
        for i, path in enumerate(sorted(frame_data_path.rglob("*.mat"))):
            frame_id = int(path.stem.split("-")[-1].split("_")[0])
            frame_dict[frame_id] = dict(path=path)

        frame_df = pd.DataFrame.from_dict(frame_dict, orient="index")

        frame_info = pd.read_csv(frame_data_path / "frames_info.csv", index_col="id")

        frame_df = frame_df.merge(
            right=frame_info, how="left", left_index=True, right_index=True
        )

        self.frame_df = frame_df

    def _load_frame(self, frame_id):
        self.frame = fr.Frame(
            id=frame_id,
            metadata=self.frame_df.loc[frame_id].copy(),
        )
        self.frame._load_vic_displacement_field_df()

    def _analyze_crack(
        self,
    ):
        """
        Values selected for Vic results
        Require a frame to be loaded first.
        """

        self.crack.iterative_crack_extraction(
            frame=self.frame,
            e1_crack_tip_guess=None,
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

    def plot(self, sample_size=None, moved=True):

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
