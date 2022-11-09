from jintegral import crack, frame, sample as sp
from jintegral import frame as fr
from jintegral import material as mt
from jintegral import jintegral as jt
import pandas as pd
import plotly.express as px
import numpy as np


class DavisSample(sp.Sample):
    def __init__(self, frame_data_path, material_name, name="davis_sample") -> None:
        super().__init__(frame_data_path, name)
        self._load_material(name=material_name)

    def _construct_frame_dict(self, frame_data_path):
        """
        Overwrite frame_dict method from parent class Sample.
        Frame id is based on sorting

        frame_data_path: Path object pointing to the frame folder
        """

        frame_dict = {}
        for i, path in enumerate(sorted(frame_data_path.rglob("*.csv"))):
            frame = int(path.stem.split("-")[-1])
            frame_dict[frame] = dict(path=path)

        return frame_dict

    def _load_frame(self, frame_id):
        self.frame = fr.Frame(id=frame_id, data_path=self.frame_dict[frame_id]["path"])
        self.frame._load_davis_displacement_field_df()

    def _analyze_crack(self, e1_crack_tip_guess=20):
        """
        Values selected for Abaqus results
        Load the proper frame first.
        """

        self.crack.iterative_crack_extraction(
            frame=self.frame,
            e1_crack_tip_guess=e1_crack_tip_guess,
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
