import scipy.io as sio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from jintegral import utils as ut
import logging


class Frame:
    """
    Private class representation of frame, aquired by DIC or extracted from FEM.
    One frame holds the displacement fields of a sample at a time t.
    The exact time t is irrevelant for the J-integral calculation and t is implicitly contained in the frame id.

    Attributes:
    id : int
        the frame id
    data_path : Path object
        the path to the frame displacement field data
    metadata : dict
        theframe metadata

    Methods:

    No public method.

    Notes
    -------

    Field data may consume a lot of memory so instantiating several frames at the same time is not recommended.
    """

    def __init__(self, id: int, metadata: dict) -> None:
        """
        Args:
        id: int
            the frame id
        metadata: dict
            the frame metadata

        Returns:
        None
        """
        logging.info(f"Initialize frame {id}")
        self.id = id
        self.data_path = metadata["path"]
        self.metadata = metadata

    def _load_vic_displacement_field_df(self):
        """
        Import data and return a dataframe after conditioning.
        """

        mat = sio.loadmat(self.data_path)
        for k in mat.keys():
            mat[k] = mat[k].reshape(-1, 1).squeeze()
            # print(k, len(mat[k]))

        df = pd.DataFrame(mat)
        # Mat files exported by vic3d includes supplementary data (unsuccessful corelations? )
        df = df.loc[((df.X != 0) & (df.Y != 0)) & (df.sigma >= 0)]
        df.reset_index(drop=True, inplace=True)

        # Remove useless columns
        df = df[["X", "Y", "Z", "U", "V", "W", "x", "y", "sigma"]]

        # Reference basis was rotated when extracting the data
        df.rename(columns={"x": "y_ref", "y": "x_ref", "sigma": "error"}, inplace=True)

        df = df.astype({k: "float" for k in ["X", "Y", "Z", "U", "V", "W"]})
        df = df.astype({k: "int" for k in ["x_ref", "y_ref"]})

        # Inverse V?
        # df["V"] = -df["V"]

        # remove NaN
        df.dropna(axis="index", inplace=True)

        flip_x = False
        if flip_x:
            # flib x axis
            df.X = abs(df.X - df.X.max())

        self.X_offset = df.X.min()
        df.X -= self.X_offset
        self.metadata["optical_crack_tip_e1"] -= self.X_offset

        self.displacement_field_df = df
        self._compute_grid_params(step_agg="mean")
        return df

    def _load_abaqus_displacement_field_df(self, crop_data=None):
        """
        Load displacement field data from an Abaqus export

        """

        displacement_field_df = pd.read_csv(self.data_path)
        displacement_field_df["x_ref"] = displacement_field_df["X"]
        displacement_field_df["y_ref"] = displacement_field_df["Y"]
        self.X_offset = displacement_field_df.X.min()
        self.metadata["optical_crack_tip_e1"] -= self.X_offset
        displacement_field_df.X -= self.X_offset
        displacement_field_df["x_u"] = (
            displacement_field_df["X"] + displacement_field_df["U"]
        )
        displacement_field_df["y_v"] = (
            displacement_field_df["Y"] + displacement_field_df["V"]
        )

        displacement_field_df["error"] = 0

        if crop_data:
            max_X = displacement_field_df["X"].max()
            max_Y = displacement_field_df["Y"].max()
            min_Y = displacement_field_df["Y"].min()
            displacement_field_df = displacement_field_df.query(
                "X <= (@max_X - @crop_data) & Y <= (@max_Y - @crop_data) & Y >= (@min_Y + @crop_data)"
            )

        self.displacement_field_df = displacement_field_df
        self._compute_grid_params(step_agg="max")
        return displacement_field_df

    def _load_davis_displacement_field_df(self):
        """
        Load displacement field data from an bristol sample

        """

        displacement_field_df = pd.read_csv(self.data_path)
        displacement_field_df.columns = [
            x if x == "error" else x.upper() for x in displacement_field_df.columns
        ]
        displacement_field_df["x_ref"] = displacement_field_df["X"]
        displacement_field_df["y_ref"] = displacement_field_df["Y"]

        displacement_field_df["x_u"] = (
            displacement_field_df["X"] + displacement_field_df["U"]
        )
        displacement_field_df["y_v"] = (
            displacement_field_df["Y"] + displacement_field_df["V"]
        )

        self.displacement_field_df = displacement_field_df
        self._compute_grid_params(step_agg="mean")
        return displacement_field_df

    def _compute_grid_params(self, step_agg):
        self.e1_min = self.displacement_field_df["X"].min()
        self.e1_max = self.displacement_field_df["X"].max()
        self.e2_min = self.displacement_field_df["Y"].min()
        self.e2_max = self.displacement_field_df["Y"].max()
        self.interp_step = ut.compute_interp_step(
            x=self.displacement_field_df["X"].to_numpy(),
            y=self.displacement_field_df["Y"].to_numpy(),
            agg=step_agg,
        )
        logging.info(f"Best interpolation step found: {self.interp_step} mm")

    def _add_noise_to_displacement_field(
        self,
        fields=["U", "V"],
        snr=None,
        noise_type="awgn",
        noise_std=None,
        uniform_bd=None,
        random_seed=None,
    ):
        # Intialise a random number generator
        random_nb_generator = np.random.default_rng(random_seed)

        for field in fields:
            if not noise_std and noise_std != 0:
                field_rms = np.sqrt(
                    (
                        (
                            self.displacement_field_df[field]
                            - self.displacement_field_df[field].mean()
                        )
                        ** 2
                    ).mean()
                )
                noise_rms = np.sqrt(field_rms**2 / snr)
                noise_std = noise_rms
            if noise_type == "awgn":

                noise = random_nb_generator.normal(
                    loc=0.0, scale=noise_std, size=self.displacement_field_df.shape[0]
                )
            elif noise_type == "uniform":
                # assume uniform distribution centered in 0
                # a = noise_std / np.sqrt(48)
                noise = random_nb_generator.uniform(
                    low=-uniform_bd,
                    high=uniform_bd,
                    size=self.displacement_field_df.shape[0],
                )
            else:
                raise ValueError("Not a defined noise type.")

            self.displacement_field_df[field] += noise
            self.displacement_field_df[f"{field}_noise"] = noise

        # Define error as the norm of noise
        self.displacement_field_df["error"] = (
            self.displacement_field_df[[f"{field}_noise" for field in fields]]
            .pow(2)
            .sum(axis=1)
            .pow(1 / 2)
        )

    def plot(self):
        fig = go.Figure()
        fig.add_contour(
            x=self.displacement_field_df["X"],
            y=self.displacement_field_df["Y"],
            z=self.displacement_field_df["V"],
        )
        fig.write_image("test.pdf")
