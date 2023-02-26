import pandas as pd
from pathlib import Path
from jintegral import crack as ck
from jintegral import material as mt
from jintegral import frame as fr
from jintegral import jintegral as jt
import jintegral
import logging


class Sample:

    """
    Represent a sample
    """

    def __init__(
        self,
        frame_data_path,
        name="tmp_sample",
    ) -> None:
        # Initialise sample parameters
        self.name = name
        self._construct_frame_df(frame_data_path)

        # Initialise output dataframes
        self.crack_tip_df = pd.DataFrame()
        self.crack_tip_df.index.name = "frame"

        self.err_df = pd.DataFrame(
            columns=[
                "frame",
                "fit_method",
                "pre_fit_method",
                "csaps_smooth",
                "gd_method",
                "spline_degree",
                "missing_values_method",
                "fitting_offset_post_crack",
                "fitting_offset_upper_crack",
                "fitting_offset_lower_crack",
                "jintegral_type",
                "offset_mode",
                "offset_AB",
                "offset_BC",
                "offset_CD",
                "offset_A",
                "width",
            ]
        )
        self.err_df.set_index(
            self.err_df.columns.to_list(),
            inplace=True,
        )
        self.err_detailed_df = pd.DataFrame(
            columns=[
                "frame",
                "fit_method",
                "pre_fit_method",
                "csaps_smooth",
                "gd_method",
                "spline_degree",
                "missing_values_method",
                "fitting_offset_post_crack",
                "fitting_offset_upper_crack",
                "fitting_offset_lower_crack",
                "jintegral_type",
                "offset_mode",
                "offset_AB",
                "offset_BC",
                "offset_CD",
                "offset_A",
                "width",
            ]
        )
        self.err_detailed_df.set_index(
            self.err_detailed_df.columns.to_list(),
            inplace=True,
        )

        # Initialize output folder
        self.result_path = Path("results") / self.name

        # Initialize logs
        log_path = self.result_path / "logs"
        log_path.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            format="%(levelname)s: %(asctime)s: %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
            level=logging.INFO,
            filename=log_path / f"{self.name}.log",
        )
        logging.info("##################################")
        logging.info(f"Initialize sample {name}.")

    def init_computing_units(
        self,
        fit_method,
        pre_fit_method,
        gd_method,
        missing_values_method,
        step_size,
        crack_tip_method,
        spline_degree=0,
        csaps_smooth=0,
        fitting_offset_post_crack=0.5,
        fitting_offset_upper_crack=0,
        fitting_offset_lower_crack=0,
    ):
        self.crack_tip_method = crack_tip_method

        # Initialize sample computing units
        self.crack = ck.Crack()

        self.jintegral = jt.Jintegral(
            spline_degree=spline_degree,
            fit_method=fit_method,
            pre_fit_method=pre_fit_method,
            gd_method=gd_method,
            missing_values_method=missing_values_method,
            csaps_smooth=csaps_smooth,
            step_size=step_size,
            fitting_offset_post_crack=fitting_offset_post_crack,
            fitting_offset_upper_crack=fitting_offset_upper_crack,
            fitting_offset_lower_crack=fitting_offset_lower_crack,
            crack_tip_method=crack_tip_method,
        )

    def prepare_data(
        self,
        frame_id=0,
        load_crack_data=False,
        dump_crack_data=True,
        dump_field_fits=True,
        load_field_fits=False,
    ):
        first_frame = False
        # Step 1: load frame data
        self._load_frame(frame_id=frame_id)

        # Step 2: derive crack information from frame data or load a previous derivation
        if load_crack_data:
            self.crack.load_crack_data(dump_path=self.result_path / "crack_data")
        else:
            self._analyze_crack()
            if dump_crack_data:
                self.crack.dump_crack_data(dump_path=self.result_path / "crack_data")

        # Store crack results at sample level: update the entry if existing or create it
        crack_tip = self.crack.get_crack_tip(
            frame_id=frame_id, method=self.crack_tip_method
        )
        self.crack_tip_df.loc[frame_id, crack_tip.keys()] = crack_tip.values()

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

    def compute_jintegral(
        self,
        offset,
        offset_mode="tip",
        jintegral_type="contour",
        plot=False,
        width=4,
    ):
        """
        Assume that a frame is already loaded"""

        # TODO add contour validation

        # Update calculation time parameters
        crack_tip = self.crack.get_crack_tip(
            frame_id=self.frame.id, method=self.crack_tip_method
        )

        if jintegral_type == "contour":
            self.jintegral.define_contour_coordinates(
                frame=self.frame,
                crack_tip=crack_tip,
                offset=offset,
                offset_mode=offset_mode,
            )
            self.jintegral.construct_jintegral_path()

            self.jintegral.compute_contour_jintegral(
                material=self.material, crack_tip=crack_tip
            )

        if jintegral_type == "area":
            self.jintegral.define_area_coordinates(
                frame=self.frame,
                crack_tip=crack_tip,
                width=width,
                offset=offset,
                offset_mode=offset_mode,
            )
            self.jintegral.construct_jintegral_area(
                width=width,
            )

            self.jintegral.compute_area_jintegral(
                material=self.material, crack_tip=crack_tip
            )

        self._update_output_dataframes(
            crack_tip=crack_tip,
            jintegral_type=jintegral_type,
            offset_mode=offset_mode,
            offset=offset,
            width=width,
        )

    def _load_material(self, name, material_data_path):
        self.material = mt.Material(name=name)
        self.material.load_properties(material_data_path=material_data_path)
        self.material.compute_compliance_tensor()

    def _update_output_dataframes(
        self, crack_tip, jintegral_type, offset_mode, offset, width
    ):
        # Update output dataframes
        self.err_detailed_df.loc[
            (
                self.frame.id,
                self.jintegral.fit_method,
                self.jintegral.pre_fit_method,
                self.jintegral.csaps_smooth,
                self.jintegral.gd_method,
                self.jintegral.spline_degree,
                self.jintegral.missing_values_method,
                self.jintegral.fitting_offset_post_crack,
                self.jintegral.fitting_offset_upper_crack,
                self.jintegral.fitting_offset_lower_crack,
                jintegral_type,
                offset_mode,
                offset["AB"],
                offset["BC"],
                offset["CD"],
                offset["A"],
                width,
            ),
            self.jintegral.J_dict.keys(),
        ] = self.jintegral.J_dict.values()

        self.err_df.loc[
            (
                self.frame.id,
                self.jintegral.fit_method,
                self.jintegral.pre_fit_method,
                self.jintegral.csaps_smooth,
                self.jintegral.gd_method,
                self.jintegral.spline_degree,
                self.jintegral.missing_values_method,
                self.jintegral.fitting_offset_post_crack,
                self.jintegral.fitting_offset_upper_crack,
                self.jintegral.fitting_offset_lower_crack,
                jintegral_type,
                offset_mode,
                offset["AB"],
                offset["BC"],
                offset["CD"],
                offset["A"],
                width,
            ),
            ["a", "J", "J_simplified", "error_tot", "error_mean"],
        ] = [
            crack_tip["e1"],
            self.jintegral.J_dict["J"],
            self.jintegral.J_dict["J_simplified"]
            if jintegral_type == "contour"
            else None,
            self.jintegral.J_dict["error_tot"],
            self.jintegral.J_dict["error_mean"],
        ]

    def save_results(self):
        self.result_path.mkdir(parents=True, exist_ok=True)

        self.crack_tip_df.to_csv(
            self.result_path / "{n}_crack_tip.csv".format(n=self.name)
        )
        self.err_df.to_csv(self.result_path / "{n}_err.csv".format(n=self.name))

        self.err_detailed_df.to_csv(
            self.result_path / "{n}_err_detailed.csv".format(n=self.name)
        )

        self.daq_df.to_csv(
            self.result_path / "{n}_load_displacement_data.csv".format(n=self.name)
        )

    ######### To rewite

    def plot(self, crack_tip):
        self.jintegral.construct_fit_plots(
            frame=self.frame,
            crack_tip=crack_tip,
        )

        self.result_path.mkdir(parents=True, exist_ok=True)
        self.jintegral.u1_fig.write_html(
            (
                self.result_path
                / f"{self.name}_{self.jintegral.fit_method}_{self.jintegral.pre_fit_method}_s-{self.jintegral.csaps_smooth}_u1_fit_frame-{self.frame.id:0>4d}.html"
            ).as_posix(),
            include_plotlyjs="directory",
        )
        self.jintegral.u2_fig.write_html(
            (
                self.result_path
                / f"{self.name}_{self.jintegral.fit_method}_{self.jintegral.pre_fit_method}_s-{self.jintegral.csaps_smooth}_u2_fit_frame-{self.frame.id:0>4d}.html"
            ).as_posix(),
            include_plotlyjs="directory",
        )

        # self.jintegral.CD_fig.write_html(
        #     (
        #         self.result_path
        #         / f"{self.name}_CD_path_frame-{self.frame.id:0>4d}.html"
        #     ).as_posix(),
        #     include_plotlyjs="directory",
        # )

    def analyze_full_crack(self, min_frame=50, max_frame=0, initial_e1_crack_tip=2):
        e1_crack_tip = initial_e1_crack_tip

        for file in self.vic_files:
            frame = int(file.stem.split(sep="-")[-1].split(sep="_")[0])

            if frame < min_frame:
                continue

            if frame > max_frame:
                break

            self.analyze_crack_tip(initial_e1_crack_tip, frame)

            initial_e1_crack_tip = self.e1_crack_tip

        self.crack_tip_df.to_csv("crack.csv")
        fig = px.line(data_frame=self.crack_tip_df, x="frame", y="e1")
        fig.show()

    def analyze_crack_tip(self, initial_e1_crack_tip, frame):
        # print("Analyzing crack in frame {f}".format(f=frame))
        self.import_vic_data(frame=frame)
        self.crack = ck.Crack(vic_df=self.vic_df)
        self.crack.iterative_crack_extraction(
            penultimate_e1_crack_tip=initial_e1_crack_tip
        )

        crack_tip = self.crack.get_crack_tip()
        crack_tip["frame"] = frame

        self.e1_crack_tip = crack_tip["e1"]

        self.crack_tip_df = self.crack_tip_df.append(crack_tip, ignore_index=True)


def cosntruct_vic_data_path(data_path, sample_id, frame_id):
    data_path = Path(data_path)
    return data_path / f"ct-spl-{sample_id:0>3d}-{frame_id:0>4d}_0.mat"
