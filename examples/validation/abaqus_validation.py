"""
Validation from abaqus files.

"""


from jintegral import abaqus_sample as asp, crack
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import queue
import multiprocessing as mp
import json


def compute_jintegral(
    jobs, results, path, name, frame_id, material_name, kwargs, crop_data=None
):
    sample = asp.AbaqusSample(
        frame_data_path=path, name=name, material_name=material_name
    )

    sample.init_computing_units(**kwargs)
    sample.prepare_data(
        frame_id=frame_id,
        dump_field_fits=False,
        load_field_fits=True,
        load_crack_data=True,
        dump_crack_data=False,
        crop_data=crop_data,
    )

    while True:
        try:
            parameters = jobs.get(block=False)
            sample.compute_jintegral(
                offset=parameters["offset"],
                jintegral_type=parameters["jintegral_type"],
                width=parameters["width"],
                plot=False,
            )

            tmp = sample.err_df.tail(1).copy()
            tmp["frame_id"] = frame_id

            tmp_detailed = sample.err_detailed_df.tail(1).copy()
            tmp_detailed["frame_id"] = frame_id
            results.put((tmp, tmp_detailed))

        except queue.Empty:
            break

        except ValueError:
            pass


def analyze_full_field():
    path = Path(__file__).parent / "data"
    name = "abaqus_sample"
    material_name = "34-700-60gsm_TP415_CP"

    for frame_id in (
        20,
        30,
        40,
    ):
        for sigma_awgn in (
            None,
            2.2e-4,
            1e-2,
            # if frame_id == 30
            # else (
            #     # None,
            #     # 2.2e-4,
            #     1e-2,
            # )
        ):
            # sigma = 2.2e-4 corresponds to sigma * pixel_size = 0.005 * 1/23 mm for Vic3D sample
            # sigma = 3.6e-2 corresponds to sigma * pixel_size = 0.03 * 1.2 mm for Davis sample

            # Frames are actually crack lengths geometrically defined
            csaps_smooths = (
                (
                    # 0.0001,
                    # 0.001,
                    # 0.01,
                    # 0.1,
                    # 0,
                    0.5,
                    1,
                )
                if frame_id == 30
                # else (
                #     # 0.0001,
                #     # 0.001,
                #     0.01,
                #     # 0.1,
                #     # 0,
                # )
                # if frame_id == 30
                else (0.1,)
            )
            for csaps_smooth in csaps_smooths:
                print("\n")
                print(f"Running {frame_id} {sigma_awgn} {csaps_smooth}")

                kwargs = dict(
                    fit_method="csaps",
                    spline_degree=0,
                    pre_fit_method="gd",
                    gd_method="cubic",
                    missing_values_method="reduce",
                    csaps_smooth=csaps_smooth,
                    step_size=0.1,
                    fitting_offset_post_crack=0.5,
                    fitting_offset_upper_crack=0,
                    fitting_offset_lower_crack=2
                    if frame_id == 40 and sigma_awgn == 1e-2
                    else 0,
                    crack_tip_method="intersec" if sigma_awgn == 1e-2 else "u2_inflex",
                )

                print(kwargs["fitting_offset_lower_crack"])

                sample = asp.AbaqusSample(
                    frame_data_path=path,
                    material_data_path=path,
                    name=name,
                    material_name=material_name,
                )
                sample.init_computing_units(**kwargs)

                result_path = sample.result_path / "err"
                result_path.mkdir(exist_ok=True, parents=True)

                plot_path = sample.result_path / "plot"
                plot_path.mkdir(exist_ok=True, parents=True)

                if sigma_awgn:
                    sample.prepare_data(
                        frame_id=frame_id,
                        load_field_fits=False,
                        dump_field_fits=True,
                        load_crack_data=False,
                        dump_crack_data=True,
                        sigma_error=sigma_awgn,
                        random_seed=2022,
                        plot_error=True,
                        overwrite_interp_step=0.3,
                    )

                else:
                    sample.prepare_data(
                        frame_id=frame_id,
                        load_field_fits=False,
                        dump_field_fits=True,
                        load_crack_data=False,
                        dump_crack_data=True,
                        overwrite_interp_step=0.3,
                    )

                crack_tip = sample.crack.get_crack_tip(
                    frame_id=frame_id, method=kwargs["crack_tip_method"]
                )

                sample.jintegral.construct_fit_3d_scatter(
                    frame=sample.frame, crack=sample.crack
                )
                sample.jintegral.u1_fig.write_html(
                    plot_path
                    / f"u1_abaqus_sample_frame-{frame_id}_sigma-{sigma_awgn}_csaps_smooth-{csaps_smooth}.html"
                )
                sample.jintegral.u2_fig.write_html(
                    plot_path
                    / f"u2_abaqus_sample_frame-{frame_id}_sigma-{sigma_awgn}_csaps_smooth-{csaps_smooth}.html"
                )

                sample.crack.plot_cod_profile(frame=sample.frame)

                jobs = mp.Queue()

                # Evaluate contour integral

                max_offset_AB = int(
                    np.floor(
                        crack_tip["e1"] - sample.frame.displacement_field_df["X"].min()
                    )
                )

                max_offset_BC = int(
                    np.floor(
                        crack_tip["e2"] - sample.frame.displacement_field_df["Y"].min()
                    )
                )

                max_offset_CD = int(
                    np.floor(
                        sample.frame.displacement_field_df["X"].max() - crack_tip["e1"]
                    )
                )

                offsets_AB = list(range(2, max_offset_AB, 1))
                offsets_AB.append(max_offset_AB)
                offsets_BC = list(range(2, max_offset_BC, 1))
                offsets_BC.append(max_offset_BC)
                offsets_CD = list(range(2, max_offset_CD, 4))
                offsets_CD.append(max_offset_CD)

                for AB in offsets_AB:
                    for BC in offsets_BC:
                        for CD in offsets_CD:
                            for A in (0.1, 1) if csaps_smooth == 0.1 else (1,):
                                jobs.put(
                                    dict(
                                        jintegral_type="contour",
                                        offset=dict(AB=AB, BC=BC, CD=CD, A=A),
                                        width=0,
                                    )
                                )

                # for A in (2, 3, 4, 5, 6, 7, 8, 9, 10):

                #     jobs.put(
                #         dict(
                #             jintegral_type="contour",
                #             offset=dict(AB=10, BC=20, CD=2, A=A),
                #             width=0,
                #         )
                #     )

                # Evaluate area integral
                if frame_id == 20:
                    widths = (1, 2, 4)
                else:
                    widths = (2, 4, 6, 8)
                for width in widths:
                    offsets_AB = list(range(2, max_offset_AB - width, 2))
                    offsets_AB.append(max_offset_AB - width)
                    offsets_BC = list(range(2, max_offset_BC - width, 2))
                    offsets_BC.append(max_offset_BC - width)
                    if frame_id == 30 and csaps_smooth == 1e-1:
                        offsets_CD = list(range(2, max_offset_CD - width, 4))
                        offsets_CD.append(max_offset_CD - width)
                    else:
                        offsets_CD = [
                            6,
                        ]

                    for AB in offsets_AB:
                        for BC in offsets_BC:
                            for CD in offsets_CD:
                                for A in (0.1, 1) if csaps_smooth == 0.1 else (1,):
                                    jobs.put(
                                        dict(
                                            jintegral_type="area",
                                            offset=dict(AB=AB, BC=BC, CD=CD, A=A),
                                            width=width,
                                        )
                                    )

                # for A in (2, 3, 4, 5, 6, 7, 8, 9, 10):

                #     jobs.put(
                #         dict(
                #             jintegral_type="area",
                #             offset=dict(AB=10, BC=20, CD=2, A=A),
                #             width=6,
                #         )
                #     )

                q_size = jobs.qsize()

                # Init progress bar
                pbar = tqdm(total=q_size, unit="run")

                results = mp.Queue()

                for _ in range(12):
                    process = mp.Process(
                        target=compute_jintegral,
                        kwargs=dict(
                            jobs=jobs,
                            results=results,
                            path=path,
                            name=name,
                            material_name=material_name,
                            frame_id=frame_id,
                            kwargs=kwargs,
                        ),
                    )
                    process.start()

                result_df = []
                detailed_result_df = []

                for _ in range(q_size):
                    result = results.get()
                    result_df.append(result[0])
                    detailed_result_df.append(result[1])
                    # with open(result_path / "tmp_results.pickle", "wb") as file:
                    #     pickle.dump(result_df, file)
                    pbar.update(1)

                result_df = pd.concat(result_df)
                result_df["sigma_awgn"] = sigma_awgn

                detailed_result_df = pd.concat(detailed_result_df)
                detailed_result_df["sigma_awgn"] = sigma_awgn

                result_df.to_csv(
                    result_path
                    / f"abaqus_sample_frame-{frame_id}_sigma-{sigma_awgn}_csaps_smooth-{csaps_smooth}.csv",
                    # index=False,
                )

                detailed_result_df.to_csv(
                    result_path
                    / f"detailed_abaqus_sample_frame-{frame_id}_sigma-{sigma_awgn}_csaps_smooth-{csaps_smooth}.csv",
                    # index=False,
                )

                metadata = {}
                metadata["crack_tip"] = crack_tip
                metadata["min_e1"] = sample.frame.displacement_field_df["X"].min()
                metadata["max_e1"] = sample.frame.displacement_field_df["X"].max()
                metadata["min_e2"] = sample.frame.displacement_field_df["Y"].min()
                metadata["max_e2"] = sample.frame.displacement_field_df["Y"].max()

                metadata_path = sample.result_path / "metadata"
                metadata_path.mkdir(parents=True, exist_ok=True)
                with open(
                    metadata_path
                    / f"metadata_abaqus_sample_frame-{frame_id}_sigma-{sigma_awgn}_csaps_smooth-{csaps_smooth}.json",
                    "w",
                    encoding="utf8",
                ) as file:
                    json.dump(metadata, file)


def analyze_cropped_data():
    path = Path(__file__).parent / "data"
    name = "abaqus_sample"
    material_name = "34-700-60gsm_TP415_CP"
    crop_data = 2

    for frame_id in (
        # 20,
        30,
        # 40,
    ):
        for sigma_awgn in (
            None,
            2.2e-4,
            1e-2,
        ):
            # sigma = 2.2e-4 corresponds to sigma * pixel_size = 0.005 * 1/23 mm for Vic3D sample
            # sigma = 3.6e-2 corresponds to sigma * pixel_size = 0.03 * 1.2 mm for Davis sample

            # Frames are actually crack lengths geometrically defined
            for csaps_smooth in (1e-1,):
                kwargs = dict(
                    fit_method="csaps",
                    spline_degree=0,
                    pre_fit_method="gd",
                    gd_method="cubic",
                    missing_values_method="reduce",
                    csaps_smooth=csaps_smooth,
                    step_size=0.1,
                    fitting_offset_post_crack=0.5,
                    fitting_offset_upper_crack=0,
                    fitting_offset_lower_crack=0,
                    crack_tip_method="intersec" if sigma_awgn == 1e-2 else "u2_inflex",
                )

                sample = asp.AbaqusSample(
                    frame_data_path=path,
                    material_data_path=path,
                    name=name,
                    material_name=material_name,
                )
                sample.init_computing_units(**kwargs)

                result_path = sample.result_path / "err"
                result_path.mkdir(exist_ok=True, parents=True)

                plot_path = sample.result_path / "plot"
                plot_path.mkdir(exist_ok=True, parents=True)

                if sigma_awgn:
                    sample.prepare_data(
                        frame_id=frame_id,
                        load_field_fits=False,
                        dump_field_fits=True,
                        load_crack_data=False,
                        dump_crack_data=True,
                        sigma_error=sigma_awgn,
                        random_seed=2022,
                        plot_error=True,
                        crop_data=crop_data,
                    )

                else:
                    sample.prepare_data(
                        frame_id=frame_id,
                        load_field_fits=False,
                        dump_field_fits=True,
                        load_crack_data=False,
                        dump_crack_data=True,
                        crop_data=crop_data,
                    )

                sample.crack.crack_data[frame_id][
                    f"{kwargs['crack_tip_method']}_e1_crack_tip"
                ] = (15.4740904935015 if sigma_awgn == 1e-2 else 14.4)

                sample.crack.dump_crack_data(
                    dump_path=sample.result_path / "crack_data"
                )

                crack_tip = sample.crack.get_crack_tip(
                    frame_id=frame_id, method=kwargs["crack_tip_method"]
                )
                print(crack_tip)

                sample.jintegral.construct_fit_3d_scatter(
                    frame=sample.frame, crack=sample.crack
                )
                sample.jintegral.u1_fig.write_html(
                    plot_path
                    / f"cropped_u1_abaqus_sample_frame-{frame_id}_sigma-{sigma_awgn}_csaps_smooth-{csaps_smooth}.html"
                )
                sample.jintegral.u2_fig.write_html(
                    plot_path
                    / f"cropped_u2_abaqus_sample_frame-{frame_id}_sigma-{sigma_awgn}_csaps_smooth-{csaps_smooth}.html"
                )

                sample.crack.plot_cod_profile(frame=sample.frame)

                jobs = mp.Queue()

                # Evaluate contour integral

                max_offset_AB = int(
                    np.floor(
                        crack_tip["e1"] - sample.frame.displacement_field_df["X"].min()
                    )
                )

                max_offset_BC = int(
                    np.floor(
                        crack_tip["e2"] - sample.frame.displacement_field_df["Y"].min()
                    )
                )

                max_offset_CD = int(
                    np.floor(
                        sample.frame.displacement_field_df["X"].max() - crack_tip["e1"]
                    )
                )

                print("\n")
                print(f"max offset BC : {max_offset_BC}")
                print(f"min Y : {sample.frame.displacement_field_df['Y'].min()}")
                print(f"crack tip : {crack_tip}")

                offsets_AB = list(range(2, max_offset_AB, 1))
                offsets_AB.append(max_offset_AB)
                offsets_BC = list(range(2, max_offset_BC, 1))
                offsets_BC.append(max_offset_BC)
                # offsets_CD = list(range(2, max_offset_CD, 2))
                # offsets_CD.append(max_offset_CD)
                offsets_CD = (6,)

                for AB in offsets_AB:
                    for BC in offsets_BC:
                        for CD in offsets_CD:
                            jobs.put(
                                dict(
                                    jintegral_type="contour",
                                    offset=dict(AB=AB, BC=BC, CD=CD, A=0.1),
                                    width=0,
                                )
                            )

                q_size = jobs.qsize()

                # Init progress bar
                pbar = tqdm(total=q_size, unit="run")

                results = mp.Queue()

                for _ in range(16):
                    process = mp.Process(
                        target=compute_jintegral,
                        kwargs=dict(
                            jobs=jobs,
                            results=results,
                            path=path,
                            name=name,
                            material_name=material_name,
                            frame_id=frame_id,
                            kwargs=kwargs,
                            crop_data=crop_data,
                        ),
                    )
                    process.start()

                result_df = []
                detailed_result_df = []

                for _ in range(q_size):
                    result = results.get()
                    result_df.append(result[0])
                    detailed_result_df.append(result[1])
                    # with open(result_path / "tmp_results.pickle", "wb") as file:
                    #     pickle.dump(result_df, file)
                    pbar.update(1)

                result_df = pd.concat(result_df)
                result_df["sigma_awgn"] = sigma_awgn

                detailed_result_df = pd.concat(detailed_result_df)
                detailed_result_df["sigma_awgn"] = sigma_awgn

                result_df.to_csv(
                    result_path
                    / f"cropped_abaqus_sample_frame-{frame_id}_sigma-{sigma_awgn}_csaps_smooth-{csaps_smooth}.csv",
                    # index=False,
                )

                detailed_result_df.to_csv(
                    result_path
                    / f"cropped_detailed_abaqus_sample_frame-{frame_id}_sigma-{sigma_awgn}_csaps_smooth-{csaps_smooth}.csv",
                    # index=False,
                )

                # metadata = {}
                # metadata["crack_tip"] = crack_tip
                # metadata["min_e1"] = sample.frame.displacement_field_df["X"].min()
                # metadata["max_e1"] = sample.frame.displacement_field_df["X"].max()
                # metadata["min_e2"] = sample.frame.displacement_field_df["Y"].min()
                # metadata["max_e2"] = sample.frame.displacement_field_df["Y"].max()

                # metadata_path = sample.result_path / "metadata"
                # metadata_path.mkdir(parents=True, exist_ok=True)
                # with open(
                #     metadata_path
                #     / f"metadata_abaqus_sample_frame-{frame_id}_sigma-{sigma_awgn}_csaps_smooth-{csaps_smooth}.json",
                #     "w",
                #     encoding="utf8",
                # ) as file:
                #     json.dump(metadata, file)


if __name__ == "__main__":
    analyze_full_field()
    # analyze_cropped_data()
