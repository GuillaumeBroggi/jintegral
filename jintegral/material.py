import pandas as pd
import numpy as np
from pathlib import Path


class Material:
    def __init__(self, name) -> None:
        self.name = name

    def load_properties(self, material_data_path):
        df = pd.read_csv(material_data_path / "materials.csv", index_col="name")

        try:
            properties = df.loc[self.name].to_dict()
        except KeyError:
            properties = df.iloc[0].to_dict()
            print(
                "Warning: {n} is not in the materials.csv. Defaulted to {d}.".format(
                    n=self.name, d=df.index[0]
                )
            )

        for k in properties:
            setattr(self, k, properties[k])

    def compute_stiffness_tensor(self, plane_stress=True):
        self.nu21 = self.nu12 * self.E2 / self.E1

        # TODO not plane stress case

        if plane_stress:
            # Define orthotropic plane stress compliance tensor
            self.S = np.array(
                [
                    [1 / self.E1, -self.nu21 / self.E2, 0],
                    [
                        -self.nu12 / self.E1,
                        1 / self.E2,
                        0,
                    ],
                    [0, 0, 1 / self.G12],
                ]
            )

    def compute_compliance_tensor(self):
        self.compute_stiffness_tensor(plane_stress=True)
        self.C = np.linalg.inv(self.S)


def main():
    material = Material(name="dummy")
    material.load_properties()
    material.compute_stiffness_tensor()
    print(material.S)


if __name__ == "__main__":
    main()
