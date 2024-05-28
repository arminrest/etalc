import os
import argparse
from icecream import ic

import numpy as np
from astropy import units as u


def gen_region_file_header():
    header = "# Region file format: DS9 version 4.1"
    header += "\n"

    header += f"global color=green dashlist=8 3 "
    header += f'width=3 font="helvetica 10 normal roman" '
    header += f"select=1 highlite=1 dash=0 fixed=0 "
    header += f"edit=1 move=1 delete=1 include=1 source=1"
    header += "\n"

    header += "fk5"
    header += "\n"
    return header


def gen_region(center_alpha, center_delta, box_size_alpha, box_size_delta):
    region = f'box({center_alpha:.7f},{center_delta:.7f},{box_size_alpha:.3f}",{box_size_delta:.3f}",0)'
    region += "\n"
    return region


def gen_region_grid(pixel_scale, grid_sep, grid_alpha_bounds, grid_delta_bounds):
    # 3 pixels (in arcsec) corresponding to `pixel_scale`.
    three_pixels = pixel_scale * 3

    vector_alpha = np.arange(grid_alpha_bounds[0], grid_alpha_bounds[1], grid_sep)
    vector_delta = np.arange(grid_delta_bounds[0], grid_delta_bounds[1], grid_sep)
    grid_alpha, grid_delta = np.meshgrid(vector_alpha, vector_delta)

    region_strings = []
    for i, (alpha, delta) in enumerate(zip(grid_alpha.flatten(), grid_delta.flatten())):
        region_str = gen_region(alpha, delta, three_pixels, three_pixels)
        region_strings.append(region_str)

    return region_strings


def make_region_file(filename, reg_file):
    with open(filename, "w") as f:
        f.write(reg_file)

    return filename


def main(
    filename,
    pixel_scale,
    grid_sep,
    grid_alpha_bounds,
    grid_delta_bounds,
):
    header = gen_region_file_header()
    region_strings = gen_region_grid(
        pixel_scale,
        grid_sep,
        grid_alpha_bounds,
        grid_delta_bounds,
    )
    reg_file = header + "".join(region_strings)
    make_region_file(filename, reg_file)
    print(f"Region file '{filename}' created.")
    return


if __name__ == "__main__":
    filename = "/Users/admin/Code/etalc/wff/sparse_grid.reg"

    region_color = "red"

    grid_alpha_bounds = [(160.9647373 * u.deg).value, (161.2446756 * u.deg).value]
    grid_delta_bounds = [(-60.2602583 * u.deg).value, (-60.1858387 * u.deg).value]

    pixel_scale = 0.810 * u.arcsec / 3  # Extracted manually, this may vary...
    grid_sep = (pixel_scale * 100).to(u.deg).value
    ic(pixel_scale)
    ic(grid_sep)
    ic(grid_alpha_bounds)
    ic(grid_delta_bounds)

    main(
        filename,
        pixel_scale.value,
        grid_sep,
        grid_alpha_bounds,
        grid_delta_bounds,
    )
