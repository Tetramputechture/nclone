"""Tests for the map -> image export (the nclone-map-render skill's backend)."""

import os

import pytest

from nclone.map_generation import compose
from nclone.map_generation.compose import render_image


def _png_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 1000


def test_render_schematic_writes_png(tmp_path):
    m, _ = compose.compose("grand_mix_2x3", seed=3)
    out = str(tmp_path / "schematic.png")
    render_image.render_schematic(m.map_data(), out, title="t")
    assert _png_nonempty(out)


def test_solid_mask_shape_and_border():
    m, _ = compose.compose("flat_intro_2x3", seed=0)
    mask = render_image._solid_mask(m.map_data(), sub=4)
    from nclone.constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT

    assert mask.shape == (MAP_TILE_HEIGHT * 4, MAP_TILE_WIDTH * 4)
    assert mask[0].all() and mask[-1].all()  # solid top/bottom border rows


def test_render_ground_truth_writes_png(tmp_path):
    m, _ = compose.compose("tunnel_curves_2x3", seed=5)
    out = str(tmp_path / "gt.png")
    render_image.render_ground_truth(m.map_data(), out)
    assert _png_nonempty(out)


@pytest.mark.parametrize("kw", [{"template": "mixed_traverse_2x3"},
                                 {"phrase": "two gaps then mines"}])
def test_cli_load_map_data(kw):
    class A:
        template = kw.get("template")
        phrase = kw.get("phrase")
        pickle = None
        json = None
        seed = 0

    md = render_image._load_map_data(A())
    assert len(md) > 1235 and md[1156] == 1  # has tiles+entities, one exit door
