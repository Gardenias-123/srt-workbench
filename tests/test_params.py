"""Tests for the GUI parameter container and Fortran input writer."""

import pytest

from app.params import SRTParams


def test_fortran_input_structure():
    p = SRTParams(slope=15, aspect=270, alpha_slope=15, beta_aspect=270)
    text = p.to_fortran_input()
    lines = text.splitlines()
    assert lines[0].startswith("# Canopy")
    assert lines[2].split()[0] == "1.0"
    assert [float(x) for x in lines[8].split()] == [15.0, 270.0, 15.0, 270.0]
    assert int(lines[14].split()[0]) == p.nfi_usr
    assert len(lines[16].split()) == p.nfi_usr
    assert len(lines[17].split()) == p.nfi_usr


def test_invalid_parameters():
    with pytest.raises(ValueError):
        SRTParams(cover=1.2)
    with pytest.raises(ValueError):
        SRTParams(d_L=0)
    with pytest.raises(ValueError):
        SRTParams(tau_L=0.8, ro_L=0.8)
    with pytest.raises(ValueError):
        SRTParams(sza=95)


def test_with_i0():
    p = SRTParams(i0=0.4)
    assert p.with_i0(0.0).i0 == 0.0
    assert p.i0 == 0.4
