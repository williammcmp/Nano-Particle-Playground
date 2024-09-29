#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday September 29 2024

Test for the centrifugation module. Ensure that the the correct unit convertions are implemented and not changed.

@author: william.mcm.p
"""

import pytest
import numpy as np
from src.Centrifugation import Centrifugation

@pytest.fixture
def cen_fixture():
    cen = Centrifugation()
    return cen

def test_sedmention_velocity(cen_fixture):
    rpm = 5000
    particle_size = 100 * 1e-9
    sed_rate, sed_velocity = cen_fixture.cal_sedimentation_rate(rpm, particle_size)

    assert sed_rate == 1.9511409710329487e-09 # Sedmentation coefficent of the particle and fluid system
    assert sed_velocity == 5.34916375412571e-05 # Sedmenation rate of the particle in the liquid while spinning
