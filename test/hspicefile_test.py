#!/usr/bin/env python


import pytest

from itertools import izip

# @pytest.mark.parametrize("ext",
#     ['tr0','sw0','ac0']
#     )

@pytest.fixture(scope="module",
        params=['tr0','sw0','ac0'])
def hspfs(request):
    import pyopus.simulator.hspicefile as hf

    objs = []
    for ver in ('9601','2001'):
        filename = '{}.{}'.format(ver,request.param)
        objs.append(hf.hspice_read(filename, debug=0)[0])

    return objs



def test_scale_name(hspfs):
    assert hspfs[0][1] == hspfs[1][1]

def test_title_string(hspfs):
    assert hspfs[0][3] == hspfs[1][3]

def test_sweep_name(hspfs):
    assert hspfs[0][0][0] == hspfs[1][0][0]

def test_sweep_params(hspfs):
    assert hspfs[0][0][1] == hspfs[1][0][1]

def test_result_names(hspfs):
    for a, b in izip(hspfs[0][0][2], hspfs[1][0][2]):
        assert a.keys() == b.keys()

def test_result_values(hspfs):
    from numpy import allclose
    for a, b in izip(hspfs[0][0][2], hspfs[1][0][2]):
        for k in a:
            assert k in b
            vala=a[k]
            valb=b[k]
            assert allclose(vala,valb)

