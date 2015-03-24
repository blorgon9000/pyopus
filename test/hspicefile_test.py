#!/usr/bin/env python

import pyopus.simulator.hspicefile as hf
# import matplotlib.pyplot as plt


for _f,_s in (
        ('9601.tr0','+'),
        ('2001.tr0','x'),
        ):
    hspf = hf.hspice_read(_f, debug=1)

    for _x in (1, 3, 4):
        print hspf[0][_x]

    results = hspf[0][0][2][0]
    scale = results[hspf[0][1]]



    print 'len(scale)=',len(scale)
    for _i in (0,1,-2,-1):
        print 'scale[{}]='.format(_i),scale[_i]

    print '-'*79

    # plt.plot(scale,results['nout1'],_s)
# plt.show()
