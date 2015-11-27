# Test performance evaluator
#import matplotlib.pyplot as plt

__all__ = [ 'heads', 'analyses', 'measures', 'variables', 'statParams', 'opParams', 'designParams' ]


heads = {
	'opus': {
		'simulator': 'SpiceOpus', 
		'settings': {
			'debug': 0
		}, 
		'moddefs': {
			'def':  { 'file': 'opamp.inc' }, 
			'tb':   { 'file': 'topdc.inc' },
			'tbrr': { 'file': 'toprr.inc' }, 
			'tm':   { 'file': 'cmos180n.lib', 'section': 'tm' },
			'wp':   { 'file': 'cmos180n.lib', 'section': 'wp' }, 
			'ws':   { 'file': 'cmos180n.lib', 'section': 'ws' }, 
			'wo':   { 'file': 'cmos180n.lib', 'section': 'wo' }, 
			'wz':   { 'file': 'cmos180n.lib', 'section': 'wz' }, 
			'wcd':  { 'file': 'cmos180n.lib', 'section': 'wcd' },
		}, 
		'options': {
			'method': 'trap'
		}, 
		'params': {
			'ibias': 100e-6, 
			'lev1': -0.5,
			'lev2': 0.5,
			'tstart': 10000e-9, 
			'tr': 1e-9, 
			'tf': 1e-9, 
			'pw': 10000e-9, 
			'rload': 100e6, 
			'cload': 1e-12
		}
	}
}

variables={
	'mosList': [ 'xmn2', 'xmn3', 'xmn1', 'xmn1b', 'xmn4', 'xmp1', 'xmp2', 'xmp3' ], 
	#'plt': plt 
}

analyses = {
	'op': {
		'head': 'opus', 
		'modules': [ 'def', 'tb' ], 
		'options': {
			'method': 'gear'
		}, 
		'params': {
			'rin': 1e6,
			'rfb': 1e6
		},
		# Save current through vdd. Save voltages at inp, inn, and out. 
		# Save vgs, vth, vds, and vdsat for mn2, mn3, and mn1. 
		'saves': [ 
			"all()", 
			"p(ipath(var['mosList'], 'x1', 'm0'), ['vgs', 'vth', 'vds', 'vdsat'])"
		],  
		'command': "op()"
	}, 
	'dc': {
		'head': 'opus', 
		'modules': [ 'def', 'tb' ], 
		'options': {
			'method': 'gear'
		},
		'params': {
			'rin': 1e6,
			'rfb': 1e6
		},	
		'saves': [ ], 
		'command': "dc(-2.0, 2.0, 'lin', 100, 'vin', 'dc')"
	}, 
	'ac': {
		'head' : 'opus', 
		'modules': [ 'def', 'tb' ], 
		'options': {
			'method': 'gear'
		}, 
		'params': {
			'rin': 1e6,
			'rfb': 1e6
		},
		'saves': [ "all()" ], 
		'command': "ac(1, 1e12, 'dec', 10)"
	}, 
	'accom': {
		'head' : 'opus', 
		'modules': [ 'def', 'tbrr' ], 
		'options': {
			'method': 'gear'
		}, 
		'params': {
			'rfb': 1e9, 
			'cin': 0.1, 
			'acdif': 0.0, 
			'accom': 1.0, 
			'acvdd': 0.0, 
			'acvss': 0.0
		},
		'saves': [ "all()" ], 
		'command': "ac(1, 1e9, 'dec', 10)"
	}, 
	'acvdd': {
		'head' : 'opus', 
		'modules': [ 'def', 'tbrr' ], 
		'options': {
			'method': 'gear'
		}, 
		'params': {
			'rfb': 1e9, 
			'cin': 0.1, 
			'acdif': 0.0, 
			'accom': 0.0, 
			'acvdd': 1.0, 
			'acvss': 0.0
		},
		'saves': [ "all()" ], 
		'command': "ac(1, 1e9, 'dec', 10)"
	}, 	
	'acvss': {
		'head' : 'opus', 
		'modules': [ 'def', 'tbrr' ], 
		'options': {
			'method': 'gear'
		}, 
		'params': {
			'rfb': 1e9, 
			'cin': 0.1, 
			'acdif': 0.0, 
			'accom': 0.0, 
			'acvdd': 0.0, 
			'acvss': 1.0
		},
		'saves': [ "all()" ], 
		'command': "ac(1, 1e9, 'dec', 10)"
	}, 	
	'tran': {
		'head' : 'opus', 
		'modules': [ 'def', 'tb' ], 
		'options': {
			'reltol': 1e-4
		},
		'params': {
			'rin': 1e6,
			'rfb': 1e6
		},
		'saves': [ "all()" ], 
		'command': "tran(param['tr']*1, param['tstart']+param['pw']*2)"
	}, 
	'translew': {
		'head' : 'opus', 
		'modules': [ 'def', 'tb' ], 
		'options': {
			'reltol': 1e-4
		},
		'params': {
			'rin': 1e6,
			'rfb': 1e6, 
			'lev1': -0.8, 
			'lev2': 0.8
		},
		'saves': [ "all()" ], 
		'command': "tran(param['tr']*1, param['tstart']+param['pw']*2)"
	}
}

# Define performance measures, dependencies, and design requirements (lower and upper bounds)
measures = {
	'isup': {
		'analysis': 'op', 
		'script': "__result=-i('vdd')", 
		'upper': 1000e-6, 
	}, 
	'out_op': {
		'analysis': 'op', 
		'expression': "v('out')", 
	},
	# Vgs overdrive (Vgs-Vth)
	'vgs_drv': {
		'analysis': 'op', 
		'expression': "array(map(m.Poverdrive(p, 'vgs', p, 'vth'), ipath(var['mosList'], 'x1', 'm0')))", 
		'vector': True, 
		'lower': 0.0, 
	}, 
	# Vds overdrive (Vds-Vdsat)
	'vds_drv': {
		'analysis': 'op', 
		'expression': "array(map(m.Poverdrive(p, 'vds', p, 'vdsat'), ipath(var['mosList'], 'x1', 'm0')))", 
		'vector': True, 
		'lower': 0.0, 
	}, 
	'swing': {
		'analysis': 'dc', 
		'expression': "m.DCswingAtGain(v('out'), v('inp', 'inn'), 0.5, 'out')", 
		'lower': 1.0, 
	},
	'gain': {
		'analysis': 'ac', 
		# 'expression': "m.ACgain(m.ACtf(v('out'), v('inp', 'inn')))", 
		'lower': 60.0, 
		'script': """
#var['plt'].semilogx(scale(), m.ACphase(m.ACtf(v('out'), v('inp', 'inn'))))
#var['plt'].show()
__result=m.ACmag(m.ACtf(v('out'), v('inp', 'inn')))[0]
"""
	},
	'gain_com': {
		'analysis': 'accom', 
		'expression': "m.ACmag(m.ACtf(v('out'), 1.0))[0]", 
	},
	'gain_vdd': {
		'analysis': 'acvdd', 
		'expression': "m.ACmag(m.ACtf(v('out'), 1.0))[0]", 
	},
	'gain_vss': {
		'analysis': 'acvss', 
		'expression': "m.ACmag(m.ACtf(v('out'), 1.0))[0]", 
	},
	'ugbw': {
		'analysis': 'ac', 
		'expression': "m.ACugbw(m.ACtf(v('out'), v('inp', 'inn')), scale())", 
		'lower': 10e6, 
	}, 
	'pm': {
		'analysis': 'ac', 
		'expression': "m.ACphaseMargin(m.ACtf(v('out'), v('inp', 'inn')))", 
		'lower': 50.0, 
	}, 
	'overshdn': {
		'analysis': 'tran', 
		'expression': "m.Tundershoot(v('out'), scale(), t1=param['tstart'], t2=(param['pw']+param['tstart']+param['tr']))", 
		'upper': 0.10, 
	},
	'overshup': {
		'analysis': 'tran', 
		#'expression': "m.Tovershoot(v('out'), scale(), t1=(param['pw']+param['tstart']+param['tr']))", 
		'upper': 0.10, 
		'script': """
__result=m.Tovershoot(v('out'), scale(), t1=(param['pw']+param['tstart']+param['tr']))
#var['plt'].plot(scale(), v('out'))
#var['plt'].show()
"""
		
	},
	'tsetdn': {
		'analysis': 'tran', 
		'expression': "m.TsettlingTime(v('out'), scale(), t1=param['tstart'], t2=(param['pw']+param['tstart']+param['tr']))", 
		'upper': 1000e-9,
	}, 
	'tsetup': {
		'analysis': 'tran', 
		'expression': "m.TsettlingTime(v('out'), scale(), t1=(param['pw']+param['tstart'])+param['tr'])", 
		'upper': 1000e-9,
	}, 
	'slewdn': {
		'analysis': 'translew', 
		'expression': "m.TslewRate('falling', v('out'), scale(), t1=param['tstart'], t2=(param['pw']+param['tstart']+param['tr']))", 
		'lower': 2e6,
	}, 
	'slewup': {
		'analysis': 'translew', 
		'expression': "m.TslewRate('rising', v('out'), scale(), t1=(param['pw']+param['tstart']+param['tr']))", 
		'lower': 2e6,
	}, 
	'cmrr': {
		'analysis': None, 
		'expression': "result['gain'][thisCorner]-result['gain_com'][thisCorner]", 
		'lower': 90.0, 
		'depends': [ 'gain', 'gain_com' ]
	}, 
	'psrr_vdd': {
		'analysis': None, 
		'expression': "result['gain'][thisCorner]-result['gain_vdd'][thisCorner]", 
		'lower': 60.0, 
		'depends': [ 'gain', 'gain_vdd' ]
	}, 
	'psrr_vss': {
		'analysis': None, 
		'expression': "result['gain'][thisCorner]-result['gain_vss'][thisCorner]", 
		'lower': 60.0, 
		'depends': [ 'gain', 'gain_vss' ]
	},
	'area': {
		'analysis': None, 
		'expression': (
			"(param['mirr_w']+param['mirr_wo'])*param['mirr_l']"
			"+param['mirr_wd']*param['mirr_ld']"
			"+param['out_w']*param['out_l']*(2)"
			"+param['load_w']*param['load_l']*(1+1)"
			"+param['dif_w']*param['dif_l']*(1+1)"
			"+param['r_out']/1e3*12e-12"
			"+param['c_out']/1e-12*100e-12"
		), 
		'upper': 9000e-12
	}
}

# Design parameters, lower bounds, upper bounds, and initial values
designParams={
	'mirr_w': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 7.46e-005,
	}, 
	#'mirr_wr': {
	#	'lo':	1e-6, 
	#	'hi':	95e-6, 
	#	'init': 7.46e-005,
	#}, 
	'mirr_wd': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 7.46e-005,
	}, 
	'mirr_wo': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 7.46e-005,
	}, 
	'mirr_l': {
		'lo':	0.18e-6, 
		'hi':	4e-6, 
		'init': 5.63e-007,
	}, 
	'mirr_ld': {
		'lo':	0.18e-6, 
		'hi':	4e-6, 
		'init': 5.63e-007,
	}, 
	'out_w': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 4.800592541419e-005,
	},
	'out_l': {
		'lo':	0.18e-6, 
		'hi':	4e-6, 
		'init': 3.750131780858e-007,
	},
	'load_w': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 3.486243671853e-005,
	},
	'load_l': {
		'lo':	0.18e-6, 
		'hi':	4e-6, 
		'init': 2.572996921261e-006,
	},
	'dif_w': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 7.728734451428e-006,
	},
	'dif_l': {
		'lo':	0.18e-6, 
		'hi':	4e-6, 
		'init': 1.082371380389e-006,
	},
	'c_out': {
		'lo':	1e-15, 
		'hi':	50e-12, 
		'init': 8.211596855053e-012,
	},
	'r_out': {
		'lo':	1, 
		'hi':	200e3, 
		'init': 1.968986740568e+001,
	}
}

# Statistical parameters, lower and upper bounds
statParams={}
for name in [ 
	'p1vt', 'p1u0', 'p2vt',  'p2u0', 
	'p3vt', 'p3u0', 'n1bvt', 'n1bu0', 
	'n1vt', 'n1u0', 'n2vt',  'n2u0',
	'n3vt', 'n3u0', 'n4vt',  'n4u0'
]: 
	statParams[name]={
		'lo': -10.0, 
		'hi': 10.0
	}

# Operating parameters definitions, lower bounds, upper bounds, and nominal values
opParams={
	'vdd': {
		'lo': 1.7, 
		'hi': 2.0, 
		'init': 1.8
	}, 
	'temperature': {
		'lo': 0.0, 
		'hi': 100.0, 
		'init': 25
	}
}
