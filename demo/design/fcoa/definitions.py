# Test performance evaluator

__all__ = [ 'heads', 'analyses', 'measures', 'variables', 'statParams', 'opParams', 'designParams' ]


heads = {
	'opus': {
		'simulator': 'SpiceOpus', 
		'settings': {
			'debug': 0
		}, 
		'moddefs': {
			'def':  { 'file': 'fcoa.inc' }, 
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
			'method': 'trap', 
			'gminsteps': 200, 
			'sollim': 1e-3, 
			'noopiter': True, 
		}, 
		'params': {
			'ibias': 5e-6, 
			'lev1': -0.5,
			'lev2': 0.5,
			'tstart': 2000e-9, 
			'tr': 1e-9, 
			'tf': 1e-9, 
			'pw': 2000e-9, 
			'rload': 100e6, 
			'cload': 2e-12, 
		}
	}
}

variables={
	'mosList': [ 
		'xp24', 'xp23', 'xp20', 'xp22', 
		'xn15', 'xn9', 
		'xn13', 'xn12', 
		'xp16', 'xp18', 'xp15', 'xp17', 
		'xn11', 'xn10', 'xn14', 'xn8'
	], 
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
		'command': "dc(-2.0, 2.0, 'lin', 400, 'vin', 'dc')"
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
		'command': "ac(1, 1e9, 'dec', 100)"
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
		'command': "ac(1, 1e9, 'dec', 100)"
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
		'command': "ac(1, 1e9, 'dec', 100)"
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
		'command': "ac(1, 1e9, 'dec', 100)"
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
	}, 
}

# Define performance measures, dependencies, and design requirements (lower and upper bounds)
measures = {
	'isup': {
		'analysis': 'op', 
		'script': "__result=-i('vdd')", 
		'upper': 400e-6, 
	}, 
	'out_offs': {
		'analysis': 'op', 
		'expression': "v('out','inp')", 
		#'lower': -20e-3, 
		#'upper': 20e-3, 
	},
	'in_offs': {
		'analysis': 'dc', 
		'expression': "m.XatI(v('inp', 'inn'), m.IatXval(v('out', 'inp'), 0.0)[0])", 
		'lower': -20e-3, 
		'upper': 20e-3, 
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
		'lower': 0.3, 
	},
	'gain': {
		'analysis': 'ac', 
		'expression': "m.ACmag(m.ACtf(v('out'), v('inp', 'inn')))[0]", 
		'lower': 70.0, 
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
		'lower': 4e6, 
	}, 
	'pm': {
		'analysis': 'ac', 
		'expression': "m.ACphaseMargin(m.ACtf(v('out'), v('inp', 'inn')))", 
		'lower': 60.0, 
	}, 
	'ovrshdn': {
		'analysis': 'tran', 
		'expression': "m.Tundershoot(v('out'), scale(), t1=param['tstart'], t2=(param['pw']+param['tstart']+param['tr']))",
		'upper': 0.10, 
	},
	'ovrshup': {
		'analysis': 'tran', 
		'expression': "m.Tovershoot(v('out'), scale(), t1=(param['pw']+param['tstart']+param['tr']))",
		'upper': 0.10, 
	},
	'tsetdn': {
		'analysis': 'tran', 
		'expression': "m.TsettlingTime(v('out'), scale(), t1=param['tstart'], t2=(param['pw']+param['tstart']+param['tr']))",
		'upper': 2500e-9,
	}, 
	'tsetup': {
		'analysis': 'tran', 
		'expression': "m.TsettlingTime(v('out'), scale(), t1=(param['pw']+param['tstart'])+param['tr'])",
		'upper': 2500e-9,
	}, 
	'slewdn': {
		'analysis': 'tran',
		'expression': "m.TslewRate('falling', v('out'), scale(), t1=param['tstart'], t2=(param['pw']+param['tstart']+param['tr']))",
		'lower': 2e6,
	}, 
	'slewup': {
		'analysis': 'tran', 
		'expression': "m.TslewRate('rising', v('out'), scale(), t1=(param['pw']+param['tstart']+param['tr']))",
		'lower': 2e6,
	}, 
	'cmrr': {
		'analysis': None, 
		'expression': "result['gain'][thisCorner]-result['gain_com'][thisCorner]", 
		'lower': 100.0, # 90
		'depends': [ 'gain', 'gain_com' ]
	}, 
	'psrr_vdd': {
		'analysis': None, 
		'expression': "result['gain'][thisCorner]-result['gain_vdd'][thisCorner]", 
		'lower': 70.0, 
		'depends': [ 'gain', 'gain_vdd' ]
	}, 
	'psrr_vss': {
		'analysis': None, 
		'expression': "result['gain'][thisCorner]-result['gain_vss'][thisCorner]", 
		'lower': 70.0, 
		'depends': [ 'gain', 'gain_vss' ]
	},
	'area': {
		'analysis': None, 
		'expression': (
			"(param['pm_w0']+param['pm_w1']+2*param['pm_w2']+4*param['pm_w3']*4)*param['pm_l']"
			"+param['nm_w']*param['nm_l']*2"
			"+param['dif_w']*param['dif_l']*4"
			"+param['nl_w']*param['nl_l']"
		), 
		'upper': 2000e-12 # 5000e-12
	}
}

# Design parameters, lower bounds, upper bounds, and initial values
designParams={
	'pm_w0': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 7.5e-6,
	}, 
	'pm_w1': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 7.5e-6,
	}, 
	'pm_w2': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 28e-6,
	}, 
	'pm_w3': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 28e-6,
	}, 
	'pm_l': {
		'lo':	0.18e-6, 
		'hi':	4e-6, 
		'init': 1e-6,
	}, 
	'nm_w': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 16e-6,
	}, 
	'nm_l': {
		'lo':	0.18e-6, 
		'hi':	4e-6, 
		'init': 1e-6,
	}, 
	'dif_w': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 12e-6,
	}, 
	'dif_l': {
		'lo':	0.18e-6, 
		'hi':	4e-6, 
		'init': 1e-6,
	}, 
	'nl_l': {
		'lo':	0.18e-6, 
		'hi':	4e-6, 
		'init': 1e-6,
	}, 
	'nl_w': {
		'lo':	1e-6, 
		'hi':	95e-6, 
		'init': 16e-6,
	}, 
}

# Statistical parameters, lower and upper bounds
statParams={}
for name in [ 
	'vt_p24', 'vt_p23', 'vt_p22',  'vt_p20', 
	'vt_n15', 'vt_n9', 
	'vt_n13', 'vt_n12', 
	'vt_p15', 'vt_p16', 'vt_p17',  'vt_p18', 
	'vt_n10', 'vt_n11', 'vt_n14',  'vt_n8',
	'u0_p24', 'u0_p23', 'u0_p22',  'u0_p20', 
	'u0_n15', 'u0_n9', 
	'u0_n13', 'u0_n12', 
	'u0_p15', 'u0_p16', 'u0_p17',  'u0_p18', 
	'u0_n10', 'u0_n11', 'u0_n14',  'u0_n8', 
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
