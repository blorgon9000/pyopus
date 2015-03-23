# Test performance evaluator

from pyopus.evaluator.performance import PerformanceEvaluator
from pyopus.evaluator.cost import formatParameters

if __name__=='__main__':
	heads = {
		'opus': {
			'simulator': 'SpiceOpus', 
			'settings': {
				'debug': 0
			}, 
			'moddefs': {
				'def':     { 'file': 'opamp.inc' }, 
				'tb':      { 'file': 'topdc.inc' }, 
				'mos_tm': { 'file': 'cmos180n.lib', 'section': 'tm' }, 
				'mos_wp': { 'file': 'cmos180n.lib', 'section': 'wp' }, 
				'mos_ws': { 'file': 'cmos180n.lib', 'section': 'ws' }, 
				'mos_wo': { 'file': 'cmos180n.lib', 'section': 'wo' }, 
				'mos_wz': { 'file': 'cmos180n.lib', 'section': 'wz' }
			}, 
			'options': {
				'method': 'trap'
			}, 
			'params': {
				'lev1': 0.0,
				'lev2': 0.5,
				'tstart': 1e-9, 
				'tr': 1e-9, 
				'tf': 1e-9, 
				'pw': 500e-9
			}
		}
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
				"i(['vdd'])", 
				"v(['inp', 'inn', 'out'])", 
				"p(ipath(['xmn2', 'xmn3', 'xmn1'], 'x1', 'm0'), ['vgs', 'vth', 'vds', 'vdsat'])"
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
		}
	}

	corners = {
		'nominal': {
			'modules': [ 'mos_tm' ], 
			'params': {
				'temperature': 25, 
				'vdd': 1.8, 
			}
		},
		'worst_power': {
			'modules': [ 'mos_wp' ], 
			'params': {
				'temperature': 100, 
				'vdd': 2.0, 
			}
		},
		'worst_speed': {
			'modules': [ 'mos_ws' ], 
			'params': {
				'temperature': 100, 
				'vdd': 1.8, 
			}
		}, 
		'worst_zero': {
			'modules': [ 'mos_wz' ], 
			'params': {
				'temperature': 100, 
				'vdd': 1.8, 
			}
		}, 
		'worst_one': {
			'modules': [ 'mos_wo' ], 
			'params': {
				'temperature': 100, 
				'vdd': 1.8, 
			}
		}
	}

	measures = {
		'isup': {
			'analysis': 'op', 
			'corners': [ 'nominal', 'worst_power', 'worst_speed' ], 
			'script': "__result=-i('vdd')"
		}, 
		'out_op': {
			'analysis': 'op', 
			'corners': [ 'nominal', 'worst_power', 'worst_speed' ], 
			'expression': "v('out')"
		},
		# Vgs overdrive (Vgs-Vth) for mn2, mn3, and mn1. 
		'vgs_drv': {
			'analysis': 'op', 
			'corners': [ 'nominal', 'worst_power', 'worst_speed', 'worst_one', 'worst_zero' ], 
			'expression': "array(map(m.Poverdrive(p, 'vgs', p, 'vth'), ipath(['xmn2', 'xmn3', 'xmn1'], 'x1', 'm0')))", 
			'vector': True
		}, 
		# Vds overdrive (Vds-Vdsat) for mn2, mn3, and mn1. 
		'vds_drv': {
			'analysis': 'op', 
			'corners': [ 'nominal', 'worst_power', 'worst_speed', 'worst_one', 'worst_zero' ], 
			'expression': "array(map(m.Poverdrive(p, 'vds', p, 'vdsat'), ipath(['xmn2', 'xmn3', 'xmn1'], 'x1', 'm0')))", 
			'vector': True
		}, 
		'swing': {
			'analysis': 'dc', 
			'corners': [ 'nominal', 'worst_power', 'worst_speed', 'worst_one', 'worst_zero' ], 
			'expression': "m.DCswingAtGain(v('out'), v('inp', 'inn'), 0.5, 'out')"
		},
		'mirr_area': {
			'analysis': None, 
			'corners': [ 'nominal' ], 
			'expression': (
				"param['mirr_w']*param['mirr_l']*(2+2+16)"
			)
		}
	}

	# Order in which mesures are printed. 
	outOrder = [ 
		'mirr_area', 'isup', 'out_op', 
		'vgs_drv', 'vds_drv', 
		'swing', 
	]

	# Input parameters
	inParams={
		'mirr_w': 7.46e-005, 
		'mirr_l': 5.63e-007
	}

	# Order in which input parameters are printed.
	inOrder=[ 'mirr_w', 'mirr_l' ]

	pe=PerformanceEvaluator(heads, analyses, corners, measures, debug=2)

	results=pe(inParams)

	print("")
	print(formatParameters(inParams, inOrder))
	print(pe.formatResults(outOrder, nMeasureName=10, nCornerName=15))

	pe.finalize()
