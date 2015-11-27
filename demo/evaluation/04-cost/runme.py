# Test cost evaluator

from pyopus.evaluator.performance import PerformanceEvaluator
from pyopus.evaluator.aggregate import *
from pyopus.evaluator.auxfunc import paramList

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
			
	variables={
		'saveInst': [ 'xmn2', 'xmn3', 'xmn1' ], 
		'saveProp': [ 'vgs', 'vth', 'vds', 'vdsat' ]
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
				"p(ipath(var['saveInst'], 'x1', 'm0'), var['saveProp'])"
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
			'expression': "array(map(m.Poverdrive(p, 'vgs', p, 'vth'), ipath(var['saveInst'], 'x1', 'm0')))", 
			'vector': True
		}, 
		# Vds overdrive (Vds-Vdsat) for mn2, mn3, and mn1. 
		'vds_drv': {
			'analysis': 'op', 
			'corners': [ 'nominal', 'worst_power', 'worst_speed', 'worst_one', 'worst_zero' ], 
			'expression': "array(map(m.Poverdrive(p, 'vds', p, 'vdsat'), ipath(var['saveInst'], 'x1', 'm0')))", 
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

	params = { 
		'mirr_w': 7.456e-005, 
		'mirr_l': 5.6e-007, 
		'out_w':  4.801e-005, 
		'out_l':  3.8e-007, 
		'load_w': 3.486e-005, 
		'load_l': 2.57e-006, 
		'dif_w':  7.73e-006, 
		'dif_l':  1.08e-006, 
	}

	# Order in which input parameters are printed.
	inOrder=params.keys()
	inOrder.sort()

	definition = [
		{
			'measure': 'isup', 			# Measurement name
			'norm': Nbelow(1e-3, 0.1e-3, 10000.0),	# Default norm is 1/10 of goal or 1 if goal is 0
								# Default failure penalization is 10000.0
			'shape': Slinear2(1.0,0.0),		# This is the default shape (linear2 with w=1 and tw=0)
			'reduce': Rexcluded()
		},
		{
			'measure': 'out_op',
			'norm': Nbelow(10, 0.1e-3),	
			'shape': Slinear2(1.0,0.0), 
			'reduce': Rexcluded()
		},
		{
			'measure': 'vgs_drv', 
			'norm': Nabove(1e-3), 			
			'reduce': Rworst()	  		# This is the default corner reduction
		},
		{
			'measure': 'vds_drv', 
			'norm': Nabove(1e-3)
		},
		{
			'measure': 'swing', 
			'norm': Nabove(1.6), 
			'shape': Slinear2(1.0,0.001), 
		},
		{
			'measure': 'mirr_area', 
			'norm': Nbelow(800e-12, 100e-12), 
			'shape': Slinear2(1.0,0.001)
		}
	]

	# Performance evaluator
	pe=PerformanceEvaluator(heads, analyses, measures, corners, variables=variables, debug=0)

	# Aggregate function
	ce=Aggregator(pe, definition, inOrder, debug=1)

	# Vectorize parameters
	x=paramList(params, inOrder)
	
	# Evaluate aggregate function at x 
	cf=ce(x)

	print("")
	print("cost=%e" % cf)
	print(ce.formatParameters())
	print(ce.formatResults(nMeasureName=10, nCornerName=15))

	pe.finalize()
