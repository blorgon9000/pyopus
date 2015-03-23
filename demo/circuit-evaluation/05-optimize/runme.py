# Optimize with cost evaluator 

from pyopus.evaluator.performance import PerformanceEvaluator
from pyopus.evaluator.cost import CostEvaluator, parameterSetup
from pyopus.optimizer import optimizerClass

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

	costInput = { 
		'mirr_w': {
			'init': 7.456e-005, 
			'lo':	1e-6, 
			'hi':	95e-6, 
			'step':	0.01e-6
		},
		'mirr_l': {
			'init': 5.6e-007, 
			'lo':	0.18e-6, 
			'hi':	4e-6, 
			'step':	0.01e-6
		},
		'out_w': {
			'init': 4.801e-005, 
			'lo':	1e-6, 
			'hi':	95e-6, 
			'step':	0.01e-6
		},
		'out_l': {
			'init': 3.8e-007, 
			'lo':	0.18e-6, 
			'hi':	4e-6, 
			'step':	0.01e-6
		},
		'load_w': {
			'init': 3.486e-005, 
			'lo':	1e-6, 
			'hi':	95e-6, 
			'step':	0.01e-6
		},
		'load_l': {
			'init': 2.57e-006, 
			'lo':	0.18e-6, 
			'hi':	4e-6, 
			'step':	0.01e-6
		},
		'dif_w': {
			'init': 7.73e-006, 
			'lo':	1e-6, 
			'hi':	95e-6, 
			'step':	0.01e-6
		},
		'dif_l': {
			'init': 1.08e-006, 
			'lo':	0.18e-6, 
			'hi':	4e-6, 
			'step':	0.01e-6
		},
	}

	# Order in which input parameters are printed.
	inOrder=[ 
		'mirr_w', 'mirr_l', 
		'out_w', 'out_l', 
		'load_w', 'load_l', 
		'dif_w', 'dif_l'
	]

	costDefinition = [
		{
			'measure': 'isup', 							# Measurement name
			'goal': 'MNbelow(1e-3, 0.1e-3, 10000.0)',	# Default norm is 1/10 of goal or 1 if goal is 0
														# Default failure penalization is 10000.0
			'shape': 'CSlinear2(1.0,0.0)',				# This is the default shape (linear2 with w=1 and tw=0)
			'reduce': 'CCexcluded()'					# Exclude from cost. 
		},
		{
			'measure': 'out_op',
			'goal': 'MNbelow(10, 0.1e-3)',	
			'shape': 'CSlinear2(1.0,0.0)', 
			'reduce': 'CCexcluded()'
		},
		{
			'measure': 'vgs_drv', 
			'goal': 'MNabove(1e-3)', 					
			'reduce': 'CCworst()'	  					# This is the default corner reduction
		},
		{
			'measure': 'vds_drv', 
			'goal': 'MNabove(1e-3)'
		},
		{
			'measure': 'swing', 
			'goal': 'MNabove(1.5)', 
			'shape': 'CSlinear2(1.0,0.001)', 
		},
		{
			'measure': 'mirr_area', 
			'goal': 'MNbelow(800e-12, 100e-12)', 
			'shape': 'CSlinear2(1.0,0.001)'
		}
	]

	# Performance and cost evaluators
	pe=PerformanceEvaluator(heads, analyses, corners, measures, debug=0)

	# Input parameter order in vector x is defined by inOrder. 
	ce=CostEvaluator(pe, inOrder, costDefinition, debug=0)

	# Initial, low, high, and step value of input parameters. 
	(xinit, xlow, xhigh, xstep)=parameterSetup(inOrder, costInput)

	# Optimizer (Hooke-Jeeves). xlo and xhi must be numpy arrays. 
	opt=optimizerClass("HookeJeeves")(ce, xlo=xlow, xhi=xhigh, maxiter=1000)

	# Set initial point. Must be a numpy array; xinit is a python list. 
	opt.reset(xinit)

	# Install reporter plugin. 
	# Print cost. Print performance every time cost is decreased. 
	opt.installPlugin(ce.getReporter())

	# Install stopper plugin. 
	# Stop when all requirements are satisfied (all cost contributions are 0). 
	opt.installPlugin(ce.getStopWhenAllSatisfied())

	# Run
	opt.run()

	# Optimization result
	xresult=opt.x
	iterresult=opt.bestIter
		
	# Final evaluation at xresult. 
	cf=ce(xresult)
		
	# Print results. 
	print("\n\nFinal cost: "+str(cf)+", found in iter "+str(iterresult)+", total "+str(opt.niter)+" iteration(s)")
	print(ce.formatParameters())
	print(ce.formatResults(nMeasureName=10, nCornerName=15))
	print("Performance in corners")
	print(pe.formatResults(outOrder, nMeasureName=10, nCornerName=15))

	pe.finalize()
