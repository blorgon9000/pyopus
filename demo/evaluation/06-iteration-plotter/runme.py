# Optimize with cost evaluator and iteration plotter

from pyopus.evaluator.performance import PerformanceEvaluator
from pyopus.evaluator.aggregate import *
from pyopus.evaluator.auxfunc import listParamDesc
from pyopus.optimizer import optimizerClass
from pyopus.visual.wxmplplotter import WxMplPlotter
from pyopus.visual.plotter import IterationPlotter

# Need to protect this from being evaluated at import. 
# If not, we get an infinite loop when multiprocessing forks the gui thread. 
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
		}, 
		'dccom': {
			'head': 'opus', 
			'modules': [ 'def', 'tb' ], 
			'options': {
				'method': 'gear'
			},
			'params': {
				'rin': 1e6,
				'rfb': 1e6
			},			
			'saves': [ 
				"v(['inp', 'inn', 'out'])", 
				"p(ipath(var['saveInst'], 'x1', 'm0'), var['saveProp'])"
			], 
			'command': "dc(0, param['vdd'], 'lin', 20, 'vcom', 'dc')"
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
		}, 
		'dcvin': {
			'analysis': 'dc', 
			'corners': [ 'nominal', 'worst_power', 'worst_speed', 'worst_one', 'worst_zero' ], 
			'expression': "v('inp', 'inn')",
			'vector': True
		},
		'dcvout': {
			'analysis': 'dc', 
			'corners': [ 'nominal', 'worst_power', 'worst_speed', 'worst_one', 'worst_zero' ], 
			'expression': "v('out')",
			'vector': True
		},
		'dccomvin': {
			'analysis': 'dccom', 
			'corners': [ 'nominal', 'worst_power', 'worst_speed', 'worst_one', 'worst_zero' ], 
			'expression': "v('inp')",
			'vector': True
		},
		'dccomvout': {
			'analysis': 'dccom', 
			'corners': [ 'nominal', 'worst_power', 'worst_speed', 'worst_one', 'worst_zero' ], 
			'expression': "v('out')",
			'vector': True
		},
		'dccom_m1vdsvdsat': {
			'analysis': 'dccom', 
			'corners': [ 'nominal', 'worst_power', 'worst_speed', 'worst_one', 'worst_zero' ], 
			'expression': "p(ipath('xmn1', 'x1', 'm0'), 'vds')-p(ipath('xmn1', 'x1', 'm0'), 'vdsat')",
			'vector': True
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
		},
		'mirr_l': {
			'init': 5.6e-007, 
			'lo':	0.18e-6, 
			'hi':	4e-6, 
		},
		'out_w': {
			'init': 4.801e-005, 
			'lo':	1e-6, 
			'hi':	95e-6, 
		},
		'out_l': {
			'init': 3.8e-007, 
			'lo':	0.18e-6, 
			'hi':	4e-6, 
		},
		'load_w': {
			'init': 3.486e-005, 
			'lo':	1e-6, 
			'hi':	95e-6, 
		},
		'load_l': {
			'init': 2.57e-006, 
			'lo':	0.18e-6, 
			'hi':	4e-6, 
		},
		'dif_w': {
			'init': 7.73e-006, 
			'lo':	1e-6, 
			'hi':	95e-6, 
		},
		'dif_l': {
			'init': 1.08e-006, 
			'lo':	0.18e-6, 
			'hi':	4e-6, 
		},
	}

	# Order in which input parameters are printed.
	inOrder=costInput.keys()
	inOrder.sort()
	
	costDefinition = [
		{
			'measure': 'isup', 				# Measurement name
			'norm': Nbelow(1e-3, 0.1e-3, 10000.0),		# Default norm is 1/10 of goal or 1 if goal is 0
									# Default failure penalization is 10000.0
			'shape': Slinear2(1.0,0.0),			# This is the default shape (linear2 with w=1 and tw=0)
			'reduce': Rexcluded()				# Exclude from cost. 
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
			'reduce': Rworst()				# This is the default corner reduction
		},
		{
			'measure': 'vds_drv', 
			'norm': Nabove(1e-3)
		},
		{
			'measure': 'swing', 
			'norm': Nabove(1.5), 
			'shape': Slinear2(1.0,0.001), 
		},
		{
			'measure': 'mirr_area', 
			'norm': Nbelow(800e-12, 100e-12), 
			'shape': Slinear2(1.0,0.001)
		}
	]

	visualisation = {
			# Window list with axes
			'graphs': {
				'dc': {
					'title': 'Amplifier DC response', 
					'shape': { 'figsize': (6,8), 'dpi': 80 }, 
					'axes': {
						'diff': {
							'subplot': (2,1,1), 
							'options': {}, 
							'gridtype': 'rect', 		# rect (default) or polar, xscale and yscale have no meaning when grid is polar
							'xscale': { 'type': 'linear' }, 	# linear by default
							'yscale': { 'type': 'linear' }, 	# linear by default
							'xlimits': (-50e-3, 50e-3), 
							'xlabel': 'Vdif=Vinp-Vinn [V]', 
							'ylabel': 'Vout [V]', 
							'title': '', 
							'legend': False, 
							'grid': True, 
						}, 
						'com': {
							'subplot': (2,1,2), 
							'options': {}, 
							'gridtype': 'rect', 		# rect (default) or polar, xscale and yscale have no meaning when grid is polar
							'xscale': { 'type': 'linear' }, 	# linear by default
							'yscale': { 'type': 'linear' }, 	# linear by default
							'xlimits': (0.0, 2.0),
							'xlabel': 'Vcom=Vinp=Vinn [V]', 
							'ylabel': 'Vout [V]', 
							'title': '', 
							'legend': False, 
							'grid': True, 
						}
					}
				},
				'm1vds': {
					'title': 'M1 Vds-Vdsat in common mode', 
					'shape': { 'figsize': (6,4), 'dpi': 80 }, 
					'axes': {
						'dc': {
							'rectangle': (0.12, 0.12, 0.76, 0.76), 
							'options': {}, 
							'gridtype': 'rect', 		# rect (default) or polar, xscale and yscale have no meaning when grid is polar
							'xscale': { 'type': 'linear' }, 	# linear by default
							'yscale': { 'type': 'linear' }, 	# linear by default
							'xlimits': (0.0, 2.0), 
							'xlabel': 'Vcom=Vinp=Vinn [V]', 
							'ylabel': 'M1 Vds-Vdsat [V]', 
							'title': '', 
							'legend': False, 
							'grid': True, 
						}, 
					}
				}
			}, 
			# (graph, axes, corner, trace) styles
			'styles': [ 
				{
					'pattern': ('^.*', '^.*', '^.*', '^.*'), 
					'style': {
						'linestyle': '-',
						'color': (0.5,0,0)
					}
				}, 
				{
					'pattern': ('^.*', '^.*', '^nom.*', '^.*'),
					'style': {
						'linestyle': '-',
						'color': (0,0.5,0)
					}
				}
			], 
			# Trace list
			'traces': {
				'dc': {
					'graph': 'dc', 
					'axes': 'diff', 
					'xresult': 'dcvin',
					'yresult': 	'dcvout', 
					'corners': [ ],	# If not specified or empty, all corners where xresult is evaluated are plotted
					'style': {	# Style is defined by style patterns that match the (graph, axis, corner, trace) tuple
						'linestyle': '-',
						'marker': '.', 
					}
				}, 
				'dccom': {
					'graph': 'dc', 
					'axes': 'com', 
					'xresult': 'dccomvin',
					'yresult': 	'dccomvout', 
					'corners': [ ],	
					'style': {	
						'linestyle': '-',
						'marker': '.', 
					}
				},
				'dc1': {
					'graph': 'dccomb', 
					'axes': 'dc', 
					'xresult': 'dcvin',
					'yresult': 	'dcvout', 
					'corners': [ 'nominal' ],	# If not specified or empty, all corners where xresult is evaluated are plotted
					'style': {	# Style is defined by style patterns that match the (graph, axis, corner, trace) tuple
						'linestyle': '-',
						'marker': '.', 
					}
				}, 
				'dccom1': {
					'graph': 'dccomb', 
					'axes': 'dc', 
					'xresult': 'dccomvin',
					'yresult': 	'dccomvout', 
					'corners': [ 'nominal' ],	
					'style': {	
						'linestyle': '-',
						'marker': '.', 
					}
				},
				'm1_vds_vdsat': {
					'graph': 'm1vds', 
					'axes': 'dc', 
					'xresult': 'dccomvin',
					'yresult': 	'dccom_m1vdsvdsat', 
					'corners': [ 'nominal' ],	
					'style': {	
						'linestyle': '-',
						'marker': '.', 
					}
				}
			}
		}

	# Performance and cost evaluators
	pe=PerformanceEvaluator(heads, analyses, measures, corners, variables=variables, debug=0)

	# Plotter
	plotter=WxMplPlotter(visualisation, pe)

	# Input parameter order in vector x is defined by inOrder. 
	ce=Aggregator(pe, costDefinition, inOrder, debug=0)

	# Initial, low, high, and step value of input parameters. 
	# Initial, low, and high value of input parameters. 
	xlow=listParamDesc(costInput, inOrder, "lo")
	xhi=listParamDesc(costInput, inOrder, "hi")
	xinit=listParamDesc(costInput, inOrder, "init")
	
	# Optimizer (Hooke-Jeeves). xlo and xhi must be numpy arrays. 
	opt=optimizerClass("HookeJeeves")(ce, xlo=xlow, xhi=xhi, maxiter=1+1*1000)

	# Set initial point. Must be a numpy array; xinit is a Python list. 
	opt.reset(xinit)

	# Install reporter plugin. 
	# Print cost. Print performance every time cost is decreased. 
	opt.installPlugin(ce.getReporter())

	# Install stopper plugin. 
	# Stop when all requirements are satisfied (all cost contributions are 0). 
	opt.installPlugin(ce.getStopWhenAllSatisfied())

	# Create and install iteration plotter plugin. 
	opt.installPlugin(IterationPlotter(plotter))

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
