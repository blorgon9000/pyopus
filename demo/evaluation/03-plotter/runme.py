# Test plotter

from pyopus.evaluator.performance import PerformanceEvaluator
from pyopus.visual.wxmplplotter import WxMplPlotter

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
		},
	}

	# Order in which mesures are printed. 
	# Do not print visualization measures
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
						'xlimits': (-3e-3, -1e-3), 
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

	pe=PerformanceEvaluator(heads, analyses, measures, corners, variables=variables, debug=0)

	plotter=WxMplPlotter(visualisation, pe, debug=1)

	results=pe(inParams)

	plotter()

	pe.finalize()
	
