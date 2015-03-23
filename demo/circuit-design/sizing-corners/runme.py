from pyopus.evaluator.performance import PerformanceEvaluator
from pyopus.evaluator.cost import CostEvaluator, parameterSetup
from pyopus.optimizer import optimizerClass
from pyopus.visual.wxmplplotter import WxMplPlotter
from pyopus.visual.plotter import IterationPlotter
from time import time
import sys

def main(simulator, method, nworkers, visualizeRun):
	# Common part of head
	common_head_sim = {
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
	
	# Simulator-dependent part
	# Spice Opus
	spiceopus_head={ 'sim': {} }
	spiceopus_head['sim'].update(common_head_sim)
	spiceopus_head['sim']['moddefs'] = {
		'def':    { 'file': 'so-opamp.inc' }, 
		'tb':     { 'file': 'so-topdc.inc' }, 
		'mos_tm': { 'file': 'so-cmos180n.lib', 'section': 'tm' }, 
		'mos_wp': { 'file': 'so-cmos180n.lib', 'section': 'wp' }, 
		'mos_ws': { 'file': 'so-cmos180n.lib', 'section': 'ws' }, 
		'mos_wo': { 'file': 'so-cmos180n.lib', 'section': 'wo' }, 
		'mos_wz': { 'file': 'so-cmos180n.lib', 'section': 'wz' }
	}
	spiceopus_head['sim']['simulator'] = 'SpiceOpus'
	spiceopus_head['sim']['settings'] = {
		'debug': 0
	}
	
	# HSPICE
	hspice_head={ 'sim': {} }
	hspice_head['sim'].update(common_head_sim)
	hspice_head['sim']['moddefs'] = {
		'def':    { 'file': 'hs-opamp.inc' }, 
		'tb':     { 'file': 'hs-topdc.inc' }, 
		'mos_tm': { 'file': 'hs-cmos180n.lib', 'section': 'tm' }, 
		'mos_wp': { 'file': 'hs-cmos180n.lib', 'section': 'wp' }, 
		'mos_ws': { 'file': 'hs-cmos180n.lib', 'section': 'ws' }, 
		'mos_wo': { 'file': 'hs-cmos180n.lib', 'section': 'wo' }, 
		'mos_wz': { 'file': 'hs-cmos180n.lib', 'section': 'wz' }
	}
	hspice_head['sim']['simulator'] = 'HSpice'
	hspice_head['sim']['settings'] = {
		'debug': 0
	}
	
	# Simulator-independent part
	analyses = {
		'op': {
			'head' : 'sim', 
			'modules': [ 'def', 'tb' ], 
			'options': {
				'method': 'gear'
			}, 
			'params': {
				'rin': 1e6,
				'rfb': 1e6
			},
			'saves': [ 
				"i(['vsrc'])", 
				"v(['out'])", 
				"p(ipath(['xmn2', 'xmn3', 'xmn1b', 'xmn1', 'xmn4', 'xmp1', 'xmp2', 'xmp3'], 'x1', 'm0'), ['vgs', 'vth', 'vds', 'vdsat'])"
				
			], 
			'command': "op()"
		}, 
		'dc': {
			'head' : 'sim', 
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
				"p(ipath(['xmn2', 'xmn3', 'xmn1b', 'xmn1', 'xmn4', 'xmp1', 'xmp2', 'xmp3'], 'x1', 'm0'), ['vgs', 'vth', 'vds', 'vdsat'])"
			], 
			'command': "dc(-2.0, 2.0, 'lin', 100, 'vin', 'dc')"
		},
		'dccom': {
			'head' : 'sim', 
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
				"p(ipath(['xmn2', 'xmn3', 'xmn1b', 'xmn1', 'xmn4', 'xmp1', 'xmp2', 'xmp3'], 'x1', 'm0'), ['vgs', 'vth', 'vds', 'vdsat'])"
			], 
			'command': "dc(0.7, param['vdd']-0.1, 'lin', 20, 'vcom', 'dc')"
		},
		'ac': {
			'head' : 'sim', 
			'modules': [ 'def', 'tb' ], 
			'options': {
				'method': 'gear'
			}, 
			'params': {
				'rin': 1e6,
				'rfb': 1e6
			},
			'saves': [ "all()" ], 
			'command': "ac(1, 1e9, 'dec', 10)"
		}, 
		'tran': {
			'head' : 'sim', 
			'modules': [ 'def', 'tb' ], 
			'options': {
				'method': 'gear'
			},
			'params': {
				'rin': 0.1e6,
				'rfb': 1e6
			},
			'saves': [ "all()" ], 
			'command': "tran(param['tr']*10, param['pw']*4)"
		}
	}

	corners = {
		'nom': {
			'modules': [ 'mos_tm' ], 
			'params': {
				'temperature': 25.0, 
				'vdd': 1.8, 
				'ibias': 100e-6, 
				'mn2vtn': 0.0,
				'mn2u0': 0.0,
				'mn3vtn': 0.0, 
				'mn3u0': 0.0
			}
		},
		'vddhi': {
			'modules': [ 'mos_wp' ], 
			'params': {
				'temperature': 100.0, 
				'vdd': 2.0, 
				'ibias': 100e-6, 
				'mn2vtn': 0.0,
				'mn2u0': 0.0,
				'mn3vtn': 0.0, 
				'mn3u0': 0.0
			}
		},
		'vddlo': {
			'modules': [ 'mos_ws' ], 
			'params': {
				'temperature': 0.0, 
				'vdd': 1.8, 
				'ibias': 100e-6, 
				'mn2vtn': 0.0,
				'mn2u0': 0.0,
				'mn3vtn': 0.0, 
				'mn3u0': 0.0
			}
		}, 
		'mm1': {
			'modules': [ 'mos_tm' ], 
			'params': {
				'temperature': 25.0, 
				'vdd': 1.8, 
				'ibias': 10e-6, 
				'mn2vtn': -3.0,
				'mn2u0': 3.0,
				'mn3vtn': 3.0, 
				'mn3u0': -3.0
			}
		}
	}

	measures = {
		'isup': {
			'analysis': 'op', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'script': "__result=-i('vsrc')"
		}, 
		'out_op': {
			'analysis': 'op', 
			'corners': [ 'nom', 'mm1' ], 
			'expression': "v('out')"
		},
		'vgs_drv': {
			'analysis': 'op', 
			'corners': [ 'nom' ], 
			'expression': "array(map(m.Poverdrive(p, 'vgs', p, 'vth'), ipath(['xmn2', 'xmn3', 'xmn1b', 'xmn1', 'xmn4', 'xmp1', 'xmp2', 'xmp3'], 'x1', 'm0')))", 
			'vector': True
		}, 
		'vds_drv': {
			'analysis': 'op', 
			'corners': [ 'nom' ], 
			'expression': "array(map(m.Poverdrive(p, 'vds', p, 'vdsat'), ipath(['xmn2', 'xmn3', 'xmn1b', 'xmn1', 'xmn4', 'xmp1', 'xmp2', 'xmp3'], 'x1', 'm0')))", 
			'vector': True
		}, 
		'vgs_drv_cm': {
			'analysis': 'dccom', 
			'corners': [ 'nom' ], 
			'expression': "array(map(m.Poverdrive(p, 'vgs', p, 'vth'), ipath(['xmn2', 'xmn3', 'xmn1b', 'xmn1', 'xmn4', 'xmp1', 'xmp2', 'xmp3'], 'x1', 'm0')))", 
			'vector': True
		},
		'vds_drv_cm': {
			'analysis': 'dccom', 
			'corners': [ 'nom' ], 
			'expression': "array(map(m.Poverdrive(p, 'vds', p, 'vdsat'), ipath(['xmn2', 'xmn3', 'xmn1b', 'xmn1', 'xmn4', 'xmp1', 'xmp2', 'xmp3'], 'x1', 'm0')))", 
			'vector': True
		},
		'swing': {
			'analysis': 'dc', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.DCswingAtGain(v('out'), v('inp', 'inn'), 0.5, 'out')"
		},
		'gain': {
			'analysis': 'ac', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.ACgain(m.ACtf(v('out'), v('inp', 'inn')))"
		},
		'ugbw': {
			'analysis': 'ac', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.ACugbw(m.ACtf(v('out'), v('inp', 'inn')), scale())"
		}, 
		'pm': {
			'analysis': 'ac', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.ACphaseMargin(m.ACtf(v('out'), v('inp', 'inn')))"
		}, 
		'undershoot': {
			'analysis': 'tran', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.Tundershoot(v('out'), scale(), t1=0.0, t2=(param['pw']+param['tstart']))"
		},
		'overshoot': {
			'analysis': 'tran', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.Tovershoot(v('out'), scale(), t1=(param['pw']+param['tstart']))"
		},
		'tf': {
			'analysis': 'tran', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.TfallTime(v('out'), scale(), t1=0.0, t2=(param['pw']+param['tstart']))"
		}, 
		'tr': {
			'analysis': 'tran', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.TriseTime(v('out'), scale(), t1=(param['pw']+param['tstart']))"
		}, 
		'tsetdn': {
			'analysis': 'tran', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.TsettlingTime(v('out'), scale(), t1=0.0, t2=(param['pw']+param['tstart']))"
		}, 
		'tsetup': {
			'analysis': 'tran', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.TsettlingTime(v('out'), scale(), t1=(param['pw']+param['tstart'])+param['tr'])"
		}, 
		'slewdn': {
			'analysis': 'tran', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.TslewRate('falling', v('out'), scale(), t1=0.0, t2=(param['pw']+param['tstart']))"
		}, 
		'slewup': {
			'analysis': 'tran', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.TslewRate('rising', v('out'), scale(), t1=(param['pw']+param['tstart']))"
		}, 
		'outoffs': {
			'analysis': None, 
			'corners': [ 'mm1' ], 
			'depends': [ 'out_op' ], 
			'expression': "abs(result['out_op'][thisCorner]-result['out_op']['nom'])"
		}, 
		'inoffs': {
			'analysis': None, 
			'corners': [ 'mm1' ], 
			'depends': [ 'out_op', ('gain', [ 'nom' ]) ], 
			'expression': "abs(result['out_op'][thisCorner]-result['out_op']['nom'])/m.dB2gain(result['gain']['nom'])"
		}, 
		'area': {
			'analysis': None, 
			'corners': [ 'nom' ], 
			'expression': (
				"param['dif_w']*param['dif_l']*(1+1)"+
				"+param['load_w']*param['load_l']*(1+1)"+
				"+param['out_w']*param['out_l']*16"+
				"+param['mirr_w']*param['mirr_l']*(2+2+16)"
			)
		}
	}

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
		'c_out': {
			'init': 8.21e-012, 
			'lo':	0.01e-12, 
			'hi':	10e-12, 
			'step':	0.01e-12
		},
		'r_out': {
			'init': 19.7, 
			'lo':	1.0, 
			'hi':	200e3, 
			'step':	0.1
		}
	}

	costDefinition = [
		{
			'measure': 'isup', 							# Measurement name
			'goal': 'MNbelow(1e-3, 0.1e-3, 10000.0)',	# Default norm is 1/10 of goal or 1 if goal is 0
														# Default failure penalization is 10000.0
			'shape': 'CSlinear2(1.0,0.0)',				# This is the default shape (linear2 with w=1 and tw=0)
			'reduce': 'CCexcluded()'					# Do not include in cost, observe only. 
		},
		{
			'measure': 'out_op',
			'goal': 'MNbelow(10, 0.1e-3)',	
			'shape': 'CSlinear2(1.0,0.0)', 
			'reduce': 'CCexcluded()'					# Do not inslude in cost, observe only. 
		},
		{
			'measure': 'vgs_drv', 
			'goal': 'MNabove(1e-3)', 					
			'reduce': 'CCworst()'	  					# This is the default corner reduction - use worst. 
		},
		{
			'measure': 'vds_drv', 
			'goal': 'MNabove(1e-3)'
		},
		{
			'measure': 'vgs_drv_cm', 
			'goal': 'MNabove(1e-3)', 					
		},
		{
			'measure': 'vds_drv_cm', 
			'goal': 'MNabove(1e-3)', 					
		},
		{
			'measure': 'swing', 
			'goal': 'MNabove(1.2)', 
			'shape': 'CSlinear2(1.0,0.001)', 
		},
		{
			'measure': 'gain', 
			'goal': 'MNabove(60.0)',  
			'shape': 'CSlinear2(1.0,0.001)',
		},
		{
			'measure': 'ugbw', 
			'goal': 'MNabove(20e6)', 
			'shape': 'CSlinear2(1.0,0.001)',
		},
		{
			'measure': 'pm', 
			'goal': 'MNabove(60.0)', 
			'shape': 'CSlinear2(1.0,0.001)',
		},
		{
			'measure': 'overshoot', 
			'goal': 'MNbelow(0.1)', 
			'shape': 'CSlinear2(1.0,0.001)',
		},
		{
			'measure': 'undershoot', 
			'goal': 'MNbelow(0.1)',  
			'shape': 'CSlinear2(1.0,0.001)',
		}, 
		{
			'measure': 'tr', 
			'goal': 'MNbelow(100e-9)', 
			'shape': 'CSlinear2(1.0,0.001)',
		},
		{
			'measure': 'tf', 
			'goal': 'MNbelow(100e-9)', 
			'shape': 'CSlinear2(1.0,0.001)',
		}, 
		{
			'measure': 'tsetdn', 
			'goal': 'MNbelow(300e-9)', 
			'shape': 'CSlinear2(1.0,0.001)',
		}, 
		{
			'measure': 'tsetup', 
			'goal': 'MNbelow(300e-9)', 
			'shape': 'CSlinear2(1.0,0.001)',
		}, 
		{
			'measure': 'slewdn', 
			'goal': 'MNabove(10e6)', 
			'shape': 'CSlinear2(1.0,0.001)',
		}, 
		{
			'measure': 'slewup', 
			'goal': 'MNabove(10e6)', 
			'shape': 'CSlinear2(1.0,0.001)',
		}, 
		{
			'measure': 'outoffs', 
			'goal': 'MNbelow(5e-3)', 
			'shape': 'CSlinear2(1.0,0.001)',
			'reduce': 'CCexcluded()'					# Do not include in cost. 
		}, 
		{
			'measure': 'inoffs', 
			'goal': 'MNbelow(5e-6)', 
			'shape': 'CSlinear2(1.0,0.001)'
		}, 
		{
			'measure': 'area', 
			'goal': 'MNbelow(1500e-12, 100e-12)', 
			'shape': 'CSlinear2(1.0,0.001)'
		}
	]

	inputOrder = [ 
		'mirr_w', 'mirr_l', 
		'out_w', 'out_l', 
		'load_w', 'load_l', 
		'dif_w', 'dif_l', 
		'c_out', 
		'r_out' 
	] 

	outputOrder = [ 
		'area', 'isup', 
		'outoffs', 'inoffs', 
		'vgs_drv', 'vds_drv', 
		'vgs_drv_cm', 'vds_drv_cm', 
		'swing', 
		'gain', 'ugbw', 'pm', 
		'tr', 'tf', 'overshoot', 'undershoot', 
		'tsetup', 'tsetdn', 'slewup', 'slewdn'
	]

	vis = {
		# Window list with axes
		'graphs': {
			'dc': {
				'title': 'Amplifier DC response', 
				'shape': { 'figsize': (6,4), 'dpi': 80 }, 
				'axes': {
					'dc': {
						'rectangle': (0.12, 0.12, 0.76, 0.76), 
						'options': {}, 
						'gridtype': 'rect', 		# rect (default) or polar, xscale and yscale have no meaning when grid is polar
						'xscale': { 'type': 'linear' }, 	# linear by default
						'yscale': { 'type': 'linear' }, 	# linear by default
						'xlimits': (-10e-3, 10e-3), 
						'xlabel': 'Vin [V]', 
						'ylabel': 'Vout [V]', 
						'title': 'DC response', 
						'legend': False, 
						'grid': True, 
					}
				}
			}, 
			'ac': {
				'title': 'Amplifier AC response', 
				'shape': { 'figsize': (6,4), 'dpi': 80 }, 
				'axes': {
					'gain': {
						'subplot': (2, 1, 1), 
						'xscale': { 
							'type': 'log', 
							'base': 10, 	# For log, default is 10
							'subticks': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 	# For log, default=10 (base=10), 12 (base=2), otherwise []
							# 'linthresh': 1e-6	# For log, turns on symlog scale at threshold
						}, 
						'ylabel': 'gain [dB]', 
						'title': 'Transfer function', 
						'grid': True, 
					}, 
					'phase': {
						'subplot': (2, 1, 2), 
						'xscale': { 'type': 'log' }, 
						'xlabel': 'f [Hz]', 
						'ylabel': 'phase [deg]', 
						'grid': True, 
					}
				}
			}, 
			'tran': {
				'title': 'Amplifier transient response', 
				'shape': { 'figsize': (6,4), 'dpi': 80 }, 
				'axes': {
					'impulse': {
						'xlabel': 'time [s]', 
						'ylabel': '[V]', 
						'title': 'Impulse response', 
					}
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
			'dcinout': {
				'graph': 'dc', 
				'axes': 'dc', 
				'xresult': 'dcvin',
				'yresult': 	'dcvout', 
				'corners': [ 'nom', 'vddhi', 'vddlo' ],	# If  no corners are specified , all corners where xresult is evaluated are plotted
				'style': {	# Style is defined by style patterns that match the (graph, axis, corner, trace) tuple
					'linestyle': '-',
					'marker': '.', 
				}
			}, 
			'acmag': {
				'graph': 'ac', 
				'axes': 'gain', 
				'xresult': 'acfreq', 
				'yresult': 'actfmag', 
			}, 
			'acphase': {
				'graph': 'ac', 
				'axes': 'phase', 
				'xresult': 'acfreq', 
				'yresult': 'actfphase', 
			}, 
			'pulsein': {
				'graph': 'tran', 
				'axes': 'impulse', 
				'xresult': 'trantime', 
				'yresult': 'tranin', 
				'style': {
					'linestyle': '-',
					'color': (0,0,0.5)
				}
			}, 
			'pulseout': {
				'graph': 'tran', 
				'axes': 'impulse', 
				'xresult': 'trantime', 
				'yresult': 'tranout', 
			}
		}
	}

	visMeasures={
		'dcvin': {
			'analysis': 'dc', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "v('inp', 'inn')",
			'vector': True
		},
		'dcvout': {
			'analysis': 'dc', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "v('out')",
			'vector': True
		},
		'acfreq': {
			'analysis': 'ac', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.abs(scale())", 
			'vector': True
			
		},
		'actfmag': { 
			'analysis': 'ac', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.ACmag(m.ACtf(v('out'), v('inp', 'inn')))", 
			'vector': True
		},
		'actfphase': { 
			'analysis': 'ac', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "m.ACphase(m.ACtf(v('out'), v('inp', 'inn')))", 
			'vector': True
		},
		'trantime': {
			'analysis': 'tran', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "scale()", 
			'vector': True
		},
		'tranin': {
			'analysis': 'tran', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "v('in')", 
			'vector': True, 
		},
		'tranout': {
			'analysis': 'tran', 
			'corners': [ 'nom', 'vddhi', 'vddlo' ], 
			'expression': "v('out')", 
			'vector': True
		},
	}
	
	# Prepare virtual machine
	try:
		from pyopus.parallel.pvm import PVM
		vm=PVM(debug=0)
		vm.spawnerBarrier()
		vm.setSpawn(mirrorMap={'*':'.'})
	except:
		print("Failed to initialize PVM. Only local runs possible.")
		vm=None

	# Update measures with visualisation measures
	measures.update(visMeasures)

	# Select simulator
	if simulator=="spiceopus": 
		head=spiceopus_head
	elif simulator=="hspice": 
		head=hspice_head
	else:
		print("Unknown simualtor '"+simulator+"'.")
		sys.exit(1)

	# Performance and cost evaluators, optimizatuion parameters
	pe=PerformanceEvaluator(head, analyses, corners, measures, debug=0)
	ce=CostEvaluator(pe, inputOrder, costDefinition, debug=0)
	(xini, xlo, xhi, xstep)=parameterSetup(inputOrder, costInput)
	
	# Plotter for plotting the performance
	if visualizeRun!='novis':
		from pyopus.visual.wxmplplotter import WxMplPlotter
		plotter=WxMplPlotter(vis, pe)

	# Select method
	if method=="hj":
		opt=optimizerClass("HookeJeeves")(ce, xlo, xhi, debug=0, maxiter=1000)
	elif method=="bc":
		opt=optimizerClass("BoxComplex")(ce, xlo, xhi, debug=0, maxiter=1000)
	elif method=="psade":
		opt=optimizerClass("ParallelSADE")(ce, xlo, xhi, debug=0, maxiter=100000, 
			vm=vm, maxSlaves=nworkers)
	else:
		print("Unknown method '"+method+"'.")
	
	# Set initial point
	opt.reset(xini)

	# Install stopper and reporter plugins
	opt.installPlugin(ce.getStopWhenAllSatisfied())	
	opt.installPlugin(ce.getReporter())
	
	# Install plotter (if requested)
	if visualizeRun=='runvis':
		opt.installPlugin(IterationPlotter(plotter))

	# Run, measure time
	t1=time()
	opt.check()
	try:
		opt.run()
	except KeyboardInterrupt:
		pe.finalize()
		raise
	except:
		raise

	# Print timing
	print("\nTime needed: %f" % (time()-t1))
	
	# Collect best point and iteration number
	xresult=opt.x
	iterresult=opt.bestIter
	
	# Reevaluate circuit
	cf=ce(xresult)
	
	# Plot final result
	if visualizeRun=='runvis' or visualizeRun=='finalvis':
		plotter(prefixText=('Final result (i=%d, cost=%f)' % (iterresult, cf)))
	
	# Print final result
	print "\nPerformance in corners"
	print pe.formatResults(outputOrder)
	print("\nFinal cost: "+str(cf)+", found in iter "+str(iterresult)+", total "+str(opt.niter)+" iteration(s)")
	# Parameters are stored in ce 
	print(ce.formatParameters())
	print(ce.formatResults()) 
	
	# Clean up and stop simulators
	pe.finalize()
	
if __name__=='__main__':
	if len(sys.argv)<2:
		print "\nNot enough arguments. Use help to get help."
		sys.exit(1)
	
	action=sys.argv[1]
	
	if action=="run":
		# Simulator
		if len(sys.argv)>2:
			simulator=sys.argv[2]
		else:
			# Default 
			simulator="spiceopus"
		
		# Method
		if len(sys.argv)>3:
			method=sys.argv[3]
		else:
			# Default 
			method="hj"
		
		# Number of workers
		if len(sys.argv)>4:
			nworkers=eval(sys.argv[4])
		else:
			# Default is 1
			nworkers=0
		
		# Visualize run
		if len(sys.argv)>5:
			visualizeRun=sys.argv[5]
		else: 
			visualizeRun='novis'
		
		main(simulator, method, nworkers, visualizeRun)
		
	else:
		if action!="help":
			print "Bad option."
		
		print """
Syntax: 
runme.py help
runme.py run <simulator> <method> <nworkers> <visualize>
  simulator: 
    hspice 
    spiceopus - default
  method: 
    hj (Hooke-Jeeves) - default
    bc (Box complex - Box constrained simplex)
    psade (Parallel simulated annealing with differential evolution)
  nworkers
    0  .. local run - default
    >0 .. parallel run (requires running PVM)
  visualize
    runvis   .. display performance plots for every improved circuit
    finalvis .. display performance plots for final circuit 
    novis    .. no performace plots - default
"""