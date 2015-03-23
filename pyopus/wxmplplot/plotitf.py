"""
**Interface to the plot window manager**

The plotting happens in a separate thread (process) so the plot windows remain 
responsive even if the main thread does some computation. No additional user 
intervention is needed (like calling some refresh function from time-to time). 

An example will explain more than ten paragraphs of text::

	import pyopus.wxmplplot as pyopl
	from numpy import arange, sin, cos, exp, pi, e

	# Initialize gui thread, clean up. 
	pyopl.init()
	pyopl.close()

	# Plot data - sin(x), cos(x), exp(x/pi) .. for x in [0, 2pi] with 0.2 step. 
	x = arange(0.0, 2*pi, 0.2)
	y1 = sin(x)
	y2 = cos(x)
	y3 = exp(x/pi)
	
	# Create first figure (plot window). This is now the active figure. 
	# Tag is assigned automatically by the system. 
	fig=pyopl.figure(windowTitle="Figure - single axes", figpx=(600,400), dpi=100)	
	
	# Lock the main GUI event loop. This implicitly disables repainting. 
	pyopl.lock(True)
	
	# If the window is closed the C++ part of the panel object is deleted, 
	# but the wxPython wrapper is still around. Accessing any attribute then 
	# results in an exception. To check if the C++ part is still there, simply 
	# use an 'if' statement. 
	if pyopl.alive(fig):
		ax=fig.add_axes((0.12,0.12,0.76,0.76))
		#ax=fig.gca()
		ax.plot(x, y1, '-o', label='sin(x)', color=(1,0,0))
		ax.hold(True)
		ax.plot(x, y2, 'rx', label='cos(x)')
		ax.plot(x, y3, '--k|', label='exp(x/pi)')
		ax.legend()
		ax.set_xlabel("x-axis")
		ax.set_ylabel("y-axis")
		ax.set_title("Axes title")
		ax.grid(True)
		ax.hold(False)
		fig.suptitle("Figure title")
		# Paint the changes on the screen. 
		pyopl.draw(fig)
		
	# Now unlock the main GUI event loop
	pyopl.lock(False)
	
	# Handle keyboard interrupts properly. 
	pyopl.join()

1. You obtain the Matplotlib :class:`Figure` object as the return value of  
   :func:`figure`. 
2. Next you have to lock the GUI event loop (call :func:`lock` with ``True``). 
   Now the GUI stops and the window is not refreshed anymore thus preventing 
   any interference from the GUI while API calls are being made. 
3. Before making API calls you have to verify that the window is not closed.  
   This can be done by calling :func:`alive` and passing it the :class:`Figure` 
   object. 
4. Do your Matplotlib API stuff on the :class:`Figure` object. 
5. Request a manual redraw of the plot window by calling :func:`draw` and 
   passing it the :class:`Figure` object. 
6. Unlock the GUI event loop by calling :func:`lock` with ``False``.
7. At the end of your program call :func:`join`. This will take proper care 
   of the keyboard interrupt when the program is finished. The program exits
   when you close the Control Window. If you don't call :func:`join` keyboard 
   interrupt is not handled properly (i.e. ignored) and the only way to exit 
   the program is to close the Control Window (or kill the interpreter). 
"""

from wxmgr import *
from matplotlib import rcParams
import os

__all__ = [ 'init', 'shutdown', 
			'lock', 'join', 
			'updateRCParams', 
			'figure', 'alive', 'draw', 
			'close', 'title', 
			'showFigure', 'raiseFigure', 'saveFigure'
		]

GUIctl=None
"The :class:`GUIControl` object for communicationg with the graphical thread."	

def init():
	"""
	Initializes the graphical thread and pops up the Control window. Closing 
	the control window closes all plots and exits the graphical thread. To 
	use graphics again a new call to :func:`init` is needed. 
	
	If the graphical therad is already running calling :func:`init` has no 
	effect. 
	"""
	global GUIctl
	
	if GUIctl is None:
		GUIctl=GUIControl()
		
	GUIctl.StartGUI()
	updateRCParams({
		'font.size': 10,
		'axes.titlesize': 10,
		'axes.labelsize': 9,
		'xtick.labelsize': 9,
		'ytick.labelsize': 9,
		'legend.fontsize': 9
	})
	
def shutdown():
	"""
	Exits the graphical thread. Equivalent to closing the Control window. 
	
	The GUI must not be locked when this function is called. See :func:`Lock`. 
	"""
	global GUIctl 
	
	if GUIctl is not None:
		GUIctl.StopGUI()

def lock(state=True):
	"""
	Calling convention: lock(*state*)

	Locks the main event loop so Matplotlib API calls can be made using the 
	matplotlib :class:`Figure` objects without interfering with the repainting 
	of the plot window. 
	
	If *state* is ``False`` the main event loop is unlocked and can procees. 
	Multiple calls with *state* set to ``True`` (or ``False``) are equovalent 
	to a single call. 
	"""
	if state is True:
		GUIctl.Lock()
	else:
		GUIctl.UnLock()

def join():
	"""
	Calling convention: join()

	Joins the GUI thread. Usually called at the end of the main thread so the 
	windows don't close immediately after the main thread is finished. 
	The waiting can be interrupted by a keyboard interrupt. 
	
	The GUI must not be locked when this function is called. See :func:`Lock`. 
	"""
	GUIctl.Join()
	
def updateRCParams(*args, **kwargs): 
	"""
	Calling convention: updateRCParams(*dict*)
	
	Updates the rcParams structure of Matplotlib with the dictionary given by 
	*dict*. Returns ``True`` on success. 
	
	The GUI must not be locked when this function is called. See :func:`Lock`. 
	"""
	return GUIctl.SendMessage({
			'cmd':['plot', 'updatercparams'], 
			'args': args, 
			'kwargs': kwargs
		}
	)

def figure(*args, **kwargs):
	"""
	Calling convention: figure(*windowTitle*, *show*, *inFront*, *figpx*, 
	*dpi*, *figsize*, ...)
	
	Create a new plot window. 
	
	*windowTitle* specifies the title of the window. 
	
	If *show* is ``True`` (default) the window will be visible as soon as it is 
	created. 
	
	If *inFront* is ``True`` (default) the window will show up on top of all 
	other windows. 
	
	*figsize* is a tuple specifying the horizontal and vertical figure size in 
	inches. *dpi* is used for obtaining the size in pixels (for the screen). 
	
	*figpx* is a tuple specifying the horizontal and vertical figure size in 
	pixels. *dpi* is used for obtaining the figure size in inches (i.e. for 
	saving the figure in a Postscript file). When specified, *figpx* overrides 
	*figsize*. 
	
	All remaining arguments are passed to the constructor of the 
	:class:`pyopus.wxmplitf.PlotFrame` object which passes them on to the 
	constructor of the :class:`pyopus.wxmplitf.Plotpanel` object. 
	
	Returns the tag of the newly created plot window which is actuaaly the 
	Matplotlib :class:`Figure` object corresponding to the plot window. 
	
	The GUI must not be locked when this function is called. See :func:`Lock`. 
	"""
	return GUIctl.SendMessage({
			'cmd':['plot', 'new'], 
			'args': args, 
			'kwargs': kwargs
		}
	)

def alive(fig):
	"""
	Returns ``True`` if the figure *fig* is alive (window is not closed). 
	
	The GUI must be locked when this function is called. See :func:`Lock`. 
	"""
	return GUIctl.FigureAlive(fig)

def draw(fig):
	"""
	Redraws figure *fig*. Does nothing if figure is not alive. 
	
	The GUI must be locked when this function is called. See :func:`Lock`. 
	"""
	return GUIctl.FigureDraw(fig)

def close(fig=None):
	"""
	Closes a plot window corresponding to figure *fig*. 
	
	If no *fig* is given, closes all plot windows. 
	
	Returns ``True`` on success. 
	
	The GUI must not be locked when this function is called. See :func:`Lock`. 
	"""
	if fig is None:
		return GUIctl.SendMessage({
				'cmd':['plot', 'closeall'], 
				'args': (), 
				'kwargs': {}
			}
		)
	else:
		return GUIctl.SendMessage({
				'cmd':['plot', 'close'], 
				'args': ( fig, ), 
				'kwargs': {}
			}
		)

def title(fig, title):
	"""
	Sets the window title of figure *fig* to *title*.
	
	This title appears in the title bar of the plot window. 
	
	The GUI must not be locked when this function is called. See :func:`Lock`. 
	"""
	return GUIctl.SendMessage({
			'cmd':['plot', 'setwindowtitle'], 
			'args': (fig, title), 
			'kwargs': {}
		}
	)

def showFigure(fig, on=True):
	"""
	Shows (*on* is ``True``) or hides the plot window corresponding to figure 
	*fig*. 
	
	Returns ``True`` on success. 
	
	The GUI must not be locked when this function is called. See :func:`Lock`. 
	"""
	return GUIctl.SendMessage({
			'cmd':['plot', 'show'], 
			'args': ( fig, on ), 
			'kwargs': {}
		}
	)

def raiseFigure(fig):
	"""
	Raises the plot window corresponding to figure *fig*. 
	
	Returns ``True`` on success. 
	
	The GUI must not be locked when this function is called. See :func:`Lock`. 
	"""
	return GUIctl.SendMessage({
			'cmd':['plot', 'raise'], 
			'args': ( fig, ), 
			'kwargs': {}
		}
	)
	
def saveFigure(fig, fileName):
	"""
	Saves the contents of a plot window corresponding to figure *fig* to a file 
	named *fileName*. 
	
	See the :meth:`FigureCanvasWxAgg.print_figure` method. The available 
	file types are listed in the :attr:`FigureCanvasWxAgg.filetypes` 
	dictionary. 
		
	Returns ``True`` on success. 
	
	The GUI must not be locked when this function is called. See :func:`Lock`. 
	"""
	# Send full path (gui process may not have the same working directory). 
	return GUIctl.SendMessage({
			'cmd':['plot', 'savefigure'], 
			'args': ( fig, os.path.realpath(fileName) ), 
			'kwargs': {}
		}
	)
