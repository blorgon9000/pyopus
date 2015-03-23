"""
.. inheritance-diagram:: pyopus.wxmplplot.wxmgr
    :parts: 1
	
**Manager for Matplotlib plot windows**

Uses the :mod:`multiprocessing` module if it is available. If not, 
:mod:`threading` is used. 

The graphical part (wxPython + matplotlib) is running in a thread (or process 
if :mod:`multiprocessing` is used) and the part that issues the plotting 
commands (defined in the :mod:`~pyopus.wxmplplot.plotitf` module) runs in 
the main thread (or process). 

The main thread uses a :class:`GUIControl` object for sending and receiving 
messages from the graphical thread. The messages are sent yo the GUI using
a :class:`Queue` object. On the graphical thread's side a :class:`ControlApp` 
application is running. The main window of the application is a 
:class:`ControlWindow` object. Commands to the GUI are sent by posting 
:class:`ControlEvent` events to the :class:`ControlApp`.

The :class:`ControlEvent` handler in :class:`ControlApp` calls the 
:meth:`ControlWindow.InterpretCommand` method that dispatches the message to 
the corresponding command handler. 
"""

import threading
import Queue
import cPickle
import traceback
import time

from matplotlib import rcParams
from matplotlib.lines import Line2D
try:
	from mpl_toolkits.mplot3d import Axes3D
except:
	print "Failed to import Axes3D. 3D plotting is not available."

import wx 
from wxmplitf import *

__all__ = [ 'GUIControl', 'GUIentry', 
			'ControlApp', 'ControlWindow', 
			'ControlEvent', 'EVT_CONTROL', 'EVT_CONTROL_ID', 'CONTROL_MSG' ]

#
# MessageCanvas providing a main window for messages
#

class ControlWindow(wx.Frame):
	"""
	This is the main GUI ControlWindow and command dispatcher. 
	
	*parent* is the parent window. *title* is the window title. 
	
	*lock* is the :class:`Lock` object that prevents other threads from 
	messing up Matplotlib objects while wxPython events are processed. 
	"""
	def __init__(self, parent, title, lock):
		wx.Frame.__init__(self, parent, -1, title, pos=(150, 150), size=(600, 300))
		
		# Store the lock
		self.lock=lock
		
		# Create the menubar
		menuBar = wx.MenuBar()

		# And a menu 
		menu = wx.Menu()
		
		# Add an item to the menu, using \tKeyName automatically
		# creates an accelerator, the third param is some help text
		# that will show up in the statusbar
		IDnewWin=menu.Append(-1, "&New plot\tCtrl-N", "New plot window")
		IDcloseAll=menu.Append(-1, "Close all plots", "Close all plot windows")
		menu.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Exit")

		# Bind the menu event to an event handler
		self.Bind(wx.EVT_MENU, self.OnNewPlot, IDnewWin)
		self.Bind(wx.EVT_MENU, self.OnCloseAll, IDcloseAll)
		self.Bind(wx.EVT_MENU, self.OnExit, id=wx.ID_EXIT)

		# Put the menu on the menubar
		menuBar.Append(menu, "&File")
		self.SetMenuBar(menuBar)

		# Create staus bar
		self.CreateStatusBar()
		
		# Now create the Panel to put the other controls on.
		panel = wx.Panel(self)

		# Create message frame
		self.messages = wx.TextCtrl(panel, -1, style=wx.TE_MULTILINE|wx.TE_RICH2) 
		self.messages.SetFont(wx.Font(8, wx.SWISS, wx.NORMAL, wx.NORMAL))
		self.messages.SetSize(self.messages.GetBestSize())
		self.messages.SetEditable(False)
		self.OutputMessage("Control window ready.\n")

		# Use a sizer to layout the controls
		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(self.messages, 1, wx.EXPAND)
		panel.SetSizer(sizer)
		panel.Layout()
		
		# Plot window storage
		self.plotWindows={}
		self.plotTagIndex={}
		
		# Initialize interpreter
		self.plot_cmdtab={}
		self.InitInterpreter()
		
		wx.EVT_WINDOW_DESTROY(self, self.OnDestroy)
		wx.EVT_CLOSE(self, self.OnClose)
		
	def InitInterpreter(self):
		"""
		Initializes the interpreter command tables. 
		
		Every command is a list of strings. The first string specifies the 
		command group and the second string the command. Currently only one 
		command group is available (``plot``). 
		
		Every command is passed an *args* tuple of positional arguments and a 
		*kwargs* dictionary of keyword arguments. 
		
		Every plot window is actually a 
		:class:`~pyopus.wxmplplot.wxmplitf.PlotFrame` object displaying a 
		Matplotlib :class:`Figure`. Every plot window is associated with a tag 
		at its creation. A tag is a hashable object (usually a string). 
		If not tag is specified at creation a tag is assigned automatically. 
		
		Every Matplotlib :class:`Figure` can have multiple :class:`Axes`. 
		Every axes represent a part of the figure where plotting takes place 
		within the coordinate system of the :class:`Axes`. Every axes in a 
		figure are associated with a numeric tag in a similar way as every plot 
		window has its tag. Axes tags are unique because they cannot be 
		assigned by the user. 
		
		Tags should be unique. If a duplicate figure tag occurs that older 
		figure with that tag becomes inaccessible. 
		
		The ``plot`` command group has the following commands:
		
		* ``exit`` - exits the interpreter
		* ``updatercparams`` - update the rc parameters with the given 
		  dictionary
		* ``new`` - create a new plot window and return its tag
		* ``close`` - close a plot window
		* ``closeall`` - close all plot windows
		* ``setwindowtitle`` - sets the title of a plot window
		* ``show`` - show or hide aplot window
		* ``raise`` - raise a plot window
		* ``savefigure`` - saves the contents of a figure to a file
		
		"""
		self.plot_cmdtab={
			'exit':				self.ExitInterpreter, 
			'updatercparams':	self.UpdateRCParams, 
			'new':				self.NewPlotWindow, 
			'close':			self.ClosePlotWindow, 
			'closeall':			self.CloseAllPlotWindows, 
			'setwindowtitle':	self.SetPlotWindowTitle, 
			'show':				self.ShowPlotWindow,
			'raise':			self.RaisePlotWindow, 
			'savefigure':		self.SaveFigure, 
		}
		
	def ExitInterpreter(self):
		"""
		Handles the ``['plot', 'exit']`` command.
		
		Exits the interpreter (thread or process).
		"""
		self.Close()
	
	def UpdateRCParams(self, dict):
		"""
		Handles the ``['plot', 'updatercparams']`` command.
		
		Update the rcParams structure of Matplotlib with the given dictionary.
		Returns ``True`` on success. 
		"""
		try:
			rcParams.update(dict);
			retval=True
		except:
			retval=False
		
		return retval
		
	def NewPlotWindow(self, windowTitle="Untitled", show=True, inFront=True, **kwargs):
		"""
		Handles the ``['plot', 'new']`` command.
		
		*windowTitle* is the title string for the plot window. 
		
		If *show* is ``True`` the window becomes visible immediately after it 
		is created. 
		
		If *inFront* is ``True`` the window is created atop of all other 
		windows. 
		
		All remaining arguments are passed on to the constructor of the 
		:class:`~pyopus.wxmplplot.wxmplitf.PlotFrame` object. 
		
		Returns the plot window's :class:`Figure` object or ``None`` on failure.
		"""
		
		# Pass the lock to the PlotFrame (plot window). 
		window=PlotFrame(self, -1, windowTitle, lock=self.lock, **kwargs)
		tag=window.get_figure()
		
		self.plotTagIndex[window]=tag
		self.plotWindows[tag]={}
		self.plotWindows[tag]['obj']=window
				
		window.Connect(window.GetId(), -1, wx.wxEVT_CLOSE_WINDOW, self.OnChildClosed)
		EVT_CLOSEALL(window, window.GetId(), self.OnCloseAll)
		
		if show:
			window.Show(True)
		
		if inFront:
			window.Raise()
		
		return tag
		
	def ClosePlotWindow(self, tag=None):
		"""
		Handles the ``['plot', 'close']`` command.
		
		Closes a plot window with given *tag*. 
		
		Returns ``True`` on success.
		"""
		if tag in self.plotWindows:
			self.plotWindows[tag]['obj'].Close()
			return True
		else:
			return False
			
	def CloseAllPlotWindows(self):
		"""
		Handles the ``['plot', 'closeall']`` command.
		
		Closes all plot windows.
		"""
		tags=self.plotWindows.keys()
		for tag in tags:
			self.plotWindows[tag]['obj'].Close()
		
		return True
	
	def SetPlotWindowTitle(self, tag, title):
		"""
		Handles the ``['plot', 'setwindowtitle']`` command.
		
		Sets the window title of the active plot window. 
		
		Returns ``True`` on success. 
		"""
		if tag in self.plotWindows:
			window=self.plotWindows[tag]['obj']
		else:
			return False
		
		window.SetTitle(title)
		
		return True
	
	def ShowPlotWindow(self, tag, on=True):
		"""
		Handles the ``['plot', 'show']`` command.
		
		Shows (*on* set to ``True``) or hides a plot window. 
		
		Returns ``True`` on success. 
		"""
		if tag in self.plotWindows:
			window=self.plotWindows[tag]['obj']
		else:
			return False
		
		window.Show(on)
		
		return True
	
	def RaisePlotWindow(self, tag):
		"""
		Handles the ``['plot', 'raise']`` command.
		
		Raises a plot window with given *tag*. 
		
		Returns ``True`` on success. 
		"""
		if tag in self.plotWindows:
			window=self.plotWindows[tag]['obj']
		else:
			return False
		
		window.Raise()
		
		return True
		
	def SaveFigure(self, tag, fileName):
		"""
		Handles the ``['plot', 'savefigure']`` command.
		
		Saves the contents of a plot window with given *tag* to a file with a 
		name given by *fileName*. File type is determined from the extension 
		in the *fileName*. 
		
		See the :meth:`FigureCanvasWxAgg.print_figure` method. The available 
		file types are listed in the :attr:`FigureCanvasWxAgg.filetypes` 
		dictionary. 
		
		Returns ``True`` on success. 
		"""
		# Get window (PlotFrame)
		if tag in self.plotWindows:
			window=self.plotWindows[tag]['obj']
		else:
			return False
		
		# Get panel (PlotPanel)
		fig=window.get_panel()
		
		# Save to file
		fig.print_figure(fileName)
		
		return True
		
	def FigureAlive(self, tag):
		"""
		Returns ``True`` if figure *tag* is alive (i.e. window is not closed). 
		"""
		if tag in self.plotWindows and self.plotWindows[tag]['obj']:
			return True
		else:
			return False
		
	def FigureDraw(self, tag):
		"""
		Redraws the figure *tag* immediately. 
		
		If *tag* is not alive doesn't do anything. 
		"""
		if tag in self.plotWindows:
			window=self.plotWindows[tag]['obj']
		
			if window:
				window.draw()
		
	def _RemovePlotWindow(self, window):
		"""
		Removes a plot window given by object *window* from the list of plot 
		windows. 
		
		Returns ``True`` on success.
		
		This should not be called directly. Call ClosePlotWindow() instead.
		"""
		if window in self.plotTagIndex:
			tag=self.plotTagIndex[window]
		else:
			return False
		
		if tag is not None:
			del self.plotWindows[tag]
			del self.plotTagIndex[window]
			
			return True
		else:
			return False
				
	def OutputMessage(self, msg, colour=(0,0,0), bgcolour=(255,255,255), fontfamily=wx.FONTFAMILY_SWISS):
		"""
		Displays a message given by *msg* in the control window. The message 
		colour and background colour are specified with the *colour* and 
		*bgcolour* arguments (tuples of RGB values between 0 and 255). 
		
		*fontfamily* specifies the font family for the message. 
		"""
		self.messages.SetInsertionPointEnd()
		style=self.messages.GetDefaultStyle()
		style.SetTextColour(wx.Colour(*colour))
		style.SetBackgroundColour(wx.Colour(*bgcolour))
		wxfont=style.GetFont()
		wxfont.SetFamily(fontfamily)
		style.SetFont(wxfont)
		self.messages.AppendText(msg)
		self.messages.SetModified(True)
			
	def InterpretCommand(self, command, args=[], kwargs={}):
		"""
		Interprets the command given by the *command* list of strings. The 
		first string is the command family and the second string is the 
		command. Currently only one command family is available (``plot``). 
		
		The arguments to the command are given by *args* and *kwargs*. 
		
		The command handlers are given in handler dictionaries with command 
		name for key. The handler dictionary for the ``plot`` command family 
		is is the :attr:`plot_cmdtab` member. 
		
		Every command is handled in a ``try-except`` block If an error occurs 
		during command execution the error is displayed in the command window. 
		
		The return value of the command handler is returned. 
		
		This method is invoked by the :class:`ControlApp` on every 
		:class:`ControlEvent`. The contents of the event specify the command, 
		the arguments and a talkback flag which is ``True`` if the return value 
		of the command handler should be sent back to the main thread 
		(process). 
		"""
		response=None
		
		if command[0]=='plot':
			# Plot commands
			# In future this will be a part of the plot controller object
			# Figure commands
			if command[1] in self.plot_cmdtab:
				try:
					# self.OutputMessage("Command: '"+str(command)+" "+str(args)+str(kwargs)+"'\n")
					response=self.plot_cmdtab[command[1]](*args, **kwargs)
					# self.OutputMessage("Response: '"+str(response)+"'\n")
				except:
					self.OutputMessage('Exception in '+str(command)+'\n', bgcolour=(255,200,200), colour=(0,0,0))
					self.OutputMessage(traceback.format_exc(), colour=(180,0,0))
					response=None
			else:
				self.OutputMessage("Unknown command [1]: '"+str(command)+"'\n")
		else:
			self.OutputMessage("Unknown command [0]: '"+str(command)+"'\n")
		
		return response
	
	def OnNewPlot(self, evt):
		"""
		Event handler for the menu option 'New plot'.
		"""

		self.NewPlotWindow()
	
	def OnCloseAll(self, evt):
		"""
		Event handler for the menu option 'Close all plots'.
		"""

		keys=self.CloseAllPlotWindows()
		
	def OnExit(self, evt):
		"""
		Event handler for the menu option 'Exit'.
		"""
		self.Close()
	
	def OnDestroy(self, evt):
		"""
		Event handler for EVT_WINDOW_DESTROY.
		"""
		pass
	
	def OnClose(self, evt):
		"""
		Event handler for EVT_CLOSE
		"""
		self.OnCloseAll(None)
		self.Destroy()
		
	def OnChildClosed(self, evt):
		"""
		Called when a child (plot window) receives a close event.
		"""
		
		# Remove it from the list
		self._RemovePlotWindow(evt.GetEventObject())
		
		evt.Skip()


#
# ControlApp providing a matplotlib canvas in a top-level wxPython window
#

class ControlApp(wx.App):
	"""
	This is the GUI control application that will run in the GUI thread. 
	
	The commands are posted as :class:`ControlEvent` events. Responses are sent 
	back through the *responseQueue*. 
	
	Any additional arguments are passed to the constructor of :class:`wx.App`. 
	"""

	def __init__(self, title="GUI Control App", responseQueue=None, lock=None, **kwds):
		self.controlWindow=None
		self.responseQueue=responseQueue
		self.lock=lock
		
		wx.App.__init__(self, **kwds)
	
	def ControlWindow(self):
		"""
		Returns the :class:`ControlWindow` object. 
		"""
		return self.controlWindow
	
	def OnInit(self):
		"""
		Called automatically when the control GUI application starts. 
		
		Initializes the main GUI control window and makes it the top window. 
		Associates the :meth:`OnControlEvent` method with :class:`ControlEvent` 
		events. 
		"""
		# Pass the lock to the ControlWindow so it can pass it on to the 
		# PlotFrame objects (plot windows). 
		self.controlWindow=controlWindow=ControlWindow(None, "GUI Control Window", lock=self.lock)
		
		EVT_CONTROL(self, CONTROL_MSG, self.OnControlEvent)
		
		self.SetTopWindow(controlWindow)
		
		controlWindow.Raise()
		controlWindow.Show(True)
		
		return True
		
	def ProcessMessage(self, message): 
		"""
		This is the function that is invoked for every command that is sent to 
		the GUI. It posts a :class:`ControlEvent` event to :class:`ControlApp`. 
		"""
		wx.PostEvent(self, ControlEvent(CONTROL_MSG, message))
		
	def OnControlEvent(self, evt):
		"""
		Handles a :class:`ControlEvent` event. 
		
		For every event it calls the :meth:`InterpretCommand` method of the 
		:class:`ControlWindow` object. If response is requested the return 
		value of the :meth:`InterpretCommand` method is sent back by 
		pickling it and putting it in the *responseQueue*. 
		"""
		response=self.controlWindow.InterpretCommand(evt.message['cmd'], evt.message['args'], evt.message['kwargs'])
		
		if self.responseQueue: 
			self.responseQueue.put(response, True)

				
#
# This is the EVT_CONTROL event. 
#

EVT_CONTROL_ID = wx.NewId()
"The wxPyhton identifier of a :class:`ControlEvent` event."

CONTROL_MSG = wx.NewId()
"The wxPyhton identifier of a message sender carried by a :class:`ControlEvent` event."

def EVT_CONTROL(win, id, func):
	"""
	Register to receive :class:`ControlEvent` events from a 
	:class:`ControlApp`.
	The events originating from *win* are to be handled by function *func*. 
	
	*id* is the identifier to be associated with the event. 
	"""
	win.Connect(id, -1, EVT_CONTROL_ID, func)

class ControlEvent(wx.PyCommandEvent):
	"""
	Control event for the ControlApp's main window. 
	
	The event carries a *message*. 
	
	*id* is the identifier associated with the event. 
	"""
	def __init__(self, id, message):
		wx.PyCommandEvent.__init__(self, EVT_CONTROL_ID, id)
		
		self.message=message
		
	def Clone(self):
		"""
		Creates a clone of the :class:`ControlEvent` object. 
		"""
		return ControlEvent(self.GetId(), self.message)

#
# This are the GUIcontrol classes.  
#

def GUIentry(queueFromGUI, lock, redirect):
	"""
	Entry point of the GUI thread. 
	
	This function creates a :class:`ControlApp` object and starts the GUI 
	application's main loop. *queueFromGUI* is the queue that is used for 
	returning values from the GUI. Commands are sent to the GUI by posting 
	events to the ControlApp object.
	
	If set to ``True`` *redirect* redirects ``stdout`` and ``stderr`` to a 
	window that pops up when needed. *redirect* is passed to the constructor of 
	:class:`wx.App`. 
	"""
	app=ControlApp(responseQueue=queueFromGUI, lock=lock, redirect=redirect)
	
	# Send the ControlApp object to the main thread. 
	queueFromGUI.put(app, True)
	
	# Enter GUI main loop. 
	app.MainLoop()

	# This is reached after main loop finishes

class GUIControl(object):
	"""
	This object is used in the main thread (or process) for sending the plot 
	commands to the graphical thread. It can also collect the value returned 
	by the functions called by the graphical thread for every command sent. 
	
	If *redirect* is ``True`` all output from ``stdout`` and ``stderr`` is 
	redirected to a window that pops up when needed. *redirect* is passed to 
	the constructor of :class:`wx.App`. 
	"""
	def __init__(self, redirect=False):
		self.redirect=redirect
		self.queueFromGUI=Queue.Queue(-1)
		self.controlThread=None
		self.lock=threading.Lock()
		self.locked=False
		self.controlApp=None
	
	def CheckIfAlive(self):
		"""
		Returns ``True`` if the graphical thread (or process) is running. 
		"""
		if self.controlThread is not None: 
			if self.controlThread.isAlive():
				return True
			else: 
				self.controlThread=None
				return False
		else:
			return False
		
	def StartGUI(self):
		"""
		Starts the graphical thread. The thread entry point is the 
		:func:`GUIentry` function. 
		"""
		if not self.CheckIfAlive():
			self.controlThread=threading.Thread(
				target=GUIentry, args=(self.queueFromGUI, self.lock, self.redirect)
			)
			# self.controlThread.daemon=True
			self.controlThread.start()
			# Get the control app from the GUI
			self.controlApp=self.queueFromGUI.get(True)
			# We pass control to the rest of the program here. GUI thread is running now. 
	
	
	def SendMessage(self, message):
		"""
		Sends a *message* (Python object) to the graphical thread. 
		
		Collects the return value of the executed graphical command and 
		returns it. 
		
		Returns ``None`` if talkback is disabled. This is also the case if 
		the graphical thread is not running. 
		"""
		if self.CheckIfAlive():
			# print "sending command", message['cmd']
			
			# Send the message to the GUI
			try:
				wx.CallAfter(self.controlApp.ProcessMessage, message)
			except (KeyboardInterrupt, SystemExit):
				# Re-reaise these two exceptions
				raise
			except:
				# Everything else is an error
				raise Exception, "Matplotlib GUI thread is not running."
			
			# print "  awaiting response"
			response=self.queueFromGUI.get(True)
			# print "    got it"
			
			return response
		else:
			return None
			
	def FigureAlive(self, tag):
		try:
			retval=self.controlApp.ControlWindow().FigureAlive(tag)
		except (KeyboardInterrupt, SystemExit):
			# Re-reaise these two exceptions
			raise
		except:
			# Everything else is an error
			raise Exception, "Matplotlib GUI thread is not running."
	
		return retval
		
	def FigureDraw(self, tag):
		try:
			self.controlApp.ControlWindow().FigureDraw(tag)
		except (KeyboardInterrupt, SystemExit):
			# Re-reaise these two exceptions
			raise
		except:
			# Everything else is an error
			raise Exception, "Matplotlib GUI thread is not running."
	
	def StopGUI(self):
		"""
		Stops the graphical thread by sending it the ``['plot', 'exit']`` 
		command. 
		"""
		self.SendMessage({
				'cmd':['plot', 'exit'], 
				'args': [], 
				'kwargs': {}
			}
		)
		
	def Lock(self):
		"""
		Marks the beginning of a section of code where Matplotlib API calls are 
		made. Locking prevents these calls from interfering with the wxPython
		event loop and crashing the application. 
		"""
		# Lock if not already locked
		if not self.locked:
			self.locked=True
			self.lock.acquire(True)
	
	def UnLock(self):
		"""
		Marks the end of a section of code where Matplotlib API calls are made. 
		It reenables the wxPython event loop.  
		"""
		if self.locked:
			self.lock.release()
			self.locked=False
			
	def Join(self):
		"""
		Waits for the GUI thread to finish. 
		
		:obj:`KeyboardInterrupt` and :obj:`SystemExit` are caught and the GUI 
		is stopped upon which the exception is re-raised. 
		"""
		try:
			while self.controlThread.is_alive():
				self.controlThread.join(timeout=0.1)
		except (KeyboardInterrupt, SystemExit):
			self.StopGUI()
			raise
	
			
