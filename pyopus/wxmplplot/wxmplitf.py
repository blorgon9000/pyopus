"""
.. inheritance-diagram:: pyopus.wxmplplot.wxmplitf
    :parts: 1
	
**wxPython canvas for displaying Matplotlib plots**

This module provides a wxPython canvas for Matplotlib to render its plots on. 
The canvas supports zooming and displays cursor position in axes coordinates 
as the cursor moves across the canvas. 

A plot window is an object of the :class:`PlotFrame` class. The canvas itself 
is an object of the :class:`PlotPanel` class. 

The module also provides print preview, printing, and saving of the plots to 
raster (e.g. PNG) or vector files (e.g. Postscript). 
"""

# Bugs: annotations, when zoomed outside visible area, cause Python to crash.
#       This is a bug in matplotlib 0.98. Hope it gets fixed. 

import wx
import sys
import os.path
import weakref

# Import dialogs
import page_setup_xrc

import matplotlib
matplotlib.use('WXAgg')
from matplotlib import rcParams
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure

__version__ = '1.0'

__all__ = [
	'PlotFrame', 'PlotPanel', 
	'AxesLimits', 'DestructableViewMixin', 'Painter', 'CrosshairPainter', 
	'RubberbandPainter', 'CursorChanger', 
	'FigurePrinter', 'FigurePrintout', 
	'EVT_POINT_ID', 'EVT_SELECTION_ID', 'EVT_COORDINATES_ID', 'EVT_CLOSEALL_ID', 
	'EVT_POINT', 'EVT_SELECTION', 'EVT_COORDINATES', 'EVT_CLOSEALL']


class AxesLimits(object):
	"""
	Stores the zoom history for Matplotlib :class:`Axes` objects. The history 
	is stored in a :class:`WeakKeyDictionary` with :class:`Axes` objects for 
	keys. 
	
	History is a list of tuples of the form (*xlim*, *ylim*) where *xlim* and 
	*ylim* are the return values of the :meth:`get_xlim` and :meth:`get_ylim` 
	methods of the corresponding :class:`Axes` object. 
	
	
	Alters the X and Y limits of C{Axes} objects while maintaining a history of
	the changes.
	"""
	def __init__(self):
		self.history = weakref.WeakKeyDictionary()

	def _get_history(self, axes):
		"""
		Returns the history list of X and Y limits associated with the *axes*
		object. 
		"""
		# Return history for axes, set it to [] if not in the dictionary
		return self.history.setdefault(axes, [])

	def zoomed(self, axes):
		"""
		Returns a boolean indicating whether *axes* has had its limits
		altered.
		"""
		# Return True if there is anything in the history for axes
		return not (not self._get_history(axes))

	def set(self, axes, xrange, yrange):
		"""
		Changes the X and Y limits of *axes* to *xrange* and *yrange*
		respectively by calling the :meth:`set_xlim` and :meth:`set_ylim` 
		methods of the *axes* object. The old state of axes is stored in the 
		history list. A boolean indicating whether or not the axes should be 
		redrawn is returned, because polar axes cannot have their limits 
		changed sensibly.
		"""
		# Can handle only rectilinear exes
		if axes.name!='rectilinear':
			return False

		# Retrieve history
		history = self._get_history(axes)
		# Do we have history for axes
		if history:
			# Yes, get current axes range as old range
			# Must copy because the returned value is always the same array with different contents
			# Need to do this because older versions of matplotlib return xlim and ylim as numpy array
			# while newer versions return a tuple
			try:
				oldRange = axes.get_xlim().copy(), axes.get_ylim().copy()
			except:
				oldRange = axes.get_xlim(), axes.get_ylim()
		else:
			# No old range needed
			oldRange = None, None

		# Store old axes range in history
		history.append(oldRange)
		
		# Set new limits
		axes.set_xlim(xrange)
		axes.set_ylim(yrange)
		
		return True

	def restore(self, axes):
		"""
		Changes the X and Y limits of C{axes} to their previous values 
		obtained from teh corresponding history list. A boolean indicating 
		whether or not the axes should be redrawn is returned, because polar 
		axes cannot have their limits changed sensibly.
		"""
		# Get history for axes
		hist = self._get_history(axes)

		if not hist:
			# Nothing in history
			return False
		else:
			# Pop history
			xrange, yrange = hist.pop()
			
			# Is entry a range
			if xrange is None and yrange is None:
				# Autoscale if both are None
				axes.autoscale_view()
				return True
			elif xrange is not None and yrange is not None:
				# Set limits if both are not None
				axes.set_xlim(*xrange)
				axes.set_ylim(*yrange)
				return True
			else:
				# One is None and the other isn't, nothing to do
				return False


class DestructableViewMixin(object):
	"""
	Utility class to break the circular reference between an object and its
	associated "view".
	"""
	def destroy(self):
		"""
		Sets this object's *view-* attribute to ``None``.
		"""
		self.view = None


#
# Components used by the PlotPanel
#

class Painter(DestructableViewMixin):
	"""
	Painters encapsulate the mechanics of drawing some value in a wxPython
	window and erasing it. Subclasses override template methods to process
	values and draw them. 
	
	The :attr:`PEN` and :attr:`BRUSH` members specify the 
	:class:`wx.Pen` and :class:`wx.Brush` objects to use for drawing (defaults 
	are :obj:`wx.BLACK_PEN` and :obj:`wx.TRANSPARENT_BRUSH`). 
	
	:attr:`FUNCTION` is the logical function to use for drawing 
	(defaults to :obj:`wx.COPY`). 
	
	The font is specified with the :attr:`FONT` member (a :class:`wx.Font` 
	object, defaults to :obj:`wx.NORMAL_FONT`). 
	
	The :attr:`TEXT_FOREGROUND` and the :attr:`TEXT_BACKGROUND` 
	members specify the :class:`wx.Colour` objects for text (defaults are 
	:obj:`wx.BLACK` and :obj:`wx.WHITE`). 
	
	*view* specifies the wxPython window to which the new :class:`Painter` 
	object is attached. The *enabled* argument sets the state of this painter 
	(``True`` for enabled). 
	"""

	PEN = wx.BLACK_PEN
	BRUSH = wx.TRANSPARENT_BRUSH
	FUNCTION = wx.COPY
	FONT = wx.NORMAL_FONT
	TEXT_FOREGROUND = wx.BLACK
	TEXT_BACKGROUND = wx.WHITE

	def __init__(self, view, enabled=True):
		self.view = view
		self.lastValue = None
		self.enabled = enabled
	
	def getEnabled(self):
		"""
		Return the enabled state of this painter.
		"""
		return self.enabled

	def setEnabled(self, state):
		"""
		Enable or disable this painter.  Disabled painters do not draw their
		values and calls to the :meth:`set` method have no effect on them.
		"""
		
		# Currently enabled, we are disabling it
		if self.enabled and not state:
			# Hide it
			self.clear()

		if not state:
			self.lastValue=None
		
		self.enabled=state
		
	def set(self, *value):
		"""
		Update this painter's value and then draw it. *value* is a tuple of 
		arguments representing the new value of the painter. The painter 
		stores the value as a formatted value in the :attr:`value` member. 
		Value may not be ``None``, which is used internally to represent the 
		absence of a current value. 
		"""
		if self.enabled:
			value = self.formatValue(value)
			self._paint(value, None)
	
	def redraw(self, dc=None, restoreValue=None):
		"""
		Redraw this painter's current value using wxPython device context *dc*. 
		If *restoreValue* is not ``None``, the formatted value is restored 
		from *restoreValue*. In the latter case the painter must be cleared 
		first. 
		"""
		if restoreValue is not None:
			value = restoreValue
		else:
			value = self.lastValue
		self.lastValue = None
		self._paint(value, dc)

	def clear(self, dc=None):
		"""
		Clear the painter's current value from the screen and the painter
		itself using wxPython device context *dc*. Returns last formatted 
		value. 
		"""
		retval=self.lastValue
		if self.lastValue is not None:
			self._paint(None, dc)
		return retval
		
	def _paint(self, value, dc):
		"""
		Draws a previously processed *value* on this painter's window using 
		wxPython device context *dc*. 
		"""
		if dc is None:
			dc = wx.ClientDC(self.view)

		dc.SetPen(self.PEN)
		dc.SetBrush(self.BRUSH)
		dc.SetFont(self.FONT)
		dc.SetTextForeground(self.TEXT_FOREGROUND)
		dc.SetTextBackground(self.TEXT_BACKGROUND)
		dc.SetLogicalFunction(self.FUNCTION)
		dc.BeginDrawing()
		
		# Do we have last value
		if self.lastValue is not None:
			# Yes, clear the value from screen
			self.clearValue(dc, self.lastValue)
			# No value on screen
			self.lastValue = None

		# Is value not None
		if value is not None:
			# Yes, display it and store it as last value
			self.drawValue(dc, value)
			self.lastValue = value

		dc.EndDrawing()

	def formatValue(self, value):
		"""
		Template method that processes the *value* tuple passed to the
		:meth:`set` method, returning the processed version.
		"""
		return value

	def drawValue(self, dc, value):
		"""
		Template method that draws a previously processed *value* using the
		wxPython device context *dc*.  This DC has already been configured, so
		calls to the :meth:`BeginDrawing` and :meth:`EndDrawing` methods may 
		not be made.
		"""
		pass

	def clearValue(self, dc, value):
		"""
		Template method that clears a previously processed *value* that was
		previously drawn, using the wxPython device context *dc*.  This DC has
		already been configured, so calls to the :meth:`BeginDrawing` and
		:meth:`EndDrawing` methods may not be made.
		"""
		pass

class CrosshairPainter(Painter):
	"""
	Draws crosshairs through the current position of the mouse.
	
	*value* is a tuple of the form (*x*, *y*) specifying the coordinates for 
	the DC where the crosshair will be painted or erased. 
	"""

	PEN = wx.WHITE_PEN
	FUNCTION = wx.XOR
	
	def drawValue(self, dc, value):
		"""
		Draws crosshairs through the ``(X, Y)`` coordinates using wxPython 
		device context *dc*. 
		"""
		dc.CrossHair(*value)
		
	def clearValue(self, dc, value):
		"""
		Clears the crosshairs drawn through the ``(X, Y)`` coordinates using 
		wxPython device context *dc*. 
		"""
		dc.CrossHair(*value)
		

class RubberbandPainter(Painter):
	"""
	Draws a selection rubberband from one point to another.
	
	*value* is a tuple of the form (*x1*, *y1*, *x2*, *y2*) specifying the 
	position of the rubberband in Matplotlib mouse coordinates. 
	"""

	PEN = wx.WHITE_PEN
	FUNCTION = wx.XOR

	def formatValue(self, value):
		"""
		Converts the ``(x1, y1, x2, y2)`` mouse coordinates from Matplotlib to
		wxPython. Basically makes sure that x1<=x2, y1<=y2, and coordinates 
		are integer. 
		"""
		x1, y1, x2, y2 = value
		if x2 < x1: x1, x2 = x2, x1
		if y2 < y1: y1, y2 = y2, y1
		return [int(z) for z in (x1, y1, x2-x1, y2-y1)]

	def drawValue(self, dc, value):
		"""
		Draws the selection rubberband around the rectangle specified by the 
		(*x1*, *y1*, *x2*, *y2*) tuple in *value* using exPython device 
		context *dc*. 
		"""
		self.PEN=wx.Pen(wx.Colour(255,70,255))
		dc.DrawRectangle(*value)
		PEN = wx.WHITE_PEN

	def clearValue(self, dc, value):
		"""
		Clears the selection rubberband around the rectangle specified by the 
		(*x1*, *y1*, *x2*, *y2*) tuple in *value* using exPython device 
		context *dc*. 
		"""
		self.PEN=wx.Pen(wx.Colour(255,70,255))
		dc.DrawRectangle(*value)
		PEN = wx.WHITE_PEN


class CursorChanger(DestructableViewMixin):
	"""
	Manages the current cursor of a wxPython window, allowing it to be switched
	between a normal arrow (when no crosshair is plotted) and a square cross
	(when crosshair is plotted).
	"""
	def __init__(self, view, enabled=True):
		self.view = view
		self.cursor = wx.CURSOR_DEFAULT
		self.enabled = enabled

	def setEnabled(self, state):
		"""
		Enable or disable this cursor changer.  When disabled, the cursor is
		reset to the normal arrow and calls to the :meth:`set` methods have no
		effect.
		"""
		oldState, self.enabled = self.enabled, state
		if oldState and not self.enabled and self.cursor != wx.CURSOR_DEFAULT:
			self.cursor = wx.CURSOR_DEFAULT
			self.view.SetCursor(wx.STANDARD_CURSOR)

	def setNormal(self):
		"""
		Change the cursor of the associated window to a normal arrow.
		"""
		if self.cursor != wx.CURSOR_DEFAULT and self.enabled:
			self.cursor = wx.CURSOR_DEFAULT
			self.view.SetCursor(wx.STANDARD_CURSOR)

	def setCross(self):
		"""
		Change the cursor of the associated window to a square cross.
		"""
		if self.cursor != wx.CURSOR_CROSS and self.enabled:
			self.cursor = wx.CURSOR_CROSS
			self.view.SetCursor(wx.CROSS_CURSOR)


#
# Printing Framework
#

# TODO: Map print quality settings onto PostScript resolutions automatically.
#	   For now, it's set to something reasonable to work around the fact that
#	   it defaults to `72' rather than `720' under wxPython 2.4.2.4
# wx.PostScriptDC_SetResolution(300)

class FigurePrinter(DestructableViewMixin):
	"""
	Provides a simplified interface to the wxPython printing framework that's
	designed for printing Matplotlib figures.
	"""

	# Collect paper sizes
	PAPER_ID={}
	PAPER_NAME={}
	for name in dir(wx):
		id=getattr(wx, name)
		if type(id) is int and name.find("PAPER_")==0:
			paper_name=name[6:]
			PAPER_ID[paper_name]=id
			PAPER_NAME[id]=paper_name
	
	# Default margins in mm
	marginL = 5
	marginT = 5
	marginR = 5
	marginB = 5
	
	# Default scaling and aspect ratio handling
	scaleFigure=True
	keepAspect=True
	
	# Default print data
	pData=wx.PrintData()
	pData.SetPaperId(wx.PAPER_A4)
	pData.SetOrientation(wx.PORTRAIT)

	def __init__(self, view):
		"""
		Create a new :class:`FigurePrinter` associated with the wxPython widget
		*view*. 
		"""
		self.view = view
		
		# Create default print data
		if FigurePrinter.pData is None:
			FigurePrinter.pData = wx.PrintData()
			
			# Set the paper size and orientation
			FigurePrinter.pData.SetPaperId(self.paperSize)
			FigurePrinter.pData.SetOrientation(self.paperOrientation)
		
	def pageSetup(self):
		"""
		Opens a page setup dialog. Collects settings and stores them as static 
		members in the :class:`FigurePrinter` class so they become persistent 
		across plot windows. 
		"""
		
		# Create dialog
		res=page_setup_xrc.get_resources()
		dlg=res.LoadDialog(self.view, "page_setup")
		dlg.FindWindowByName('ID_OK').SetId(wx.ID_OK)
		dlg.FindWindowByName('ID_CANCEL').SetId(wx.ID_CANCEL)
		
		# Fill dialog
		paper_selector=dlg.FindWindowByName('paper_size')
		names=self.PAPER_ID.keys()
		names.sort()
		for name in names: 
			paper_selector.Append(name)
		paper_selector.SetStringSelection(self.PAPER_NAME[self.pData.GetPaperId()])
		
		if self.pData.GetOrientation() == wx.PORTRAIT:
			dlg.FindWindowByName('portrait').SetValue(True)
			dlg.FindWindowByName('landscape').SetValue(False)
		else: 
			dlg.FindWindowByName('portrait').SetValue(False)
			dlg.FindWindowByName('landscape').SetValue(True)
		
		dlg.FindWindowByName('left').SetValue(self.marginL)
		dlg.FindWindowByName('right').SetValue(self.marginR)
		dlg.FindWindowByName('top').SetValue(self.marginT)
		dlg.FindWindowByName('bottom').SetValue(self.marginB)
		
		dlg.FindWindowByName('scale_to_page').SetValue(self.scaleFigure)
		dlg.FindWindowByName('keep_aspect').SetValue(self.keepAspect)
		
		# Run dialog
		if dlg.ShowModal() == wx.ID_OK:
			# Collect data
			FigurePrinter.pData.SetPaperId(self.PAPER_ID[paper_selector.GetStringSelection()])
			
			if dlg.FindWindowByName('portrait').GetValue():
				FigurePrinter.pData.SetOrientation(wx.PORTRAIT)
			else:
				FigurePrinter.pData.SetOrientation(wx.LANDSCAPE)
				
			FigurePrinter.marginL=dlg.FindWindowByName('left').GetValue()
			FigurePrinter.marginR=dlg.FindWindowByName('right').GetValue()
			FigurePrinter.marginT=dlg.FindWindowByName('top').GetValue()
			FigurePrinter.marginB=dlg.FindWindowByName('bottom').GetValue()
			
			FigurePrinter.scaleFigure=dlg.FindWindowByName('scale_to_page').GetValue()
			FigurePrinter.keepAspect=dlg.FindWindowByName('keep_aspect').GetValue()
			
		dlg.Destroy()
		
	def previewFigure(self, figure, title=None):
		"""
		Open a "Print Preview" window for the matplotlib *figure*.  The
		keyword argument *title* provides the printing framework with a title
		for the print job.
		"""
		# Make a copy of print data in print dialog data
		data = wx.PrintDialogData(wx.PrintData(self.pData))
		printout1 = FigurePrintout(figure, "Matplotlib Figure", 
						(self.marginL, self.marginT), (self.marginR, self.marginB), 
						scaleFigure=self.scaleFigure, keepAspect=self.keepAspect
					)
		printout2 = None #TextDocPrintout(text, "title", self.margins)
		preview = wx.PrintPreview(printout1, printout2, data)
		if not preview.Ok():
			wx.MessageBox("Unable to create PrintPreview!", "Error")
		else:
			# Create the preview frame
			frame = wx.PreviewFrame(preview, self.view, "Matplotlib Figure Print Preview")
			
			# Initialize frame so that the dimensions get calculated
			frame.Initialize()
			
			# Get the canvas showing the preview
			canvas=preview.GetCanvas()
			# Get the actual size of the canvas
			canvasVirtualSize=wx.Size(*(canvas.GetVirtualSize())) # virtual size is a tuple, convert to wxSize
			# Calculate overhead between frame and canvas size
			overhead=frame.GetSize()-canvas.GetSize()
			# Calculate window size so that the whole canvas is showed
			newSize=canvasVirtualSize+overhead+wx.Size(20,20)
			# Limit it to screen size
			newSize.DecTo(wx.ScreenDC().GetSize())
			# Resize frame
			frame.SetSize(newSize)
		
			frame.Show()
		
	def printFigure(self, figure, title=None):
		"""
		Open a "Print" dialog to print the matplotlib chart *figure*.  The
		keyword argument *title* provides the printing framework with a title
		for the print job.
		"""
		# Make a copy of print data in print dialog data
		data = wx.PrintDialogData(wx.PrintData(self.pData))
		printer = wx.Printer(data)
		printout = FigurePrintout(figure, "title", 
						(self.marginL, self.marginT), (self.marginR, self.marginB), 
						scaleFigure=self.scaleFigure, keepAspect=self.keepAspect
					)
		useSetupDialog = True
		if not printer.Print(self.view, printout, useSetupDialog) and printer.GetLastError() == wx.PRINTER_ERROR:
			wx.MessageBox(
				"There was a problem printing.\n"
				"Perhaps your current printer is not set correctly?",
				"Printing Error", wx.OK
			)
		else:
			data = printer.GetPrintDialogData()
			
			# Copy PrintData
			FigurePrinter.pData = wx.PrintData(data.GetPrintData()) 
			
			# Collect paper size and orientation
			FigurePrinter.paperSize=self.pData.GetPaperId()
			FigurePrinter.paperOrientation=self.pData.GetOrientation()
		
		printout.Destroy()
		

class FigurePrintout(wx.Printout):
	"""
	Render a matplotlib :class:`Figure` object *figure* to a page or file using 
	wxPython's printing framework. *title* is used for the print job title. 
	
	*marginTL* and *marginBR* specify the top-left and the bottom-right 
	margins in the form of tuples with two members (i.e. (*left*, *top*) and 
	(*right*, *bottom*)). The margins are in millimeters. 
	
	If *scaleFigure* is ``True`` the figure is scaled to the paper size. 
	If *keepAspect* is not ``True`` the aspect ration of the figure can be 
	broken when the figure is scaled to the paper size. 
	
	The figure is rendered as a raster image. The DPI setting of the printer 
	is used for image rasterization. 
	
	Use the :meth:`print_figure` method of the :class:`PlotPanel` class 
	(inherited from its base class) for saving vector images in a vector 
	format (i.e. Postscript). 
	"""

	def __init__(self, figure, title=None, marginTL=(5,5), marginBR=(5,5), scaleFigure=True, keepAspect=True):
		self.figure = figure
		
		# Store margins in inches
		self.mLeft=marginTL[0]/25.4
		self.mTop=marginTL[1]/25.4
		self.mRight=marginBR[0]/25.4
		self.mBottom=marginBR[1]/25.4
		
		# Figure scaling
		self.scaleFigure=scaleFigure
		self.keepAspect=keepAspect

		figTitle = figure.gca().title.get_text()
		if not figTitle:
			figTitle = title or 'Matplotlib Figure'

		wx.Printout.__init__(self, figTitle)

	def GetPageInfo(self):
		"""
		Overrides wx.Printout.GetPageInfo() to provide the printing framework
		with the number of pages in this print job.
		"""
		return (1, 1, 1, 1)

	def OnPrintPage(self, pageNumber):
		"""
		Overrides wx.Printout.OnPrintPage to render the matplotlib figure to
		a printing device context.
		"""

		# Device context to draw the page
		dc = self.GetDC()

		# PPI_P: Pixels Per Inch of the Printer
		wPPI_P, hPPI_P = [float(x) for x in self.GetPPIPrinter()]
		
		# PPI: Pixels Per Inch of the DC
		if self.IsPreview():
			wPPI, hPPI = [float(x) for x in self.GetPPIScreen()]
		else:
			wPPI, hPPI = wPPI_P, hPPI_P

		# Pg_Px: Size of the page (pixels)
		wPg_Px,  hPg_Px  = [float(x) for x in self.GetPageSizePixels()]

		# Dev_Px: Size of the DC (pixels)
		wDev_Px, hDev_Px = [float(x) for x in self.GetDC().GetSize()]

		# Pg: Size of the page (inches)
		wPg = wPg_Px / wPPI_P
		hPg = hPg_Px / hPPI_P

		# Area: printable area within the margins (inches)
		wArea = wPg - self.mLeft - self.mRight
		hArea = hPg - self.mTop - self.mBottom
		
		# If area is below 10% of page size, set it to 10%
		if wArea<0.1*wPg:
			wArea=0.1*wPg
		if hArea<0.1*hPg:
			hArea=0.1*hPg

		# Get figure dimensions
		wFig=self.figure.get_figwidth()
		hFig=self.figure.get_figheight()
		
		# Figure aspect ratio
		aspectFig=wFig/hFig
		
		# Scale figure to paper size (if needed)
		if self.scaleFigure:
			if self.keepAspect:
				# Keep aspect ratio
				# Try wFig=wArea
				wFig=wArea
				hFig=wArea/aspectFig
				if hFig>hArea:
					# Try hFig=hArea
					hFig=hArea
					wFig=hArea*aspectFig
			else:
				# Fit to page
				wFig=wArea
				hFig=hArea
		else:
			# No scaling, keep original size (if possible)
			if self.keepAspect:
				# Must keep aspect ratio
				if wFig>wArea:
					# Scale down width
					sf=wArea*1.0/wFig
					wFig=wArea
					hFig=hFig*sf
				if hFig>hArea: 
					# Scale down height
					sf=hArea*1.0/hFig
					wFig=wFig*sf
					hFig=hArea
			else:
				# No need to keep aspect ratio
				if wFig>wArea:
					# Clip width
					wFig=wArea
				if hFig>hArea:
					# Clip height
					hFig=hArea

		# scale factor = device size / page size (equals 1.0 for real printing)
		wS = (wDev_Px/wPPI)/wPg
		hS = (hDev_Px/hPPI)/hPg

		# Fig_Dx: scaled printing size of the figure (device pixels)
		# M_Dx: scaled minimum margins (device pixels)
		wFig_Dx = int(wS * wPPI * wFig)
		hFig_Dx = int(hS * hPPI * hFig)
		wM_Dx = int(wS * wPPI * self.mLeft)
		hM_Dx = int(hS * hPPI * self.mTop)

		# Render figure on bitmap using PPI of the DC
		image = self.render_figure_as_image(wFig, hFig, (wPPI + hPPI)/2.0)

		# Rescale if this is the preview
		if self.IsPreview():
			image = image.Scale(wFig_Dx, hFig_Dx)
		
		# Draw bitmap
		self.GetDC().DrawBitmap(image.ConvertToBitmap(), wM_Dx, hM_Dx, False)

		return True

	# This renders an image and returns it as wxImage
	def render_figure_as_image(self, wFig, hFig, dpi):
		"""
		Renders a Matplotlib figure using the Agg backend and stores the result
		in a class:`wx.Image`.  The arguments *wFig* and *hFig* are the width 
		and the height of the figure, and *dpi* is the dots-per-inch to render 
		at.
		"""
		figure = self.figure

		# Set new DPI, width, and height in inches
		old_dpi = figure.get_dpi()
		figure.set_dpi(dpi)
		old_width = figure.get_figwidth()
		figure.set_figwidth(wFig)
		old_height = figure.get_figheight()
		figure.set_figheight(hFig)
		old_frameon = figure.get_frameon()
		figure.set_frameon(False)

		# Width and height in pixels
		wFig_Px = int(figure.bbox.width)
		hFig_Px = int(figure.bbox.height)

		# Get renderer and use it to draw the figure
		# agg = RendererAgg(wFig_Px, hFig_Px, Value(dpi))
		agg = RendererAgg(wFig_Px, hFig_Px, dpi)
		figure.draw(agg)

		# reset back to old DPI, width and height in inches
		figure.set_dpi(old_dpi)
		figure.set_figwidth(old_width)
		figure.set_figheight(old_height)
		figure.set_frameon(old_frameon)

		# Create an empty image and set it to rendered image
		image = wx.EmptyImage(wFig_Px, hFig_Px)
		image.SetData(agg.tostring_rgb())
		return image


#
# wxPython event interface for the PlotPanel and PlotFrame
#

EVT_POINT_ID = wx.NewId()
"The wxPython identifier of a :class:`PointEvent` event."

def EVT_POINT(win, id, func):
	"""
	Register to receive wxPython :class:`PointEvent` events from a 
	:class:`PlotPanel` or :class:`PlotFrame`. 
	The events originating from *win* are to be handled by function *func*. 
	
	*id* is the identifier to be associated with the event. 
	"""
	win.Connect(id, -1, EVT_POINT_ID, func)


class PointEvent(wx.PyCommandEvent):
	"""
	wxPython event emitted when a left-click-release occurs in a Matplotlib
	axes of a window without an area selection.

	The :attr:`axes` member holds the :class:`Axes` object on which the 
	click occurred. :attr:`xdata` and :attr:`ydata` are the axes 
	coordinates of the click. 
	
	*id* is the identifier to be associated with the event. 
	"""
	def __init__(self, id, axes, xdata, ydata):
		wx.PyCommandEvent.__init__(self, EVT_POINT_ID, id)
		self.axes = axes
		self.xdata = xdata
		self.ydata = ydata

	def Clone(self):
		"""
		Creates a clone of the :class:`PointEvent` object. 
		"""
		return PointEvent(self.GetId(), self.axes, self.xdata, self.ydata)


EVT_SELECTION_ID = wx.NewId()
"The wxPython identifier of a :class:`SelectionEvent` event."

def EVT_SELECTION(win, id, func):
	"""
	Register to receive wxPython :class:`SelectionEvent` events from a 
	:class:`PlotPanel` or :class:`PlotFrame`. 
	The events originating from *win* are to be handled by function *func*. 
	
	*id* is the identifier to be associated with the event. 
	"""
	win.Connect(id, -1, EVT_SELECTION_ID, func)


class SelectionEvent(wx.PyCommandEvent):
	"""
	wxPython event emitted when an area selection occurs in a Matplotlib axes
	of a window for which zooming has been enabled.  The selection is
	described by a rectangle from (*x1*, *y1*) to (*x2*, *y2*), of which only
	one point is required to be inside the axes.

	The :attr:`axes` member holds the :class:`Axes` object on which the 
	selection occurred. :attr:`x1data`, :attr:`y1data`, 
	:attr:`x2data`, and :attr:`y2data` are the axes coordinates of
	the selection. 
	
	*id* is the identifier to be associated with the event. 
	"""
	def __init__(self, id, axes, x1data, y1data, x2data, y2data):
		wx.PyCommandEvent.__init__(self, EVT_SELECTION_ID, id)
		self.axes = axes
		self.x1data = x1data
		self.y1data = y1data
		self.x2data = x2data
		self.y2data = y2data

	def Clone(self):
		"""
		Creates a clone of the :class:`SelectionEvent` object. 
		"""
		return SelectionEvent(self.GetId(), self.axes, self.x1data, self.y1data, self.x2data, self.y2data)


#
# wxPython event interface for the PlotPanel and PlotFrame
#

EVT_COORDINATES_ID = wx.NewId()
"The wxPython identifier of a :class:`CoordinatesEvent` event."

def EVT_COORDINATES(win, id, func):
	"""
	Register to receive wxPython :class:`CoordinatesEvent` events from a 
	:class:`PlotPanel` or :class:`PlotFrame`. 
	The events originating from *win* are to be handled by function *func*. 
	
	*id* is the identifier to be associated with the event. 
	"""
	win.Connect(id, -1, EVT_COORDINATES_ID, func)


class CoordinatesEvent(wx.PyCommandEvent):
	"""
	wxPython event emitted when cursor moves over axes. 
	
	The :attr:`cotype` member is a string describing the type of 
	coordinates given by :attr:`c1` and :attr:`c2` members. 
	:attr:`str` holds the formatted coordinates. 
	"""
	def __init__(self, id, cotype, c1, c2, str):
		"""
		Create a new C{CoordinatesEvent}. 
		"""
		wx.PyCommandEvent.__init__(self, EVT_COORDINATES_ID, id)
		self.cotype = cotype
		self.c1 = c1
		self.c2 = c2
		self.str = str

	def Clone(self):
		"""
		Creates a clone of the :class:`CoordinatesEvent` object. 
		"""
		return CoordinatesEvent(self.GetId(), self.cotype, self.c1, self.c2, self.str)

		
#
# wxPython close all event
#

EVT_CLOSEALL_ID = wx.NewId()
"The wxPython identifier of a :class:`CloseAll` event."

def EVT_CLOSEALL(win, id, func):
	"""
	Register to receive wxPython :class:`CloseAllEvent` events from a 
	:class:`PlotPanel` or :class:`PlotFrame`.  
	The events originating from *win* are to be handled by function *func*. 
	
	*id* is the identifier to be associated with the event. 
	"""
	win.Connect(id, -1, EVT_CLOSEALL_ID, func)


class CloseAllEvent(wx.PyCommandEvent):
	"""
	wxPython event emitted when all plot windows should be closed. 
	"""
	def __init__(self, id):
		"""
		Create a new C{CloseAllEvent}. 
		"""
		wx.PyCommandEvent.__init__(self, EVT_CLOSEALL_ID, id)

	def Clone(self):
		"""
		Creates a clone of the :class:`CloseAllEvent` object. 
		"""
		return CoordinatesEvent(self.GetId())

#
# Matplotlib canvas in a wxPython window
#

class PlotPanel(FigureCanvasWxAgg):
	"""
	A Matplotlib canvas suitable for embedding in wxPython applications. 
	
	Setting *cursor*, *crosshairs*, and *rubberband* to ``True`` enables the 
	corresponding facilities of the canvas. 
	
	By setting *point* and *selection* to ``True`` the canvas emits 
	:class:`PointEvent` and :class:`SelectionEvent` events. 
	
	Setting *zoom* to ``True`` enables zooming. 
	
	*figsize* is a tuple specifying the figure width and height in inches. 
	Together with *dpi* they define the size of the figure in pixels. 
	
	*figpx* is a tuple with the horizontal and vertical size of the figure 
	in pixels. If it is given it overrides the *figsize* setting. *dpi* is used 
	for obtaining the figure size in inches. 
	
	For *parent* and *id* see wxPython documentation. 
	
	If neither *figsize* nor *figpx* are given the settings from matplotlibrc 
	are used. The same holds for *dpi*. 
	
	Holding down the left button and moving the mouse selects the area to be 
	zoomed. The zooming is performed when the button is released. 
	
	Right-clicking zooms out to the previous view. 
	
	Pressing the ``I`` key identifies the nearest curve. 
	"""
	def __init__(self, parent, id, 
					cursor=True, crosshairs=True, rubberband=True, point=True, selection=True, zoom=True, 
					figpx=None, figsize=None, dpi=None):
					
		# If no figsize is given, use figure.figsize from matplotlibrc
		if figsize is None:
			figsize=rcParams['figure.figsize']
		
		# If no dpi is given, use figure.dpi from matplotlibrc
		if dpi is None:
			dpi=rcParams['figure.dpi']
		
		# When given, figpx overrides figsize. 
		# figsize is calculated from figpx and dpi. 
		if figpx is not None:
			figsize=(figpx[0]*1.0/dpi, figpx[1]*1.0/dpi)
		
		FigureCanvasWxAgg.__init__(self, parent, id, Figure(figsize, dpi))
		
		self.cursor = CursorChanger(self, cursor)
		self.crosshairs = CrosshairPainter(self, crosshairs)
		self.rubberband = RubberbandPainter(self, rubberband)
		self.point = point
		self.selection = selection
		self.zoom = zoom

		self.figure.set_edgecolor('black')
		self.figure.set_facecolor('white')
		self.SetBackgroundColour(wx.WHITE)
		
		# Turn on repaint
		self.repaintEnabled=True
		
		# Zoom corner 1, data and wx coordinates
		self.axes1 = None
		self.zoom1 = None
		self.point1 = None
		
		# Axes history
		self.limits=AxesLimits()
		
		# Connect matplotlib event handlers
		self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion_notify_event)
		self.figure.canvas.mpl_connect('button_press_event', self.on_button_press_event)
		self.figure.canvas.mpl_connect('button_release_event', self.on_button_release_event)
		self.figure.canvas.mpl_connect('pick_event', self.on_pick_event)
		self.figure.canvas.mpl_connect('key_press_event', self.on_key_press_event)
		
		# find the toplevel parent window and register an activation event
		# handler that is keyed to the id of this PlotPanel
		topwin = self._get_toplevel_parent()
		
		# Connect wx event handlers
		# topwin is a PlotFrame object - the toplevel window of a figure
		topwin.Connect(self.GetId(), self.GetId(), wx.wxEVT_ACTIVATE, self.OnActivate)

		wx.EVT_ERASE_BACKGROUND(self, self.OnEraseBackground)
		wx.EVT_WINDOW_DESTROY(self, self.OnDestroy)
		
	def _get_toplevel_parent(self):
		"""
		Returns the first toplevel parent of this window.
		"""
		topwin = self.GetParent()
		while not isinstance(topwin, (wx.Frame, wx.Dialog)):
			topwin = topwin.GetParent()
		return topwin	   
	
	def _to_data_coords(self, axes, x, y):
		"""
		Takes coordinates in wxWindows system and converts them to 
		axes coordinates. Returns a tuple of two values. 
		"""
		
		# No axes, nothing to do
		if axes is None:
			return (None, None)
			
		# Canvas coordinates have origin in bottom left corner
		y1 = self.figure.bbox.height - y
		
		# Convert to coordinates on axes
		try:
			xdata, ydata = axes.transData.inverted().transform_point((x, y1))
		except ValueError:
			return (None, None)
		else:
			return (xdata, ydata)

	#
	# Matplotlib event handling 
	#
	
	def on_motion_notify_event(self, event):
		"""
		A handler for matplotlib ``motion_notify_event`` events. 
		Invoked every time mouse moves across the canvas and on 
		when a mouse button is released.  
		"""
		axes=event.inaxes
		x = event.guiEvent.GetX()
		y = event.guiEvent.GetY()
		xdata = event.xdata
		ydata = event.ydata
		
		# If we are in selection mode we must draw a rubberband
		if self.axes1 is not None:
			# Yes, draw rubberband
			x0, y0 = self.point1
			self.rubberband.set(x0, y0, x, y)
			
		# If we are inside axes 
		if axes is not None:
			# Set a cross cursor and draw crosshairs
			self.cursor.setCross()
			self.crosshairs.set(x, y)			
			
			# We post coordinates events
			wx.PostEvent(self, 
				CoordinatesEvent(self.GetId(), 
					axes.name, 
					xdata, ydata, 
					axes.format_coord(xdata, ydata)
				)
			)
		else:
			# Outside axes
			# If there is any rubberband, it remains where it was
			
			# Normal cursor, no crosshairs
			self.cursor.setNormal()
			self.crosshairs.clear()
			
			# Post coordinates event
			wx.PostEvent(self, CoordinatesEvent(self.GetId(), 'No axes', None, None, 'unknown'))
			
	def on_button_press_event(self, event):
		"""
		A handler for matplotlib ``button_press_event`` events. 
		Invoked every time a mouse button is pressed. 
		"""
		axes=event.inaxes
		x = event.guiEvent.GetX()
		y = event.guiEvent.GetY()
		xdata = event.xdata
		ydata = event.ydata
		
		if event.button==1:
			# Left button pressed
			
			# Change cursor
			self.cursor.setCross()
			
			# Are we inside axes
			if axes is not None:
				# Are the axes rectilinear
				if axes.name=='rectilinear':
					# OK, we have zoom point 1
					self.axes1 = axes
					self.zoom1 = xdata, ydata
					self.point1 = x, y
		elif event.button==3:
			# Right button pressed and zooming enabled
			if axes is not None:
				if self.zoom and self.limits.restore(axes):
					# We have axes and zoom out requires a redraw
					self.forceDraw()
	
	def on_button_release_event(self, event):
		"""
		A handler for matplotlib ``button_release_event`` events. 
		Invoked every time a mouse button is released. 
		"""
		
		axes=event.inaxes
		x = event.guiEvent.GetX()
		y = event.guiEvent.GetY()
		xdata = event.xdata
		ydata = event.ydata
		
		if event.button==1:
			# Left button released
			
			# If we are in selection mode, clear rubberband
			if self.axes1 is not None:
				self.rubberband.clear()
			
			# Calculate second point coordinates from x,y based on self.axes1
			# This way we get to zoom beyond axes. 
			actualxdata, actualydata = self._to_data_coords(self.axes1, x, y)
			
			# Are we in selection mode and do we have a second point
			if self.axes1 is not None and actualxdata is not None and actualydata is not None:
				# Prepare ranges
				xrange=self.zoom1[0], actualxdata
				yrange=self.zoom1[1], actualydata
					
				# Fix ranges
				if xrange[0]>xrange[1]:
					xrange=xrange[-1::-1]
				if yrange[0]>yrange[1]:
					yrange=yrange[-1::-1]
					
				# Is the range nonzero?
				if xrange[1]-xrange[0]>0 and yrange[1]-yrange[0]>0:
					# Yes, it is
					# Emit a selection event if selection events are enabled
					if self.selection: 
						wx.PostEvent(self, SelectionEvent(self.GetId(), self.axes1, xrange[0], yrange[0], xrange[1], yrange[1]))
						
					# Are coordinates rectilinear and is zoom allowed. 
					if self.axes1.name=='rectilinear' and self.zoom:
						# Yes, zoom ... 
						if self.limits.set(self.axes1, xrange, yrange):
							# ... and redraw if needed
							self.forceDraw()
				else:
					# We have no range, just a point
					# Emit a point event if point events are enabled
					if self.point:
						wx.PostEvent(self, PointEvent(self.GetId(), self.axes1, xdata, ydata))
					
			# Reset zoom point 1, leave selection mode
			self.axes1 = None
			self.zoom1 = None
			self.point1 = None
			
		# Normal cursor if no axes
		if axes is None:
			self.cursor.setNormal()
		else:
			# Cross cursor if we have axes
			self.cursor.setCross()
	
	def on_key_press_event(self, event):
		"""
		A handler for matplotlib ``key_press_event`` events. 
		Invoked every time a key is pressed. 
		"""
		if event.key=='i':
			# Identification
			if event.inaxes is not None:
				# Tell artists to fire pick events
				event.inaxes.pick(event)
				
	def on_pick_event(self, event):
		"""
		A handler for matplotlib ``pick_event`` events. 
		Invoked every time user picks a location close to some object. 
		"""
		self.SetToolTipString("'"+event.artist.get_label()+"'")
	
	def _onPaint(self, evt):
		"""
		Overrides the :class:`FigureCanvasWxAgg` paint event to redraw the
		crosshairs, etc.
		"""
		
		# We have now 
		dc = wx.ClientDC(self)
		if self.repaintEnabled:
			# Clear the crosshairs and the rubberband from the whole area
			# This will damage the area that will be repainted, but after repaint
			# it will be OK. 
			ch=self.crosshairs.clear(dc)
			rb=self.rubberband.clear(dc)
			del dc
			
			FigureCanvasWxAgg._onPaint(self, evt)
			# At this point the figure should be complete clear of any crosshairs or rubberband
		
			# Now paint the crosshairs and the rubberband on the whole area
			dc = wx.ClientDC(self)
			self.crosshairs.redraw(dc, ch)
			self.rubberband.redraw(dc, rb)
			del dc
		
		dc = wx.PaintDC(self)

	#
	# wxPython event handling 
	#
	
	def OnActivate(self, evt):
		"""
		Handles the wxPython window activation event.
		"""
		
		active=evt.GetActive()
		if not active:
			# On deactivation make cursor normal
			self.cursor.setNormal()
		
		evt.Skip()

	def OnEraseBackground(self, evt):
		"""
		Overrides the wxPython backround repainting event to reduce flicker.
		"""
		pass

	def OnDestroy(self, evt):
		"""
		Handles the wxPython window destruction event.
		"""
		
		if self.GetId() == evt.GetEventObject().GetId():
			objects = [self.cursor, self.rubberband, self.crosshairs]
			for obj in objects:
				obj.destroy()

			# unregister the activation event handler for this PlotPanel
			topwin = self._get_toplevel_parent()
			topwin.Disconnect(-1, self.GetId(), wx.wxEVT_ACTIVATE)

	#
	# Force drawing 
	#
	
	def forceDraw(self):
		"""
		Forces the drawing of the associated :class:`Figure` onto the canvas. 
		"""
		if self.repaintEnabled:
			dc = wx.ClientDC(self)
			ch=self.crosshairs.clear(dc)
			rb=self.rubberband.clear(dc)
			del dc
			
			self.draw()
			
			dc = wx.ClientDC(self)
			self.crosshairs.redraw(dc, ch)
			self.rubberband.redraw(dc, rb)
			del dc
	
	#
	# Getters and setters
	#
	
	def get_figure(self):
		"""
		Returns the figure associated with this canvas.
		"""
		return self.figure
		
	def set_cursor(self, state):
		"""
		Enable or disable the changing mouse cursor.  When enabled, the cursor
		changes from the normal arrow to a square cross when the mouse enters a
		Matplotlib axes on this canvas.
		"""
		self.cursor.setEnabled(state)

	def set_crosshairs(self, state):
		"""
		Enable or disable drawing crosshairs through the mouse cursor when it
		is inside a matplotlib axes.
		"""
		self.crosshairs.setEnabled(state)
	
	def set_rubberband(self, state):
		"""
		Enable or disable drawing a rubberband. 
		"""
		self.rubberband.setEnabled(state)

	def set_point(self, state):
		"""
		Enable or disable point events.
		"""
		self.point = state
		
	def set_selection(self, state):
		"""
		Enable or disable selection events. 
		"""
		self.selection = state

	def set_zoom(self, state):
		"""
		Enable or disable zooming in/out when the user makes an area selection 
		or right-clicks the axes. 
		"""
		self.zoom=state
	
	def set_repaint(self, state):
		"""
		Enable or disable repainting. 
		"""
		self.repaintEnabled=state

#
# A class derived from PyEvtHandler that overrides the EventHandler
#

class ProtectedEvtHandler(wx.PyEvtHandler):
	"""
	An event handler object that wraps the EventHandler of a wxPython object 
	and inserts a :class:`Lock` that prevents other threads from accessing 
	Matplotlib objects while wxPython events are handled. 
	
	*lock* is the :class:`Lock` object. 
	
	*obj* is the wxPython object whose events are going to be intercepted. 
	"""
	def __init__(self, lock, obj):
		wx.PyEvtHandler.__init__(self)
		self.lock=lock
		self.obj=obj
		
	def ProcessEvent(self, evt):
		"""
		Wrapper for the EventHandler. 
		"""
		if self.lock:
			self.lock.acquire()
		retval=self.obj.ProcessEvent(evt)
		if self.lock:
			self.lock.release()
		return retval

		
#
# Matplotlib canvas in a top-level wxPython window
#

class PlotFrame(wx.Frame):
	"""
	A matplotlib canvas embedded in a wxPython window.

	*title* is the title of the window. See wxPython documentation for *parent* 
	and *id*. 
	
	*lock* is a :class:`Lock` object that prevents other threads from accessing 
	Matplotlib objects while wxPython events are handled. It is passed to the 
	:class:`PlotPanel`. 
	
	All remaining arguments are passed to the :class:`PlotPanel` constructor. 
	"""
	def __init__(self, parent, id, title, lock, **kwargs):
		wx.Frame.__init__(self, parent, id, title)
		# We are the PlotPanel's parent, id is chosen automatically
		# kwargs go to PlotPanel
		self.panel = PlotPanel(self, -1, **kwargs)

		# Push a new event handler that wraps ProcesEvent and inserts a lock 
		# thereby preventing other threads from accessing Matplotlib objects 
		# while wxPython events are processed. 
		self.peh=ProtectedEvtHandler(lock, self.panel)
		self.panel.SetEventHandler(self.peh)
		
		# Get the calculated panel size
		calculatedSize=self.panel.GetSize()
		
		self.CreateStatusBar()
		self.GetStatusBar().SetFieldsCount(1)
		self.GetStatusBar().SetStatusWidths([-1])
		
		self.printer = FigurePrinter(self)

		self.create_menus()
		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(self.panel, 1, wx.ALL|wx.EXPAND, 0)
		self.SetSizer(sizer)
		self.Fit()
		
		# Resize the panel back to its original size
		# Now the panelSize and panelDPI will result in the actual panel panelPixels
		overhead=self.GetSize()-self.panel.GetSize()
		self.SetSize(calculatedSize+overhead)

		wx.EVT_WINDOW_DESTROY(self, self.OnDestroy)
		
		# Make OnCoordinates handle CoordinateEvent events originating from 
		# the PlotPanel object. 
		EVT_COORDINATES(self.panel, self.panel.GetId(), self.OnCoordinates)
	
	ABOUT_TITLE = 'About wxmpl.PlotFrame'
	ABOUT_MESSAGE = ('wxmpl.PlotFrame %s\n' %  __version__
		+ 'Written by Ken McIvor <mcivor@iit.edu>\n'
		+ 'Copyright 2005 Illinois Institute of Technology'
		+ '\n'
		+ 'Adapted by Arpad Buermen'
		+ 'Copyright 2011 Arpad Buermen')
	
	def create_menus(self):
		"""
		Creates the main menu of the window. 
		"""
		mainMenu = wx.MenuBar()
		menu = wx.Menu()

		id = wx.NewId()
		menu.Append(id, '&Save As...\tCtrl+S',
			'Save a copy of the current plot')
		wx.EVT_MENU(self, id, self.OnMenuFileSave)

		# Printing under OSX doesn't work well because the DPI of the
		# printer is always reported as 72.  It will be disabled until print
		# qualities are mapped onto wx.PostScriptDC resolutions.

		if not sys.platform.startswith('darwin'):
			menu.AppendSeparator()

			id = wx.NewId()
			menu.Append(id, 'Page Set&up...',
				'Set the size and margins of the printed figure')
			wx.EVT_MENU(self, id, self.OnMenuFilePageSetup)

			id = wx.NewId()
			menu.Append(id, 'Print Pre&view...',
				'Preview the print version of the current plot')
			wx.EVT_MENU(self, id, self.OnMenuFilePrintPreview)

			id = wx.NewId()
			menu.Append(id, '&Print...\tCtrl+P', 'Print the current plot')
			wx.EVT_MENU(self, id, self.OnMenuFilePrint)

		menu.AppendSeparator()

		id = wx.NewId()
		menu.Append(id, '&Close Window\tCtrl+W',
			'Close the current plot window')
		wx.EVT_MENU(self, id, self.OnMenuFileClose)
		
		id = wx.NewId()
		menu.Append(id, 'Close All Windows',
			'Close the all plot windows')
		wx.EVT_MENU(self, id, self.OnMenuFileCloseAll)

		mainMenu.Append(menu, '&File')
		menu = wx.Menu()

		id = wx.NewId()
		menu.Append(id, '&About...', 'Display version information')
		wx.EVT_MENU(self, id, self.OnMenuHelpAbout)

		mainMenu.Append(menu, '&Help')
		self.SetMenuBar(mainMenu)

	def PrintCoordinates(self, text):
		"""
		Displays the formatted coordinates given by *text* in the status bar. 
		"""
		self.GetStatusBar().SetStatusText(text)
		
	def OnCoordinates(self, evt):
		"""
		Handler for :class:`CoordinatesEvent` events coming from the 
		:class:`PlotPanel`. 
		"""
		self.GetStatusBar().SetStatusText(
				"%s: %s" % (evt.cotype, evt.str)
		)
		
	def OnDestroy(self, evt):
		"""
		Handler for wxPython window destruction events. 
		"""
		if self.GetId() == evt.GetEventObject().GetId():
			self.printer.destroy()

	def OnMenuFileSave(self, evt):
		"""
		Handles File->Save menu events.
		"""
		# Build list of supported formats
		wildcards=''
		for extension, description in FigureCanvasWxAgg.filetypes.iteritems():
			if len(wildcards)>0:
				wildcards+='|'
			wildcards+=description+' (*.'+extension+')|*.'+extension
		
		fileName = wx.FileSelector('Save Plot', default_extension='png',
			wildcard=wildcards, 
			parent=self, flags=wx.SAVE|wx.OVERWRITE_PROMPT)

		if not fileName:
			return

		path, ext = os.path.splitext(fileName)
		ext = ext[1:].lower()

		# figpx (figsize*dpi) is used for raster images
		# figsize and dpi are used for postscript and pdf
		try:
			self.panel.print_figure(fileName)
		except IOError, e:
			if e.strerror:
				err = e.strerror
			else:
				err = e

			wx.MessageBox('Could not save file: %s' % err, 'Error - plotit', parent=self, style=wx.OK|wx.ICON_ERROR)

	def OnMenuFilePageSetup(self, evt):
		"""
		Handles File->Page Setup menu events
		"""
		self.printer.pageSetup()

	def OnMenuFilePrintPreview(self, evt):
		"""
		Handles File->Print Preview menu events
		"""
		self.printer.previewFigure(self.get_figure())

	def OnMenuFilePrint(self, evt):
		"""
		Handles File->Print menu events
		"""
		self.printer.printFigure(self.get_figure())

	def OnMenuFileClose(self, evt):
		"""
		Handles File->Close menu events.
		"""
		self.Close()
	
	def OnMenuFileCloseAll(self, evt):
		"""
		Handles File->Close All menu events. 
		"""
		wx.PostEvent(self, CloseAllEvent(self.GetId()))

	def OnMenuHelpAbout(self, evt):
		"""
		Handles Help->About menu events.
		"""
		wx.MessageBox(self.ABOUT_MESSAGE, self.ABOUT_TITLE, parent=self,
			style=wx.OK)

	def get_figure(self):
		"""
		Returns the figure associated with this canvas.
		"""
		return self.panel.get_figure()
	
	def get_panel(self):
		"""
		Returns the plot panel object associated with this frame and its canvas.  
		"""
		return self.panel

	def draw(self):
		"""
		Draw the associated :class:`Figure` onto the screen. 
		Shortcut to the PlotPanel's :meth:`forceDraw` method. 
		"""
		self.panel.forceDraw()
