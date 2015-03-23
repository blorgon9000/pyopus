import pyopus.wxmplplot as pyopl
from numpy import arange, sin, cos, exp, pi, e

if __name__ == '__main__':
	# Initialize gui thread, clean up. 
	pyopl.init()
	pyopl.close()

	# Plot data - sin(x), cos(x), exp(x/pi) .. for x in [0, 2pi] with 0.2 step. 
	x = arange(0.0, 2*pi, 0.2)
	y1 = sin(x)
	y2 = cos(x)
	y3 = exp(x/pi)

	# Create figure. Tag is assigned automatically by the system. Do not show it. 
	f1=pyopl.figure(windowTitle="Figure - single axes", show=False, figpx=(600,400), dpi=100)	
	
	# Lock GUI
	pyopl.lock(True)
	
	# Check if figure is alive
	if pyopl.alive(f1):
		# Create axes in active figure. 
		ax1=f1.gca()
		
		# Add traces, legend, labels, axes title, grid, figure title
		ax1.plot(x, y1, '-o', label='sin(x)', color=(1,0,0))
		ax1.hold(True)
		ax1.plot(x, y2, 'rx', label='cos(x)')
		ax1.plot(x, y3, '--k|', label='exp(x/pi)')
		ax1.legend()
		ax1.set_xlabel('x-axis')
		ax1.set_ylabel('y-axis')
		ax1.set_title('Axes title. ')
		ax1.grid(True)
		ax1.hold(False)
		
		f1.suptitle("Figure title")
		
		# Draw figure on screen
		pyopl.draw(f1)
		
	# Unlock GUI
	pyopl.lock(False)
	
	# Figure is still invisible, save it as png. 
	# Size will be the figure size in pixels given by figpx 
	# (if not given, calculated from figsize and dpi). 
	pyopl.saveFigure(f1, "demo.png")
	
	# Save it as eps
	# Size will be figure size in inches given by figsize
	# (if not given, calculated from figpx and dpi). 
	pyopl.saveFigure(f1, "demo.eps")
	
	# Now show and raise figure
	pyopl.showFigure(f1, True)
	pyopl.raiseFigure(f1)
		
	# Save again
	pyopl.saveFigure(f1, "demo-visible.png")
	pyopl.saveFigure(f1, "demo-visible.eps")
		
	# Close figure
	pyopl.close(f1)
	
	# Handle keyboard interrupts properly. 
	pyopl.join()
