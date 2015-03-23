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
	
	# Plot window
	f1=pyopl.figure(windowTitle="Subplots", figpx=(600,600), dpi=100)
	
	# Lock GUI
	pyopl.lock(True)
	
	# Check if figure is alive
	if pyopl.alive(f1):
		# Create 2 subplots, vertical stack of 2
		ax1=f1.add_subplot(2,1,1)
		ax2=f1.add_subplot(2,1,2)
	
		# First axes
		ax1.hold(True)
		ax1.plot(x, y1, '-o', label='sin(x)', color=(1,0,0))
		ax1.plot(x, 2*y1, '-o', label='2sin(x)', color=(1,0,0))
		ax1.grid(True)
		ax1.set_xlabel('x')
		ax1.set_ylabel('sin(x), 2sin(x)')
		# Titles on vertically stacked axes are a bad idea, because they overlap with x-axis labels. 
		# ax1.set_title('Axes 1')
		ax1.hold(False)
		
		# Second axes
		ax2.hold(True)
		ax2.plot(x, y1, '-', label='sin(x)', color=(1,0,0))
		ax2.plot(x, y2, '-', label='cos(x)', color=(0,1,0))
		ax2.grid(True)
		ax2.set_xlabel('x')
		ax2.set_ylabel('sin(x), cos(x)')
		# ax2.set_title('Axes 2')
		ax2.hold(False)
		
		# Delete axes 
		# f1.delaxes(ax2)
		
		# Clear axes 
		# ax1.clear()
		
		# Draw figure on screen
		pyopl.draw(f1)
		
	# Unlock GUI
	pyopl.lock(False)
	
	# Handle keyboard interrupts properly. 
	pyopl.join()