import pyopus.wxmplplot as pyopl
from numpy import arange, sin, cos, exp, pi, e

if __name__ == '__main__':
	# Initialize gui thread, clean up. 
	pyopl.init()
	pyopl.close()

	# Plot data - sin(x), cos(x), exp(x/pi) .. for x in [0, 2pi] with 0.2 step. 
	x = arange(0.1, 100.0, 0.2)
	y1 = x**0.8 
	y2 = x 
	y3 = x**1.2 
	
	# Plot window
	f1=pyopl.figure(windowTitle="Log plots", figpx=(600,600), dpi=100)
	
	# Lock GUI
	pyopl.lock(True)
	
	# Check if figure is alive
	if pyopl.alive(f1):
		# Create 4 subplots, 2x2
		ax1=f1.add_subplot(2,2,1)
		ax2=f1.add_subplot(2,2,2)
		ax3=f1.add_subplot(2,2,3)
		ax4=f1.add_subplot(2,2,4)
		
		# First axes
		ax1.hold(True)
		ax1.plot(x, y1, '-', label='x**0.8')
		ax1.plot(x, y2, '-', label='x')
		ax1.plot(x, y3, '-', label='x**1.2')
		ax1.grid(True)
		ax1.hold(False)
		
		# Second axes
		ax2.hold(True)
		ax2.semilogx(x, y1, '-', label='x**0.8')
		ax2.semilogx(x, y2, '-', label='x')
		ax2.semilogx(x, y3, '-', label='x**1.2')
		ax2.grid(True)
		ax2.hold(False)

		# Third axes
		ax3.hold(True)
		ax3.semilogy(x, y1, '-', label='x**0.8')
		ax3.semilogy(x, y2, '-', label='x')
		ax3.semilogy(x, y3, '-', label='x**1.2')
		ax3.grid(True)
		ax3.hold(False)

		# Fourth axes
		ax4.hold(True)
		ax4.loglog(x, y1, '-', label='x**0.8')
		ax4.loglog(x, y2, '-', label='x')
		ax4.loglog(x, y3, '-', label='x**1.2')
		# ax4.set_xscale('linear') # Change x axis scale
		# ax4.set_yscale('log')	   # Change y axis scale
		ax4.grid(True)
		ax4.hold(False)

		# Figure title
		f1.suptitle("Log plots")
		
		# Draw figure on screen
		pyopl.draw(f1)
		
	# Unlock GUI
	pyopl.lock(False)
	
	# Handle keyboard interrupts properly. 
	pyopl.join()