import pyopus.wxmplplot as pyopl
from numpy import arange, sin, cos, exp, pi, e

if __name__ == '__main__':
	# Initialize gui thread, clean up. 
	pyopl.init()
	pyopl.close()

	# Plot data - sin(x), cos(x), exp(x/pi) .. for x in [0, 2pi] with 0.2 step. 
	t = arange(0.0, 10*pi, 0.2)
	r = 100/(10+t)
	y = sin(t)*r
	x = cos(t)*r
	
	# Plot window
	f1=pyopl.figure(windowTitle="xy and polar plot", figpx=(800,400), dpi=100)
	
	# Lock GUI
	pyopl.lock(True)
	
	# Check if figure is alive
	if pyopl.alive(f1):
		# Create 2 subplots, horizontal stack of 2
		ax1=f1.add_subplot(1,2,1, aspect='equal')	# Equal aspect ratio
		ax2=f1.add_subplot(1,2,2, aspect='equal', projection='polar')	# Equal aspect ratio, polar plot
		
		# First axes
		ax1.hold(True)
		ax1.plot(x, y, '-', label='sin(x)', color=(1,0,0))
		ax1.grid(True)
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_title('Equal aspect ratio, x-y plot')
		ax1.hold(False)
		
		# Second axes
		ax2.hold(True)
		ax2.plot(t, r, '-', label='r=r(t)', color=(1,0,0))
		ax2.grid(True)
		
		# Get r limits
		r1=ax2.get_rmin()
		r2=ax2.get_rmax()
		print("R limits: %f..%f" % (r1, r2))
		# Reduce rmax by 20%, set rmin to 5.0. The latter means that only 
		# points with r>=5 are plotted. Points with r=5 are at the origin. 
		ax2.set_rmin(5.0)
		ax2.set_rmax(r2*0.8)
		ax2.set_title('Equal aspect ratio, polar plot r=r(t)')
		
		ax2.hold(False)
		
		# Draw figure on screen
		pyopl.draw(f1)
		
	# Unlock GUI
	pyopl.lock(False)
	
	# Handle keyboard interrupts properly. 
	pyopl.join()