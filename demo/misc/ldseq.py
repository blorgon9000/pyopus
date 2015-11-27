# Low discrepancy Halton/Sobol sequence for generating uniformly distributed points on a hypersphere

import numpy as np
import pyopus.wxmplplot as pyopl
import pyopus.misc.ghalton as ghalton
import pyopus.misc.sobol as sobol

if __name__=='__main__':
	# Number of points
	N=100
	
	# Dimension of space (plotting makes sense only with 2)
	n=2
	
	# Round n to next greater or equal even number (nn)
	nn=int(np.ceil(n/2)*2)
	
	# Halton/Sobol sequence generator
	gen=ghalton.Halton(nn)
	# gen=sobol.Sobol(nn)
	
	# Skip first nn entries
	gen.get(nn)
	
	# Initialize gui thread, clean up.
	pyopl.init()
	pyopl.close()
	
	# Create first figure (plot window). This is now the active figure.
	f1=pyopl.figure(windowTitle="Random points inside a disc", figpx=(600,400), dpi=100)
	
	# Lock GUI
	pyopl.lock(True)
	
	# Check if figure is alive
	if pyopl.alive(f1):
		ax1=f1.add_axes((0.12,0.12,0.76,0.76))
		
		ax1.hold(True)

		ii=0
		rskips=0
		mskips=0
		while ii<150:
			# Get vector of length nn
			x=np.array(gen.get(1)[0])
			
			# Generate normal random numbers
			
			# Marsaglia method
			#x2k=x[::2]*2-1.0
			#x2k1=x[1::2]*2-1.0
			#s=x2k**2+x2k1**2
			#if (s>=1.0).any():
			#	mskips+=1
			#	continue
			#y2k=x2k*(-2*np.log(s)/s)
			#y2k1=x2k1*(-2*np.log(s)/s)
			
			# Box Mueller method (nn normal random numbers)
			x2k=x[::2]*2
			x2k1=x[1::2]*2
			s=-2*np.log(x2k)
			y2k=s*np.cos(2*np.pi*x2k1)
			y2k1=s*np.sin(2*np.pi*x2k1)
			
			# Join them in a vector, truncate its length to n
			xn=x.copy()
			xn[::2]=y2k
			xn[1::2]=y2k1
			xn=xn[:n]
			
			# Random point on a unit sphere
			r=(xn**2).sum()**0.5
			if (r<1e-7).any():
				rskips+=1
				continue
			xs=xn/r
			
			# Red for Halton sequence
			# Green for normal random numbers
			# Blue for points on a hypersphere
			ax1.plot(x[0], x[1], 'o', color=(1,0,0))
			ax1.plot(xn[0], xn[1], 'o', color=(0,1,0))
			ax1.plot(xs[0], xs[1], 'o', color=(0,0,1))
			
			ii+=1
		
		if mskips>0:
			print "# of skipped points due to s>=1   :", mskips
		if rskips>0:
			print "# of skipped points due to small r:", rskips
		
		ax1.set_xlim(-1.2, 1.2)
		ax1.set_ylim(-1.2, 1.2)
		
		ax1.grid(True)
		
		ax1.hold(False)
		
		# Draw figure on screen
		pyopl.draw(f1)
		
		# Unlock GUI
		pyopl.lock(False)
		
		# Handle keyboard interrupts properly.
		pyopl.join()
	
	