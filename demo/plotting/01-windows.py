import pyopus.wxmplplot as pyopl

if __name__ == '__main__':
	# Initialize gui thread. The thread exits when control window is closed. 
	pyopl.init()
	
	# If the GUI thread was running before init(), 
	# there might be some plot windows around. 
	# Close them. 
	pyopl.close()
	
	# Create a figure with default size (matplotlibrc). 
	f0=pyopl.figure()
	print("pyopl.figure() returned: "+str(f0))
	
	# Create a figure with 400x300 pixels, 100dpi. This gives a 4x3 inch image. 
	f1=pyopl.figure(windowTitle="Window title 1", figpx=(400,300), dpi=100)	
	print("pyopl.figure() returned: "+str(f1))
	
	# Change window title. 
	pyopl.title(f1, "Changed window title 2")
	
	# Close a figure. 
	# pyopl.close(f1)
	
	# Close all figures
	# pyopl.close()
	
	# Show/hide (True/False) figure 
	# pyopl.showFigure(f0, False)
	# pyopl.showFigure(f0, True)
	
	# Raise figure / active figure 
	# pyopl.raiseFigure(f0)
	
	# Wait for the user to close the GUI Control Window. 
	pyopl.join()