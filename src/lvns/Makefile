CC=gcc
INCLUDES=-I/usr/include/python2.7 \
	-I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/ \
	-I../libf2c
CCOPTS=-fPIC -DLINUX -g
LINKER=gcc
LINKOPTS=-shared
LINKLIBS=../libf2c/libf2c.a -lm
LIBNAME=_lvns
OBJECTS=lvns.o lvlcmm.o lvumm.o lvuns.o
INSTALLDIR=../../pyopus/problems

all: library 

library: $(LIBNAME).so

%.o: %.c
	$(CC) $(CCOPTS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CCOPTS) $(INCLUDES) -c $< -o $@

$(LIBNAME).so: $(OBJECTS)
	$(LINKER) $(LINKOPTS) -o $@ $(OBJECTS) $(LINKLIBS)
	cp $@ $(INSTALLDIR)

clean:
	rm -f *.o *.so
