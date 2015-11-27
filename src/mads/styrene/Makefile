CC=gcc
CPP=g++
INCLUDES=
CCOPTS=-fPIC -DLINUX -O3 $(INCLUDES) -Wall
AR=ar
RANLIB=ranlib
LIBNAME=sty
OBJECTS=bb.o burner.o cashflow.o chemical.o \
	column.o combrx.o flash.o heatx.o mix.o pfr.o \
	profitability.o pump.o reaction.o \
	servor.o split.o stream.o \
	thermolib.o

all: library

library: $(LIBNAME).a

%.o: %.cpp
	$(CXX) $(CCOPTS) $(INCLUDES) -c $< -o $@

$(LIBNAME).a: $(OBJECTS)
	$(AR) rs $(LIBNAME).a $(OBJECTS)

clean:
	rm -f *.o *.a
	