"""
**Parallel computation support module**

In PyOPUS a parallel execution environment is abstracted in form of a 
**virtual machine**. Virtual machines are composed of **hosts** which run 
**tasks**. Tasks communicate with each other using basic virtual machine 
communication facilities. A **message** is an abstraction of one of these 
facilities which provides sending and receiving data between two hosts. 

Messages are used at higher levels of abstraction. See 
the :mod:`pyopus.parallel.cooperative` module which is a cooperative 
multitasking OS capable of dispatching jobs to available computing nodes. 
It can run a parallel algorithms locally or on a virtual machine via 
e.g. MPI. Parallel algorithms are described in a UNIX-like manner 
with spawn/join system calls. The OS also provides a simple asynchronous 
job dispatching facility. 

Nothing from this module's submodules is imported into the main 
:mod:`parallel` module.
"""

# Export only portable stuff.
__all__ = [ ]
