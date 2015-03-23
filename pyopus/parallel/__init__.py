"""
**Parallel computation support module**

In PyOPUS a parallel execution environment is abstracted in form of a 
**virtual machine**. Virtual machines are composed of **hosts** which run 
**tasks**. Tasks communicate with each other using basic virtual machine 
communication facilities. A **message** is an abstraction of one of these 
facilities which provides sending and receiving data between two hosts. 

Messages are used at higher levels of abstraction. See 
:mod:`pyopus.parallel.evtdrvms`. 

Nothing from this module's submodules is imported into the main 
:mod:`parallel` module.
"""

# Export only portable stuff.
__all__ = [ ]
