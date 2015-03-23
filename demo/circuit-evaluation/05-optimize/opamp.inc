* Simple opamp, 0.18u, 1.8V, SpiceOpus

* Any include or lib that is set from cirsimlib (specified in problem definition) 
* must not be specified here. 

* Include and lib files - set from cirsimlib
* .lib 'cmos180n.lib' tt

* Include files - fixed
.include 'mosmm.inc'

* Any parameter that is set from circimlib (specified in problem definition) 
* must not be specified here. 

* Operating conditions - set from cirsimlib
* .param lev1=0
* .param lev2=0.5
* .param tstart=1e-9
* .param tr=1e-9
* .param tf=1e-9
* .param pw=500e-9
* .param vdd=1.800000000000e+000 
* .option temp=2.500000000000e+001 
* .param rfb=1e6
* .param rin=1e6

* Design parameters - set from cirsimlib
* .param mirr_w=7.455884130590e-005
* .param mirr_l=5.627763640504e-007
* .param out_w=4.800592541419e-005
* .param out_l=3.750131780858e-007
* .param load_w=3.486243671853e-005
* .param load_l=2.572996921261e-006
* .param dif_w=7.728734451428e-006
* .param dif_l=1.082371380389e-006

* Operating conditions - fixed
.param ibias=1.000000000000e-004 

* Design parameters - fixed
.param c_out=8.211596855053e-012
.param r_out=1.968986740568e+001

* inp inn out vdd vss bias slp slpx
.subckt amp 3 4 5 1 2 7 11 12
xmp1 9 10 1 1  submodp w={load_w} l={load_l} m=2
xmp2 10 10 1 1 submodp w={load_w} l={load_l} m=2
xmp1s 9 12 1 1 submodp w=1u        l=0.5u
xmp3 5 9 1 1   submodp w={out_w}   l={out_l}   m=16
xmn2 9 3 8 2   submodn w={dif_w}   l={dif_l} 
xmn3 10 4 8 2  submodn w={dif_w}   l={dif_l} 
xmn1s 7 11 2 2 submodn w=1u        l=0.5u
xmn1b 7 7 2 2  submodn w={mirr_w}  l={mirr_l}  m=2
xmn1 8 7 2 2   submodn w={mirr_w}  l={mirr_l}  m=2
xmn4 5 7 2 2   submodn w={mirr_w}  l={mirr_l}  m=16
cout 5a 9 c={c_out}
rout 5 5a r={r_out}
.ends









