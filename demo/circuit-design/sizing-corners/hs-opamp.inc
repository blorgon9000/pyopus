* Simple opamp, 0.18u, 1.8V

.include hs-mosmm.inc

*RangeParam
*.param vdd=1.800000000000e+000 
*.param ibias=1.000000000000e-004 
** .temp 2.500000000000e+001 

*StatParam
.param mp1vtn=    0.000000000000e+000 
.param mp1u0=    0.000000000000e+000 
.param mp2vtn=    0.000000000000e+000 
.param mp2u0=    0.000000000000e+000 
.param mp3vtn=    0.000000000000e+000 
.param mp3u0=    0.000000000000e+000 
.param mn1vtn=    0.000000000000e+000 
.param mn1u0=    0.000000000000e+000 
* .param mn2vtn=    0.000000000000e+000 
.param mn2u0=    0.000000000000e+000 
* .param mn3vtn=    0.000000000000e+000 
.param mn3u0=    0.000000000000e+000 
.param mn4vtn=    0.000000000000e+000 
.param mn4u0=    0.000000000000e+000 
.param mn1bvtn=    0.000000000000e+000 
.param mn1bu0=    0.000000000000e+000 

*DesignParam
* .param mirr_w=7.455884130590e-005
* .param mirr_l=5.627763640504e-007
* .param out_w=4.800592541419e-005
* .param out_l=3.750131780858e-007
* .param load_w=3.486243671853e-005
* .param load_l=2.572996921261e-006
* .param dif_w=7.728734451428e-006
* .param dif_l=1.082371380389e-006
* .param c_out=8.211596855053e-012
* .param r_out=1.968986740568e+001

* inp inn out vdd vss bias slp slpx
.subckt amp 3 4 5 1 2 7 11 12
xmp1 9 10 1 1  submodp w=load_w l=load_l m=2  u0mm=mp1u0   vtmm=mp1vtn
xmp2 10 10 1 1 submodp w=load_w l=load_l m=2  u0mm=mp2u0   vtmm=mp2vtn
xmp1s 9 12 1 1 submodp w=1u l=0.5u
xmp3 5 9 1 1   submodp w=out_w   l=out_l   m=16 vtmm=mp3vtn  u0mm=mp3u0    
xmn2 9 3 8 2   submodn w=dif_w   l=dif_l        vtmm=mn2vtn  u0mm=mn2u0
xmn3 10 4 8 2  submodn w=dif_w   l=dif_l        vtmm=mn3vtn  u0mm=mn3u0
xmn1s 7 11 2 2 submodn w=1u l=0.5u
xmn1b 7 7 2 2  submodn w=mirr_w  l=mirr_l  m=2  vtmm=mn1bvtn u0mm=mn1bu0 
xmn1 8 7 2 2   submodn w=mirr_w  l=mirr_l  m=2  vtmm=mn1vtn  u0mm=mn1u0
xmn4 5 7 2 2   submodn w=mirr_w  l=mirr_l  m=16 vtmm=mn4vtn  u0mm=mn4u0
cout 5a 9 c_out
rout 5 5a r_out
.ends


* inp inn out vdd vss bias slp slpx
.subckt ampref 3 4 5 1 2 7 11 12
xmp1 9 10 1 1  submodp w=load_w l=load_l m=2
xmp2 10 10 1 1 submodp w=load_w l=load_l m=2
xmp1s 9 12 1 1 submodp w=1u      l=0.5u
xmp3 5 9 1 1   submodp w=out_w   l=out_l   m=16
xmn2 9 3 8 2   submodn w=dif_w   l=dif_l 
xmn3 10 4 8 2  submodn w=dif_w   l=dif_l 
xmn1s 7 11 2 2 submodn w=1u      l=0.5u
xmn1b 7 7 2 2  submodn w=mirr_w  l=mirr_l  m=2
xmn1 8 7 2 2   submodn w=mirr_w  l=mirr_l  m=2
xmn4 5 7 2 2   submodn w=mirr_w  l=mirr_l  m=16
cout 5a 9 c_out
rout 5 5a r_out
.ends
