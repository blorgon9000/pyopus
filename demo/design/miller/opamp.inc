* Miller OTA, 0.18u, 1.8V, SpiceOpus

* Any include or lib that is set from pyopus (specified in problem definition) 
* must not be specified here. 

* Include files - fixed
.include 'mosmm.inc'

* Any parameter that is set from pyopus (specified in problem definition) 
* must not be specified here. 

* inp inn out vdd vss bias slp slpx
.subckt amp 3 4 5 1 2 7 11 12
xmp1 9 10 1 1  submodp w={load_w}  l={load_l}  m=1  vtmm={p1vt}  u0mm={p1u0}
xmp2 10 10 1 1 submodp w={load_w}  l={load_l}  m=1  vtmm={p2vt}  u0mm={p2u0}
xmp1s 9 12 1 1 submodp w=1u        l=0.5u           
xmp3 5 9 1 1   submodp w={out_w}   l={out_l}   m=2  vtmm={p3vt}  u0mm={p3u0}
xmn2 9 3 8 2   submodn w={dif_w}   l={dif_l}        vtmm={n2vt}  u0mm={n2u0}
xmn3 10 4 8 2  submodn w={dif_w}   l={dif_l}        vtmm={n3vt}  u0mm={n3u0}
xmn1s 7 11 2 2 submodn w=1u        l=0.5u           
xmn1b 7 7 2 2  submodn w={mirr_w}  l={mirr_l}  m=1  vtmm={n1bvt} u0mm={n1bu0}
xmn1 8 7 2 2   submodn w={mirr_wd} l={mirr_ld} m=1  vtmm={n1vt}  u0mm={n1u0}
xmn4 5 7 2 2   submodn w={mirr_wo} l={mirr_l}  m=1  vtmm={n4vt}  u0mm={n4u0}
cout 5a 9 c={c_out}
rout 5 5a r={r_out}
.ends
