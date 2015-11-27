* Folded cascode OTA, 0.18u, 1.8V, SpiceOpus

* Any include or lib that is set from pyopus (specified in problem definition) 
* must not be specified here. 

* Include files - fixed
.include 'mosmm.inc'

* Any parameter that is set from pyopus (specified in problem definition) 
* must not be specified here. 

* inp inn out vdd vss bias
.subckt fcoa 3 4 5 1 2 6 
xp24 6  6  8  8  submodp w={pm_w0} l={pm_l} vtmm={vt_p24} u0mm={u0_p24}
xp20 7  6  9  9  submodp w={pm_w2} l={pm_l} vtmm={vt_p20} u0mm={u0_p20}
xp23 8  8  1  1  submodp w={pm_w1} l={pm_l} vtmm={vt_p23} u0mm={u0_p23}
xp22 9  8  1  1  submodp w={pm_w2} l={pm_l} vtmm={vt_p22} u0mm={u0_p22}

xn15 7  7  2  2  submodn w={nm_w}  l={nm_l} vtmm={vt_n15} u0mm={u0_n15}
xn9  10 7  2  2  submodn w={nm_w}  l={nm_l} vtmm={vt_n9}  u0mm={u0_n9}

xn13 11 3  10 2  submodn w={dif_w} l={dif_l} m=4 vtmm={vt_n13} u0mm={u0_n13}
xn12 12 4  10 2  submodn w={dif_w} l={dif_l} m=4 vtmm={vt_n12} u0mm={u0_n12}

xp16 20 6  11 11 submodp w={pm_w3} l={pm_l} m=4 vtmm={vt_p16} u0mm={u0_p16}
xp15 5  6  12 12 submodp w={pm_w3} l={pm_l} m=4 vtmm={vt_p15} u0mm={u0_p15}
xp18 11 8  1  1  submodp w={pm_w3} l={pm_l} m=4 vtmm={vt_p18} u0mm={u0_p18}
xp17 12 8  1  1  submodp w={pm_w3} l={pm_l} m=4 vtmm={vt_p17} u0mm={u0_p17}

xn11 20 20 21 2  submodn w={nl_w}  l={nl_l} vtmm={vt_n11} u0mm={u0_n11}
xn14 5  20 22 2  submodn w={nl_w}  l={nl_l} vtmm={vt_n14} u0mm={u0_n14}
xn10 21 21 2  2  submodn w={nl_w}  l={nl_l} vtmm={vt_n10} u0mm={u0_n10}
xn8  22 21 2  2  submodn w={nl_w}  l={nl_l} vtmm={vt_n8}  u0mm={u0_n8}
.ends
