* Test topology - dc, ac, and tran with feedback

x1 (inp inn out vdd 0 bias) fcoa
vdd (vdd 0) dc={vdd}
ibias (bias 0) dc={ibias}
rfb (out1 inn) r={rfb}
efb (out1 0 out 0) gain=1
rin (in inn) r={rin}
vcom (inp 0) dc={vdd/2}
vin (in inp) dc=0 acmag=1 pulse=({lev1} {lev2} {tstart} {tr} {tf} {pw})
rload (out 0) r={rload}
cload (out 0) c={cload}

