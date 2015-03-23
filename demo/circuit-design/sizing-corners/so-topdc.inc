* Test topology

x1 (inp inn out vdd 0 bias 0 vdd) amp
vsrc (vdd 0) dc={vdd}
ibias (vdd bias) dc={ibias}
rfb (out inn) r={rfb}
rin (in inn) r={rin}
vcom (inp 0) dc={vdd/2}
vin (in inp) dc=0 acmag=1 pulse=({lev1} {lev2} {tstart} {tr} {tf} {pw})
rload (out 0) r=100k
cload (out 0) c=0.5p
