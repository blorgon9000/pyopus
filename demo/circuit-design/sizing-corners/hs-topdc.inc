* Test topology

x1 inp inn out vdd 0 bias 0 vdd amp
vsrc vdd 0 dc=vdd
ibias vdd bias dc=ibias
rfb out inn rfb
rin in inn rin
vcom inp 0 dc='vdd/2'
vin in inp dc=0 ac=1 pulse (lev1 lev2 tstart tr tf pw)
rload out 0 100k
cload out 0 0.5p
