* Test topology - rejection ratios

x1 (inp inn out vdd vss bias) fcoa
vdd (vdd 0) dc={vdd} ac={acvdd}
vss (vss 0) dc=0 ac={acvss}
ibias (vdd bias) dc={ibias}
rfb (out1 inn) r={rfb}
efb (out1 0 out 0) gain=1
cin (inn com) c={cin}
vcom (com 0) dc={vdd/2} ac={accom}
vdif (inp com) dc=0 ac={acdif}
rload (out 0) r={rload}
cload (out 0) c={cload}
