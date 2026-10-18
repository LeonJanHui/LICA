# Full model Track-C attention — 4j3l
reinitialize
bg_color white
load /home/dldx/Boltz-EGNN-Affinity/results/interpretability/trackc_response/4j3l/full/4j3l_attention.pdb, cplx
hide everything, cplx
show sticks, polymer and cplx
color gray70, polymer and cplx
set stick_radius, 0.12, polymer and cplx
show cartoon, polymer and cplx
set cartoon_transparency, 0.8
color gray85, polymer and cplx and cartoon
show sticks, resn LIG and cplx
util.cnc resn LIG and cplx
set stick_radius, 0.20, resn LIG and cplx
select attn_atoms, cplx and b > 5
show sticks, attn_atoms
hide spheres, cplx
set stick_radius, 0.18, attn_atoms and polymer
spectrum b, blue_white_red, (attn_atoms or (resn LIG)), minimum=0, maximum=100
color gray70, polymer and cplx and not (b > 5)
color gray85, polymer and cplx and cartoon
set ray_opaque_background, on
set antialias, 2
orient (resn LIG) or attn_atoms
zoom (resn LIG) or attn_atoms, 8
hide labels, cplx
distance tc00, cplx and id 393, cplx and id 33
set dash_width, 6.00, tc00
set dash_color, red, tc00
hide labels, tc00
distance tc01, cplx and id 379, cplx and id 33
set dash_width, 4.72, tc01
set dash_color, red, tc01
hide labels, tc01
distance tc02, cplx and id 396, cplx and id 270
set dash_width, 4.13, tc02
set dash_color, tv_orange, tc02
hide labels, tc02
distance tc03, cplx and id 385, cplx and id 117
set dash_width, 3.18, tc03
set dash_color, tv_orange, tc03
hide labels, tc03
distance tc04, cplx and id 392, cplx and id 249
set dash_width, 2.88, tc04
set dash_color, tv_orange, tc04
hide labels, tc04
distance tc05, cplx and id 396, cplx and id 299
set dash_width, 2.80, tc05
set dash_color, tv_orange, tc05
hide labels, tc05
distance tc06, cplx and id 393, cplx and id 28
set dash_width, 2.64, tc06
set dash_color, salmon, tc06
hide labels, tc06
distance tc07, cplx and id 396, cplx and id 302
set dash_width, 2.46, tc07
set dash_color, salmon, tc07
hide labels, tc07
distance tc08, cplx and id 386, cplx and id 309
set dash_width, 2.41, tc08
set dash_color, salmon, tc08
hide labels, tc08
distance tc09, cplx and id 393, cplx and id 26
set dash_width, 2.23, tc09
set dash_color, salmon, tc09
hide labels, tc09
distance tc10, cplx and id 392, cplx and id 22
set dash_width, 2.11, tc10
set dash_color, salmon, tc10
hide labels, tc10
distance tc11, cplx and id 390, cplx and id 181
set dash_width, 2.05, tc11
set dash_color, salmon, tc11
hide labels, tc11
distance tc12, cplx and id 396, cplx and id 300
set dash_width, 1.99, tc12
set dash_color, salmon, tc12
hide labels, tc12
distance tc13, cplx and id 393, cplx and id 32
set dash_width, 1.94, tc13
set dash_color, salmon, tc13
hide labels, tc13
distance tc14, cplx and id 375, cplx and id 107
set dash_width, 1.82, tc14
set dash_color, salmon, tc14
hide labels, tc14
distance tc15, cplx and id 378, cplx and id 278
set dash_width, 1.79, tc15
set dash_color, salmon, tc15
hide labels, tc15
distance tc16, cplx and id 394, cplx and id 33
set dash_width, 1.76, tc16
set dash_color, salmon, tc16
hide labels, tc16
distance tc17, cplx and id 379, cplx and id 28
set dash_width, 1.73, tc17
set dash_color, salmon, tc17
hide labels, tc17
select hi_res, cplx and id 30+31+32+33+20+21+22+23+24+25+26+27+28+29+267+268+269+270+271+272+273+274+275+276+277+278+302+303+304+305+306+307+308+309
show sticks, hi_res
set stick_radius, 0.22, hi_res
spectrum b, blue_white_red, hi_res, minimum=0, maximum=100
pseudoatom lab0, pos=[-4.659,-14.211,-15.468]
label lab0, "Gly1032"
hide nonbonded, lab0
set label_size, 11, lab0
set label_color, black, lab0
pseudoatom lab1, pos=[-4.569,-9.328,-17.043]
label lab1, "His1031"
hide nonbonded, lab1
set label_size, 11, lab1
set label_color, black, lab1
pseudoatom lab2, pos=[3.179,-19.232,-13.976]
label lab2, "Tyr1071"
hide nonbonded, lab2
set label_size, 11, lab2
set label_color, black, lab2
pseudoatom lab3, pos=[6.839,-19.819,-5.228]
label lab3, "Ile1075"
hide nonbonded, lab3
set label_size, 11, lab3
set label_color, black, lab3
# hi-weight residues in PLIP: ['A:GLY1032', 'A:TYR1071', 'A:ILE1075']
png /home/dldx/Boltz-EGNN-Affinity/results/interpretability/trackc_response/4j3l/full/4j3l_attention.png, dpi=300, ray=1, width=2000, height=1500
# quit
