# Full model Track-C attention — 2qbr
reinitialize
bg_color white
load /home/dldx/Boltz-EGNN-Affinity/results/interpretability/trackc_response/2qbr/full/2qbr_attention.pdb, cplx
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
distance tc00, cplx and id 371, cplx and id 51
set dash_width, 6.00, tc00
set dash_color, red, tc00
hide labels, tc00
distance tc01, cplx and id 378, cplx and id 344
set dash_width, 3.05, tc01
set dash_color, tv_orange, tc01
hide labels, tc01
distance tc02, cplx and id 389, cplx and id 320
set dash_width, 2.68, tc02
set dash_color, tv_orange, tc02
hide labels, tc02
distance tc03, cplx and id 388, cplx and id 317
set dash_width, 2.42, tc03
set dash_color, salmon, tc03
hide labels, tc03
distance tc04, cplx and id 388, cplx and id 66
set dash_width, 2.33, tc04
set dash_color, salmon, tc04
hide labels, tc04
distance tc05, cplx and id 378, cplx and id 256
set dash_width, 2.25, tc05
set dash_color, salmon, tc05
hide labels, tc05
distance tc06, cplx and id 387, cplx and id 66
set dash_width, 2.16, tc06
set dash_color, salmon, tc06
hide labels, tc06
distance tc07, cplx and id 364, cplx and id 51
set dash_width, 1.99, tc07
set dash_color, salmon, tc07
hide labels, tc07
distance tc08, cplx and id 378, cplx and id 345
set dash_width, 1.92, tc08
set dash_color, salmon, tc08
hide labels, tc08
distance tc09, cplx and id 390, cplx and id 320
set dash_width, 1.87, tc09
set dash_color, salmon, tc09
hide labels, tc09
distance tc10, cplx and id 364, cplx and id 206
set dash_width, 1.85, tc10
set dash_color, salmon, tc10
hide labels, tc10
distance tc11, cplx and id 369, cplx and id 51
set dash_width, 1.77, tc11
set dash_color, salmon, tc11
hide labels, tc11
distance tc12, cplx and id 364, cplx and id 50
set dash_width, 1.74, tc12
set dash_color, salmon, tc12
hide labels, tc12
distance tc13, cplx and id 378, cplx and id 244
set dash_width, 1.73, tc13
set dash_color, salmon, tc13
hide labels, tc13
distance tc14, cplx and id 382, cplx and id 261
set dash_width, 1.69, tc14
set dash_color, salmon, tc14
hide labels, tc14
distance tc15, cplx and id 389, cplx and id 319
set dash_width, 1.64, tc15
set dash_color, salmon, tc15
hide labels, tc15
select hi_res, cplx and id 40+41+42+43+44+45+46+47+48+49+50+51+319+320+321+322+339+340+341+342+343+344+345+346+347+63+64+65+66+67+68+69+70
show sticks, hi_res
set stick_radius, 0.22, hi_res
spectrum b, blue_white_red, hi_res, minimum=0, maximum=100
pseudoatom lab0, pos=[36.969,10.106,0.851]
label lab0, "Tyr46"
hide nonbonded, lab0
set label_size, 11, lab0
set label_color, black, lab0
pseudoatom lab1, pos=[52.516,7.057,11.270]
label lab1, "Gly259"
hide nonbonded, lab1
set label_size, 11, lab1
set label_color, black, lab1
pseudoatom lab2, pos=[55.984,13.572,5.839]
label lab2, "Gln262"
hide nonbonded, lab2
set label_size, 11, lab2
set label_color, black, lab2
pseudoatom lab3, pos=[43.328,1.503,-0.673]
label lab3, "Asp48"
hide nonbonded, lab3
set label_size, 11, lab3
set label_color, black, lab3
# hi-weight residues in PLIP: ['A:TYR46', 'A:GLN262', 'A:ASP48']
png /home/dldx/Boltz-EGNN-Affinity/results/interpretability/trackc_response/2qbr/full/2qbr_attention.png, dpi=300, ray=1, width=2000, height=1500
# quit
