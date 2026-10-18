# Full model Track-C attention — 4tmn
reinitialize
bg_color white
load /home/dldx/Boltz-EGNN-Affinity/results/interpretability/trackc_response/4tmn/full/4tmn_attention.pdb, cplx
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
distance tc00, cplx and id 538, cplx and id 525
set dash_width, 6.00, tc00
set dash_color, red, tc00
hide labels, tc00
distance tc01, cplx and id 536, cplx and id 281
set dash_width, 4.57, tc01
set dash_color, red, tc01
hide labels, tc01
distance tc02, cplx and id 556, cplx and id 39
set dash_width, 4.29, tc02
set dash_color, tv_orange, tc02
hide labels, tc02
distance tc03, cplx and id 540, cplx and id 525
set dash_width, 4.07, tc03
set dash_color, tv_orange, tc03
hide labels, tc03
distance tc04, cplx and id 537, cplx and id 44
set dash_width, 3.35, tc04
set dash_color, tv_orange, tc04
hide labels, tc04
distance tc05, cplx and id 538, cplx and id 44
set dash_width, 3.11, tc05
set dash_color, tv_orange, tc05
hide labels, tc05
distance tc06, cplx and id 529, cplx and id 226
set dash_width, 3.00, tc06
set dash_color, tv_orange, tc06
hide labels, tc06
distance tc07, cplx and id 529, cplx and id 227
set dash_width, 2.89, tc07
set dash_color, tv_orange, tc07
hide labels, tc07
distance tc08, cplx and id 548, cplx and id 39
set dash_width, 2.74, tc08
set dash_color, tv_orange, tc08
hide labels, tc08
distance tc09, cplx and id 549, cplx and id 39
set dash_width, 2.65, tc09
set dash_color, salmon, tc09
hide labels, tc09
distance tc10, cplx and id 552, cplx and id 39
set dash_width, 2.54, tc10
set dash_color, salmon, tc10
hide labels, tc10
distance tc11, cplx and id 529, cplx and id 228
set dash_width, 2.41, tc11
set dash_color, salmon, tc11
hide labels, tc11
distance tc12, cplx and id 543, cplx and id 40
set dash_width, 2.39, tc12
set dash_color, salmon, tc12
hide labels, tc12
distance tc13, cplx and id 535, cplx and id 480
set dash_width, 2.26, tc13
set dash_color, salmon, tc13
hide labels, tc13
distance tc14, cplx and id 529, cplx and id 229
set dash_width, 2.22, tc14
set dash_color, salmon, tc14
hide labels, tc14
distance tc15, cplx and id 539, cplx and id 310
set dash_width, 2.05, tc15
set dash_color, salmon, tc15
hide labels, tc15
distance tc16, cplx and id 552, cplx and id 204
set dash_width, 2.04, tc16
set dash_color, salmon, tc16
hide labels, tc16
distance tc17, cplx and id 543, cplx and id 38
set dash_width, 1.99, tc17
set dash_color, salmon, tc17
hide labels, tc17
distance tc18, cplx and id 538, cplx and id 203
set dash_width, 1.92, tc18
set dash_color, salmon, tc18
hide labels, tc18
distance tc19, cplx and id 538, cplx and id 204
set dash_width, 1.88, tc19
set dash_color, salmon, tc19
hide labels, tc19
distance tc20, cplx and id 543, cplx and id 39
set dash_width, 1.84, tc20
set dash_color, salmon, tc20
hide labels, tc20
distance tc21, cplx and id 556, cplx and id 421
set dash_width, 1.81, tc21
set dash_color, salmon, tc21
hide labels, tc21
distance tc22, cplx and id 532, cplx and id 60
set dash_width, 1.78, tc22
set dash_color, salmon, tc22
hide labels, tc22
distance tc23, cplx and id 541, cplx and id 421
set dash_width, 1.76, tc23
set dash_color, salmon, tc23
hide labels, tc23
distance tc24, cplx and id 550, cplx and id 421
set dash_width, 1.73, tc24
set dash_color, salmon, tc24
hide labels, tc24
select hi_res, cplx and id 33+34+35+36+37+38+39+40+525+220+221+222+223+224+225+226+227+228+229+41+42+43+44+45
show sticks, hi_res
set stick_radius, 0.22, hi_res
spectrum b, blue_white_red, hi_res, minimum=0, maximum=100
pseudoatom lab0, pos=[46.059,14.788,1.264]
label lab0, "Asn112"
hide nonbonded, lab0
set label_size, 11, lab0
set label_color, black, lab0
pseudoatom lab1, pos=[57.774,22.595,-7.066]
label lab1, "Zn5"
hide nonbonded, lab1
set label_size, 11, lab1
set label_color, black, lab1
pseudoatom lab2, pos=[54.600,28.538,-10.369]
label lab2, "His146"
hide nonbonded, lab2
set label_size, 11, lab2
set label_color, black, lab2
pseudoatom lab3, pos=[46.232,20.709,0.840]
label lab3, "Ala113"
hide nonbonded, lab3
set label_size, 11, lab3
set label_color, black, lab3
# hi-weight residues in PLIP: ['E:ASN112']
png /home/dldx/Boltz-EGNN-Affinity/results/interpretability/trackc_response/4tmn/full/4tmn_attention.png, dpi=300, ray=1, width=2000, height=1500
# quit
