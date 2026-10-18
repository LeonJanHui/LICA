# w/o Track C (Track A+B atom attention) — 4j3l
reinitialize
bg_color white
load /home/dldx/Boltz-EGNN-Affinity/results/interpretability/trackc_response/4j3l/wo_track_c/4j3l_attention.pdb, cplx
hide everything, cplx
show sticks, polymer and cplx
color gray70, polymer and cplx
set stick_radius, 0.12, polymer and cplx
show cartoon, polymer and cplx
set cartoon_transparency, 0.8
color gray85, polymer and cplx and cartoon
show sticks, resn LIG and cplx
util.cnc resn LIG and cplx
select attn_atoms, cplx and b > 5
show sticks, attn_atoms
hide spheres, cplx
spectrum b, blue_white_red, (attn_atoms or (resn LIG)), minimum=0, maximum=100
color gray70, polymer and cplx and not (b > 5)
orient (resn LIG) or attn_atoms
zoom (resn LIG) or attn_atoms, 8
png /home/dldx/Boltz-EGNN-Affinity/results/interpretability/trackc_response/4j3l/wo_track_c/4j3l_attention.png, dpi=300, ray=1, width=2000, height=1500
# quit
