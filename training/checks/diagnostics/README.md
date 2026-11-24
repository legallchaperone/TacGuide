# Evaluation diagnostics (2026-11-24)

One-off analysis scripts kept for reference; paths inside point at
`/data2/cwx/TacGuide/logs/eval_debug` and the two policy servers on ports 8000 (gc_B) and 8001 (gc_A).

Findings they support:

1. Render mismatch. Without DLSS (missing NGX library on the sim host) camera images are about 7 grey
   levels brighter and 20% sharper than the public dataset, with identical geometry (`geom.py`,
   `variants_png.py`, `collect_vs_eval.py`). On a training scene this moves the predicted grasp height by
   up to 27 mm (`domain.py`); brightness, blur and h264 emulation do not fix it (`sens.py`, `domain2.py`,
   `domain3.py`, `perturb.py`). With DLSS active the difference drops to about 2.5 levels and the
   prediction shift to 4 mm and below (`cmp_sources.py`).
2. Start-up check. At step 0 both tactile images must show exactly 63 markers (`tacref.py`).
3. Per-chunk offset. On frames where the demonstration is at rest the models predict a constant offset
   of about 1 mrad on joints 1, 4 and 7 (about -0.4 mm in flange x per chunk), independent of chunk
   index, noise, precision and number of denoising steps (`bias.py`, `bias2.py`, `cpu_bias.py`). With
   re-planning every 20 steps and no state input it accumulates to a 5 to 8 mm miss (`c2.py`, `c3.py`,
   `miss_table.py`).
4. Execution is exact: commanded and achieved joints agree to 0.0 mrad in free motion (`track.py`), and
   the scripted expert grasps exactly above the chip centre at flange z of about 145 mm (`expert.py`,
   `fk_cmp.py`).
