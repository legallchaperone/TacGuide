"""TacGuide additions on top of the bundled OpenVTLA training code.

distill.py        distillation / extrapolation loss (experiments C and D)
guided.py         guided inference, used for CFG(A, B) and zero-training extrapolation
configs.py        experiment names (gc_A, gc_B, gc_C, gc_D_w*) -> training configs
train_distill.py  train C or D
serve_guided.py   policy server for guided inference
gpu_queue.py      shared-GPU job scheduler (standard library only)
"""
