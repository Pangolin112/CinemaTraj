# Vendored GenDoP

Subset of [nerfstudio-project GenDoP](https://github.com/3DTopia/GenDoP)
(Zhang et al., ICCV 2025) vendored for CinemaTraj's `no_parametric` ablation,
which uses `core.models.LMM` through
`baselines/ChatCam_GenDoP/cinegpt_wrapper.py`.

Removed from the upstream tree because CinemaTraj does not use them:
`dataset/` (DataDoP construction tooling), `evaluate/`,
`Blender_visualization/`, `assets/`, `outputs/`.

Model weights are not redistributed — see the repository README for the
download link. Upstream license is kept in `LICENSE`.
