# SRT Workbench · 可微分 SRT 冠层辐射传输工作台

可微分随机辐射传输（SRT / TopoSRT）冠层模型的在线演示：正演、梯度反演与
敏感性/可辨识性分析。

- JAX 正演与 Fortran 参考模型对齐到 BRF **~1e-5**（球/柱/锥/椭球冠层、LAD 1–9、平坦/坡面）
- 反演：LAI、覆盖度、H（可选高斯先验）
- 敏感性：Jacobian / Fisher / 条件数

本演示只运行 JAX 引擎；Fortran 参考程序不在本空间分发。

源码与文档见 GitHub：<https://github.com/Gardenias-123/srt-workbench>
