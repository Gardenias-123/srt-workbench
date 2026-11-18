# 完整 JAX SRT 移植：验证报告

> 2026-11-18。对应 Fortran 参考：`SRT_diag`（生产 `SRT_topo` 的副本 + 只读诊断输出），
> 标准案例：`i0=1, SZA=45°, SAA=270°, TreeType=4（球冠）, H=5, d_L=1,
> cover=0.49397, hot_spot=8, τ_L=0.4843, ρ_L=0.4746, ρ_soil=0.3379, nz=20`。

## 1. 结论

`diff_srt/ref_srt.py`（NumPy 逐行对照实现）与 `diff_srt/jax_srt.py`（JAX 可微实现）
已经复现 Fortran 完整 SRT 链路，包括 **四项散射矩阵** `ScatDD/DU/UD/UU`：

| 量（天底） | Fortran | JAX | 差 |
|---|---:|---:|---:|
| BS_dir BRF | 0.19209 | 0.192109 | 1.9e-5 |
| S 问题 BRF | 0.74666 | 0.746630 | 3.0e-5 |
| S 问题底部下行通量（R_S） | 0.28667 | 0.286671 | 1e-6 |
| BS_dir 透过率 T_BS | 0.68982 | 0.689825 | 5e-6 |
| **总 BRF（含土壤多次反射）** | **0.38480** | **0.384809** | **9e-6** |

全 20 个观测角：

| 情形 | max abs diff | RMSE |
|---|---:|---:|
| i0 = 1（纯直射） | 8.9e-6 | 5.4e-6 |
| i0 = 0（纯漫射） | 1.1e-5 | 5.6e-6 |

对生产 golden `data/reference/cases/base/brf.csv`：max diff 8.9e-6。

## 2. 之前 S 问题偏高 3 倍的原因

`get_fk_lin` / `get_intpS_lin` 的散射源是 **两项之和**：

```fortran
xxD = xxD + (ScatUD(m1,m)*etaU + ScatDD(m1,m)*etaD)*w0(m1)
xxU = xxU + (ScatUU(m1,m)*etaU + ScatDU(m1,m)*etaD)*w0(m1)
```

旧 `fortran_style.py` 用 `where(mu_q>0, etaD, etaU)` 只保留一支，等价于丢掉
“上行辐射向下散射 / 下行辐射向上散射”的项。修正后 S 问题、总 BRF 同时对齐。

顺带修正的三个实现问题：

1. `EQN2PI` 的方位角分块步长是 `π/2`（Fortran 里局部变量 `pi=acos(-1)/2`），
   不是 `π`；方位角公式的分母是 `w-2i`（0 基）而非 `w-2i+2`。
2. Volterra 对角步使用 `(1-e^{-xx})/xx` 本身，而 j 循环内用 `1-(1-e^{-xx})/xx`。
3. `get_f0_Fi0` / `get_Idif` 上行部分的层索引（`ih`、`jh`）有 0/1 基错位；
   `ScatUD/ScatUU` 的元组顺序曾经接反（`DD,DU,UD,UU` vs `DD,DU,UU,UD`）。

## 3. 诊断数据

`SRT_diag/test.f` 在每一轮、每一步、每次迭代都 dump 中间量到
`SRT_diag/000output/DIAG_r{1,2,3}_*.txt`（`dump1r/dump2r/dumpmatr/dump2ri` 在
`SRT_diag/diag_dump.f`）。这套数据让 NumPy 对照实现可以逐函数对齐，而不是只看
最终 BRF。重新生成：

```bash
cd SRT_diag
PATH="/d/tools/msys64/mingw64/bin:$PATH" \
  gfortran -static -std=legacy -fallow-argument-mismatch -O2 -o exe_diag.exe *.f
./exe_diag.exe
```

## 4. 数值说明

Fortran 全程 `real(4)`。对齐残差（约 1e-5 BRF）主要来自单精度求和顺序；把
JAX 切到 float32 与 float64 得到的 BRF 相同（0.38391078 vs 0.38391089），
说明这 **不是** 精度问题，而是 Fortran 输出的 5 位小数 + 单精度舍入。

## 5. 可微性

`jax.grad` 对 `H, cover, d_L, ρ_L, τ_L` 全部有限且与有限差分一致
（H 的有限差分在步长 1e-3~1e-2 时收敛到 0.1309，与自动微分一致；小步长因
`cond_pcf` 的分段判据而噪声较大，属正常）。

## 5.1 坡面（球冠）

坡面链路全部对齐（`slope`/`aspect` 通过 `slope_geometry` 变换到坡面系，
`par_pcf(11)=slope` 进入 `cond_pcf` 的 lam，球冠按 `cover×cos(slope)` 缩放，
最后直射部分乘 `cos(SZA_T)/cos(SZA)` 回到平面系）。

| 案例（生产 `data/reference/cases`） | max abs diff | RMSE |
|---|---:|---:|
| 平坦（golden） | 6.7e-6 | 3.7e-6 |
| slope15 aspect0 | 1.0e-5 | 3.9e-6 |
| slope30 aspect0 | 7.9e-6 | 3.4e-6 |
| slope45 aspect0 | 5.0e-6 | 2.8e-6 |
| slope15 aspect180 | 6.8e-6 | 4.3e-6 |
| slope15 aspect270 | 1.1e-5 | 5.6e-6 |
| 850 nm slope15 aspect270（26 角） | 9.4e-6 | 5.4e-6 |

`SRT_diag_slope15` 逐量对照：K/kkH 4.3e-7，散射矩阵 3e-8，用户方向场量 ~1e-7，
BRF 1e-5。

**实现细节**：`cond_pcf` 的分支判据（`lam <= rz2` 等）在冠层相切处是
measure-zero 边沿，Fortran 的 `real(4)` 舍入决定走哪支；JAX 里将
`lam/rz1/rz2` 先舍入到 float32 再做判据（算术仍用 float64，保证自动微分稳定），
否则单条方向会出现 O(1) 的 K 偏差并放大成 ~1e-4 的 BRF 误差。

## 5.2 柱冠（TreeType=1）

`cond_pcf` case(1)：`lam` 与坡面因子同上，另加
`d = sqrt(1 - cos²(fi) sin²(alpha)) * lam / R` 与两圆重叠的 kappa；
圆柱 case 只检查上边界（没有 `|H-z|` 项），p(z)=常数 cover。
非球冠的坡面规则：`H×cos(slope)`、cover 不变、hot_spot×cos(slope)。

| 案例 | max abs diff | RMSE |
|---|---:|---:|
| 平坦柱冠（SRT_diag_cyl_flat） | 9.6e-6 | — |
| slope15 aspect0 柱冠（SRT_diag_cyl15） | 8.5e-6 | 4.8e-6 |

逐量对照：K/kkH 6.4e-7，BRF 8.5e-6。

## 5.3 锥冠 / 椭球冠 / 非球面 LAD

新增 `cond_pcf` case(2) 圆锥与 case(3) 椭球（`aratio=diam0/H·cos(AlphaSlope)`，
注意椭球用的是直径、圆柱用的是半径；case 2 只检查上边界，case 3 检查上下边界），
以及 `get_G_fun` / `get_APF` 的 LAD 1–5 数值积分和 LAD 7/8/9 的解析式。
LAD 1–5 的面积相位函数对 (ρ,τ) 线性，几何矩阵在主机侧预计算一次后作为常数
传入 JAX（否则 JAX 图里会生成 217×217×3600 的张量）。

| 案例（slope15, aspect0） | max abs diff | RMSE |
|---|---:|---:|
| 锥冠（TreeType=2） | 1.0e-5 | 4.1e-6 |
| 椭球冠（TreeType=3） | 1.2e-5 | 5.1e-6 |
| 球冠 + planophile LAD（iorien=1） | 7.1e-6 | 4.2e-6 |
| 球冠 + vertical LAD（iorien=8） | 9.5e-6 | 4.1e-6 |

逐量对照：锥/椭 K 1e-6，LAD1/8 的 sigma ~2e-6、散射矩阵 ~1e-7。
LAD8 在热点方向曾差 0.7%（BRF），原因是 Fortran 的 `gind` 用 `real(4)` 方位角
做 `|φ1-φ|>π` 的折返判据；用 float32 复现太阳方位角的 roundtrip 后对齐到 1e-5。

## 6. 当前边界

- 支持 **球/柱/锥/椭冠 × LAD 1–9 × 平坦/坡面**。
  非球冠的坡面规则为 `H×cos(slope)`、cover 不变、`hot_spot×cos(slope)`。
- `slope`/`aspect` 在完整模型里走主机侧几何变换，因此只能**固定**，不能作为
  反演自由参数；需要反演坡度时仍用 reduced 模型。
- 217 方向 Carlson 求积，JIT 编译约 26 s（前向）/ 2 min（梯度），之后单次前向 ~1 ms。
- GUI 的 JAX 引擎在 `slope==0 && tree_type==4 && lad==6` 且 slope/aspect 不作为
  自由参数时自动启用完整模型，否则回退 reduced 模型并在 `model_note` 中说明。


## 7. 反演可辨识性（补充）

完整模型对 `H -> aH, d_L -> d_L/a`（cover 不变）严格不变（数值精确 0 差异），
因此单波段 BRF 只能辨识 `LAI = d_L·H·Λ(cover)` 与 cover。已实现
`d_L = LAI/(H·Λ(cover))` 的 LAI 参数化；合成实验精确回收（LAI 误差 <1e-5，
cond≈3.4），H 的 Jacobian 列范数为 ~0。详见
`data_zzl/inversion_experiment/report.md` 与 `lai_report.md`。
