# 论文等价性结果：权威复现链（REPRODUCIBILITY）

> 本清单是论文数值等价性结果（30 天受控、1 年自主、10 年链式、性能）的
> **唯一权威生成路径**。所有数字与图都由本清单的脚本产出，**不要再用临时
> 脚本重算**——历史上多次误差（如把闰年错位当成物理分叉）都源于临时脚本
> 绕开了本清单里的对齐约定。
>
> 服务器 Python：`/home/qixiang/.conda/envs/yjh/bin/python`
> 服务器 runner 目录：`/data/yangjinhui/surf_pytorch/cmfd_offline_runner/`
> 服务器脚本目录：`/data/yangjinhui/surf_pytorch/surf_paper/scripts/`

---

## 0. 共享初始场（唯一权威）

所有等价性实验（30 天、1 年、10 年、性能）共用同一个五年 spin-up 平衡初态：

```
/data/yangjinhui/surf_pytorch/surf_paper/experiments/spinup_era5land_2005_2009/restartin_era5land_2009_eq.nc
```

ERA5-Land 2005 状态映射到 97709 列 → CMFD 强迫 2005–2009 连续积分 → 2009 末态。
**任何等价性对比都必须从它起步**，否则两侧轨迹不可比。

> 注意：`initial_states/native_20d_terminal_complete.nc` 是另一套（欠 spin-up）
> 初态，仅供历史参考，**不用于等价性对比**。

---

## 1. 闰年对齐约定（极易错，务必遵守）

- 链式多年运行按**真实日历年**积分：平年 17520 步（365 天），闰年 17568 步
  （366 天；2009–2018 中 2012、2016 为闰年）。torch `global_step` 逐年累加。
- Fortran 每年 NSTOP = 17520 / 17568（与 torch 一致），但每年 `o_gg.nc` 只
  存档到 480 步的最后一个倍数（平年 17280），**真正的年末态在 `restartout.nc`**。
- 因此年末对齐 = torch 累计步（年末）↔ Fortran `<year>/restartout.nc`。
- 权威实现：`compare_multiyear_terminal.py` 用 `metadata.ntime × 6` 累积每年
  真实步数，并读取 `restartout.nc`。**对齐逻辑以它为准。**

---

## 2. 论文等价性实验的权威生成脚本

### 2.1 30 天受控边界驱动（Table controlled-30d, Fig controlled-*）
- 积分：`run_full_domain_offline.py`（消费 Fortran 捕获边界）
- 对比出图：`compare_full_domain_controlled.py`
- Fortran 参考：`fortran_runs/era5land_30d_eq/`（+ `era5land_30d_capture/`）

### 2.2 1 年自主积分（Table annual-autonomous, Fig annual_*）
- torch 积分：`run_full_domain_offline.py`，初态 = spinup_2009_eq，2018 全年
  （17520 步），stride 480
- Fortran 参考：`fortran_runs/era5land_1y_2018_eq/`（终态）+
  `era5land_1y_2018_eq_10d/`（逐 10 天）
- 对比出图：`compare_autonomous_annual.py`
  （env：`SURF_TORCH_ANNUAL`/`SURF_FORTRAN_10D`/`SURF_FORTRAN_FINAL`/`SURF_DATA`/`SURF_COMPARE_OUTPUT`）
- 水量能量收支图：`compute_budget_year.py` → `plot_water_energy_budget.py`

### 2.3 10 年链式（Table multiyear-autonomous, Fig multiyear-*）
- torch 积分：`run_torch_multiyear_prod.py`，初态 = spinup_2009_eq，
  2009–2018 逐年链接，`SURF_YEARS=2009:2018`，`SURF_PRELOAD_FORCING=1`
- Fortran 参考：`fortran_runs/era5land_multiyear_eq/years/YYYY/`（每年
  `restartout.nc` 为真末态）
- 对比出图：**`compare_multiyear_terminal.py`**（闰年对齐已内置）
  （env：`SURF_TORCH_MULTIYEAR`/`SURF_FORTRAN_YEARS`/`SURF_MULTIYEAR_DATA`/`SURF_COMPARE_OUTPUT`/`SURF_YEARS`）

### 2.4 性能扩展性（Table performance-scaling, Fig surf_fortran_graph_scaling）
- torch 侧：`benchmark_scaling.py`（GPU A100，240 步，3 重复，不计启动；
  eager 与 `SURF_CUDA_GRAPH=1` 图模式各一轮），产物在
  `experiments/bench_final_cpu_gpu/`（eager）与 `experiments/bench_final_graph/`（graph）
- Fortran 侧：`fortran_runs/bench_f90_*col/`（osmMASTER，32 线程，240 步，
  计 `Time step loop` 段）
- 出图：`plot_performance_scaling_graph.py`（Fortran + eager + graph 三路对比）

---

## 3. 观测评估 / 校准（MODIS，附录）

**观测算子存档（唯一权威路径）**：`cmfd_offline_runner/run_modis_observables_gpunative.sh`
（GPU 原生 + spinup 初态 + `SURF_MODIS_TARGETS` 启用观测算子）→
`<case>/surf_modis_observables.nc`。**每次复现用此脚本，不要手改环境。**

后续脚本见 `USAGE.md` §2–§3；权威脚本均在 `scripts/`：
`compare_modis_validation.py`（验证地图/季节循环）、
`modis_slab_maps.py`、`modis_slab_asym_train.py`、`slab_seasonal_figures.py`、
`slab_observables_year.py`、`modis_structural_closure.py`、`modis_calibration_heldout.py`、
`make_closure_schematic.py`、`asym_train_replot.py`、`combine_modis_identifiability.py`、
`plot_regional/seasonal_tair_sensitivity.py`、`fortran_guided_initial_state_inversion.py`、
`gradient_validation.py`。

---

## 4. 论文重建

```bash
cd surf_paper/manuscript && latexmk -pdf -g -interaction=nonstopmode main.tex
```

25 张引用图全部在 `manuscript/figures/` 下。

---

## 5. 临时/调试脚本（不属于复现链）

`surf_paper/tmp/`、本地 `tmp/patch_f90chain/`、本地 `scripts_tmp/` 下的脚本是
调试过程产物，**不复现论文结果**，可归档后删除。权威对齐模块
`align_multiyear.py` 已并入 `compare_multiyear_terminal.py` 的约定，无需单独保留。
