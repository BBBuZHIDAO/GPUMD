# GPUMD 5.9.1：独立 NEP + IWP M6X6 模块

本分支基于 GPUMD v5.9.1，集成了适用于 M6X6 纳米线的独立 NEP-IWP 混合势模块。

## 文件和入口

- `src/force/iwp_nep_m6x6.cu`：独立 `IWP_NEP_M6X6 : public Potential` 类，包含势参数读取、邻居搜索、NEP 和 IWP 能量/力/virial。
- `src/force/iwp_nep_m6x6.cuh`：独立类/数据结构及径向法向量解析导数；不依赖 `iwp_normal.cuh` 或 `ilp_nep.cuh`。
- `src/force/force.cu`：注册 `nep_iwp`，也接受别名 `nep_iwp_m6x6`。
- `potentials/iwp` 和 `tools/iwp`：保留旧 Mo6S6/Mo6Se6 参数与转换工具。

原 5.9.1 的 `ilp_nep.cu/.cuh` 和 `nep_ilp` 实现保留。新模块的类型、宏与非静态辅助函数单独命名，避免重复符号。
5.9.1 的 makefile 已用 `$(wildcard force/*.cu)` 和 `.cu -> .o/.obj` 规则，新增文件会自动参与编译，无需手动写入对象列表。

## 元素和物理定义

允许的元素集合为 **Mo W S Se Te**，最大 5 种。可按实际体系使用其中一个子集，元素不能重复。
势文件指定的元素顺序用于参数矩阵顺序；代码按元素名称映射至每个 NEP 的类型，不要求 NEP 文件采用同样顺序。

- Mo/W 的径向法向量符号为 +1；S/Se/Te 为 -1。
- 在同一 wire 内，以对应原子对的 `rcut_normal` 为搜索上限，选择最近的四个 M-X 原子。
- 法向量为四个单位位移向量的带符号和归一化；中心原子与四个邻居的解析导数进入原子力和 virial。
- IWP 仅作用于不同 wire 的原子对；NEP 计算同一 wire 的内部相互作用。
- NEP 径向/角向截断与法向量截断分别过滤。cell 搜索覆盖三者所需范围，周期 cell 遍历避免重复。
- 法向量近邻允许非互惠；导数力直接散射到实际近邻。

## 五元素势文件

首两行为：

```text
nep_iwp 5 Mo W S Se Te
0
```

第二行 `0` 表示 model.xyz 中的 group method 0，标签识别每根 wire。
后续必须有 **25 行 × 12 个数值**，不写行内或独立 `#` 注释。
每行按以下顺序：

```text
z0 alpha delta epsilon C d sR reff C6 S rcut_normal rcut_global
```

矩阵行顺序为：

```text
Mo-Mo Mo-W Mo-S Mo-Se Mo-Te
W-Mo  W-W  W-S  W-Se  W-Te
S-Mo  S-W  S-S  S-Se  S-Te
Se-Mo Se-W Se-S Se-Se Se-Te
Te-Mo Te-W Te-S Te-Se Te-Te
```

每个组合占一行，以上五行只是顺序说明。模板为
`potentials/iwp/iwp_M6X6_5elements.template`，其中尖括号字段必须用拟合数值替换后另存为 `.txt` 才能运行。
`z0/delta/sR/reff` 与两个截断必须为正，所有 12 个数值必须有限。
截断和长度采用 Å；epsilon、C 采用 meV，C6 采用 meV·Å^6，代码乘 `1e-3*S` 转为 eV；S 是无量纲缩放。
按通常无向原子对模型，A-B 与 B-A 参数保持一致。
本版本沿用 5.9.1 混合势的要求：全局 IWP 截断不小于所用 NEP 的最大截断。

**支持元素不等于已有该元素的拟合参数。** 旧包只有 Mo–Mo、Mo–S、Mo–Se、S–Se 的四组拟合参数。
旧三元素矩阵的 S–S/Se–Se 行为特定异质双线体系的占位行，不可直接作为这些相互作用的可靠参数。
W、Te 相关对及其他实际出现的缺失原子对必须提供拟合参数；本包没有复制 Mo/S/Se 参数来伪造 W/Te 参数。

## run.in 和 NEP map

```text
potential iwp_M6X6.txt nep_map.txt
```

一个通用五元素 NEP4 给两根 wire 使用的 map 示例：

```text
0 1 nep.txt
2
0
0
```

这里 NEP group method 为 0，有一个 NEP 文件，两个 group 均用 NEP 0。
若每根 wire 使用自己的 NEP4，可以保留旧包的多 NEP map 格式。
每个 NEP 必须包含分配给它的 wire 内实际存在的全部元素。
本独立模块保留 **5.9.1 的 NEP4 解析与参数布局**；旧包 NEP3/NEP5 的扩展路径没有移入。
遇到不足四个正常近邻或退化法向量时，代码沿用旧 IWP 的 ERROR 输出行为；应停止使用该结果并修正 wire 标签、结构或 rcut_normal。

## 编译

Linux/WSL，在具有 nvcc 的环境中：

```bash
cd src
make -j4 gpumd CUDA_ARCH=-arch=sm_120
```

`sm_120` 适用于 RTX 50 系列 GPU；其他 GPU 需按实际架构设置。
单独生成对象：

```bash
make force/iwp_nep_m6x6.o CUDA_ARCH=-arch=sm_120
```

本次实际使用 `/usr/local/cuda-12.9/bin/nvcc`。如 nvcc 不在 PATH，增加
`CC=/usr/local/cuda-12.9/bin/nvcc`。若 WSL 继承了 Windows 的 OS 环境变量，使用 `env -u OS make ...`。
Windows 原 makefile 使用 `.obj`，因此对应文件为 `iwp_nep_m6x6.obj`。
双精度 atomicAdd 路径要求 CUDA sm>=60。HIP/AMD 与 gpumd_replica 没有在本次验证中运行；replica 本身是另一套独立势接口，不包含本 IWP。

## 验证

```bash
python3 tests/iwp/run_cpu_tests.py
IWP_TEST_NEP=/path/to/five-element-nep4.txt \
IWP_TEST_GPUMD=$PWD/src/gpumd \
python3 -m unittest discover -s tests/iwp -p test_iwp_normal_cutoff.py -v
```

GPU 测试 NEP 要包含 Mo W S Se Te，径向/角向截断 4 Å。测试用的 IWP 系数是合成值，仅用于代码验证。
CPU 导数/符号、参数转换和有向能量分解测试通过；13 项 GPU 回归测试通过，包含六种 M-X 组合、五元素 5×5 矩阵重排、NEP4 能量与力一致性、法向量力有限差分、virial 应变有限差分、周期边界、独立 normal cutoff 及近邻非互惠。
5.9.1 已删除 `dump_force`，示例或旧 run.in 应改为 `dump_xyz 1 forces.xyz force`；新 thermo.out 带注释头，后处理需忽略以 # 开始的行。

## 源码来源与许可

GPUMD 5.9.1 的原始文件保留其原许可。移入的新 IWP 文件含旧 GPUMD 4.0 定制包的 GPL-3.0-or-later 代码，相关许可证保存在 `LICENCE_IWP_GPL3`。

独立代码审查另发现并修复了邻居归约 kernel 的尾部线程检查（改为 n1 < N2）。
尝试 Compute Sanitizer 时，本机 WSL/WDDM 返回 `Failed to initialize WDDM debugger interface` 和 `Device not supported`；
因此未完成 GPU 内存检查，不将此项标为通过，也没有修改系统调试/管理员设置。
