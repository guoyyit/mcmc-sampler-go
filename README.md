# mcmc-sampler-go

本地多链马尔可夫链蒙特卡洛（MCMC）采样服务。仅需要目标分布的（未归一化）对数概率密度，
支持随机游走 Metropolis-Hastings 与哈密顿蒙特卡洛（HMC），多链并行、预热自适应、
R-hat/ESS 收敛诊断、原子检查点与崩溃恢复、取消/超时以及有界内存的采样历史。

## 快速开始

```go
target := mcmc.Target{
    Dim: 2,
    LogProb: func(theta []float64) (float64, bool) {
        return -0.5*(theta[0]*theta[0] + theta[1]*theta[1]), true
    },
    GradLogProb: func(theta []float64) ([]float64, bool) { // 可选
        return []float64{-theta[0], -theta[1]}, true
    },
}
cfg := mcmc.Config{
    Target: target, Algo: mcmc.AlgoHMC,
    Seed: 20240101, Chains: 4,
    Iterations: 2000, Warmup: 1000,
    Adapt: mcmc.AdaptStepSizeMass,
    RHatThreshold: 1.01, ESSTarget: 400,
    Checkpoint: mcmc.CheckpointConfig{Path: "run.json", Every: 200},
    History:    mcmc.HistoryConfig{Mode: mcmc.RetainAll},
}
s, _ := mcmc.New(cfg)
res, _ := s.Run(mcmc.RunOptions{Ctx: ctx})
if !res.Converged {
    log.Fatal(res.Reason) // 明确的未收敛结论
}
// res.Diagnostics.RHat / ESS / ESSTail / Mean / Std
// res.Samples[chain][draw]
```

崩溃后恢复（新进程，同一份代码与种子）：

```go
res, err := s.Run(mcmc.RunOptions{Ctx: ctx, Resume: "run.json"})
```

## 算法

### 随机游走 Metropolis-Hastings（`AlgoMH`）
对称高斯提议 `theta' ~ N(theta, s^2 I)`，接受概率
`alpha = min(1, exp(logp(theta') - logp(theta)))`。提议核对称，无需 Hastings 校正。
所有接受判定都在对数空间完成：比较 `log(u) <= log alpha`，不取 `exp` 到 0，
因此极小接受率不会因下溢被错误接受。

### 哈密顿蒙特卡洛（`AlgoHMC`）
对角质量矩阵 `M`，动量 `p ~ N(0, M)`，leapfrog 蛙跳积分 Hamilton 方程
`H(q,p) = -logp(q) + p^T M^{-1} p / 2`。步数固定（`TrajectorySteps`）或在
`[TrajectoryMin, TrajectoryMax]` 内由每链 RNG 随机选取。接受率
`alpha = min(1, exp(H0 - H1))`。

- 目标未提供 `GradLogProb` 时使用步长自适应的前向有限差分（`FiniteDiffEps`）。
- 轨迹中任一点落在支撑外（`ok=false`）、返回非有限值或数值溢出，即判定为
  发散轨迹：拒绝该提议、停在当前点，接受率记 0，链永不进入非有限状态。
- 支撑外点一律表示为 `-Inf`；当前点为 `-Inf` 而提议点有限时接受，两者皆 `-Inf` 时拒绝。

### 非法参数
`LogDensityFunc` 返回 `(lp, ok)`，`ok=false` 表示参数在支撑外，按 `logp=-Inf` 处理。
含 `NaN`/`Inf` 的入参向量、返回值为 `NaN` 或 `+Inf` 的对数密度、维度不符均返回
`ErrIllegalParams` / `ErrNonFiniteLogProb`，不会被静默接受。

## 随机数与可复现性

- 每链持有独立的 SplitMix64 RNG（全状态为一个 `uint64`），种子由
  `chainSeed(masterSeed, chainIndex, chainCount)` 经固定混合函数导出；
  不使用时间、PID、goroutine ID、map 迭代顺序或调度顺序。
- 初始点：用户提供时原样使用；否则由该链 RNG 以 `InitScale`（默认各维 1）
  的高斯偏移确定性生成，因此各链天然分散且可逐位复现。
- 相同 `Config`、种子、初值下，样本序列（含预热）逐条一致；检查点包含 RNG 全状态，
  恢复后续抽样本与不间断运行完全一致（见 `TestCrashRecoverySampleEquivalence`）。

## 预热自适应（仅在预热内发生）

采用 Hoffman & Gelman (2014) 的双平均（dual averaging）步长调节，
`TargetAcceptLow/High` 的中点作为目标接受率 δ。

- `AdaptNone`：不调节。
- `AdaptStepSize`（MH 与 HMC）：调节 MH 提议标准差 / HMC 的 `epsilon`。
  接受率高于目标则增大步长，低于目标则减小；预热末使用双平均的平滑估计
  `exp(log_eps_bar)` 作为最终步长，并夹在 `[1e-8, 1e3]`。
- `AdaptStepSizeMass`（仅 HMC）：在上述步长调节之外，按 Stan 式窗口安排
  （15 步快速前缀 + 倍增规则窗口 + 15 步终端缓冲）用 Welford 在线矩估计各维
  样本方差，结合强度为 5 的单位信息先验正则化，更新对角质量矩阵；
  每个窗口结束时重启双平均。

马尔可夫性质：所有适应都只使用截至当前迭代的转移统计，预热结束的那一刻调用
`Freeze()` 冻结步长与质量矩阵，采样阶段参数不再改变，之后的链是固定核上的
马尔可夫链。预热样本默认不进入收敛诊断。

## 收敛诊断

`assess` 对每个坐标计算：

- split R-hat：每条链先对半拆分，再做秩归一化（inverse-normal 变换）后的
  R-hat（Vehtari, Gelman, Gabry 2021）；
- bulk ESS：秩归一化后按多链自相关正序列截断的有效样本量（Gelman 2013 公式 11.8）；
- tail ESS：5% 与 95% 分位指示变量的最小 ESS。

判定：`max(RHat) <= RHatThreshold`（默认 1.01）且每个坐标
`ESS >= max(ESSTarget, 5*chains)` 才 `Converged=true`。
未达标时 `Result.Converged=false` 且 `Result.Reason` 给出量化原因（最大 R-hat、
最小 ESS 与阈值），绝不静默输出“已收敛”。取消或不足 2 条完成链时不输出任何
部分诊断（`Diagnostics=nil`）。

## 检查点与恢复

- 每 `Checkpoint.Every` 次转移，协调器收集所有链在**同一逻辑边界**的快照后，
  以“临时文件 + fsync + rename”原子落盘；任何已提交文件都完整、可加载。
  单链异常或取消只会让本次全局检查点缺失，不会破坏上一个文件。
- 检查点内容（`Checkpoint`）：格式版本、种子、校验用配置标量、每链
  RNG 全状态、当前位置与 logp、引擎参数（步长/质量矩阵）、适配器内部状态
  （双平均累积量、Welford 统计、窗口位置）、保留的样本历史。
  目标分布是 Go 函数，不参与序列化，恢复时由调用方用同一份代码重建，并校验
  算法、维度、迭代数、种子等标量一致，不一致返回 `ErrCheckpointCorrupt`。
- 恢复后从最近检查点继续，RNG 状态保证逐样本一致；对已完成的运行重复恢复是幂等的。

## 取消、超时与内存

- `RunOptions.Ctx` 取消或超时后，各链在下一转移边界立即停止并标记
  `canceled`；协调器不再产生新文件，已提交检查点保持有效；不输出部分诊断。
- 单链 panic 被该链 goroutine 的 recover 隔离，标记为 `failed` 并附带错误信息，
  其他链照常完成；失败链数导致无法评估时给出明确原因。
- 历史保留策略（`HistoryConfig.Mode`）：
  - `all`：保留全部预热后样本（上界为迭代数）；
  - `thinned`：每 `Thin` 个保留一个；
  - `window`：环形缓冲仅保留最近 `Window` 个，长运行内存恒定；
  - `KeepWarmup` 控制预热样本是否保留（默认不保留、不进入诊断）。

## 参数默认值

| 参数 | 默认 |
| --- | --- |
| `Algo` | `metropolis-hastings` |
| `MHStepSize` / `HMCStepSize` | `0.1` / `1.0`（开启自适应时为初值） |
| `TrajectorySteps` | 10 |
| `FiniteDiffEps` | 1e-6 |
| `Adapt` | `stepsize`（无预热时强制 `none`） |
| `TargetAcceptLow/High` | 0.6 / 0.9 |
| `InitScale` | 各维 1.0 |
| `RHatThreshold` / `ESSTarget` | 1.01 / 100 |
| `History.Mode` | `all` |

## 测试

`go test ./...`（另建议 `-race`）。覆盖：非法/极端数值与对数空间接受
（`numeric_test.go`）、种子复现与初值确定性（`reproduce_test.go`）、
步长与质量矩阵自适应（`adapt_test.go`）、R-hat/ESS 与明确未收敛
（`diagnose_test.go`）、检查点恢复逐样本一致与损坏检测（`checkpoint_test.go`）、
取消/超时/故障隔离（`cancel_test.go`）、保留策略与有界内存（`history_test.go`）。
