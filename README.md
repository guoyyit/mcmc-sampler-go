# mcmc-sampler-go

A local Markov-Chain Monte-Carlo sampling service. Given only a function that
evaluates an (unnormalized) log probability density, it draws samples with
either **random-walk Metropolis–Hastings (MH)** or **Hamiltonian Monte Carlo
(HMC)**, runs several chains in parallel, adapts proposal parameters during a
warmup phase, quantifies convergence, and persists full checkpoints so a
crashed or cancelled job resumes bit-for-bit identically to an uninterrupted
run.

## Package layout

| Package | Responsibility |
| --- | --- |
| `target` | `Target` / `GradientTarget` interfaces, dimension and value validation |
| `rng` | Deterministic, serializable per-chain random streams |
| `sampler` | `Sampler` interface, `MH` and `HMC` implementations, log-space acceptance |
| `adapt` | Warmup step-size (dual averaging) and diagonal mass-matrix adaptation |
| `diagnostics` | Split R-hat, rank-normalized bulk/tail ESS, explicit convergence verdict |
| `history` | Thinned, size-capped sample retention with copy-on-write storage |
| `checkpoint` | Gob-encoded, atomically replaced on-disk checkpoints |
| `chain` | Single-chain runner tying sampler + adaptation + history together |
| `service` | Multi-chain orchestration: parallelism, isolation, checkpointing, cancellation |

## Quick start

```go
import (
    "context"

    "github.com/guoyyit/mcmc-sampler-go/adapt"
    "github.com/guoyyit/mcmc-sampler-go/diagnostics"
    "github.com/guoyyit/mcmc-sampler-go/history"
    "github.com/guoyyit/mcmc-sampler-go/sampler"
    "github.com/guoyyit/mcmc-sampler-go/service"
)

type gaussian struct{}
func (gaussian) Dim() int { return 2 }
func (gaussian) LogDensity(x []float64) float64 { return -0.5*(x[0]*x[0]+x[1]*x[1]) }
func (gaussian) Gradient(x, g []float64)        { g[0], g[1] = -x[0], -x[1] }

cfg := service.Config{
    JobID:     "example",
    NumChains: 4,
    Draws:     2000,   // total iterations per chain
    Warmup:    500,    // adaptive iterations; draws 501..2000 are retained
    Seed:      42,     // single seed; everything is derived from it
    Sampler: sampler.Params{
        Algorithm:    sampler.AlgoHMC,
        Initial:      []float64{0, 0},
        InitJitter:   []float64{1, 1}, // disperse chains deterministically
        StepSize:     0.1,
        NumLeapfrog:  20,
    },
    Adapt:            adapt.Config{Warmup: 500, TargetAcceptMin: 0.7, TargetAcceptMax: 0.9, AdaptMass: true},
    History:          history.Policy{Thin: 1, MaxSamples: 0}, // 0 = keep all
    Thresholds:       diagnostics.Thresholds{MaxRhat: 1.01, MinESS: 100},
    CheckpointDir:    "./.mcmc-checkpoints",
    CheckpointEvery:  100,
    CheckpointInterval: time.Second,
}

result, err := service.Run(context.Background(), cfg, gaussian{})
```

On success `result.Draws[chain][draw][dimension]` holds post-warmup draws.
On non-convergence `service.Run` returns an error wrapping
`service.ErrNotConverged` and **no samples are returned**; the error lists
every failing variable and metric.

## Algorithms

### Random-walk Metropolis–Hastings (`sampler.AlgoMH`)

Proposal: `x' = x + L * z`, `z ~ N(0, I)`, where `L` is a per-dimension
diagonal proposal scale (`ProposalScale`). The proposal is symmetric, so the
acceptance probability reduces to the density ratio:

```
log alpha = min(0, log p(x') - log p(x))
```

The decision is made entirely in log space by comparing `log u` (with
`u ~ U(0,1)`) against `log alpha`; `exp` is never taken of a density ratio.
`math.Inf(-1)` log densities (impossible regions) are legal and always
rejected. NaN and `+Inf` log densities are programming errors and are rejected
loudly.

### Hamiltonian Monte Carlo (`sampler.AlgoHMC`)

The target must implement `GradientTarget`. Each iteration samples momentum
`p ~ N(0, I)`, integrates Hamiltonian dynamics with `NumLeapfrog` leapfrog
steps of size `StepSize` using a diagonal (inverse) mass matrix, and applies a
Metropolis correction on the energy difference:

```
log alpha = min(0, log p(q') - log p(q) - (K(p') - K(p)))
```

If any intermediate position, momentum or gradient becomes non-finite, the
transition is marked `Divergent` in `Stats` and the chain stays at its previous
position (rather than emitting a non-finite state). Illegal target values
(NaN/`+Inf` log density) are returned as errors, never swallowed.

## Reproducibility

* A single job `Seed` expands to a root stream; each chain gets a child stream
  derived by `rng.Child(root, chainIndex)`. Derivation happens in fixed index
  order, independent of goroutine scheduling.
* The RNG is `math/rand/v2.ChaCha8`, whose **full internal state** is
  serialized in checkpoints. Restoring it reproduces the exact future variate
  sequence.
* Chain starting points are `Initial + InitJitter * N(0,I)` drawn from that
  chain's stream before any proposal, so chains disperse deterministically.
* Same seed + same config ⇒ draw-by-draw identical output, asserted in
  `service.TestSeedReproducibility` and
  `TestHMCEndToEndAdaptationAndReproducibility`.

## Warmup adaptation

Adaptation runs on iterations `1..Warmup` and is then **frozen**; only
proposal parameters change, never the chain state itself, so after warmup the
chain is an ordinary time-homogeneous Markov chain (the adaptation does not
alter transition kernels of subsequent draws).

* **Step size** uses dual averaging (Hoffman & Gelman, Algorithm 6), driving
  the empirical acceptance probability toward the midpoint of
  `[TargetAcceptMin, TargetAcceptMax]`. The smoothed `epsilon_bar` becomes the
  frozen step size.
* **Mass matrix** (optional, `AdaptMass`) accumulates a Welford online
  covariance of warmup positions and commits the diagonal variance at the end
  of warmup, lightly regularized toward 1 so a near-constant dimension cannot
  collapse to zero.
* Parameters: `Gamma=0.05`, `Kappa=0.75`, `T0=10`, target acceptance interval
  default `[0.8,0.8]`, initial step size `1`. All are configurable and
  validated.

## Convergence diagnostics

`diagnostics` implements the Vehtari–Gelman–Gabry (2021) recommendations:

* **Split R-hat**: every chain is split in half, values are rank-normalized,
  and R-hat is computed from between/within-chain variances.
* **Bulk ESS**: effective sample size of rank-normalized draws using Geyer's
  initial positive (monotone) sequence on pooled autocorrelations.
* **Tail ESS**: minimum of the 5th/95th percentile indicator ESS.

`Evaluate` returns a `Report` whose `Converged` flag is always explicit and
whose `Reasons` slice enumerates every threshold violation. Defaults:
`MaxRhat = 1.01`, `MinESS = min(100, 4 * numChains)`.

## Checkpointing and recovery

* A `Checkpoint` contains, per chain: chain ID, iteration, full RNG state,
  sampler state (current position/log-prob/iteration), adaptation engine
  state, and retained history — everything required to continue exactly.
* `FileStore.Save` writes gob to a temp file, `fsync`s, then atomically
  `rename`s, so a crash mid-write leaves the previous checkpoint intact.
* Restart: calling `service.Run` with the same `CheckpointDir` loads the
  checkpoint, verifies job ID/seed/chain count, restores every chain, and runs
  to completion. The resulting draws are asserted identical to an
  uninterrupted run in `service.TestCrashRecoveryMatchesUninterrupted`.
* A checkpoint must match the configuration (job id, seed, chain count); a
  mismatch is rejected rather than silently mixing jobs.

## Cancellation, timeouts and memory

* `service.Run` takes a `context.Context`; cancellation or deadline stops all
  chains at their next iteration boundary (asserted in
  `TestCancellationStopsAllChains`).
* On cancellation, the last committed checkpoint is (re)written and remains
  valid; no diagnostics are computed and no partial `Result` is returned.
* A failing chain cancels the run and its error is surfaced with the chain id;
  each chain is fully independent (own state and RNG) and the other chains'
  already-produced results stay intact (`TestFailingChainDoesNotCorruptOthers`).
* History memory is bounded by `history.Policy`: `Thin` keeps every Nth
  post-warmup draw and `MaxSamples` hard-caps retained draws (newest kept).
  Every stored position is copied, so sampler scratch memory can never alias
  retained samples.

## Numerical-stability guarantees

* All acceptance comparisons are in log space; `-Inf` densities reject
  proposals and the `-Inf - -Inf` (0/0 probability) case is handled
  explicitly instead of producing NaN.
* Positions and parameters are validated for finiteness; illegal proposal
  scales, step sizes and mass entries are rejected up front.
* See `sampler/stability_test.go`.

## Tests

```sh
go test -race ./...
```

Coverage includes numerical stability/underflow, seed reproducibility,
dispersed initialization, adaptive step-size and mass-matrix behavior,
R-hat/ESS on synthetic converged/offset/autocorrelated chains, history
bounds, atomic checkpointing, bit-exact crash recovery, cancellation, and
cross-chain fault isolation.
