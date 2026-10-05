package mcmc

import "math"

// Draw is the output of a single kernel transition.
type Draw struct {
	// Theta is the parameter vector.
	Theta []float64
	// LogProb is the target log density at Theta.
	LogProb float64
	// Accepted reports whether the proposal was accepted (MH) or the
	// trajectory stayed finite (HMC).
	Accepted bool
	// AcceptProb is the Metropolis acceptance probability.
	AcceptProb float64
	// Warmup marks transitions produced during warmup.
	Warmup bool
	// Energy is the HMC Hamiltonian value (0 for MH).
	Energy float64
}

// Params are the tunable kernel parameters frozen after warmup.
type Params struct {
	// StepSize is the MH proposal standard deviation or HMC epsilon.
	StepSize float64 `json:"step_size"`
	// Mass is the diagonal mass matrix (per-coordinate variances) for
	// HMC; nil for MH.
	Mass []float64 `json:"mass,omitempty"`
}

// Engine is a sampling kernel for one chain. All randomness comes from the
// supplied rng64; engines must never use a global random source.
type Engine interface {
	// Init places the chain at theta with log density lp.
	Init(theta []float64, lp float64)
	// Step performs one transition. warmup=true lets the adapter record
	// the transition so it can adapt parameters.
	Step(rng rng64, warmup bool, adapter Adapter) (Draw, error)
	// Params returns the current tunable parameters.
	Params() Params
	// SetParams overwrites the tunable parameters (warmup freeze/resume).
	SetParams(p Params)
}

// newEngine builds an engine for the configured algorithm.
func newEngine(cfg *Config) (Engine, error) {
	switch cfg.Algo {
	case AlgoMH:
		return newMH(cfg)
	case AlgoHMC:
		return newHMC(cfg)
	default:
		return nil, ErrInvalidConfig
	}
}

// logAccept computes the Metropolis acceptance log-probability from the
// target log-density ratio. Both values may be -infinity:
// proposal from an illegal point (-Inf - finite) rejects; an illegal
// proposal (finite - -Inf, i.e. +Inf) is capped so the test stays finite
// but still accepts — which is correct when the current point itself has
// zero density. NaN anywhere rejects.
func logAccept(lpProposal, lpCurrent float64) float64 {
	if math.IsNaN(lpProposal) || math.IsNaN(lpCurrent) {
		return math.Inf(-1)
	}
	switch {
	case math.IsInf(lpCurrent, -1):
		if math.IsInf(lpProposal, -1) {
			return math.Inf(-1) // both impossible: reject
		}
		return 0 // current point impossible, proposal has density: accept
	case math.IsInf(lpProposal, -1):
		return math.Inf(-1)
	default:
		d := lpProposal - lpCurrent
		if math.IsNaN(d) {
			return math.Inf(-1)
		}
		if d > 0 {
			d = 0
		}
		return d
	}
}

// acceptWith draws the accept decision in log space from uniform u:
// accept iff log(u) <= logAlpha. Never underflows for extremely small
// alpha because no exp is taken.
func acceptWith(rng rng64, logAlpha float64) (accepted bool, alpha float64) {
	if math.IsInf(logAlpha, -1) || math.IsNaN(logAlpha) {
		return false, 0
	}
	if logAlpha >= 0 {
		return true, 1
	}
	u := rng.Float64()
	if u == 0 {
		u = twoPow53
	}
	alpha = math.Exp(logAlpha)
	return math.Log(u) <= logAlpha, alpha
}

func cloneFloat64(x []float64) []float64 {
	y := make([]float64, len(x))
	copy(y, x)
	return y
}
