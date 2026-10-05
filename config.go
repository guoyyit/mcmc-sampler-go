package mcmc

import (
	"context"
	"fmt"
)

// Algorithm selects a sampling kernel.
type Algorithm string

const (
	// AlgoMH is the random-walk Metropolis-Hastings algorithm.
	AlgoMH Algorithm = "metropolis-hastings"
	// AlgoHMC is Hamiltonian Monte Carlo with leapfrog integration.
	AlgoHMC Algorithm = "hmc"
)

// AdaptMode selects what warmup adaptation adjusts.
type AdaptMode string

const (
	// AdaptNone disables adaptation.
	AdaptNone AdaptMode = "none"
	// AdaptStepSize adapts step size only (MH proposal scale / HMC epsilon).
	AdaptStepSize AdaptMode = "stepsize"
	// AdaptStepSizeMass adapts step size and the (diagonal) mass matrix.
	AdaptStepSizeMass AdaptMode = "stepsize-mass"
)

// RetainMode selects which samples are kept in memory.
type RetainMode string

const (
	// RetainAll keeps every retained post-warmup draw.
	RetainAll RetainMode = "all"
	// RetainThinned keeps one draw every Thin post-warmup draws.
	RetainThinned RetainMode = "thinned"
	// RetainWindow keeps only the most recent Window draws (bounded memory).
	RetainWindow RetainMode = "window"
)

// CheckpointConfig configures periodic state persistence.
type CheckpointConfig struct {
	// Path is the checkpoint file. Empty disables checkpointing.
	Path string
	// Every is the number of transitions between checkpoints; 0 disables
	// periodic checkpointing (an explicit Save remains possible).
	Every int
}

// HistoryConfig configures sample retention.
type HistoryConfig struct {
	// Mode is the retention strategy.
	Mode RetainMode
	// Thin is used by RetainThinned (1 keeps every draw).
	Thin int
	// Window is the maximum number of retained draws per chain for
	// RetainWindow.
	Window int
	// KeepWarmup retains warmup draws when true.
	KeepWarmup bool
}

// Config configures a Sampler.
type Config struct {
	// Target is the distribution to sample.
	Target Target
	// Algo selects the kernel.
	Algo Algorithm
	// Seed is the master random seed.
	Seed uint64
	// Chains is the number of parallel chains (>= 1).
	Chains int
	// Iterations is the total number of transitions per chain (warmup
	// included).
	Iterations int
	// Warmup is the number of warmup transitions per chain.
	Warmup int

	// MHStepSize is the MH proposal standard deviation. Zero requests
	// adaptation (or a default when adaptation is disabled).
	MHStepSize float64
	// HMCStepSize is the HMC leapfrog step size. Zero requests adaptation.
	HMCStepSize float64
	// TrajectorySteps is the fixed leapfrog count for HMC.
	TrajectorySteps int
	// TrajectoryMin/TrajectoryMax randomize the leapfrog count in the
	// inclusive range when TrajectoryMax > TrajectoryMin.
	TrajectoryMin int
	TrajectoryMax int
	// FiniteDiffEps sets the forward finite-difference step used when the
	// target provides no gradient.
	FiniteDiffEps float64

	// Adapt selects warmup adaptation.
	Adapt AdaptMode
	// TargetAcceptLow/High bound the target acceptance-probability range.
	TargetAcceptLow  float64
	TargetAcceptHigh float64
	// InitScale is the standard deviation used when scattering initial
	// points of chains that do not provide one.
	InitScale []float64

	// RHatThreshold and ESSTarget control the convergence verdict.
	RHatThreshold float64
	ESSTarget     float64

	// Checkpoint configures persistence.
	Checkpoint CheckpointConfig
	// History configures retention.
	History HistoryConfig
}

// Validate checks the configuration and fills derived defaults in place.
func (c *Config) Validate() error {
	if c.Target.Dim < 1 {
		return fmt.Errorf("%w: target dimension must be >= 1", ErrInvalidConfig)
	}
	if c.Target.LogProb == nil {
		return fmt.Errorf("%w: target LogProb is required", ErrInvalidConfig)
	}
	switch c.Algo {
	case AlgoMH, AlgoHMC:
	case "":
		c.Algo = AlgoMH
	default:
		return fmt.Errorf("%w: unknown algorithm %q", ErrInvalidConfig, c.Algo)
	}
	if c.Chains < 1 {
		return fmt.Errorf("%w: chains must be >= 1", ErrInvalidConfig)
	}
	if c.Iterations < 1 {
		return fmt.Errorf("%w: iterations must be >= 1", ErrInvalidConfig)
	}
	if c.Warmup < 0 || c.Warmup > c.Iterations {
		return fmt.Errorf("%w: warmup must be in [0, iterations]", ErrInvalidConfig)
	}

	switch c.Algo {
	case AlgoMH:
		if c.MHStepSize < 0 {
			return fmt.Errorf("%w: MHStepSize must be >= 0", ErrInvalidConfig)
		}
		if c.MHStepSize == 0 {
			c.MHStepSize = 0.1 // replaced by adaptation when enabled
		}
	case AlgoHMC:
		if c.HMCStepSize < 0 {
			return fmt.Errorf("%w: HMCStepSize must be >= 0", ErrInvalidConfig)
		}
		if c.HMCStepSize == 0 {
			c.HMCStepSize = 1.0
		}
		randomTraj := c.TrajectoryMax > c.TrajectoryMin
		if randomTraj {
			if c.TrajectoryMin < 1 {
				return fmt.Errorf("%w: TrajectoryMin must be >= 1", ErrInvalidConfig)
			}
		} else if c.TrajectorySteps < 1 {
			c.TrajectorySteps = 10
		}
		if c.FiniteDiffEps == 0 {
			c.FiniteDiffEps = 1e-6
		}
	}

	switch c.Adapt {
	case AdaptNone, AdaptStepSize, AdaptStepSizeMass:
	case "":
		c.Adapt = AdaptStepSize
	default:
		return fmt.Errorf("%w: unknown adapt mode %q", ErrInvalidConfig, c.Adapt)
	}
	if c.Adapt == AdaptStepSizeMass && c.Algo != AlgoHMC {
		return fmt.Errorf("%w: mass adaptation requires HMC", ErrInvalidConfig)
	}
	if c.Warmup == 0 {
		c.Adapt = AdaptNone
	}
	if c.TargetAcceptLow == 0 && c.TargetAcceptHigh == 0 {
		c.TargetAcceptLow, c.TargetAcceptHigh = 0.6, 0.9
	}
	if !(0 < c.TargetAcceptLow && c.TargetAcceptLow <= c.TargetAcceptHigh && c.TargetAcceptHigh < 1) {
		return fmt.Errorf("%w: require 0 < TargetAcceptLow <= TargetAcceptHigh < 1", ErrInvalidConfig)
	}

	if c.RHatThreshold == 0 {
		c.RHatThreshold = 1.01
	}
	if c.ESSTarget == 0 {
		c.ESSTarget = 100
	}
	if c.RHatThreshold <= 1 {
		return fmt.Errorf("%w: RHatThreshold must be > 1", ErrInvalidConfig)
	}

	if len(c.InitScale) == 0 {
		c.InitScale = defaultScale(c.Target.Dim)
	} else if len(c.InitScale) != c.Target.Dim {
		return fmt.Errorf("%w: InitScale length must match Dim", ErrInvalidConfig)
	}

	switch c.History.Mode {
	case RetainAll, RetainThinned, RetainWindow:
	case "":
		c.History.Mode = RetainAll
	default:
		return fmt.Errorf("%w: unknown retain mode %q", ErrInvalidConfig, c.History.Mode)
	}
	if c.History.Mode == RetainThinned {
		if c.History.Thin < 1 {
			c.History.Thin = 1
		}
	}
	if c.History.Mode == RetainWindow && c.History.Window < 1 {
		c.History.Window = 1000
	}
	if c.Checkpoint.Every < 0 {
		return fmt.Errorf("%w: checkpoint Every must be >= 0", ErrInvalidConfig)
	}
	return nil
}

// withDefault returns v when non-zero, else def.
func withDefault(v, def float64) float64 {
	if v == 0 {
		return def
	}
	return v
}

// RunOptions carries per-run dynamic options.
type RunOptions struct {
	// Ctx cancels all chains (timeout or explicit cancellation).
	Ctx context.Context
	// InitialPoints optionally fixes each chain's initial point; entries
	// may be nil to request deterministic scattering.
	InitialPoints [][]float64
	// Resume loads this checkpoint path before running; empty disables.
	Resume string
}
