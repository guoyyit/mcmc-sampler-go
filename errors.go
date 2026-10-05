package mcmc

import "errors"

var (
	errStub = errors.New("mcmc: not implemented")

	// ErrInvalidConfig is returned when a Config fails validation.
	ErrInvalidConfig = errors.New("mcmc: invalid configuration")
	// ErrIllegalParams is returned when a target log density is evaluated at
	// parameters outside the target's support.
	ErrIllegalParams = errors.New("mcmc: illegal parameters")
	// ErrNonFiniteLogProb indicates that a target returned a non-finite,
	// non-negative-infinity log probability.
	ErrNonFiniteLogProb = errors.New("mcmc: non-finite log probability")
	// ErrChainDiverged indicates that a chain produced a non-finite draw.
	ErrChainDiverged = errors.New("mcmc: chain diverged")
	// ErrCheckpointCorrupt indicates that a checkpoint file could not be loaded.
	ErrCheckpointCorrupt = errors.New("mcmc: corrupt checkpoint")
	// ErrNoSamples indicates that an assessment was requested with no samples.
	ErrNoSamples = errors.New("mcmc: no samples available")
)
