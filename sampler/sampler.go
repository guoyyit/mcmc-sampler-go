package sampler

import (
	"context"
	"errors"
	"fmt"

	"github.com/guoyyit/mcmc-sampler-go/rng"
	"github.com/guoyyit/mcmc-sampler-go/target"
)

var (
	// ErrUnknownAlgorithm is returned for an unrecognized algorithm name.
	ErrUnknownAlgorithm = errors.New("sampler: unknown algorithm")
	// ErrInvalidParams is returned when sampler parameters are invalid.
	ErrInvalidParams = errors.New("sampler: invalid parameters")
	// ErrDensityInvalid is returned when the target yields NaN or +Inf.
	ErrDensityInvalid = errors.New("sampler: target returned an illegal log density")
	// ErrDiverged signals an HMC integrator divergence (non-finite state).
	ErrDiverged = errors.New("sampler: integrator diverged")
)

// Algorithm names.
const (
	AlgoMH  = "metropolis-hastings"
	AlgoHMC = "hmc"
)

// Params holds the (mutable) tuning parameters of a sampler.
type Params struct {
	Algorithm string
	Initial   []float64
	// InitJitter, when positive, disperses chain starting points: the actual
	// initial position is Initial + InitJitter * N(0,1) drawn from the chain's
	// deterministic random stream before any proposal consumes randomness.
	// Zero means every chain starts exactly at Initial.
	InitJitter    []float64
	ProposalScale []float64 // per-dimension proposal standard deviation (MH)
	StepSize      float64   // integrator step size (HMC)
	NumLeapfrog   int       // leapfrog steps per HMC proposal
	InverseMass   []float64 // diagonal inverse mass matrix (HMC)
}

// Stats records proposal-level diagnostics for one iteration.
type Stats struct {
	Accepted      bool
	LogAcceptProb float64 // log MH acceptance probability, capped at 0
	Divergent     bool    // HMC integrator divergence
}

// State is one draw produced by a sampler.
type State struct {
	Iteration int
	Position  []float64
	LogProb   float64
	Stats     Stats
}

// Sampler advances one Markov chain by one iteration.
type Sampler interface {
	Step(ctx context.Context) (State, error)
	Current() State
	Params() Params
	SetParams(Params) error
	// Marshal encodes the sampler's mutable state (current position), but not
	// its random stream.
	Marshal() ([]byte, error)
	Unmarshal([]byte) error
}

// New constructs the requested sampler.
func New(t target.Target, r *rng.Source, p Params) (Sampler, error) {
	if t == nil {
		return nil, fmt.Errorf("%w: nil target", ErrInvalidParams)
	}
	if r == nil {
		return nil, fmt.Errorf("%w: nil random source", ErrInvalidParams)
	}
	if err := target.ValidatePosition(t.Dim(), p.Initial); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidParams, err)
	}
	switch p.Algorithm {
	case AlgoMH:
		return newMH(t, r, p)
	case AlgoHMC:
		gt, ok := t.(target.GradientTarget)
		if !ok {
			return nil, fmt.Errorf("%w: %s requires a target implementing Gradient", ErrInvalidParams, AlgoHMC)
		}
		return newHMC(gt, r, p)
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnknownAlgorithm, p.Algorithm)
	}
}
