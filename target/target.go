package target

import (
	"errors"
	"fmt"
	"math"
)

var (
	// ErrInvalidDimension is returned when a target or position has an
	// unsupported dimension.
	ErrInvalidDimension = errors.New("target: dimension must be positive")
	// ErrNilFunc is returned when a nil log-density function is supplied.
	ErrNilFunc = errors.New("target: log-density function must not be nil")
	// ErrWrongDimension is returned when a position length does not match the
	// target dimension.
	ErrWrongDimension = errors.New("target: position has wrong dimension")
	// ErrNonFinitePosition is returned when a position contains NaN or Inf.
	ErrNonFinitePosition = errors.New("target: position contains a non-finite value")
	// ErrNonFiniteLogProb is returned when a log density is NaN or +Inf.
	ErrNonFiniteLogProb = errors.New("target: log density must be finite or -Inf, never NaN or +Inf")
)

// LogProbFunc returns the log density log p(x).
//
// math.Inf(-1) is a legal return value (a point with zero probability).
// NaN and math.Inf(+1) are illegal and must be rejected.
type LogProbFunc func(x []float64) float64

// Target is a distribution specified only through its unnormalized log density.
type Target interface {
	Dim() int
	LogDensity(x []float64) float64
}

// GradientTarget is implemented by targets that can evaluate the gradient of
// log p(x), which is required by Hamiltonian Monte Carlo.
type GradientTarget interface {
	Target
	Gradient(x []float64, g []float64)
}

type basicTarget struct {
	dim int
	f   LogProbFunc
}

// New wraps a plain log-density function as a Target.
func New(dim int, f LogProbFunc) (Target, error) {
	if dim <= 0 {
		return nil, ErrInvalidDimension
	}
	if f == nil {
		return nil, ErrNilFunc
	}
	return &basicTarget{dim: dim, f: f}, nil
}

func (b *basicTarget) Dim() int { return b.dim }

func (b *basicTarget) LogDensity(x []float64) float64 { return b.f(x) }

// ValidatePosition checks that x has the target dimension and contains only
// finite values.
func ValidatePosition(dim int, x []float64) error {
	if len(x) != dim {
		return fmt.Errorf("%w: got %d, want %d", ErrWrongDimension, len(x), dim)
	}
	for i, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("%w at component %d", ErrNonFinitePosition, i)
		}
	}
	return nil
}

// ValidateLogProb checks a log-density value. -Inf is legal; NaN and +Inf are
// not.
func ValidateLogProb(lp float64) error {
	if math.IsNaN(lp) || math.IsInf(lp, +1) {
		return ErrNonFiniteLogProb
	}
	return nil
}
