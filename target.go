package mcmc

import "math"

// LogDensityFunc evaluates the (unnormalized) log probability density at
// theta. ok=false means theta is outside the support (illegal parameters);
// such points are treated as log-probability -infinity and are never
// accepted.
type LogDensityFunc func(theta []float64) (float64, bool)

// GradientFunc evaluates the gradient of the log density at theta.
// ok=false signals illegal parameters.
type GradientFunc func(theta []float64) (grad []float64, ok bool)

// Target describes a distribution to sample from.
type Target struct {
	// Dim is the dimensionality of the parameter vector.
	Dim int
	// LogProb is the log density. Required.
	LogProb LogDensityFunc
	// GradLogProb is the log-density gradient. Optional; when nil, HMC uses
	// a finite-difference approximation based on LogProb.
	GradLogProb GradientFunc
}

// Eval returns the log density at theta, mapping illegal parameters and
// non-finite return values to -infinity.
func (t *Target) Eval(theta []float64) (float64, error) {
	if t == nil || t.LogProb == nil {
		return math.Inf(-1), ErrInvalidConfig
	}
	if len(theta) != t.Dim {
		return math.Inf(-1), ErrIllegalParams
	}
	for _, x := range theta {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return math.Inf(-1), ErrIllegalParams
		}
	}
	lp, ok := t.LogProb(theta)
	switch {
	case !ok:
		return math.Inf(-1), nil
	case math.IsInf(lp, -1):
		return math.Inf(-1), nil
	case math.IsNaN(lp) || math.IsInf(lp, +1):
		return math.Inf(-1), ErrNonFiniteLogProb
	default:
		return lp, nil
	}
}

// EvalGrad returns the log density and its gradient at theta.
func (t *Target) EvalGrad(theta []float64) (float64, []float64, error) {
	lp, err := t.Eval(theta)
	if err != nil {
		return math.Inf(-1), nil, err
	}
	if math.IsInf(lp, -1) {
		return lp, nil, nil
	}
	if t.GradLogProb == nil {
		return math.Inf(-1), nil, ErrInvalidConfig
	}
	grad, ok := t.GradLogProb(theta)
	if !ok {
		return math.Inf(-1), nil, nil
	}
	if len(grad) != t.Dim {
		return math.Inf(-1), nil, ErrNonFiniteLogProb
	}
	for _, g := range grad {
		if math.IsNaN(g) || math.IsInf(g, 0) {
			return math.Inf(-1), nil, ErrNonFiniteLogProb
		}
	}
	return lp, grad, nil
}
