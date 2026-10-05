package mcmc

import "math"

// Initial points are zero (or the user-supplied point) plus independent
// Gaussian offsets drawn from each chain's own seeded RNG. Because the
// per-chain seeds differ deterministically with the chain index, the
// initial points are dispersed and do not depend on goroutine scheduling,
// launch order, time or the host environment.

// initialPoint returns a deterministic, dispersed initial point for chain
// chainIndex of chainCount. When supplied is non-nil it is used verbatim;
// otherwise the point is generated solely from the per-chain rng64 and the
// configured InitScale, so initialization never depends on the host,
// scheduling or launch order.
func initialPoint(dim int, supplied []float64, scale []float64, chainIndex, chainCount int, rng rng64) []float64 {
	theta := make([]float64, dim)
	if len(supplied) == dim {
		copy(theta, supplied)
		return theta
	}
	for i := 0; i < dim; i++ {
		s := 1.0
		if i < len(scale) && scale[i] > 0 {
			s = scale[i]
		}
		theta[i] = rng.NormalFloat64() * s
	}
	return theta
}

// defaultScale returns per-dimension standard deviations for init scatter.
func defaultScale(dim int) []float64 {
	s := make([]float64, dim)
	for i := range s {
		s[i] = 1
	}
	return s
}

// legalInitialPoint rejects NaN/Inf initial points by checking the target.
func legalInitialPoint(t *Target, theta []float64) (float64, bool) {
	lp, err := t.Eval(theta)
	if err != nil {
		return 0, false
	}
	if !isFinite(lp) {
		return lp, false
	}
	return lp, true
}

// isFinite reports whether x is neither NaN nor +/-Inf.
func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}
