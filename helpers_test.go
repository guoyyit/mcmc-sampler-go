package mcmc

import (
	"math"
)

// normalTarget builds a diagonal-Gaussian target with the given means and
// standard deviations, plus an analytic gradient.
func normalTarget(mean, std []float64) Target {
	dim := len(mean)
	prec := make([]float64, dim)
	for i := range prec {
		prec[i] = 1 / (std[i] * std[i])
	}
	logConst := 0.0
	for _, s := range std {
		logConst -= math.Log(2*math.Pi) / 2
		logConst -= math.Log(s)
	}
	t := Target{Dim: dim}
	t.LogProb = func(theta []float64) (float64, bool) {
		lp := logConst
		for i := 0; i < dim; i++ {
			d := theta[i] - mean[i]
			lp -= 0.5 * d * d * prec[i]
		}
		return lp, true
	}
	t.GradLogProb = func(theta []float64) ([]float64, bool) {
		g := make([]float64, dim)
		for i := range g {
			g[i] = -(theta[i] - mean[i]) * prec[i]
		}
		return g, true
	}
	return t
}

// halfPlaneTarget rejects points with negative first coordinate.
func halfPlaneTarget(dim int) Target {
	t := Target{
		Dim: dim,
		LogProb: func(theta []float64) (float64, bool) {
			if theta[0] < 0 {
				return 0, false
			}
			return -0.5 * theta[0] * theta[0], true
		},
	}
	t.GradLogProb = func(theta []float64) ([]float64, bool) {
		if theta[0] < 0 {
			return nil, false
		}
		g := make([]float64, dim)
		g[0] = -theta[0]
		return g, true
	}
	return t
}

func baseConfig(t Target, algo Algorithm) Config {
	cfg := Config{
		Target:           t,
		Algo:             algo,
		Seed:             42,
		Chains:           2,
		Iterations:       400,
		Warmup:           200,
		Adapt:            AdaptStepSize,
		RHatThreshold:    2.0,
		ESSTarget:        10,
		TargetAcceptLow:  0.6,
		TargetAcceptHigh: 0.9,
		History:          HistoryConfig{Mode: RetainAll},
	}
	return cfg
}

func floatsApprox(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}
