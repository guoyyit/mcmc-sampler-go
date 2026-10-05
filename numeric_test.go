package mcmc

import (
	"math"
	"testing"
)

func TestEvalIllegalAndNonFinite(t *testing.T) {
	target := Target{
		Dim: 1,
		LogProb: func(theta []float64) (float64, bool) {
			if theta[0] < 0 {
				return 0, false
			}
			return -1e100, true
		},
	}
	// illegal support -> -infinity, no error-accept path
	lp, err := target.Eval([]float64{-1})
	if err != nil || !math.IsInf(lp, -1) {
		t.Fatalf("illegal point: lp=%v err=%v", lp, err)
	}
	// very small but finite log prob stays finite
	lp, err = target.Eval([]float64{1})
	if err != nil || lp != -1e100 {
		t.Fatalf("finite point: lp=%v err=%v", lp, err)
	}
	// NaN/Inf parameter rejected explicitly
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := target.Eval([]float64{bad}); err == nil {
			t.Fatalf("expected rejection for %v", bad)
		}
	}
}

func TestEvalRejectsNonFiniteReturn(t *testing.T) {
	for _, bad := range []float64{math.NaN(), math.Inf(1)} {
		target := Target{
			Dim:     1,
			LogProb: func([]float64) (float64, bool) { return bad, true },
		}
		if _, err := target.Eval([]float64{0}); err == nil {
			t.Fatalf("expected error for return value %v", bad)
		}
	}
}

func TestLogAcceptNoUnderflow(t *testing.T) {
	// extremely negative log probabilities must not produce bogus accepts
	if la := logAccept(-1e308, 0); la != -1e308 {
		t.Fatalf("expected -1e308, got %v", la)
	}
	if la := logAccept(math.Inf(-1), 0); !math.IsInf(la, -1) {
		t.Fatalf("expected -inf, got %v", la)
	}
	rng := newSplitMix(1)
	if got, _ := acceptWith(rng, -1e308); got {
		t.Fatal("tiny alpha must never be accepted via underflow")
	}
	// current point impossible, proposal legal -> accept
	if la := logAccept(-1, math.Inf(-1)); la != 0 {
		t.Fatalf("expected 0, got %v", la)
	}
	// both impossible -> reject
	if la := logAccept(math.Inf(-1), math.Inf(-1)); !math.IsInf(la, -1) {
		t.Fatalf("expected -inf, got %v", la)
	}
	// NaN -> reject
	if la := logAccept(math.NaN(), 0); !math.IsInf(la, -1) {
		t.Fatalf("expected -inf for NaN")
	}
}

func TestMHNeverLeavesHalfPlane(t *testing.T) {
	target := halfPlaneTarget(1)
	cfg := Config{
		Target:     target,
		Algo:       AlgoMH,
		Seed:       7,
		Chains:     1,
		Iterations: 500,
		Warmup:     0,
		Adapt:      AdaptNone,
		MHStepSize: 5.0,
		History:    HistoryConfig{Mode: RetainAll},
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// force a legal but close-to-boundary start
	res, err := s.Run(RunOptions{InitialPoints: [][]float64{{0.01}}})
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range res.Samples[0] {
		if d.Theta[0] < 0 || math.IsNaN(d.Theta[0]) {
			t.Fatalf("chain left support: %v", d.Theta[0])
		}
	}
}

func TestHMCNeverDivergesToNonFinite(t *testing.T) {
	target := halfPlaneTarget(2)
	cfg := Config{
		Target:          target,
		Algo:            AlgoHMC,
		Seed:            9,
		Chains:          1,
		Iterations:      300,
		Warmup:          100,
		Adapt:           AdaptStepSize,
		HMCStepSize:     0.05,
		TrajectorySteps: 20,
		RHatThreshold:   10,
		ESSTarget:       1,
		History:         HistoryConfig{Mode: RetainAll},
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Run(RunOptions{InitialPoints: [][]float64{{0.5, 0}}})
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range res.Samples[0] {
		for _, x := range d.Theta {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				t.Fatalf("non-finite draw: %v", d.Theta)
			}
		}
		if d.Theta[0] < 0 {
			t.Fatalf("left support: %v", d.Theta)
		}
	}
}
