package mcmc

import (
	"math"
	"testing"
)

func TestRhatOneForIdenticalStationaryChains(t *testing.T) {
	rng := newSplitMix(11)
	chains := make([][]float64, 4)
	for c := range chains {
		chains[c] = make([]float64, 2000)
		for i := range chains[c] {
			chains[c][i] = rng.NormalFloat64()
		}
	}
	r := splitRhat(chains)
	if math.IsNaN(r) || math.Abs(r-1) > 0.02 {
		t.Fatalf("R-hat for iid chains = %v, want ~1", r)
	}
}

func TestRhatLargeForShiftedChains(t *testing.T) {
	rng := newSplitMix(12)
	chains := make([][]float64, 4)
	for c := range chains {
		chains[c] = make([]float64, 1000)
		shift := float64(c * 3)
		for i := range chains[c] {
			chains[c][i] = shift + rng.NormalFloat64()
		}
	}
	if r := splitRhat(chains); r < 1.5 {
		t.Fatalf("R-hat for shifted chains = %v, want > 1.5", r)
	}
}

func TestESSReasonableForIID(t *testing.T) {
	rng := newSplitMix(13)
	chains := make([][]float64, 4)
	for c := range chains {
		chains[c] = make([]float64, 2000)
		for i := range chains[c] {
			chains[c][i] = rng.NormalFloat64()
		}
	}
	if e := essBulk(chains); e < 5000 {
		t.Fatalf("bulk ESS for iid = %v, want > 5000", e)
	}
	if e := essTail(chains); e < 3000 {
		t.Fatalf("tail ESS for iid = %v, want > 3000", e)
	}
}

func TestNonConvergenceIsExplicit(t *testing.T) {
	// very few post-warmup draws + strict threshold => explicit failure
	cfg := baseConfig(normalTarget([]float64{0}, []float64{1}), AlgoMH)
	cfg.Chains = 2
	cfg.Warmup = 10
	cfg.Iterations = 20
	cfg.ESSTarget = 100000
	cfg.RHatThreshold = 1.001
	cfg.MHStepSize = 0.5
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Run(RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Converged {
		t.Fatal("expected non-converged verdict")
	}
	if res.Reason == "" || res.Diagnostics == nil {
		t.Fatal("non-convergence must be explicit with diagnostics")
	}
}

func TestConvergedRun(t *testing.T) {
	cfg := baseConfig(normalTarget([]float64{0}, []float64{1}), AlgoMH)
	cfg.Chains = 4
	cfg.Warmup = 500
	cfg.Iterations = 3000
	cfg.MHStepSize = 0.9
	cfg.Adapt = AdaptNone
	cfg.RHatThreshold = 1.05
	cfg.ESSTarget = 200
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Run(RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Converged {
		t.Fatalf("expected convergence, reason: %s", res.Reason)
	}
	if math.Abs(res.Diagnostics.Mean[0]) > 0.15 {
		t.Fatalf("posterior mean off: %v", res.Diagnostics.Mean[0])
	}
}
