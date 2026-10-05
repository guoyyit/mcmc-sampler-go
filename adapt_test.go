package mcmc

import (
	"math"
	"testing"
)

func TestStepSizeAdaptMovesToAcceptRange(t *testing.T) {
	// 1-D standard normal, random walk MH: too-small proposals have
	// acceptance ~1, so adaptation must grow the step size away from the
	// tiny initial value.
	cfg := baseConfig(normalTarget([]float64{0}, []float64{1}), AlgoMH)
	cfg.Chains = 1
	cfg.Warmup = 300
	cfg.Iterations = 350
	cfg.MHStepSize = 0.001
	cfg.Adapt = AdaptStepSize
	cfg.TargetAcceptLow = 0.3
	cfg.TargetAcceptHigh = 0.5
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Run(RunOptions{InitialPoints: [][]float64{{0}}})
	if err != nil {
		t.Fatal(err)
	}
	frozen := s.chains[0].engine.Params().StepSize
	if frozen <= 0.01 {
		t.Fatalf("step size did not grow: %v", frozen)
	}
	if math.IsNaN(frozen) {
		t.Fatal("frozen step is NaN")
	}
	if res.WarmupParams[0].StepSize != frozen {
		t.Fatal("params changed after freeze")
	}
}

func TestStepSizeAdaptShrinks(t *testing.T) {
	// huge proposals in 1-D are almost always rejected: step must shrink
	cfg := baseConfig(normalTarget([]float64{0}, []float64{1}), AlgoMH)
	cfg.Chains = 1
	cfg.Warmup = 300
	cfg.Iterations = 350
	cfg.MHStepSize = 100
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(RunOptions{InitialPoints: [][]float64{{0}}}); err != nil {
		t.Fatal(err)
	}
	frozen := s.chains[0].engine.Params().StepSize
	if frozen >= 100 {
		t.Fatalf("step size did not shrink: %v", frozen)
	}
}

func TestHMCMassAdaptation(t *testing.T) {
	target := normalTarget([]float64{0, 0}, []float64{1, 5})
	cfg := baseConfig(target, AlgoHMC)
	cfg.Chains = 1
	cfg.Warmup = 300
	cfg.Iterations = 400
	cfg.Adapt = AdaptStepSizeMass
	cfg.HMCStepSize = 0.1
	cfg.TrajectorySteps = 15
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Run(RunOptions{InitialPoints: [][]float64{{0, 0}}})
	if err != nil {
		t.Fatal(err)
	}
	mass := res.WarmupParams[0].Mass
	if len(mass) != 2 {
		t.Fatalf("expected diagonal mass of dim 2, got %v", mass)
	}
	if !(mass[1] > mass[0]) {
		t.Fatalf("mass did not learn anisotropy: %v", mass)
	}
}

func TestAdaptNoneKeepsStep(t *testing.T) {
	cfg := baseConfig(normalTarget([]float64{0}, []float64{1}), AlgoMH)
	cfg.Chains = 1
	cfg.Warmup = 0
	cfg.Iterations = 50
	cfg.MHStepSize = 0.25
	cfg.Adapt = AdaptNone
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(RunOptions{InitialPoints: [][]float64{{0}}}); err != nil {
		t.Fatal(err)
	}
	if got := s.chains[0].engine.Params().StepSize; got != 0.25 {
		t.Fatalf("step changed without adaptation: %v", got)
	}
}

func TestAdapterRestoreSnapshot(t *testing.T) {
	cfg := baseConfig(normalTarget([]float64{0}, []float64{1}), AlgoMH)
	a1 := newAdapter(&cfg, 1, AlgoMH, 0.1)
	for i := 0; i < 20; i++ {
		a1.Observe(Draw{Accepted: true, AcceptProb: 0.95}, []float64{0}, -1, nil)
	}
	snap := a1.Snapshot()
	a2 := newAdapter(&cfg, 1, AlgoMH, 0.1)
	if err := a2.Restore(snap); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		a1.Observe(Draw{Accepted: false, AcceptProb: 0.1}, []float64{0}, -1, nil)
		a2.Observe(Draw{Accepted: false, AcceptProb: 0.1}, []float64{0}, -1, nil)
	}
	if a1.Params().StepSize != a2.Params().StepSize {
		t.Fatalf("adapter replay diverged: %v vs %v",
			a1.Params().StepSize, a2.Params().StepSize)
	}
}
