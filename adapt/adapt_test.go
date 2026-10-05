package adapt

import (
	"math"
	"testing"
)

func TestStepSizeMovesTowardTarget(t *testing.T) {
	e, err := New(Config{
		Warmup:          200,
		TargetAcceptMin: 0.8,
		TargetAcceptMax: 0.8,
		InitStepSize:    1,
		AdaptMass:       false,
	}, 2)
	if err != nil {
		t.Fatal(err)
	}
	start := e.StepSize()
	// Observed acceptance 0.3 is far below target: step size must shrink.
	for iter := 1; iter <= 200; iter++ {
		e.Observe(iter, 0.3, []float64{0, 0})
	}
	e.Freeze()
	final := e.FinalStepSize()
	if !(final < start) {
		t.Fatalf("expected step size to shrink: start %v final %v", start, final)
	}
	if !e.Frozen() {
		t.Fatal("engine should be frozen")
	}
	// After freeze, Observe must not change parameters.
	e.Observe(201, 0.01, []float64{0, 0})
	if e.FinalStepSize() != final {
		t.Fatal("frozen parameters changed after warmup")
	}
}

func TestStepSizeGrowsWhenAcceptanceHigh(t *testing.T) {
	e, _ := New(Config{Warmup: 200, InitStepSize: 0.01, AdaptMass: false}, 1)
	for iter := 1; iter <= 200; iter++ {
		e.Observe(iter, 0.99, []float64{0})
	}
	e.Freeze()
	if !(e.FinalStepSize() > 0.01) {
		t.Fatalf("expected step size to grow, got %v", e.FinalStepSize())
	}
}

func TestMassMatrixLearnsVariance(t *testing.T) {
	dim := 2
	e, _ := New(Config{
		Warmup:    5000,
		AdaptMass: true,
	}, dim)
	// Feed draws from an anisotropic Gaussian: variances 1 and 4.
	rng := newDeterministicMix()
	for iter := 1; iter <= 5000; iter++ {
		u := rng()
		v := rng()
		e.Observe(iter, 0.8, []float64{u, 2 * v})
	}
	e.Freeze()
	mass := e.InverseMass()
	if math.Abs(mass[0]-1) > 0.15 {
		t.Fatalf("inverse mass dim0 = %.3f, want ~1", mass[0])
	}
	if math.Abs(mass[1]-4) > 0.6 {
		t.Fatalf("inverse mass dim1 = %.3f, want ~4", mass[1])
	}
}

func TestInvalidConfig(t *testing.T) {
	if _, err := New(Config{Warmup: -1}, 2); err == nil {
		t.Fatal("expected error for negative warmup")
	}
	if _, err := New(Config{Warmup: 10, TargetAcceptMin: 0.9, TargetAcceptMax: 0.1}, 2); err == nil {
		t.Fatal("expected error for inverted acceptance interval")
	}
	if _, err := New(Config{Warmup: 10, InitStepSize: -1}, 2); err == nil {
		t.Fatal("expected error for negative step size")
	}
	if _, err := New(Config{Warmup: 10}, 0); err == nil {
		t.Fatal("expected error for zero dimension")
	}
}

// newDeterministicMix returns a simple deterministic pseudo-normal stream
// (sum of uniform-like values) good enough for variance tests.
func newDeterministicMix() func() float64 {
	state := uint64(0x12345)
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(state>>11) / float64(1<<53)
	}
	return func() float64 {
		// 12 uniforms - 6 has mean 0 and variance 1.
		return (next() + next() + next() + next() +
			next() + next() + next() + next() +
			next() + next() + next() + next() - 6)
	}
}
