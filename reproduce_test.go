package mcmc

import (
	"reflect"
	"testing"
)

func runOnce(t *testing.T, algo Algorithm, seed uint64, opts RunOptions) *Result {
	t.Helper()
	cfg := baseConfig(normalTarget([]float64{1, -2}, []float64{1, 2}), algo)
	cfg.Seed = seed
	cfg.RHatThreshold = 100
	cfg.ESSTarget = 1
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Run(opts)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func TestSeedReproducibilityMH(t *testing.T) {
	opts := RunOptions{}
	r1 := runOnce(t, AlgoMH, 123, opts)
	r2 := runOnce(t, AlgoMH, 123, opts)
	r3 := runOnce(t, AlgoMH, 999, opts)
	if !reflect.DeepEqual(r1.Samples, r2.Samples) {
		t.Fatal("same seed produced different MH sample sequences")
	}
	if reflect.DeepEqual(r1.Samples, r3.Samples) {
		t.Fatal("different seeds produced identical MH sequences")
	}
}

func TestSeedReproducibilityHMC(t *testing.T) {
	opts := RunOptions{}
	r1 := runOnce(t, AlgoHMC, 321, opts)
	r2 := runOnce(t, AlgoHMC, 321, opts)
	if !reflect.DeepEqual(r1.Samples, r2.Samples) {
		t.Fatal("same seed produced different HMC sample sequences")
	}
}

func TestInitialPointsDeterministic(t *testing.T) {
	// two samplers created independently must scatter chains identically
	dim := 3
	rng1 := newSplitMix(chainSeed(5, 1, 4))
	rng2 := newSplitMix(chainSeed(5, 1, 4))
	p1 := initialPoint(dim, nil, defaultScale(dim), 1, 4, rng1)
	p2 := initialPoint(dim, nil, defaultScale(dim), 1, 4, rng2)
	if !reflect.DeepEqual(p1, p2) {
		t.Fatal("initial points depend on something other than seed/index")
	}
	rngOther := newSplitMix(chainSeed(5, 2, 4))
	p3 := initialPoint(dim, nil, defaultScale(dim), 2, 4, rngOther)
	if reflect.DeepEqual(p1, p3) {
		t.Fatal("chains were not dispersed to different initial points")
	}
}

func TestRNGSnapshotReproduces(t *testing.T) {
	r := newSplitMix(77)
	seq := make([]float64, 10)
	for i := 0; i < 5; i++ {
		seq[i] = r.Float64()
	}
	snap := r.Snapshot()
	for i := 5; i < 10; i++ {
		seq[i] = r.Float64()
	}
	r2 := newSplitMix(0)
	if err := r2.Restore(snap); err != nil {
		t.Fatal(err)
	}
	for i := 5; i < 10; i++ {
		if r2.Float64() != seq[i] {
			t.Fatalf("RNG restore mismatch at %d", i)
		}
	}
}

func TestAdaptiveWarmupFreezesParams(t *testing.T) {
	cfg := baseConfig(normalTarget([]float64{0}, []float64{1}), AlgoMH)
	cfg.Chains = 1
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.Run(RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// after warmup, step size must be frozen: all post-warmup engine
	// parameters equal the single reported frozen value
	ch := s.chains[0]
	step := ch.engine.Params().StepSize
	if step <= 0 {
		t.Fatal("non-positive frozen step")
	}
	if !ch.adapter.Frozen() {
		t.Fatal("adapter not frozen after warmup")
	}
	if res.WarmupParams[0].StepSize != step {
		t.Fatal("reported warmup params diverge from engine")
	}
}
