package mcmc

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func recoveryConfig(t *testing.T, path string) Config {
	t.Helper()
	cfg := baseConfig(normalTarget([]float64{0.5, -1}, []float64{1, 1.5}), AlgoHMC)
	cfg.Chains = 3
	cfg.Warmup = 100
	cfg.Iterations = 400
	cfg.HMCStepSize = 0.1
	cfg.TrajectorySteps = 10
	cfg.Adapt = AdaptStepSize
	cfg.RHatThreshold = 100
	cfg.ESSTarget = 1
	cfg.Checkpoint = CheckpointConfig{Path: path, Every: 25}
	return cfg
}

func TestCrashRecoverySampleEquivalence(t *testing.T) {
	dir := t.TempDir()
	cp := filepath.Join(dir, "state.json")

	// uninterrupted reference run
	refCfg := recoveryConfig(t, "")
	refS, err := New(refCfg)
	if err != nil {
		t.Fatal(err)
	}
	ref, err := refS.Run(RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// interrupted run: run to the checkpoint, then resume in a fresh
	// Sampler process-equivalent (new object, only file + seed available)
	cfg := recoveryConfig(t, cp)
	s1, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s1.Run(RunOptions{}); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(cp)
	if err != nil || info.Size() == 0 {
		t.Fatalf("checkpoint not written: %v", err)
	}

	cfg2 := recoveryConfig(t, cp)
	s2, err := New(cfg2)
	if err != nil {
		t.Fatal(err)
	}
	res2, err := s2.Run(RunOptions{Resume: cp})
	if err != nil {
		t.Fatal(err)
	}
	if !res2.Resumed {
		t.Fatal("resume flag not set")
	}

	// resumed chains complete the full iteration count
	for _, cr := range res2.Chains {
		if cr.Iteration != cfg2.Iterations {
			t.Fatalf("chain %d stopped at %d", cr.Index, cr.Iteration)
		}
	}

	// resumed retained samples must equal the uninterrupted sequence
	if !reflect.DeepEqual(ref.Samples, res2.Samples) {
		for c := 0; c < cfg.Chains; c++ {
			a, b := ref.Samples[c], res2.Samples[c]
			if len(a) != len(b) {
				t.Fatalf("chain %d length %d vs %d", c, len(a), len(b))
			}
			for i := range a {
				if !reflect.DeepEqual(a[i].Theta, b[i].Theta) {
					t.Fatalf("chain %d draw %d diverged after resume: %v vs %v",
						c, i, a[i].Theta, b[i].Theta)
				}
			}
		}
	}
}

func TestResumeIdempotent(t *testing.T) {
	dir := t.TempDir()
	cp := filepath.Join(dir, "state.json")
	cfg := recoveryConfig(t, cp)
	s1, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	res1, err := s1.Run(RunOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// resuming a completed run changes nothing
	s2, err := New(recoveryConfig(t, cp))
	if err != nil {
		t.Fatal(err)
	}
	res2, err := s2.Run(RunOptions{Resume: cp})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(res1.Samples, res2.Samples) {
		t.Fatal("resuming a completed run altered samples")
	}
}

func TestCorruptCheckpointRejected(t *testing.T) {
	dir := t.TempDir()
	cp := filepath.Join(dir, "state.json")
	if err := os.WriteFile(cp, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	s, err := New(recoveryConfig(t, cp))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(RunOptions{Resume: cp}); err == nil {
		t.Fatal("corrupt checkpoint must be rejected")
	}
}

func TestSeedMismatchRejected(t *testing.T) {
	dir := t.TempDir()
	cp := filepath.Join(dir, "state.json")
	cfg := recoveryConfig(t, cp)
	s1, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s1.Run(RunOptions{}); err != nil {
		t.Fatal(err)
	}
	cfg2 := recoveryConfig(t, cp)
	cfg2.Seed++
	s2, err := New(cfg2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s2.Run(RunOptions{Resume: cp}); err == nil {
		t.Fatal("seed mismatch must be rejected")
	}
}
