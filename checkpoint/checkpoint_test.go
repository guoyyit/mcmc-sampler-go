package checkpoint

import (
	"errors"
	"testing"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	s, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Load(); !errors.Is(err, ErrNoCheckpoint) {
		t.Fatalf("expected ErrNoCheckpoint, got %v", err)
	}
	cp := &Checkpoint{
		JobID:     "job-1",
		Seed:      77,
		Draws:     100,
		Warmup:    20,
		NumChains: 2,
		Chains: []ChainSnapshot{
			{ID: 0, Iteration: 5, RNG: []byte{1, 2}, Sampler: []byte{3}, History: []byte{4}},
			{ID: 1, Iteration: 6, RNG: []byte{5, 6}, Sampler: []byte{7}, History: []byte{8}},
		},
	}
	if err := s.Save(cp); err != nil {
		t.Fatal(err)
	}
	got, err := s.Load()
	if err != nil {
		t.Fatal(err)
	}
	if got.JobID != "job-1" || got.Seed != 77 || got.NumChains != 2 {
		t.Fatalf("header mismatch: %+v", got)
	}
	if got.Version != Version {
		t.Fatalf("version %d", got.Version)
	}
	if len(got.Chains[1].RNG) != 2 || got.Chains[1].RNG[0] != 5 {
		t.Fatalf("chain payload not restored: %+v", got.Chains[1])
	}
}

func TestAtomicReplace(t *testing.T) {
	dir := t.TempDir()
	s, _ := NewFileStore(dir)
	cp := &Checkpoint{JobID: "j", NumChains: 1, Chains: []ChainSnapshot{{ID: 0}}}
	for i := 0; i < 20; i++ {
		cp.Chains[0].Iteration = i
		if err := s.Save(cp); err != nil {
			t.Fatal(err)
		}
	}
	got, err := s.Load()
	if err != nil {
		t.Fatal(err)
	}
	if got.Chains[0].Iteration != 19 {
		t.Fatalf("final checkpoint iteration %d", got.Chains[0].Iteration)
	}
}
