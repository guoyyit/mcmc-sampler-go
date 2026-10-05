package chain

import (
	"context"
	"testing"

	"github.com/guoyyit/mcmc-sampler-go/history"
	"github.com/guoyyit/mcmc-sampler-go/sampler"
)

type norm1 struct{}

func (norm1) Dim() int                       { return 1 }
func (norm1) LogDensity(x []float64) float64 { return -0.5 * x[0] * x[0] }
func (norm1) Gradient(x, g []float64)        { g[0] = -x[0] }

func cfg() Config {
	return Config{
		ID:            0,
		Seed:          4242,
		Draws:         300,
		Warmup:        50,
		Sampler:       sampler.Params{Algorithm: sampler.AlgoMH, Initial: []float64{0}, ProposalScale: []float64{1}},
		HistoryPolicy: history.Policy{},
	}
}

func TestChainSnapshotRestoreContinuation(t *testing.T) {
	// Uninterrupted reference.
	ref, err := New(cfg(), norm1{})
	if err != nil {
		t.Fatal(err)
	}
	if err := ref.Run(context.Background()); err != nil {
		t.Fatal(err)
	}

	// Run halfway, snapshot, restore, finish.
	c, err := New(cfg(), norm1{})
	if err != nil {
		t.Fatal(err)
	}
	half := cfg()
	half.Draws = 150
	c.cfg.Draws = 150
	if err := c.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	snap, err := c.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if snap.Iteration != 150 {
		t.Fatalf("snapshot iteration %d", snap.Iteration)
	}
	fullCfg := cfg()
	c2, err := Restore(fullCfg, norm1{}, snap)
	if err != nil {
		t.Fatal(err)
	}
	if err := c2.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	if c2.Iteration() != 300 {
		t.Fatalf("restored chain iteration %d", c2.Iteration())
	}

	rd, cd := ref.Draws(), c2.Draws()
	if len(rd) != len(cd) {
		t.Fatalf("draw count %d != %d", len(cd), len(rd))
	}
	for i := range rd {
		if rd[i].Position[0] != cd[i].Position[0] || rd[i].Iteration != cd[i].Iteration {
			t.Fatalf("draw %d mismatch after restore: %v != %v", i, cd[i], rd[i])
		}
	}
}

func TestChainCancel(t *testing.T) {
	c, err := New(cfg(), norm1{})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := c.Run(ctx); err == nil {
		t.Fatal("expected cancellation error")
	}
	if c.Iteration() != 0 {
		t.Fatal("no iterations should have run")
	}
}
