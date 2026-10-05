package chain

import (
	"context"
	"testing"
)

func TestZeroWarmupRetainsAllDraws(t *testing.T) {
	cfg := cfg()
	cfg.Warmup = 0
	c, err := New(cfg, norm1{})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := len(c.Draws()); got != 300 {
		t.Fatalf("retained %d draws with warmup=0, want 300", got)
	}
	if c.Draws()[0].Iteration != 1 {
		t.Fatalf("first retained iteration %d, want 1", c.Draws()[0].Iteration)
	}
}
