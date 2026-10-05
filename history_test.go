package mcmc

import "testing"

func drawAt(x float64, warmup bool) Draw {
	return Draw{Theta: []float64{x}, LogProb: -x * x / 2, Warmup: warmup}
}

func TestRetainAll(t *testing.T) {
	h := newHistory(HistoryConfig{Mode: RetainAll})
	for i := 0; i < 100; i++ {
		h.Append(drawAt(float64(i), false))
	}
	if got := len(h.Draws()); got != 100 {
		t.Fatalf("retained %d, want 100", got)
	}
	if h.Draws()[99].Theta[0] != 99 {
		t.Fatal("order broken")
	}
}

func TestRetainThinned(t *testing.T) {
	h := newHistory(HistoryConfig{Mode: RetainThinned, Thin: 3})
	for i := 0; i < 10; i++ {
		h.Append(drawAt(float64(i), false))
	}
	d := h.Draws()
	if len(d) != 4 || d[0].Theta[0] != 0 || d[3].Theta[0] != 9 {
		t.Fatalf("thinning wrong: %v", d)
	}
}

func TestRetainWindowBoundedAndOrdered(t *testing.T) {
	h := newHistory(HistoryConfig{Mode: RetainWindow, Window: 5})
	for i := 0; i < 100000; i++ {
		h.Append(drawAt(float64(i), false))
	}
	d := h.Draws()
	if len(d) != 5 {
		t.Fatalf("window unbounded: %d", len(d))
	}
	for k := range d {
		if d[k].Theta[0] != float64(99995+k) {
			t.Fatalf("window order/content wrong at %d: %v", k, d[k].Theta)
		}
	}
	if h.TotalPostWarmup() != 100000 {
		t.Fatal("total counter must include dropped draws")
	}
}

func TestWarmupKeptOnlyWhenConfigured(t *testing.T) {
	h := newHistory(HistoryConfig{Mode: RetainAll, KeepWarmup: true})
	h.Append(drawAt(0, true))
	h.Append(drawAt(1, false))
	if len(h.WarmupDraws()) != 1 || len(h.Draws()) != 1 {
		t.Fatal("warmup retention wrong")
	}
	h2 := newHistory(HistoryConfig{Mode: RetainAll})
	h2.Append(drawAt(0, true))
	if len(h2.WarmupDraws()) != 0 {
		t.Fatal("warmup should be dropped by default")
	}
}

func TestHistorySnapshotRoundTrip(t *testing.T) {
	h := newHistory(HistoryConfig{Mode: RetainWindow, Window: 3})
	for i := 0; i < 10; i++ {
		h.Append(drawAt(float64(i), false))
	}
	snap := h.Snapshot()
	h2 := newHistory(HistoryConfig{Mode: RetainWindow, Window: 3})
	if err := h2.Restore(snap); err != nil {
		t.Fatal(err)
	}
	d1, d2 := h.Draws(), h2.Draws()
	if len(d1) != len(d2) {
		t.Fatal("length mismatch")
	}
	for i := range d1 {
		if d1[i].Theta[0] != d2[i].Theta[0] {
			t.Fatalf("content mismatch at %d", i)
		}
	}
	// appending after restore keeps the ring semantics
	h2.Append(drawAt(100, false))
	last := h2.Draws()
	if last[len(last)-1].Theta[0] != 100 {
		t.Fatal("append after restore broken")
	}
}
