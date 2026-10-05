package mcmc

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func TestCancelStopsImmediatelyNoDiagnostics(t *testing.T) {
	dir := t.TempDir()
	cfg := baseConfig(normalTarget([]float64{0}, []float64{1}), AlgoMH)
	cfg.Chains = 3
	cfg.Warmup = 50
	cfg.Iterations = 1000000
	cfg.Checkpoint = CheckpointConfig{Path: dir + "/cp.json", Every: 50000}
	cfg.History = HistoryConfig{Mode: RetainWindow, Window: 200}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()
	start := time.Now()
	res, err := s.Run(RunOptions{Ctx: ctx})
	if err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)
	if elapsed > 2*time.Second {
		t.Fatal("cancel did not stop chains promptly")
	}
	if res.Diagnostics != nil {
		t.Fatal("partial diagnostics must not be emitted on cancel")
	}
	if res.Converged {
		t.Fatal("canceled run cannot be converged")
	}
	if res.Reason == "" {
		t.Fatal("canceled run needs an explicit reason")
	}
	for _, cr := range res.Chains {
		if cr.Status != ChainCanceled {
			t.Fatalf("chain %d status %s", cr.Index, cr.Status)
		}
	}
	// checkpoint from before cancellation remains loadable
	resumeStart := time.Now()
	s2, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	resumeCtx, resumeCancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
	defer resumeCancel()
	if _, err := s2.Run(RunOptions{Resume: dir + "/cp.json", Ctx: resumeCtx}); err != nil {
		t.Fatalf("committed checkpoint invalid after cancel: %v", err)
	}
	t.Logf("cancel elapsed=%v resume=%v", elapsed, time.Since(resumeStart))
}

func TestTimeoutCancelsAllChains(t *testing.T) {
	cfg := baseConfig(normalTarget([]float64{0}, []float64{1}), AlgoMH)
	cfg.Chains = 4
	cfg.Warmup = 10
	cfg.Iterations = 1000000
	cfg.History = HistoryConfig{Mode: RetainWindow, Window: 100}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	res, err := s.Run(RunOptions{Ctx: ctx})
	if err != nil {
		t.Fatal(err)
	}
	canceled := 0
	for _, cr := range res.Chains {
		if cr.Status == ChainCanceled {
			canceled++
		}
	}
	if canceled != cfg.Chains {
		t.Fatalf("expected %d canceled chains, got %d", cfg.Chains, canceled)
	}
}

func TestOneChainFailureIsolatesOthers(t *testing.T) {
	var chain0Panic bool
	var evalCount int64
	target := Target{
		Dim: 1,
		LogProb: func(theta []float64) (float64, bool) {
			x := theta[0]
			n := atomic.AddInt64(&evalCount, 1)
			if chain0Panic && x > 100 && n > 3 {
				panic("boom from target")
			}
			return -0.5 * theta[0] * theta[0], true
		},
	}
	cfg := baseConfig(target, AlgoMH)
	cfg.Chains = 3
	cfg.Warmup = 10
	cfg.Iterations = 60
	cfg.RHatThreshold = 100
	cfg.ESSTarget = 1
	inits := [][]float64{{1}, {-1}, {-1}}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	chain0Panic = true
	// chain 0 starts at theta=101: its first proposal evaluation panics;
	// other chains live around -1 and never reach that region.
	inits = [][]float64{{101}, {-1}, {-1}}
	res, err := s.Run(RunOptions{InitialPoints: inits})
	if err != nil {
		t.Fatal(err)
	}
	if res.Chains[0].Status != ChainFailed || res.Chains[0].Error == "" {
		t.Fatalf("chain 0 should fail: %+v", res.Chains[0])
	}
	for c := 1; c < 3; c++ {
		if res.Chains[c].Status != ChainComplete {
			t.Fatalf("chain %d affected by chain 0 failure: %s", c, res.Chains[c].Status)
		}
	}
	if res.Converged {
		t.Fatal("run with a failed chain must not report convergence")
	}
}
