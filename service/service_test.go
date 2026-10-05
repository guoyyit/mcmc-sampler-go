package service

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"testing"
	"time"

	"github.com/guoyyit/mcmc-sampler-go/checkpoint"
	"github.com/guoyyit/mcmc-sampler-go/diagnostics"
	"github.com/guoyyit/mcmc-sampler-go/history"
	"github.com/guoyyit/mcmc-sampler-go/sampler"
)

// stdNorm is a 2D standard normal with analytic gradient.
type stdNorm struct{}

func (stdNorm) Dim() int { return 2 }
func (stdNorm) LogDensity(x []float64) float64 {
	return -0.5 * (x[0]*x[0] + x[1]*x[1])
}
func (stdNorm) Gradient(x, g []float64) {
	g[0], g[1] = -x[0], -x[1]
}

func testConfig(seed uint64) Config {
	return Config{
		JobID:     "repro",
		NumChains: 4,
		Draws:     600,
		Warmup:    200,
		Seed:      seed,
		Sampler: sampler.Params{
			Algorithm:     sampler.AlgoMH,
			Initial:       []float64{0, 0},
			InitJitter:    []float64{1, 1},
			ProposalScale: []float64{1, 1},
		},
		History:            history.Policy{},
		Thresholds:         diagnostics.Thresholds{MaxRhat: 1.05, MinESS: 50},
		CheckpointEvery:    1,
		CheckpointInterval: 5 * time.Millisecond,
	}
}

func TestSeedReproducibility(t *testing.T) {
	cfg1 := testConfig(12345)
	cfg1.CheckpointDir = ""
	cfg2 := testConfig(12345)
	cfg2.CheckpointDir = ""

	r1, err := Run(context.Background(), cfg1, stdNorm{})
	if err != nil {
		t.Fatal(err)
	}
	r2, err := Run(context.Background(), cfg2, stdNorm{})
	if err != nil {
		t.Fatal(err)
	}
	for c := range r1.Draws {
		if len(r1.Draws[c]) != len(r2.Draws[c]) {
			t.Fatalf("chain %d draw count differs", c)
		}
		for i := range r1.Draws[c] {
			for d := 0; d < 2; d++ {
				if r1.Draws[c][i][d] != r2.Draws[c][i][d] {
					t.Fatalf("chain %d draw %d dim %d differs: %.17g != %.17g",
						c, i, d, r1.Draws[c][i][d], r2.Draws[c][i][d])
				}
			}
		}
	}
}

func TestDifferentSeedsDiffer(t *testing.T) {
	cfg1, cfg2 := testConfig(1), testConfig(2)
	cfg1.Thresholds.MaxRhat, cfg2.Thresholds.MaxRhat = 1.2, 1.2
	r1, err := Run(context.Background(), cfg1, stdNorm{})
	if err != nil {
		t.Fatal(err)
	}
	r2, err := Run(context.Background(), cfg2, stdNorm{})
	if err != nil {
		t.Fatal(err)
	}
	same := true
	for c := range r1.Draws {
		for i := range r1.Draws[c] {
			if r1.Draws[c][i][0] != r2.Draws[c][i][0] {
				same = false
			}
		}
	}
	if same {
		t.Fatal("different seeds produced identical sample sequences")
	}
}

func TestChainsAreDispersed(t *testing.T) {
	cfg := testConfig(77)
	cfg.CheckpointEvery = 0
	r, err := Run(context.Background(), cfg, stdNorm{})
	if err != nil {
		t.Fatal(err)
	}
	first := make([]float64, cfg.NumChains)
	for c := 0; c < cfg.NumChains; c++ {
		first[c] = r.Draws[c][0][0]
	}
	for i := 1; i < len(first); i++ {
		if first[i] == first[0] {
			t.Fatalf("chains %d and 0 started identically: %v", i, first)
		}
	}
}

func TestCrashRecoveryMatchesUninterrupted(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "cp")
	cfg := testConfig(999)
	cfg.CheckpointDir = dir
	cfg.Draws = 200_000
	cfg.Warmup = 50_000
	cfg.Thresholds = diagnostics.Thresholds{MaxRhat: 1.2, MinESS: 1}

	// Uninterrupted reference.
	ref, err := Run(context.Background(), cfg, stdNorm{})
	if err != nil {
		t.Fatal(err)
	}

	// Interrupted run: cancel shortly after start, then resume.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	_, err = Run(ctx, cfg, stdNorm{})
	cancel()
	if err == nil {
		t.Fatal("expected cancellation error")
	}
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected error: %v", err)
	}

	// The checkpoint must exist and be internally valid.
	resumed, err := Run(context.Background(), cfg, stdNorm{})
	if err != nil {
		t.Fatalf("resume failed: %v", err)
	}
	if !resumed.Resumed {
		t.Fatal("resumed result not flagged as resumed")
	}

	// Every post-warmup draw must match the uninterrupted run bit-for-bit.
	for c := range ref.Draws {
		if len(resumed.Draws[c]) != len(ref.Draws[c]) {
			t.Fatalf("chain %d draw count %d != reference %d",
				c, len(resumed.Draws[c]), len(ref.Draws[c]))
		}
		for i := range ref.Draws[c] {
			for d := 0; d < 2; d++ {
				if resumed.Draws[c][i][d] != ref.Draws[c][i][d] {
					t.Fatalf("chain %d draw %d dim %d mismatch after recovery: %.17g != %.17g",
						c, i, d, resumed.Draws[c][i][d], ref.Draws[c][i][d])
				}
			}
		}
	}
}

func TestCancellationStopsAllChains(t *testing.T) {
	cfg := testConfig(42)
	cfg.CheckpointEvery = 0
	cfg.Draws = 200_000
	cfg.Warmup = 50_000
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()
	start := time.Now()
	_, err := Run(ctx, cfg, stdNorm{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if time.Since(start) > 2*time.Second {
		t.Fatal("chains did not stop promptly after cancellation")
	}
}

func TestNonConvergenceIsExplicit(t *testing.T) {
	cfg := testConfig(5)
	cfg.Thresholds = diagnostics.Thresholds{MaxRhat: 1.0001, MinESS: 1_000_000}
	res, err := Run(context.Background(), cfg, stdNorm{})
	if err == nil {
		t.Fatal("expected non-convergence error")
	}
	if !errors.Is(err, ErrNotConverged) {
		t.Fatalf("expected ErrNotConverged, got %v", err)
	}
	if res != nil {
		t.Fatal("samples must not be returned when not converged")
	}
}

func TestInvalidConfigRejected(t *testing.T) {
	cfg := testConfig(1)
	cfg.NumChains = 1
	if _, err := Run(context.Background(), cfg, stdNorm{}); err == nil {
		t.Fatal("expected error for single chain")
	}
	cfg = testConfig(1)
	cfg.Warmup = cfg.Draws
	if _, err := Run(context.Background(), cfg, stdNorm{}); err == nil {
		t.Fatal("expected error when warmup == draws")
	}
}

func TestResultsFinite(t *testing.T) {
	cfg := testConfig(2024)
	r, err := Run(context.Background(), cfg, stdNorm{})
	if err != nil {
		t.Fatal(err)
	}
	for c := range r.Draws {
		for _, x := range r.Draws[c] {
			for _, v := range x {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatal("non-finite value in results")
				}
			}
		}
	}
}

// explodingTarget returns NaN density once a chain crosses x[0] > 0.4,
// causing that chain to fail deterministically while others keep sampling.
type explodingTarget struct{}

func (explodingTarget) Dim() int { return 1 }
func (explodingTarget) LogDensity(x []float64) float64 {
	if x[0] > 0.4 {
		return math.NaN()
	}
	return -0.5 * x[0] * x[0]
}

func TestFailingChainDoesNotCorruptOthers(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "cp")
	cfg := Config{
		JobID:              "fault",
		NumChains:          3,
		Draws:              200_000,
		Warmup:             50_000,
		Seed:               31337,
		Sampler:            sampler.Params{Algorithm: sampler.AlgoMH, Initial: []float64{0}, ProposalScale: []float64{2}},
		History:            history.Policy{},
		Thresholds:         diagnostics.Thresholds{MaxRhat: 2, MinESS: 1},
		CheckpointDir:      dir,
		CheckpointEvery:    1,
		CheckpointInterval: 5 * time.Millisecond,
	}
	_, err := Run(context.Background(), cfg, explodingTarget{})
	if err == nil {
		t.Fatal("expected an error from the exploding chain")
	}

	// All chains must have committed snapshots; the chains that never hit the
	// NaN region must hold finite, well-formed state.
	fs, lerr := checkpointStore(dir)
	if lerr != nil {
		t.Fatal(lerr)
	}
	cp, lerr := fs.Load()
	if lerr != nil {
		t.Fatal(lerr)
	}
	if len(cp.Chains) != 3 {
		t.Fatalf("expected 3 snapshots, got %d", len(cp.Chains))
	}
	for i, snap := range cp.Chains {
		if snap.ID != i {
			t.Fatalf("snapshot order/id mismatch at %d: %d", i, snap.ID)
		}
		if len(snap.RNG) == 0 || len(snap.Sampler) == 0 {
			t.Fatalf("chain %d snapshot incomplete", i)
		}
	}
}

func TestCancelledCheckpointResumesCleanly(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "cp-cancel")
	cfg := testConfig(555)
	cfg.CheckpointDir = dir
	cfg.Draws = 400_000
	cfg.Warmup = 100_000
	cfg.Thresholds = diagnostics.Thresholds{MaxRhat: 1.2, MinESS: 1}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(40 * time.Millisecond)
		cancel()
	}()
	if _, err := Run(ctx, cfg, stdNorm{}); err == nil {
		t.Fatal("expected cancellation error")
	}

	// The committed checkpoint must load directly from disk.
	fs, err := checkpoint.NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cp, err := fs.Load()
	if err != nil {
		t.Fatalf("checkpoint invalid after cancellation: %v", err)
	}
	if cp.NumChains != cfg.NumChains {
		t.Fatalf("chain count %d", cp.NumChains)
	}
	progressed := false
	for _, snap := range cp.Chains {
		if snap.Iteration > 0 {
			progressed = true
		}
	}
	if !progressed {
		t.Fatal("no chain made progress before cancellation")
	}

	// Resume to completion without error.
	res, err := Run(context.Background(), cfg, stdNorm{})
	if err != nil {
		t.Fatalf("resume after cancellation failed: %v", err)
	}
	if !res.Converged {
		t.Fatal("resumed run did not converge")
	}
}

func TestChainsProduceIndependentSequences(t *testing.T) {
	cfg := testConfig(707)
	r, err := Run(context.Background(), cfg, stdNorm{})
	if err != nil {
		t.Fatal(err)
	}
	// No two chains may share an identical sequence.
	for a := 0; a < cfg.NumChains; a++ {
		for b := a + 1; b < cfg.NumChains; b++ {
			identical := true
			for i := 0; i < len(r.Draws[a]); i++ {
				if r.Draws[a][i][0] != r.Draws[b][i][0] {
					identical = false
					break
				}
			}
			if identical {
				t.Fatalf("chains %d and %d produced identical sequences", a, b)
			}
		}
	}
}
