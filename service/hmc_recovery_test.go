package service

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/guoyyit/mcmc-sampler-go/checkpoint"
)

func TestHMCCrashRecoveryMatchesUninterrupted(t *testing.T) {
	mkCfg := func(dir string) Config {
		return Config{
			JobID:              "hmc-crash",
			NumChains:          4,
			Draws:              120_000,
			Warmup:             30_000,
			Seed:               31415,
			Sampler:            sampler2D(),
			Adapt:              adaptCfg(30_000),
			History:            historyPol(),
			Thresholds:         thr(),
			CheckpointDir:      dir,
			CheckpointEvery:    1,
			CheckpointInterval: 5 * time.Millisecond,
		}
	}

	refDir := filepath.Join(t.TempDir(), "ref")
	ref, err := Run(context.Background(), mkCfg(refDir), anisotropic{})
	if err != nil {
		t.Fatal(err)
	}

	// Interrupt deterministically: cancel as soon as the first checkpoint is
	// durable, guaranteeing at least one resumable state was committed.
	intDir := filepath.Join(t.TempDir(), "int")
	cfg := mkCfg(intDir)
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		path := filepath.Join(intDir, checkpoint.FileName)
		deadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(deadline) {
			if _, err := os.Stat(path); err == nil {
				time.Sleep(5 * time.Millisecond) // let a few more commits land
				cancel()
				return
			}
			time.Sleep(time.Millisecond)
		}
		cancel()
	}()
	if _, err := Run(ctx, cfg, anisotropic{}); err == nil ||
		(!errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded)) {
		t.Fatalf("expected interruption, got %v", err)
	}

	resumed, err := Run(context.Background(), mkCfg(intDir), anisotropic{})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if !resumed.Resumed {
		t.Fatal("not flagged as resumed")
	}
	for c := range ref.Draws {
		if len(resumed.Draws[c]) != len(ref.Draws[c]) {
			t.Fatalf("chain %d count %d != %d", c, len(resumed.Draws[c]), len(ref.Draws[c]))
		}
		for i := range ref.Draws[c] {
			for d := 0; d < 2; d++ {
				if resumed.Draws[c][i][d] != ref.Draws[c][i][d] {
					t.Fatalf("HMC recovery mismatch chain %d draw %d dim %d", c, i, d)
				}
			}
		}
	}
}
