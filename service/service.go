package service

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/guoyyit/mcmc-sampler-go/adapt"
	"github.com/guoyyit/mcmc-sampler-go/chain"
	"github.com/guoyyit/mcmc-sampler-go/checkpoint"
	"github.com/guoyyit/mcmc-sampler-go/diagnostics"
	"github.com/guoyyit/mcmc-sampler-go/history"
	"github.com/guoyyit/mcmc-sampler-go/rng"
	"github.com/guoyyit/mcmc-sampler-go/sampler"
	"github.com/guoyyit/mcmc-sampler-go/target"
)

var (
	// ErrInvalidConfig is returned for malformed service configuration.
	ErrInvalidConfig = errors.New("service: invalid configuration")
	// ErrNotConverged is returned when diagnostics reject the run. Samples are
	// deliberately not returned in that case.
	ErrNotConverged = errors.New("service: chains did not converge")
)

// Config configures a sampling job.
type Config struct {
	JobID           string
	NumChains       int
	Draws           int
	Warmup          int
	Seed            uint64
	Sampler         sampler.Params
	Adapt           adapt.Config
	History         history.Policy
	Thresholds      diagnostics.Thresholds
	CheckpointDir   string
	CheckpointEvery int
	// CheckpointInterval is the wall-clock polling interval for periodic
	// checkpoints. Zero defaults to one second.
	CheckpointInterval time.Duration
}

// Result is the outcome of a completed (or resumed) run.
type Result struct {
	JobID     string
	Converged bool
	Report    diagnostics.Report
	Draws     [][][]float64 // [chain][draw][dim]
	Resumed   bool
}

// Service runs multi-chain MCMC jobs with adaptation, diagnostics,
// checkpointing and cancellation.
type Service struct {
	cfg Config
	t   target.Target
}

// New validates the configuration.
func New(cfg Config, t target.Target) error {
	if t == nil {
		return fmt.Errorf("%w: nil target", ErrInvalidConfig)
	}
	if cfg.NumChains < 2 {
		return fmt.Errorf("%w: need at least 2 chains (got %d)", ErrInvalidConfig, cfg.NumChains)
	}
	if cfg.Draws <= 0 {
		return fmt.Errorf("%w: draws must be positive", ErrInvalidConfig)
	}
	if cfg.Warmup < 0 || cfg.Warmup >= cfg.Draws {
		return fmt.Errorf("%w: warmup must be in [0, draws)", ErrInvalidConfig)
	}
	if err := target.ValidatePosition(t.Dim(), cfg.Sampler.Initial); err != nil {
		return fmt.Errorf("%w: %v", ErrInvalidConfig, err)
	}
	if cfg.Sampler.Algorithm == sampler.AlgoHMC {
		cfg.Adapt.Defaults()
		if err := cfg.Adapt.Validate(); err != nil {
			return fmt.Errorf("%w: %v", ErrInvalidConfig, err)
		}
	}
	if cfg.CheckpointEvery < 0 {
		return fmt.Errorf("%w: checkpoint interval must be non-negative", ErrInvalidConfig)
	}
	if cfg.History.MaxSamples < 0 || cfg.History.Thin < 0 {
		return fmt.Errorf("%w: negative history retention values", ErrInvalidConfig)
	}
	return nil
}

type chainOutcome struct {
	id  int
	ch  *chain.Chain
	err error
}

// Run starts or resumes a job and blocks until completion, cancellation or a
// chain failure. Chains run concurrently and fully independently: a failing
// chain cancels the whole run but its own error is reported without
// corrupting the other chains' already-validated results.
func Run(ctx context.Context, cfg Config, t target.Target) (*Result, error) {
	if err := New(cfg, t); err != nil {
		return nil, err
	}

	var store checkpoint.Store
	if cfg.CheckpointDir != "" {
		fs, err := checkpoint.NewFileStore(cfg.CheckpointDir)
		if err != nil {
			return nil, err
		}
		store = fs
	}

	// Derive per-chain seeds deterministically from the single job seed. The
	// root stream is advanced in fixed index order 0..N-1, independent of
	// goroutine scheduling.
	root := rng.NewRoot(cfg.Seed)
	chainSeedUint := make([]uint64, cfg.NumChains)
	for i := 0; i < cfg.NumChains; i++ {
		child := rng.Child(root, uint64(i))
		chainSeedUint[i] = child.Uint64()
	}

	chainCfgs := make([]chain.Config, cfg.NumChains)
	for i := range chainCfgs {
		params := cfg.Sampler
		chainCfgs[i] = chain.Config{
			ID:            i,
			Seed:          chainSeedUint[i],
			Draws:         cfg.Draws,
			Warmup:        cfg.Warmup,
			Sampler:       params,
			Adapt:         cfg.Adapt,
			HistoryPolicy: cfg.History,
		}
	}

	resumed := false
	chains := make([]*chain.Chain, cfg.NumChains)
	if store != nil {
		if cp, err := store.Load(); err == nil {
			if cp.JobID != cfg.JobID || cp.NumChains != cfg.NumChains || cp.Seed != cfg.Seed {
				return nil, fmt.Errorf("%w: checkpoint belongs to a different job", ErrInvalidConfig)
			}
			for i, snap := range cp.Chains {
				c, err := chain.Restore(chainCfgs[snap.ID], t, &cp.Chains[i])
				if err != nil {
					return nil, err
				}
				chains[snap.ID] = c
			}
			resumed = true
		} else if !errors.Is(err, checkpoint.ErrNoCheckpoint) {
			return nil, err
		}
	}
	for i := range chains {
		if chains[i] == nil {
			c, err := chain.New(chainCfgs[i], t)
			if err != nil {
				return nil, err
			}
			chains[i] = c
		}
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	outcomes := make(chan chainOutcome, cfg.NumChains)
	allDone := make(chan struct{})
	var wg sync.WaitGroup
	for _, c := range chains {
		wg.Add(1)
		go func(ch *chain.Chain) {
			defer wg.Done()
			err := ch.Run(runCtx)
			outcomes <- chainOutcome{id: ch.ID(), ch: ch, err: err}
		}(c)
	}
	go func() {
		wg.Wait()
		close(allDone)
	}()

	// Periodic checkpointing. The loop snapshots every CheckpointEvery
	// completed iterations of the slowest chain; snapshots from independent
	// chains are mutually consistent because each chain state is self
	// contained (sampling streams do not interact).
	lastCheckpointIter := 0
	if resumed {
		lastCheckpointIter = minIteration(chains)
	}
	checkpointDone := make(chan struct{})
	go func() {
		defer close(checkpointDone)
		if store == nil || cfg.CheckpointEvery <= 0 {
			select {
			case <-runCtx.Done():
			case <-allDone:
			}
			return
		}
		interval := cfg.CheckpointInterval
		if interval <= 0 {
			interval = time.Second
		}
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-runCtx.Done():
				return
			case <-allDone:
				return
			case <-ticker.C:
				it := minIteration(chains)
				if it-lastCheckpointIter >= cfg.CheckpointEvery {
					if err := saveCheckpoint(store, cfg, chains, 0); err != nil {
						cancel()
						return
					}
					lastCheckpointIter = it
				}
			}
		}
	}()

	finished := make(map[int]chainOutcome)
	var firstErr error
	for len(finished) < cfg.NumChains {
		select {
		case oc := <-outcomes:
			finished[oc.id] = oc
			if oc.err != nil && !errors.Is(oc.err, context.Canceled) && firstErr == nil {
				firstErr = fmt.Errorf("chain %d failed: %w", oc.id, oc.err)
				cancel()
			}
		case <-ctx.Done():
			cancel()
		}
	}
	wg.Wait()
	<-checkpointDone

	if ctx.Err() != nil {
		// Persist whatever state is committed so a later resume is valid.
		if store != nil {
			_ = saveCheckpoint(store, cfg, chains, 0)
		}
		return nil, ctx.Err()
	}
	if firstErr != nil {
		if store != nil {
			_ = saveCheckpoint(store, cfg, chains, 0)
		}
		return nil, firstErr
	}
	// Build sample matrix and run convergence diagnostics.
	samples := make([][][]float64, cfg.NumChains)
	for i, c := range chains {
		draws := c.Draws()
		samples[i] = make([][]float64, len(draws))
		for j, d := range draws {
			samples[i][j] = d.Position
		}
	}
	report := diagnostics.Evaluate(samples, cfg.Thresholds)
	if !report.Converged {
		return nil, fmt.Errorf("%w: %s", ErrNotConverged, joinReasons(report.Reasons))
	}

	if store != nil {
		if err := saveCheckpoint(store, cfg, chains, 0); err != nil {
			return nil, err
		}
	}

	return &Result{
		JobID:     cfg.JobID,
		Converged: true,
		Report:    report,
		Draws:     samples,
		Resumed:   resumed,
	}, nil
}

func saveCheckpoint(store checkpoint.Store, cfg Config, chains []*chain.Chain, _ int) error {
	cp := &checkpoint.Checkpoint{
		JobID:     cfg.JobID,
		Seed:      cfg.Seed,
		Draws:     cfg.Draws,
		Warmup:    cfg.Warmup,
		NumChains: cfg.NumChains,
		Chains:    make([]checkpoint.ChainSnapshot, len(chains)),
	}
	for i, c := range chains {
		snap, err := c.Snapshot()
		if err != nil {
			return fmt.Errorf("snapshot chain %d: %w", i, err)
		}
		cp.Chains[i] = *snap
	}
	return store.Save(cp)
}

func minIteration(chains []*chain.Chain) int {
	min := chains[0].Iteration()
	for _, c := range chains[1:] {
		if it := c.Iteration(); it < min {
			min = it
		}
	}
	return min
}

func joinReasons(reasons []string) string {
	out := ""
	for i, r := range reasons {
		if i > 0 {
			out += "; "
		}
		out += r
	}
	return out
}
