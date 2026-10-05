package mcmc

import (
	"context"
	"fmt"
	"sync"
)

// Result is the output of Sampler.Run.
type Result struct {
	// Diagnostics holds the pooled convergence assessment; nil when
	// Converged is false due to cancellation or insufficient chains.
	Diagnostics *Diagnostics
	// Converged is true only when every configured threshold is met.
	Converged bool
	// Reason explains a non-converged verdict ("" when converged).
	Reason string
	// Samples[c][i] is the i-th retained draw of chain c.
	Samples [][]Draw
	// Chains reports per-chain status.
	Chains []ChainResult
	// WarmupParams are the frozen per-chain kernel parameters.
	WarmupParams []Params
	// Resumed reports whether the run started from a checkpoint.
	Resumed bool
}

// Sampler runs parallel MCMC chains for one configuration.
type Sampler struct {
	cfg     Config
	chains  []*chain
	resumed bool
}

// New validates cfg and constructs a Sampler.
func New(cfg Config) (*Sampler, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Sampler{cfg: cfg}, nil
}

// Run executes sampling. On resume, already-completed iterations are not
// repeated. Cancellation stops chains immediately, leaves committed
// checkpoints valid and suppresses partial diagnostics.
func (s *Sampler) Run(opts RunOptions) (*Result, error) {
	if opts.Ctx == nil {
		opts.Ctx = context.Background()
	}
	if opts.Resume != "" {
		if err := s.loadResume(opts.Resume); err != nil {
			return nil, err
		}
	} else {
		if err := s.buildChains(opts); err != nil {
			return nil, err
		}
	}

	cancel := waitCancel(opts.Ctx)
	checkpointing := s.cfg.Checkpoint.Path != "" && s.cfg.Checkpoint.Every > 0
	type cpEvent struct {
		chain    int
		boundary int
		snap     chainSnapshot
	}
	var cpEvents chan cpEvent
	var coordinatorDone chan struct{}
	if checkpointing {
		cpEvents = make(chan cpEvent, 4096)
		coordinatorDone = make(chan struct{})
		// The coordinator is the only goroutine that receives chain
		// snapshots; each snapshot is produced by its own chain before
		// the chain continues, so a committed file never mixes state
		// observed at different logical boundaries.
		go func() {
			defer close(coordinatorDone)
			reach := make([]int, s.cfg.Chains) // max boundary reached + 1
			latest := make([]chainSnapshot, s.cfg.Chains)
			flushed := -1
			for {
				select {
				case ev, ok := <-cpEvents:
					if !ok {
						return
					}
					latest[ev.chain] = ev.snap
					if ev.boundary+1 > reach[ev.chain] {
						reach[ev.chain] = ev.boundary + 1
					}
					minReach := reach[0]
					for c := 1; c < s.cfg.Chains; c++ {
						if reach[c] < minReach {
							minReach = reach[c]
						}
					}
					target := minReach - 1
					if target > flushed {
						cp := &Checkpoint{
							Version: checkpointVersion,
							Seed:    s.cfg.Seed,
							Config:  persistConfig(&s.cfg),
							Chains:  make([]chainSnapshot, s.cfg.Chains),
						}
						copy(cp.Chains, latest)
						_ = saveCheckpoint(s.cfg.Checkpoint.Path, cp)
						flushed = target
					}
				}
			}
		}()
	}

	var wg sync.WaitGroup
	results := make([]ChainResult, s.cfg.Chains)
	wg.Add(s.cfg.Chains)
	for c := 0; c < s.cfg.Chains; c++ {
		ch := s.chains[c]
		go func() {
			defer wg.Done()
			var res ChainResult
			defer func() {
				if r := recover(); r != nil {
					res = ChainResult{
						Index:     ch.index,
						Status:    ChainFailed,
						Iteration: ch.iter,
						Accepted:  ch.accepted,
						Error:     fmt.Sprintf("panic: %v", r),
					}
					results[ch.index] = res
				}
			}()
			var cb func(chainSnapshot)
			if checkpointing {
				cb = func(snap chainSnapshot) {
					boundary := snap.Iteration/s.cfg.Checkpoint.Every - 1
					select {
					case cpEvents <- cpEvent{chain: ch.index, boundary: boundary, snap: snap}:
					default:
					}
					// Do not let a slow disk extend a canceled run: if a
					// checkpoint send raced with cancellation, the chain
					// still returns to its loop-top cancel check promptly.
				}
			}
			res = ch.run(&s.cfg, cancel, cb)
			results[ch.index] = res
		}()
	}
	wg.Wait()
	if checkpointing {
		close(cpEvents)
		<-coordinatorDone
	}

	res := &Result{
		Chains:  results,
		Resumed: s.resumed,
		Samples: make([][]Draw, s.cfg.Chains),
	}
	for _, r := range results {
		if r.Status == ChainCanceled {
			res.Reason = "sampling canceled: diagnostics suppressed"
			return res, nil
		}
	}
	// Explicit final checkpoint so completed runs can resume cheaply.
	if s.cfg.Checkpoint.Path != "" {
		if err := s.Save(s.cfg.Checkpoint.Path); err != nil {
			return nil, err
		}
	}

	histories := make([]*History, s.cfg.Chains)
	okChains := 0
	for c, ch := range s.chains {
		histories[c] = ch.history
		res.Samples[c] = ch.history.Draws()
		if results[c].Status == ChainComplete {
			okChains++
		}
		res.WarmupParams = append(res.WarmupParams, ch.engine.Params())
	}
	if okChains < 2 && s.cfg.Chains >= 2 {
		res.Reason = "fewer than 2 chains completed: cannot assess convergence"
		return res, nil
	}
	diag, err := assess(histories, s.cfg.Target.Dim)
	if err != nil {
		res.Reason = fmt.Sprintf("convergence assessment unavailable: %v", err)
		return res, nil
	}
	res.Diagnostics = diag
	if !diag.Converged(s.cfg.RHatThreshold, s.cfg.ESSTarget, okChains) {
		res.Converged = false
		res.Reason = fmt.Sprintf(
			"NOT CONVERGED: max R-hat %.4f (threshold %.3f), min bulk ESS %.1f (target %.0f)",
			maxSlice(diag.RHat), s.cfg.RHatThreshold,
			minSlice(diag.ESS), s.cfg.ESSTarget)
		return res, nil
	}
	res.Converged = true
	return res, nil
}

// Save writes a checkpoint immediately.
func (s *Sampler) Save(path string) error {
	cp := &Checkpoint{
		Version: checkpointVersion,
		Seed:    s.cfg.Seed,
		Config:  persistConfig(&s.cfg),
		Chains:  make([]chainSnapshot, len(s.chains)),
	}
	for i, ch := range s.chains {
		cp.Chains[i] = ch.snapshot()
	}
	return saveCheckpoint(path, cp)
}

// buildChains constructs (or reconstructs) the chain set.
func (s *Sampler) buildChains(opts RunOptions) error {
	s.chains = make([]*chain, s.cfg.Chains)
	for c := 0; c < s.cfg.Chains; c++ {
		var init []float64
		if c < len(opts.InitialPoints) {
			init = opts.InitialPoints[c]
		}
		ch, err := newChain(&s.cfg, c, init)
		if err != nil {
			return err
		}
		s.chains[c] = ch
	}
	return nil
}

func (s *Sampler) loadResume(path string) error {
	cp, err := loadCheckpoint(path)
	if err != nil {
		return err
	}
	if cp.Seed != s.cfg.Seed {
		return fmt.Errorf("%w: seed mismatch", ErrCheckpointCorrupt)
	}
	if err := cp.Config.matches(&s.cfg); err != nil {
		return err
	}
	if len(cp.Chains) != s.cfg.Chains {
		return fmt.Errorf("%w: chain count mismatch", ErrCheckpointCorrupt)
	}
	s.chains = make([]*chain, s.cfg.Chains)
	for _, snap := range cp.Chains {
		ch, err := restoreChain(&s.cfg, snap)
		if err != nil {
			return err
		}
		if snap.Index < 0 || snap.Index >= s.cfg.Chains {
			return fmt.Errorf("%w: chain index out of range", ErrCheckpointCorrupt)
		}
		s.chains[snap.Index] = ch
	}
	for c, ch := range s.chains {
		if ch == nil {
			return fmt.Errorf("%w: missing chain %d", ErrCheckpointCorrupt, c)
		}
	}
	s.resumed = true
	return nil
}

// waitCancel returns a closed channel when ctx is nil.
func waitCancel(ctx context.Context) <-chan struct{} {
	if ctx == nil {
		ch := make(chan struct{})
		close(ch)
		return ch
	}
	return ctx.Done()
}

func maxSlice(x []float64) float64 {
	m := x[0]
	for _, v := range x[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func minSlice(x []float64) float64 {
	m := x[0]
	for _, v := range x[1:] {
		if v < m {
			m = v
		}
	}
	return m
}
