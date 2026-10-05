package chain

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/guoyyit/mcmc-sampler-go/adapt"
	"github.com/guoyyit/mcmc-sampler-go/checkpoint"
	"github.com/guoyyit/mcmc-sampler-go/history"
	"github.com/guoyyit/mcmc-sampler-go/rng"
	"github.com/guoyyit/mcmc-sampler-go/sampler"
	"github.com/guoyyit/mcmc-sampler-go/target"
)

// ErrMismatchedCheckpoint is returned when a snapshot does not belong to the
// chain configuration it is applied to.
var ErrMismatchedCheckpoint = errors.New("chain: checkpoint does not match chain configuration")

// Config configures one chain.
type Config struct {
	ID            int
	Seed          uint64 // chain-specific stream seed
	Draws         int    // total iterations including warmup
	Warmup        int
	Sampler       sampler.Params
	Adapt         adapt.Config
	HistoryPolicy history.Policy
}

// Chain runs one Markov chain.
type Chain struct {
	mu   sync.RWMutex
	cfg  Config
	t    target.Target
	rng  *rng.Source
	s    sampler.Sampler
	eng  *adapt.Engine
	hist *history.Buffer
}

// New constructs a chain at iteration 0. The provided seed must already be
// chain-specific (the service derives one seed per chain deterministically).
func New(cfg Config, t target.Target) (*Chain, error) {
	if cfg.Draws <= 0 {
		return nil, fmt.Errorf("chain: draws must be positive, got %d", cfg.Draws)
	}
	if cfg.Warmup < 0 || cfg.Warmup > cfg.Draws {
		return nil, fmt.Errorf("chain: warmup must be in [0, draws]")
	}
	if cfg.Sampler.Algorithm == sampler.AlgoHMC {
		cfg.Adapt.Defaults()
	}
	stream := rng.NewRoot(cfg.Seed)
	s, err := sampler.New(t, stream, cfg.Sampler)
	if err != nil {
		return nil, err
	}
	c := &Chain{
		cfg:  cfg,
		t:    t,
		rng:  stream,
		s:    s,
		hist: history.New(cfg.HistoryPolicy),
	}
	if cfg.Sampler.Algorithm == sampler.AlgoHMC && cfg.Warmup > 0 {
		eng, err := adapt.New(cfg.Adapt, t.Dim())
		if err != nil {
			return nil, err
		}
		c.eng = eng
	}
	return c, nil
}

// Run advances the chain until Draws iterations complete or ctx is cancelled.
// Cancellation returns ctx.Err() immediately; the already-committed state
// (available through Snapshot) stays consistent.
func (c *Chain) Run(ctx context.Context) error {
	for {
		c.mu.RLock()
		done := c.s.Current().Iteration >= c.cfg.Draws
		c.mu.RUnlock()
		if done {
			break
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		c.mu.Lock()
		iterBefore := c.s.Current().Iteration
		st, err := c.s.Step(ctx)
		if err != nil {
			c.mu.Unlock()
			return fmt.Errorf("chain %d iteration %d: %w", c.cfg.ID, c.s.Current().Iteration+1, err)
		}

		iter := st.Iteration
		// Only observe iterations executed by this runner. After a resume the
		// first state is the restored one (iter == iterBefore) and must not be
		// observed a second time, or dual averaging/mass statistics would
		// diverge from the uninterrupted run.
		if c.eng != nil && iter > iterBefore && c.eng.Active(iter) {
			// Feed the realized acceptance probability (exp of capped log
			// probability) into dual averaging.
			acc := math.Exp(st.Stats.LogAcceptProb)
			c.eng.Observe(iter, acc, st.Position)
			p := c.s.Params()
			p.StepSize = c.eng.StepSize()
			p.InverseMass = c.eng.InverseMass()
			if err := c.s.SetParams(p); err != nil {
				return fmt.Errorf("chain %d: apply adaptation: %w", c.cfg.ID, err)
			}
			if iter == c.cfg.Warmup {
				c.eng.Freeze()
				frozen := c.s.Params()
				frozen.StepSize = c.eng.FinalStepSize()
				frozen.InverseMass = c.eng.InverseMass()
				if err := c.s.SetParams(frozen); err != nil {
					return fmt.Errorf("chain %d: freeze adaptation: %w", c.cfg.ID, err)
				}
			}
		}

		// Only post-warmup draws enter retained history.
		if iter > c.cfg.Warmup {
			c.hist.Add(history.Draw{Iteration: iter, Position: st.Position, LogProb: st.LogProb})
		}
		c.mu.Unlock()
	}
	return nil
}

// Draws returns the retained post-warmup draws.
func (c *Chain) Draws() []history.Draw {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.hist.Draws()
}

// Iteration returns the number of completed iterations.
func (c *Chain) Iteration() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.s.Current().Iteration
}

// ID returns the chain index.
func (c *Chain) ID() int { return c.cfg.ID }

// Snapshot captures the complete resumable state of the chain.
func (c *Chain) Snapshot() (*checkpoint.ChainSnapshot, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	rngState, err := c.rng.Marshal()
	if err != nil {
		return nil, err
	}
	samplerState, err := c.s.Marshal()
	if err != nil {
		return nil, err
	}
	histState, err := c.hist.Marshal()
	if err != nil {
		return nil, err
	}
	snap := &checkpoint.ChainSnapshot{
		ID:        c.cfg.ID,
		Iteration: c.s.Current().Iteration,
		RNG:       rngState,
		Sampler:   samplerState,
		History:   histState,
	}
	if c.eng != nil {
		engState, err := c.eng.Marshal()
		if err != nil {
			return nil, err
		}
		snap.Adapt = engState
	}
	return snap, nil
}

// Restore rebuilds a chain from a snapshot and returns it.
func Restore(cfg Config, t target.Target, snap *checkpoint.ChainSnapshot) (*Chain, error) {
	c, err := New(cfg, t)
	if err != nil {
		return nil, err
	}
	if snap.ID != cfg.ID {
		return nil, fmt.Errorf("%w: snapshot chain id %d != config id %d", ErrMismatchedCheckpoint, snap.ID, cfg.ID)
	}
	// c.s was built by New against c.rng (which consumed deterministic
	// jitter draws, exactly as in the original run). Now reset that same
	// stream object in place to its checkpointed position and overwrite the
	// sampler's mutable state, so no extra random draws are spent during
	// recovery and the next step continues the exact original sequence.
	if err := c.rng.Restore(snap.RNG); err != nil {
		return nil, fmt.Errorf("chain %d: restore rng: %w", cfg.ID, err)
	}
	if err := c.s.Unmarshal(snap.Sampler); err != nil {
		return nil, fmt.Errorf("chain %d: restore sampler: %w", cfg.ID, err)
	}
	if c.s.Current().Iteration != snap.Iteration {
		return nil, fmt.Errorf("%w: sampler iteration %d != snapshot %d",
			ErrMismatchedCheckpoint, c.s.Current().Iteration, snap.Iteration)
	}

	if c.eng != nil {
		if len(snap.Adapt) == 0 {
			return nil, fmt.Errorf("%w: missing adaptation state", ErrMismatchedCheckpoint)
		}
		if err := c.eng.Unmarshal(snap.Adapt); err != nil {
			return nil, fmt.Errorf("chain %d: restore adaptation: %w", cfg.ID, err)
		}
		// Re-apply the persisted proposal parameters. While still adapting
		// this is the instantaneous step size; after freeze it is the
		// smoothed final value. Without this a mid-warmup resume would take
		// its next step with a different step size than the uninterrupted run.
		p := c.s.Params()
		if c.eng.Frozen() {
			p.StepSize = c.eng.FinalStepSize()
		} else {
			p.StepSize = c.eng.StepSize()
		}
		p.InverseMass = c.eng.InverseMass()
		if err := c.s.SetParams(p); err != nil {
			return nil, err
		}
	}
	if err := c.hist.Unmarshal(snap.History); err != nil {
		return nil, fmt.Errorf("chain %d: restore history: %w", cfg.ID, err)
	}
	return c, nil
}
