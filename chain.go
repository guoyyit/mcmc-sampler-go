package mcmc

import "fmt"

// ChainStatus labels the state of a single chain.
type ChainStatus string

const (
	ChainRunning  ChainStatus = "running"
	ChainComplete ChainStatus = "complete"
	ChainCanceled ChainStatus = "canceled"
	ChainFailed   ChainStatus = "failed"
)

// ChainResult summarizes one chain's run.
type ChainResult struct {
	Index     int         `json:"index"`
	Status    ChainStatus `json:"status"`
	Iteration int         `json:"iteration"`
	Accepted  int         `json:"accepted"`
	Error     string      `json:"error,omitempty"`
}

// chain is the full serializable state of one Markov chain.
type chain struct {
	index    int
	engine   Engine
	adapter  Adapter
	rng      rng64
	history  *History
	theta    []float64
	lp       float64
	iter     int
	accepted int
	status   ChainStatus
}

// chainSnapshot is the on-disk representation of a chain.
type chainSnapshot struct {
	Index     int             `json:"index"`
	Iteration int             `json:"iteration"`
	Accepted  int             `json:"accepted"`
	Theta     []float64       `json:"theta"`
	LogProb   float64         `json:"log_prob"`
	RNG       RNGSnapshot     `json:"rng"`
	Params    Params          `json:"params"`
	Adapter   AdaptSnapshot   `json:"adapter"`
	History   HistorySnapshot `json:"history"`
}

// run drives the chain until completion, cancellation or error.
// onCheckpoint, when non-nil, is invoked with this chain's snapshot
// every cfg.Checkpoint.Every transitions; the caller serializes all
// chains atomically.
func (ch *chain) run(cfg *Config, cancel <-chan struct{}, onCheckpoint func(chainSnapshot)) ChainResult {
	for ch.iter < cfg.Iterations {
		select {
		case <-cancel:
			ch.status = ChainCanceled
			return ch.result()
		default:
		}
		warmup := ch.iter < cfg.Warmup
		d, err := ch.engine.Step(ch.rng, warmup, ch.adapter)
		if err != nil {
			ch.status = ChainFailed
			return ChainResult{
				Index:     ch.index,
				Status:    ChainFailed,
				Iteration: ch.iter,
				Accepted:  ch.accepted,
				Error:     err.Error(),
			}
		}
		if warmup && ch.iter+1 == cfg.Warmup {
			frozen := ch.adapter.Freeze()
			ch.engine.SetParams(frozen)
		}
		if d.Accepted {
			ch.accepted++
		}
		if len(d.Theta) == cfg.Target.Dim {
			ch.theta = cloneFloat64(d.Theta)
			ch.lp = d.LogProb
		}
		ch.history.Append(d)
		ch.iter++
		if onCheckpoint != nil && cfg.Checkpoint.Every > 0 &&
			ch.iter%cfg.Checkpoint.Every == 0 {
			select {
			case <-cancel:
				ch.status = ChainCanceled
				return ch.result()
			default:
			}
			onCheckpoint(ch.snapshot())
		}
	}
	ch.status = ChainComplete
	return ch.result()
}

func (ch *chain) result() ChainResult {
	return ChainResult{
		Index:     ch.index,
		Status:    ch.status,
		Iteration: ch.iter,
		Accepted:  ch.accepted,
	}
}

func (ch *chain) snapshot() chainSnapshot {
	snap := chainSnapshot{
		Index:     ch.index,
		Iteration: ch.iter,
		Accepted:  ch.accepted,
		Theta:     cloneFloat64(ch.theta),
		LogProb:   ch.lp,
		RNG:       ch.rng.Snapshot(),
		Params:    ch.engine.Params(),
		History:   ch.history.Snapshot(),
	}
	if ch.adapter != nil {
		snap.Adapter = ch.adapter.Snapshot()
	}
	return snap
}

// restoreChain rebuilds a chain from a checkpoint snapshot. The restored
// RNG state makes subsequent draws identical to an uninterrupted run.
func restoreChain(cfg *Config, snap chainSnapshot) (*chain, error) {
	engine, err := newEngine(cfg)
	if err != nil {
		return nil, err
	}
	engine.Init(snap.Theta, snap.LogProb)
	engine.SetParams(snap.Params)

	var adapter Adapter
	if cfg.Warmup > 0 && snap.Iteration < cfg.Warmup {
		initial := cfg.MHStepSize
		if cfg.Algo == AlgoHMC {
			initial = cfg.HMCStepSize
		}
		adapter = newAdapter(cfg, cfg.Target.Dim, cfg.Algo, initial)
		if err := adapter.Restore(snap.Adapter); err != nil {
			return nil, err
		}
	} else if snap.Iteration >= cfg.Warmup && cfg.Warmup > 0 {
		adapter = newFrozenAdapter(cfg, snap)
	}

	seed := chainSeed(cfg.Seed, snap.Index, cfg.Chains)
	rng := newSplitMix(0)
	if err := rng.Restore(snap.RNG); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCheckpointCorrupt, err)
	}
	_ = seed
	hist := newHistory(cfg.History)
	if err := hist.Restore(snap.History); err != nil {
		return nil, err
	}
	return &chain{
		index:    snap.Index,
		engine:   engine,
		adapter:  adapter,
		rng:      rng,
		history:  hist,
		theta:    cloneFloat64(snap.Theta),
		lp:       snap.LogProb,
		iter:     snap.Iteration,
		accepted: snap.Accepted,
	}, nil
}

// newChain builds a fresh chain at its deterministic initial point.
func newChain(cfg *Config, index int, initial []float64) (*chain, error) {
	engine, err := newEngine(cfg)
	if err != nil {
		return nil, err
	}
	rng := newSplitMix(chainSeed(cfg.Seed, index, cfg.Chains))
	theta := initialPoint(cfg.Target.Dim, initial, cfg.InitScale,
		index, cfg.Chains, rng)
	lp, err := cfg.Target.Eval(theta)
	if err != nil {
		return nil, err
	}
	if !isFinite(lp) {
		return nil, fmt.Errorf("%w: chain %d initial point outside support",
			ErrIllegalParams, index)
	}
	engine.Init(theta, lp)
	var adapter Adapter
	if cfg.Warmup > 0 {
		initialStep := cfg.MHStepSize
		if cfg.Algo == AlgoHMC {
			initialStep = cfg.HMCStepSize
		}
		adapter = newAdapter(cfg, cfg.Target.Dim, cfg.Algo, initialStep)
	}
	return &chain{
		index:   index,
		engine:  engine,
		adapter: adapter,
		rng:     rng,
		history: newHistory(cfg.History),
		theta:   theta,
		lp:      lp,
		status:  ChainRunning,
	}, nil
}
