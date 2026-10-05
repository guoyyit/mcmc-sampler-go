package mcmc

// The history never grows beyond the configured limit: RetainAll keeps at
// most Iterations-Warmup draws, RetainThinned divides that by Thin, and
// RetainWindow keeps at most History.Window draws per chain regardless of
// run length.

// History stores retained draws for one chain under the configured
// retention policy, bounding memory for long runs.
type History struct {
	cfg HistoryConfig

	// post-warmup draws
	draws []Draw
	// warmup draws (only when KeepWarmup)
	warmup []Draw
	// ring bookkeeping for RetainWindow
	ringHead int
	ringFull bool

	// total post-warmup transitions observed (may exceed len(draws))
	totalPostWarmup int
	// index of the next post-warmup transition, 0-based
	nextIndex int
}

func newHistory(cfg HistoryConfig) *History {
	return &History{cfg: cfg}
}

// Append stores d according to the retention policy.
func (h *History) Append(d Draw) {
	if d.Warmup {
		if h.cfg.KeepWarmup {
			h.warmup = append(h.warmup, d)
		}
		return
	}
	idx := h.nextIndex
	h.nextIndex++
	h.totalPostWarmup++
	switch h.cfg.Mode {
	case RetainThinned:
		if idx%h.cfg.Thin != 0 {
			return
		}
		h.draws = append(h.draws, d)
	case RetainWindow:
		if !h.ringFull && len(h.draws) < h.cfg.Window {
			h.draws = append(h.draws, d)
			if len(h.draws) == h.cfg.Window {
				h.ringFull = true
			}
		} else {
			h.draws[h.ringHead] = d
			h.ringHead = (h.ringHead + 1) % h.cfg.Window
		}
	default:
		h.draws = append(h.draws, d)
	}
}

// Draws returns the retained post-warmup draws in transition order.
func (h *History) Draws() []Draw {
	if h.cfg.Mode != RetainWindow || !h.ringFull {
		return h.draws
	}
	ordered := make([]Draw, len(h.draws))
	n := len(h.draws)
	for i := range ordered {
		ordered[i] = h.draws[(h.ringHead+i)%n]
	}
	return ordered
}

// WarmupDraws returns the retained warmup draws (empty unless KeepWarmup).
func (h *History) WarmupDraws() []Draw { return h.warmup }

// TotalPostWarmup reports how many post-warmup transitions occurred.
func (h *History) TotalPostWarmup() int { return h.totalPostWarmup }

// HistorySnapshot is the serializable retained history.
type HistorySnapshot struct {
	Draws           []Draw `json:"draws"`
	Warmup          []Draw `json:"warmup,omitempty"`
	TotalPostWarmup int    `json:"total_post_warmup"`
	NextIndex       int    `json:"next_index"`
	RingFull        bool   `json:"ring_full,omitempty"`
}

func (h *History) Snapshot() HistorySnapshot {
	return HistorySnapshot{
		Draws:           cloneDraws(h.Draws()),
		Warmup:          cloneDraws(h.warmup),
		TotalPostWarmup: h.totalPostWarmup,
		NextIndex:       h.nextIndex,
		RingFull:        h.ringFull && len(h.draws) == h.cfg.Window,
	}
}

func (h *History) Restore(s HistorySnapshot) error {
	h.draws = cloneDraws(s.Draws)
	h.warmup = cloneDraws(s.Warmup)
	h.totalPostWarmup = s.TotalPostWarmup
	h.nextIndex = s.NextIndex
	h.ringHead = 0
	h.ringFull = s.RingFull
	return nil
}

func cloneDraws(in []Draw) []Draw {
	if in == nil {
		return nil
	}
	out := make([]Draw, len(in))
	for i := range in {
		out[i] = in[i]
		out[i].Theta = cloneFloat64(in[i].Theta)
	}
	return out
}
