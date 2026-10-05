package sampler

import (
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/guoyyit/mcmc-sampler-go/rng"
	"github.com/guoyyit/mcmc-sampler-go/target"
)

// mhState is the gob-encoded mutable state of an MH sampler.
type mhState struct {
	Iteration     int
	Position      []float64
	LogProb       float64
	ProposalScale []float64
}

// MH is a random-walk Metropolis-Hastings sampler with a fixed (or adapted)
// diagonal Gaussian proposal.
type MH struct {
	t     target.Target
	r     *rng.Source
	p     Params
	iter  int
	cur   []float64
	curLP float64
}

func newMH(t target.Target, r *rng.Source, p Params) (*MH, error) {
	scale := p.ProposalScale
	if scale == nil {
		scale = make([]float64, t.Dim())
		for i := range scale {
			scale[i] = 1
		}
	}
	if len(scale) != t.Dim() {
		return nil, fmt.Errorf("%w: proposal scale dimension %d != target dimension %d", ErrInvalidParams, len(scale), t.Dim())
	}
	for i, s := range scale {
		if !(s > 0) || math.IsInf(s, 0) {
			return nil, fmt.Errorf("%w: proposal scale at %d must be finite and positive, got %v", ErrInvalidParams, i, s)
		}
	}
	initial := append([]float64(nil), p.Initial...)
	if len(p.InitJitter) == t.Dim() {
		for i, j := range p.InitJitter {
			if j < 0 || math.IsInf(j, 0) || math.IsNaN(j) {
				return nil, fmt.Errorf("%w: init jitter at %d must be finite and non-negative", ErrInvalidParams, i)
			}
			initial[i] += j * r.Gaussian()
		}
		if err := target.ValidatePosition(t.Dim(), initial); err != nil {
			return nil, fmt.Errorf("%w: jittered initial position: %v", ErrInvalidParams, err)
		}
	} else if len(p.InitJitter) != 0 {
		return nil, fmt.Errorf("%w: init jitter dimension %d != target dimension %d", ErrInvalidParams, len(p.InitJitter), t.Dim())
	}
	m := &MH{
		t:     t,
		r:     r,
		p:     p,
		iter:  0,
		cur:   initial,
		curLP: 0,
	}
	lp := t.LogDensity(initial)
	if err := target.ValidateLogProb(lp); err != nil {
		return nil, fmt.Errorf("%w: initial log density: %v", ErrDensityInvalid, err)
	}
	m.curLP = lp
	return m, nil
}

func (m *MH) Step(ctx context.Context) (State, error) {
	if err := ctx.Err(); err != nil {
		return State{}, err
	}
	dim := m.t.Dim()
	prop := make([]float64, dim)
	for i := 0; i < dim; i++ {
		prop[i] = m.cur[i] + m.p.ProposalScale[i]*m.r.Gaussian()
	}

	var st Stats
	next := m.cur
	nextLP := m.curLP

	if err := target.ValidatePosition(dim, prop); err != nil {
		// An invalid (non-finite) proposal is rejected outright rather than
		// evaluated.
		st.LogAcceptProb = math.Inf(-1)
	} else {
		propLP := m.t.LogDensity(prop)
		if err := target.ValidateLogProb(propLP); err != nil {
			return State{}, fmt.Errorf("%w: %v", ErrDensityInvalid, err)
		}
		// The symmetric Gaussian proposal cancels in the ratio; the decision
		// is computed entirely in log space: log a = min(0, lp_prop - lp_cur).
		// If lp_prop is -Inf the difference is -Inf and the proposal is
		// rejected even if lp_cur is also -Inf (0/0 probabilities must never
		// become an acceptance of 1).
		diff := propLP - m.curLP
		if math.IsNaN(diff) || math.IsInf(diff, +1) {
			// Guard against -Inf - -Inf = NaN and cur -Inf / prop finite
			// producing +Inf: both are forced to a safe acceptance rule.
			switch {
			case math.IsInf(propLP, -1) && math.IsInf(m.curLP, -1):
				diff = math.Inf(-1)
			case math.IsInf(m.curLP, -1) && !math.IsInf(propLP, -1):
				diff = 0 // any finite proposal beats an impossible point
			default:
				return State{}, fmt.Errorf("%w: non-finite acceptance log-probability", ErrDensityInvalid)
			}
		}
		logAccept := math.Min(0, diff)
		st.LogAcceptProb = logAccept
		if math.Log(m.r.Float64()) < logAccept {
			next = prop
			nextLP = propLP
			st.Accepted = true
		}
	}

	m.cur = next
	m.curLP = nextLP
	m.iter++
	return State{Iteration: m.iter, Position: append([]float64(nil), m.cur...), LogProb: m.curLP, Stats: st}, nil
}

func (m *MH) Current() State {
	return State{Iteration: m.iter, Position: append([]float64(nil), m.cur...), LogProb: m.curLP}
}

func (m *MH) Params() Params {
	p := m.p
	p.ProposalScale = append([]float64(nil), m.p.ProposalScale...)
	p.Initial = append([]float64(nil), m.cur...)
	return p
}

func (m *MH) SetParams(p Params) error {
	if p.Algorithm != "" && p.Algorithm != AlgoMH {
		return fmt.Errorf("%w: cannot switch algorithm on MH", ErrInvalidParams)
	}
	if len(p.ProposalScale) != m.t.Dim() {
		return fmt.Errorf("%w: proposal scale dimension mismatch", ErrInvalidParams)
	}
	for i, s := range p.ProposalScale {
		if !(s > 0) || math.IsInf(s, 0) {
			return fmt.Errorf("%w: proposal scale at %d must be finite and positive", ErrInvalidParams, i)
		}
	}
	scale := append([]float64(nil), p.ProposalScale...)
	m.p.ProposalScale = scale
	return nil
}

func (m *MH) Marshal() ([]byte, error) {
	var sb strings.Builder
	if err := gob.NewEncoder(&sb).Encode(mhState{
		Iteration:     m.iter,
		Position:      m.cur,
		LogProb:       m.curLP,
		ProposalScale: m.p.ProposalScale,
	}); err != nil {
		return nil, fmt.Errorf("sampler: marshal MH: %w", err)
	}
	return []byte(sb.String()), nil
}

func (m *MH) Unmarshal(data []byte) error {
	var st mhState
	if err := gob.NewDecoder(strings.NewReader(string(data))).Decode(&st); err != nil {
		return fmt.Errorf("sampler: unmarshal MH: %w", err)
	}
	if err := target.ValidatePosition(m.t.Dim(), st.Position); err != nil {
		return fmt.Errorf("sampler: unmarshal MH: %w", err)
	}
	if err := target.ValidateLogProb(st.LogProb); err != nil {
		return fmt.Errorf("sampler: unmarshal MH: %w", err)
	}
	if st.Iteration < 0 {
		return errors.New("sampler: unmarshal MH: negative iteration count")
	}
	m.iter = st.Iteration
	m.cur = st.Position
	m.curLP = st.LogProb
	if len(st.ProposalScale) == m.t.Dim() {
		m.p.ProposalScale = st.ProposalScale
	}
	return nil
}
