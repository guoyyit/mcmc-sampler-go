package sampler

import (
	"context"
	"encoding/gob"
	"fmt"
	"math"
	"strings"

	"github.com/guoyyit/mcmc-sampler-go/rng"
	"github.com/guoyyit/mcmc-sampler-go/target"
)

// hmcState is the gob-encoded mutable state of an HMC sampler.
type hmcState struct {
	Iteration   int
	Position    []float64
	LogProb     float64
	StepSize    float64
	NumLeapfrog int
	InverseMass []float64
}

// HMC is Hamiltonian Monte Carlo with a diagonal mass matrix and leapfrog
// integration.
type HMC struct {
	t     target.GradientTarget
	r     *rng.Source
	p     Params
	iter  int
	cur   []float64
	curLP float64
}

func validateHMCParams(dim int, p Params) error {
	if p.NumLeapfrog <= 0 {
		return fmt.Errorf("%w: num leapfrog steps must be positive, got %d", ErrInvalidParams, p.NumLeapfrog)
	}
	if !(p.StepSize > 0) || math.IsInf(p.StepSize, 0) {
		return fmt.Errorf("%w: step size must be finite and positive, got %v", ErrInvalidParams, p.StepSize)
	}
	if len(p.InverseMass) != dim {
		return fmt.Errorf("%w: inverse mass dimension %d != target dimension %d", ErrInvalidParams, len(p.InverseMass), dim)
	}
	for i, m := range p.InverseMass {
		if !(m > 0) || math.IsInf(m, 0) {
			return fmt.Errorf("%w: inverse mass at %d must be finite and positive, got %v", ErrInvalidParams, i, m)
		}
	}
	return nil
}

func newHMC(t target.GradientTarget, r *rng.Source, p Params) (*HMC, error) {
	dim := t.Dim()
	if len(p.InverseMass) == 0 {
		p.InverseMass = make([]float64, dim)
		for i := range p.InverseMass {
			p.InverseMass[i] = 1
		}
	}
	if p.NumLeapfrog == 0 {
		p.NumLeapfrog = 10
	}
	if err := validateHMCParams(dim, p); err != nil {
		return nil, err
	}
	initial := append([]float64(nil), p.Initial...)
	if len(p.InitJitter) == dim {
		for i, j := range p.InitJitter {
			if j < 0 || math.IsInf(j, 0) || math.IsNaN(j) {
				return nil, fmt.Errorf("%w: init jitter at %d must be finite and non-negative", ErrInvalidParams, i)
			}
			initial[i] += j * r.Gaussian()
		}
		if err := target.ValidatePosition(dim, initial); err != nil {
			return nil, fmt.Errorf("%w: jittered initial position: %v", ErrInvalidParams, err)
		}
	} else if len(p.InitJitter) != 0 {
		return nil, fmt.Errorf("%w: init jitter dimension %d != target dimension %d", ErrInvalidParams, len(p.InitJitter), dim)
	}
	h := &HMC{t: t, r: r, p: p, cur: initial}
	lp := t.LogDensity(initial)
	if err := target.ValidateLogProb(lp); err != nil {
		return nil, fmt.Errorf("%w: initial log density: %v", ErrDensityInvalid, err)
	}
	h.curLP = lp
	return h, nil
}

// gradient evaluates and validates the gradient at x.
func (h *HMC) gradient(x []float64) ([]float64, bool) {
	dim := len(x)
	g := make([]float64, dim)
	h.t.Gradient(x, g)
	for _, v := range g {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, false
		}
	}
	return g, true
}

func (h *HMC) Step(ctx context.Context) (State, error) {
	if err := ctx.Err(); err != nil {
		return State{}, err
	}
	dim := h.t.Dim()

	// Fresh momentum from the standard normal; mass matrix is diagonal.
	momentum := make([]float64, dim)
	for i := range momentum {
		momentum[i] = h.r.Gaussian()
	}

	st := Stats{}
	q := append([]float64(nil), h.cur...)
	lp := h.curLP

	// Current Hamiltonian components: kinetic energy = 0.5 * sum p_i^2.
	var curKinetic float64
	for _, pi := range momentum {
		curKinetic += 0.5 * pi * pi
	}

	g, ok := h.gradient(q)
	if !ok || math.IsNaN(lp) || math.IsInf(lp, +1) {
		st.Divergent = true
		st.LogAcceptProb = math.Inf(-1)
		h.iter++
		return State{Iteration: h.iter, Position: append([]float64(nil), h.cur...), LogProb: h.curLP, Stats: st}, nil
	}

	eps := h.p.StepSize
	mInv := h.p.InverseMass

	// Leapfrog integration.
	// p <- p + (eps/2) * grad U  (grad log p = -grad U, so U update uses +g)
	for i := 0; i < dim; i++ {
		momentum[i] += 0.5 * eps * g[i]
	}
	diverged := false
	for step := 0; step < h.p.NumLeapfrog; step++ {
		for i := 0; i < dim; i++ {
			q[i] += eps * mInv[i] * momentum[i]
		}
		for _, v := range q {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				diverged = true
			}
		}
		if diverged {
			break
		}
		lpNew := h.t.LogDensity(q)
		if err := target.ValidateLogProb(lpNew); err != nil {
			// Illegal density (NaN/+Inf) is a target error, not a
			// divergence: reject the proposal loudly rather than silently.
			return State{}, fmt.Errorf("%w: %v", ErrDensityInvalid, err)
		}
		lp = lpNew
		g, ok = h.gradient(q)
		if !ok {
			diverged = true
			break
		}
		if step < h.p.NumLeapfrog-1 {
			for i := 0; i < dim; i++ {
				momentum[i] += eps * g[i]
			}
		}
	}

	if diverged {
		st.Divergent = true
		st.LogAcceptProb = math.Inf(-1)
	} else {
		// Final momentum half-step.
		for i := 0; i < dim; i++ {
			momentum[i] += 0.5 * eps * g[i]
		}
		// Negate momentum for time reversibility (irrelevant to acceptance).
		var propKinetic float64
		for _, pi := range momentum {
			propKinetic += 0.5 * pi * pi
		}
		// log acceptance = lp_new - lp_cur - (K_new - K_cur), all in log
		// space and capped at 0.
		diff := lp - h.curLP - (propKinetic - curKinetic)
		if math.IsNaN(diff) {
			st.Divergent = true
			st.LogAcceptProb = math.Inf(-1)
		} else {
			logAccept := math.Min(0, diff)
			st.LogAcceptProb = logAccept
			if math.Log(h.r.Float64()) < logAccept {
				h.cur = q
				h.curLP = lp
				st.Accepted = true
			}
		}
	}

	h.iter++
	return State{Iteration: h.iter, Position: append([]float64(nil), h.cur...), LogProb: h.curLP, Stats: st}, nil
}

func (h *HMC) Current() State {
	return State{Iteration: h.iter, Position: append([]float64(nil), h.cur...), LogProb: h.curLP}
}

func (h *HMC) Params() Params {
	p := h.p
	p.InverseMass = append([]float64(nil), h.p.InverseMass...)
	p.Initial = append([]float64(nil), h.cur...)
	return p
}

func (h *HMC) SetParams(p Params) error {
	if p.Algorithm != "" && p.Algorithm != AlgoHMC {
		return fmt.Errorf("%w: cannot switch algorithm on HMC", ErrInvalidParams)
	}
	merged := h.p
	if p.NumLeapfrog > 0 {
		merged.NumLeapfrog = p.NumLeapfrog
	}
	if p.StepSize > 0 {
		merged.StepSize = p.StepSize
	}
	if len(p.InverseMass) == h.t.Dim() {
		merged.InverseMass = append([]float64(nil), p.InverseMass...)
	}
	if err := validateHMCParams(h.t.Dim(), merged); err != nil {
		return err
	}
	h.p = merged
	return nil
}

func (h *HMC) Marshal() ([]byte, error) {
	var sb strings.Builder
	if err := gob.NewEncoder(&sb).Encode(hmcState{
		Iteration:   h.iter,
		Position:    h.cur,
		LogProb:     h.curLP,
		StepSize:    h.p.StepSize,
		NumLeapfrog: h.p.NumLeapfrog,
		InverseMass: h.p.InverseMass,
	}); err != nil {
		return nil, fmt.Errorf("sampler: marshal HMC: %w", err)
	}
	return []byte(sb.String()), nil
}

func (h *HMC) Unmarshal(data []byte) error {
	var st hmcState
	if err := gob.NewDecoder(strings.NewReader(string(data))).Decode(&st); err != nil {
		return fmt.Errorf("sampler: unmarshal HMC: %w", err)
	}
	if err := target.ValidatePosition(h.t.Dim(), st.Position); err != nil {
		return fmt.Errorf("sampler: unmarshal HMC: %w", err)
	}
	if err := target.ValidateLogProb(st.LogProb); err != nil {
		return fmt.Errorf("sampler: unmarshal HMC: %w", err)
	}
	if st.Iteration < 0 {
		return fmt.Errorf("sampler: unmarshal HMC: negative iteration count")
	}
	h.iter = st.Iteration
	h.cur = st.Position
	h.curLP = st.LogProb
	if st.StepSize > 0 {
		h.p.StepSize = st.StepSize
	}
	if st.NumLeapfrog > 0 {
		h.p.NumLeapfrog = st.NumLeapfrog
	}
	if len(st.InverseMass) == h.t.Dim() {
		h.p.InverseMass = st.InverseMass
	}
	return nil
}
