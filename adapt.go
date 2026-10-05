package mcmc

import "math"

// Adapter performs warmup adaptation and freezes at warmup end. Every
// decision is a deterministic function of observed accept statistics and
// draws, so identical chains reproduce identical parameter trajectories.
type Adapter interface {
	// Observe records one warmup transition, in iteration order.
	Observe(d Draw, theta []float64, lp float64, grad []float64)
	// Params returns the parameters the engine should use right now.
	Params() Params
	// Frozen reports whether adaptation has ended.
	Frozen() bool
	// Freeze ends adaptation and returns the final frozen parameters.
	Freeze() Params
	// Snapshot captures the complete adaptation state.
	Snapshot() AdaptSnapshot
	// Restore overwrites the adaptation state.
	Restore(s AdaptSnapshot) error
}

// AdaptSnapshot is the serializable adapter state.
type AdaptSnapshot struct {
	Mode          AdaptMode `json:"mode"`
	Frozen        bool      `json:"frozen"`
	Iteration     int       `json:"iteration"`
	StepSize      float64   `json:"step_size"`
	Hbar          float64   `json:"hbar"`
	LogEpsBar     float64   `json:"log_eps_bar"`
	Mu            float64   `json:"mu"`
	M             []float64 `json:"m,omitempty"`
	M2            []float64 `json:"m2,omitempty"`
	SampleCount   int       `json:"sample_count"`
	WindowStart   int       `json:"window_start"`
	WindowEnd     int       `json:"window_end"`
	NextWindowEnd int       `json:"next_window_end"`
	Term          bool      `json:"term"`
	Mass          []float64 `json:"mass,omitempty"`
}

// dualAveraging implements the Hoffman-Gelman (2014) dual-averaging step
// size adaptation, optionally combined with diagonal (windowed) variance
// adaptation for the HMC mass matrix.
type dualAveraging struct {
	mode       AdaptMode
	dim        int
	warmup     int
	targetLow  float64
	targetHigh float64
	initial    float64

	frozen bool
	iter   int
	step   float64

	// dual averaging accumulators (Hoffman-Gelman Algorithm 5)
	hbar      float64
	logEpsBar float64
	mu        float64
	k         float64
	gamma     float64
	t0        float64

	// diagonal mass matrix window statistics (Welford online moments)
	mean        []float64
	m2          []float64
	sampleCount int
	windowStart int
	windowEnd   int
	nextEnd     int
	useMass     bool
	total       int
	initBuf     int
	termBuf     int
	mass        []float64
	term        bool
}

func newAdapter(cfg *Config, dim int, algo Algorithm, initial float64) Adapter {
	a := &dualAveraging{
		mode:       cfg.Adapt,
		dim:        dim,
		warmup:     cfg.Warmup,
		targetLow:  cfg.TargetAcceptLow,
		targetHigh: cfg.TargetAcceptHigh,
		initial:    initial,
		step:       initial,
		k:          0.75,
		gamma:      0.05,
		t0:         10,
		mu:         math.Log(10 * initial),
		logEpsBar:  math.Log(initial),
		total:      cfg.Warmup,
	}
	a.useMass = cfg.Adapt == AdaptStepSizeMass && algo == AlgoHMC
	if a.useMass {
		a.mass = defaultScale(dim)
		a.mean = make([]float64, dim)
		a.m2 = make([]float64, dim)
		a.initWindows()
	}
	return a
}

// newFrozenAdapter rebuilds an adapter whose adaptation has ended; its
// only job is to hold the frozen parameters for re-checkpointing.
func newFrozenAdapter(cfg *Config, snap chainSnapshot) Adapter {
	a := &dualAveraging{
		mode:    cfg.Adapt,
		dim:     cfg.Target.Dim,
		warmup:  cfg.Warmup,
		frozen:  true,
		iter:    snap.Iteration,
		step:    snap.Params.StepSize,
		useMass: cfg.Adapt == AdaptStepSizeMass && cfg.Algo == AlgoHMC,
	}
	if a.useMass {
		a.mass = cloneFloat64(snap.Params.Mass)
		a.mean = make([]float64, a.dim)
		a.m2 = make([]float64, a.dim)
	}
	return a
}

func (a *dualAveraging) Observe(d Draw, theta []float64, lp float64, grad []float64) {
	if a.frozen || a.mode == AdaptNone {
		return
	}
	i := a.iter
	a.iter++

	// Dual-averaging step-size adaptation (Hoffman & Gelman 2014,
	// Algorithm 5). Acceptance probability is bounded in [0,1].
	acc := d.AcceptProb
	if acc < 0 {
		acc = 0
	} else if acc > 1 {
		acc = 1
	}
	m := a.windowCount(i)
	a.hbar = (1-1/(m+a.t0))*a.hbar + (1/(m+a.t0))*(a.targetAccept()-acc)
	logEps := a.mu - math.Sqrt(m)/a.gamma*a.hbar
	a.logEpsBar = math.Pow(m, -a.k)*logEps + (1-math.Pow(m, -a.k))*a.logEpsBar
	a.step = clampStep(math.Exp(logEps))

	if a.useMass {
		a.observeMass(i, theta)
	}
}

func (a *dualAveraging) Params() Params {
	p := Params{StepSize: a.step}
	if a.useMass {
		p.Mass = cloneFloat64(a.mass)
	}
	return p
}

func (a *dualAveraging) Frozen() bool { return a.frozen }

func (a *dualAveraging) Freeze() Params {
	if a.frozen {
		return a.Params()
	}
	a.frozen = true
	if a.mode != AdaptNone {
		a.step = clampStep(math.Exp(a.logEpsBar))
	}
	return a.Params()
}

func (a *dualAveraging) Snapshot() AdaptSnapshot {
	return AdaptSnapshot{
		Mode:          a.mode,
		Frozen:        a.frozen,
		Iteration:     a.iter,
		StepSize:      a.step,
		Hbar:          a.hbar,
		LogEpsBar:     a.logEpsBar,
		Mu:            a.mu,
		M:             cloneFloat64(a.mean),
		M2:            cloneFloat64(a.m2),
		SampleCount:   a.sampleCount,
		WindowStart:   a.windowStart,
		WindowEnd:     a.windowEnd,
		NextWindowEnd: a.nextEnd,
		Term:          a.term,
		Mass:          cloneFloat64(a.mass),
	}
}

func (a *dualAveraging) Restore(s AdaptSnapshot) error {
	a.mode = s.Mode
	a.frozen = s.Frozen
	a.iter = s.Iteration
	a.step = s.StepSize
	a.hbar = s.Hbar
	a.logEpsBar = s.LogEpsBar
	a.mu = s.Mu
	a.sampleCount = s.SampleCount
	a.windowStart = s.WindowStart
	a.windowEnd = s.WindowEnd
	a.nextEnd = s.NextWindowEnd
	a.term = s.Term
	if a.useMass {
		if len(s.M) != a.dim || len(s.M2) != a.dim || len(s.Mass) != a.dim {
			return ErrCheckpointCorrupt
		}
		a.mean = cloneFloat64(s.M)
		a.m2 = cloneFloat64(s.M2)
		a.mass = cloneFloat64(s.Mass)
	}
	return nil
}

// targetAccept returns the midpoint of the configured acceptance range.
// initWindows lays out mass-estimation windows: a fast stepsize-only
// prefix, doubling regular windows, and a stepsize-only terminal suffix.
func (a *dualAveraging) initWindows() {
	const minTotal = 40
	if a.total < minTotal {
		a.windowEnd = -1 // no room: stepsize adaptation only
		return
	}
	a.initBuf = 15
	a.termBuf = 15
	avail := a.total - a.initBuf - a.termBuf
	w := 25
	if w > avail {
		w = avail
	}
	a.windowStart = a.initBuf
	a.windowEnd = a.initBuf + w
	a.planNextWindow()
}

// planNextWindow sets nextEnd: the following window doubles until the
// terminal suffix; the final regular window absorbs the slack so that it
// ends exactly at total-termBuf.
func (a *dualAveraging) planNextWindow() {
	if a.windowEnd >= a.total-a.termBuf {
		a.nextEnd = -1
		return
	}
	cur := a.windowEnd - a.windowStart
	next := a.windowEnd + 2*cur
	if next > a.total-a.termBuf {
		next = a.total - a.termBuf
	}
	a.nextEnd = next
}

// windowCount returns the 1-based dual-averaging counter, reset whenever a
// mass window restarts (Hoffman-Gelman Algorithm 6).
func (a *dualAveraging) windowCount(i int) float64 {
	if !a.useMass || (!a.term && i < a.windowStart) {
		return float64(i + 1)
	}
	return float64(i - a.windowStart + 1)
}

func (a *dualAveraging) observeMass(i int, theta []float64) {
	if a.windowEnd < 0 || i < a.windowStart || i >= a.finalWindowEnd() {
		return
	}
	a.sampleCount++
	n := float64(a.sampleCount)
	for j := 0; j < a.dim; j++ {
		x := theta[j]
		delta := x - a.mean[j]
		a.mean[j] += delta / n
		a.m2[j] += delta * (x - a.mean[j])
	}
	if i+1 == a.windowEnd {
		a.closeWindow()
	}
}

func (a *dualAveraging) finalWindowEnd() int {
	if a.windowEnd < 0 {
		return -1
	}
	if a.nextEnd < 0 {
		return a.windowEnd
	}
	return a.total - a.termBuf
}

// closeWindow turns the window's sample covariance into a regularized
// diagonal mass matrix and starts the next window.
func (a *dualAveraging) closeWindow() {
	if a.sampleCount > 1 {
		a.mass = a.estimateMass()
	}
	if a.nextEnd < 0 {
		// Restart dual averaging for the terminal stepsize-only phase.
		a.hbar = 0
		a.mu = math.Log(10 * a.step)
		a.windowStart = a.total - a.termBuf
		a.term = true
		a.windowEnd = -1
		return
	}
	a.windowStart = a.windowEnd
	a.windowEnd = a.nextEnd
	a.sampleCount = 0
	a.mean = make([]float64, a.dim)
	a.m2 = make([]float64, a.dim)
	// Restart dual averaging for the new window (Algorithm 6).
	a.hbar = 0
	a.mu = math.Log(10 * a.step)
	a.planNextWindow()
}

// estimateMass regularizes the sample diagonal variance toward 1 with a
// unit-information prior of strength 5, matching Stan's prior.
func (a *dualAveraging) estimateMass() []float64 {
	mass := make([]float64, a.dim)
	n := float64(a.sampleCount)
	for j := range mass {
		v := a.m2[j] / (n - 1)
		if !(v > 1e-8) {
			v = 1
		}
		v = (n-1)/(n+5)*v + 5/(n+5)
		mass[j] = v
	}
	return mass
}

func (a *dualAveraging) targetAccept() float64 { return (a.targetLow + a.targetHigh) / 2 }

func clampStep(x float64) float64 {
	const minStep, maxStep = 1e-8, 1e3
	if math.IsNaN(x) || x < minStep {
		return minStep
	}
	if x > maxStep {
		return maxStep
	}
	return x
}
