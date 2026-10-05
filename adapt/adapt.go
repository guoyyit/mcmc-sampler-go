package adapt

import (
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"strings"
)

var (
	// ErrInvalidConfig is returned for malformed adaptation configuration.
	ErrInvalidConfig = errors.New("adapt: invalid configuration")
)

const (
	defaultGamma = 0.05
	defaultKappa = 0.75
	defaultT0    = 10
)

// Config controls warmup adaptation.
type Config struct {
	Warmup          int
	TargetAcceptMin float64
	TargetAcceptMax float64
	InitStepSize    float64
	AdaptMass       bool
	InitInverseMass []float64
	// Dual-averaging coefficients (Hoffman & Gelman 2014).
	Gamma float64
	Kappa float64
	T0    float64
}

// Engine adapts the HMC (or MH) step size during warmup and optionally the
// diagonal inverse mass matrix. Once Freeze is called all parameters are
// fixed for the remainder of the run.
type Engine struct {
	cfg Config

	// Dual averaging state (Hoffman & Gelman 2014, Algorithm 6).
	stepCount int     // m: number of accepted step-size observations
	logEps    float64 // log current step size
	hBar      float64 // Hbar
	logEpsBar float64 // log epsilon_bar (smoothed step size)
	mu        float64 // mu

	// Mass-matrix window adaptation.
	windowCount int       // observations accumulated in the current mass window
	mean        []float64 // running sample mean (Welford)
	m2          []float64 // running sum of squared deviations (Welford)

	committedMass []float64 // inverse mass exposed to the sampler

	frozen bool
}

// Defaults fills zero-valued tuning parameters with standard values.
func (cfg *Config) Defaults() {
	if cfg.Gamma == 0 {
		cfg.Gamma = defaultGamma
	}
	if cfg.Kappa == 0 {
		cfg.Kappa = defaultKappa
	}
	if cfg.T0 == 0 {
		cfg.T0 = defaultT0
	}
	if cfg.TargetAcceptMin == 0 {
		cfg.TargetAcceptMin = 0.8
	}
	if cfg.TargetAcceptMax == 0 {
		cfg.TargetAcceptMax = 0.8
	}
	if cfg.InitStepSize == 0 {
		cfg.InitStepSize = 1
	}
}

// Validate checks the configuration.
func (cfg Config) Validate() error {
	if cfg.Warmup < 0 {
		return fmt.Errorf("%w: warmup must be non-negative", ErrInvalidConfig)
	}
	if cfg.TargetAcceptMin < 0 || cfg.TargetAcceptMin > 1 ||
		cfg.TargetAcceptMax < 0 || cfg.TargetAcceptMax > 1 ||
		cfg.TargetAcceptMin > cfg.TargetAcceptMax {
		return fmt.Errorf("%w: target acceptance interval must lie within [0,1] with min <= max", ErrInvalidConfig)
	}
	if cfg.Gamma <= 0 || cfg.Kappa <= 0 || cfg.Kappa >= 1 || cfg.T0 <= 0 {
		return fmt.Errorf("%w: dual-averaging coefficients out of range", ErrInvalidConfig)
	}
	if cfg.InitStepSize <= 0 || math.IsInf(cfg.InitStepSize, 0) {
		return fmt.Errorf("%w: initial step size must be finite and positive", ErrInvalidConfig)
	}
	return nil
}

// New constructs an adaptation engine for a dim-dimensional target.
func New(cfg Config, dim int) (*Engine, error) {
	cfg.Defaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if dim <= 0 {
		return nil, fmt.Errorf("%w: dimension must be positive", ErrInvalidConfig)
	}
	invMass := append([]float64(nil), cfg.InitInverseMass...)
	if len(invMass) == 0 {
		invMass = make([]float64, dim)
		for i := range invMass {
			invMass[i] = 1
		}
	}
	if len(invMass) != dim {
		return nil, fmt.Errorf("%w: initial inverse mass dimension %d != %d", ErrInvalidConfig, len(invMass), dim)
	}
	for i, m := range invMass {
		if !(m > 0) || math.IsInf(m, 0) {
			return nil, fmt.Errorf("%w: initial inverse mass at %d must be positive", ErrInvalidConfig, i)
		}
	}
	e := &Engine{
		cfg:       cfg,
		logEps:    math.Log(cfg.InitStepSize),
		logEpsBar: math.Log(cfg.InitStepSize),
		mu:        math.Log(10 * cfg.InitStepSize),
		mean:      make([]float64, dim),
		m2:        make([]float64, dim),
	}
	// The frozen inverse mass is exposed only through InverseMass, which
	// returns the last committed estimate. Keep a committed copy.
	e.committedMass = append([]float64(nil), invMass...)
	return e, nil
}

// Active reports whether adaptation is running at the given 1-based
// iteration. Iterations 1..Warmup are adaptive; later iterations are frozen.
func (e *Engine) Active(iteration int) bool {
	return !e.frozen && iteration <= e.cfg.Warmup && iteration > 0
}

// Observe feeds one iteration's acceptance statistic (a log acceptance
// probability in [-inf, 0] or its probability form) and the accepted
// position. It must be called exactly once per warmup iteration, in order.
//
// Adaptation affects only proposal parameters (step size, mass matrix); it
// never edits the current chain state itself, so once Freeze is called the
// chain continues as an ordinary time-homogeneous Markov chain.
func (e *Engine) Observe(iteration int, acceptStat float64, position []float64) {
	if e.frozen || !e.Active(iteration) {
		return
	}
	if math.IsNaN(acceptStat) || math.IsInf(acceptStat, +1) {
		return
	}

	// Track the empirical acceptance probability against the configured
	// interval. acceptStat may be a probability in [0,1] or log-prob in
	// (-inf,0]; callers pass probabilities via ObserveAcceptance.
	target := 0.5 * (e.cfg.TargetAcceptMin + e.cfg.TargetAcceptMax)

	// Dual averaging (Hoffman & Gelman 2014, Algorithm 6).
	e.stepCount++
	m := float64(e.stepCount)
	e.hBar = (1-1/(m+e.cfg.T0))*e.hBar + (1/(m+e.cfg.T0))*(target-acceptStat)
	e.logEps = e.mu - e.hBar*math.Sqrt(m)/e.cfg.Gamma
	weight := math.Pow(m, -e.cfg.Kappa)
	e.logEpsBar = weight*e.logEps + (1-weight)*e.logEpsBar

	// Mass-matrix adaptation uses Welford's online covariance over warmup
	// draws; the estimate is committed at the end of warmup.
	if e.cfg.AdaptMass && len(position) == len(e.mean) {
		e.windowCount++
		n := float64(e.windowCount)
		for i, x := range position {
			delta := x - e.mean[i]
			e.mean[i] += delta / n
			e.m2[i] += delta * (x - e.mean[i])
		}
	}
}

// StepSize returns the instantaneous adapted step size.
func (e *Engine) StepSize() float64 {
	eps := math.Exp(e.logEps)
	if !(eps > 0) || math.IsInf(eps, 0) {
		return e.cfg.InitStepSize
	}
	return eps
}

// FinalStepSize returns the dual-averaging-smoothed step size, which is what
// gets frozen at the end of warmup.
func (e *Engine) FinalStepSize() float64 {
	eps := math.Exp(e.logEpsBar)
	if !(eps > 0) || math.IsInf(eps, 0) {
		return e.cfg.InitStepSize
	}
	return eps
}

// InverseMass returns the current committed diagonal inverse mass matrix.
func (e *Engine) InverseMass() []float64 {
	return append([]float64(nil), e.committedMass...)
}

// Frozen reports whether adaptation has ended.
func (e *Engine) Frozen() bool { return e.frozen }

// Freeze ends adaptation: the smoothed step size and the empirical variance
// diagonal become fixed for all subsequent iterations.
func (e *Engine) Freeze() {
	if e.frozen {
		return
	}
	e.logEps = math.Log(e.FinalStepSize())
	if e.cfg.AdaptMass && e.windowCount > 1 {
		n := float64(e.windowCount)
		variance := make([]float64, len(e.mean))
		for i := range variance {
			v := e.m2[i] / (n - 1)
			// Regularized diagonal: blend empirical variance with 1 to avoid
			// collapsing a dimension seen as constant.
			v = (n*v + 1e-3) / (n + 5e-3)
			if !(v > 1e-12) {
				v = 1e-3
			}
			variance[i] = v
		}
		e.committedMass = variance
	}
	e.frozen = true
}

type engineState struct {
	Config        Config
	StepCount     int
	LogEps        float64
	HBar          float64
	LogEpsBar     float64
	Mu            float64
	WindowCount   int
	Mean          []float64
	M2            []float64
	CommittedMass []float64
	Frozen        bool
}

// Marshal encodes the complete adaptation state.
func (e *Engine) Marshal() ([]byte, error) {
	var sb strings.Builder
	err := gob.NewEncoder(&sb).Encode(engineState{
		Config:        e.cfg,
		StepCount:     e.stepCount,
		LogEps:        e.logEps,
		HBar:          e.hBar,
		LogEpsBar:     e.logEpsBar,
		Mu:            e.mu,
		WindowCount:   e.windowCount,
		Mean:          e.mean,
		M2:            e.m2,
		CommittedMass: e.committedMass,
		Frozen:        e.frozen,
	})
	if err != nil {
		return nil, fmt.Errorf("adapt: marshal engine: %w", err)
	}
	return []byte(sb.String()), nil
}

// Unmarshal restores an engine previously produced by Marshal.
func (e *Engine) Unmarshal(data []byte) error {
	var st engineState
	if err := gob.NewDecoder(strings.NewReader(string(data))).Decode(&st); err != nil {
		return fmt.Errorf("adapt: unmarshal engine: %w", err)
	}
	e.cfg = st.Config
	e.stepCount = st.StepCount
	e.logEps = st.LogEps
	e.hBar = st.HBar
	e.logEpsBar = st.LogEpsBar
	e.mu = st.Mu
	e.windowCount = st.WindowCount
	e.mean = st.Mean
	e.m2 = st.M2
	e.committedMass = st.CommittedMass
	e.frozen = st.Frozen
	return nil
}
