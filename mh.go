package mcmc

import "math"

// mhEngine implements symmetric random-walk Metropolis-Hastings with a
// Gaussian proposal N(theta, StepSize^2 I).
type mhEngine struct {
	target   *Target
	dim      int
	theta    []float64
	lp       float64
	stepSize float64
}

func newMH(cfg *Config) (Engine, error) {
	return &mhEngine{
		target:   &cfg.Target,
		dim:      cfg.Target.Dim,
		stepSize: cfg.MHStepSize,
	}, nil
}

func (m *mhEngine) Init(theta []float64, lp float64) {
	m.theta = cloneFloat64(theta)
	m.lp = lp
}

func (m *mhEngine) Step(rng rng64, warmup bool, adapter Adapter) (Draw, error) {
	proposal := make([]float64, m.dim)
	for i := range proposal {
		proposal[i] = m.theta[i] + m.stepSize*rng.NormalFloat64()
		if math.IsNaN(proposal[i]) || math.IsInf(proposal[i], 0) {
			return Draw{}, ErrChainDiverged
		}
	}
	lpProp, err := m.target.Eval(proposal)
	if err != nil {
		return Draw{}, err
	}
	logAlpha := logAccept(lpProp, m.lp)
	accepted, alpha := acceptWith(rng, logAlpha)
	if accepted {
		m.theta = proposal
		m.lp = lpProp
	}
	d := Draw{
		Theta:      cloneFloat64(m.theta),
		LogProb:    m.lp,
		Accepted:   accepted,
		AcceptProb: alpha,
		Warmup:     warmup,
	}
	if adapter != nil && warmup {
		adapter.Observe(d, m.theta, m.lp, nil)
		m.stepSize = adapter.Params().StepSize
	}
	return d, nil
}

func (m *mhEngine) Params() Params { return Params{StepSize: m.stepSize} }

func (m *mhEngine) SetParams(p Params) {
	if p.StepSize > 0 && isFinite(p.StepSize) {
		m.stepSize = p.StepSize
	}
}
