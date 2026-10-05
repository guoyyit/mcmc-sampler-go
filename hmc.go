package mcmc

import "math"

// hmcEngine implements Hamiltonian Monte Carlo with a diagonal mass matrix
// and leapfrog integration.
type hmcEngine struct {
	target *Target
	dim    int

	theta    []float64
	lp       float64
	grad     []float64
	stepSize float64
	mass     []float64

	trajMin int
	trajMax int
	fdEps   float64
}

func newHMC(cfg *Config) (Engine, error) {
	dim := cfg.Target.Dim
	h := &hmcEngine{
		target:   &cfg.Target,
		dim:      dim,
		stepSize: cfg.HMCStepSize,
		mass:     defaultScale(dim),
		trajMin:  cfg.TrajectoryMin,
		trajMax:  cfg.TrajectoryMax,
		fdEps:    withDefault(cfg.FiniteDiffEps, 1e-6),
	}
	if cfg.TrajectoryMax <= cfg.TrajectoryMin {
		h.trajMin = cfg.TrajectorySteps
		h.trajMax = cfg.TrajectorySteps
	}
	return h, nil
}

func (h *hmcEngine) Init(theta []float64, lp float64) {
	h.theta = cloneFloat64(theta)
	h.lp = lp
	if g, err := h.gradient(h.theta); err == nil && g != nil {
		h.grad = g
	}
}

func (h *hmcEngine) Step(rng rng64, warmup bool, adapter Adapter) (Draw, error) {
	if adapter != nil {
		p := adapter.Params()
		if p.StepSize > 0 {
			h.stepSize = p.StepSize
		}
		if len(p.Mass) == h.dim {
			h.mass = cloneFloat64(p.Mass)
		}
	}
	if h.grad == nil {
		g, err := h.gradient(h.theta)
		if err != nil {
			return Draw{}, err
		}
		h.grad = g
	}

	momentum := make([]float64, h.dim)
	for i := range momentum {
		momentum[i] = rng.NormalFloat64()
		if math.IsNaN(momentum[i]) || math.IsInf(momentum[i], 0) {
			return Draw{}, ErrChainDiverged
		}
	}
	kineticStart := h.kinetic(momentum)
	nsteps := h.trajMin
	if h.trajMax > h.trajMin {
		nsteps += int(rng.Float64() * float64(h.trajMax-h.trajMin+1))
		if nsteps > h.trajMax {
			nsteps = h.trajMax
		}
	}

	newTheta, newMomentum, newGrad, lpNew, finite := h.leapfrog(h.theta, momentum, nsteps)
	d := Draw{Warmup: warmup}
	if !finite {
		// Divergent/illegal trajectory: stay at the current point. The
		// proposal has probability zero, so acceptance probability is 0.
		d.Theta = cloneFloat64(h.theta)
		d.LogProb = h.lp
		d.Accepted = false
		d.AcceptProb = 0
		d.Energy = math.Inf(+1)
		if adapter != nil && warmup {
			adapter.Observe(d, h.theta, h.lp, h.grad)
		}
		return d, nil
	}

	kineticEnd := h.kinetic(newMomentum)
	hStart := -h.lp + kineticStart
	hEnd := -lpNew + kineticEnd
	d.Energy = hEnd
	// log alpha = H(start) - H(end), entirely in log space.
	logAlpha := hStart - hEnd
	if math.IsNaN(logAlpha) || math.IsInf(logAlpha, +1) {
		logAlpha = 0
	}
	// Averaged acceptance proxy (Hoffman-Gelman): min(1, exp(logAlpha)).
	alpha := 1.0
	if logAlpha < 0 {
		alpha = math.Exp(logAlpha)
	}
	accepted, _ := acceptWith(rng, logAlpha)
	d.AcceptProb = alpha
	if accepted {
		h.theta = newTheta
		h.lp = lpNew
		h.grad = newGrad
		d.Accepted = true
	} else {
		d.Accepted = false
	}
	d.Theta = cloneFloat64(h.theta)
	d.LogProb = h.lp
	if adapter != nil && warmup {
		adapter.Observe(d, h.theta, h.lp, h.grad)
	}
	return d, nil
}

func (h *hmcEngine) Params() Params {
	return Params{StepSize: h.stepSize, Mass: cloneFloat64(h.mass)}
}

func (h *hmcEngine) SetParams(p Params) {
	if isFinite(p.StepSize) && p.StepSize > 0 {
		h.stepSize = p.StepSize
	}
	if len(p.Mass) == h.dim {
		for _, m := range p.Mass {
			if !(m > 0) || !isFinite(m) {
				return
			}
		}
		h.mass = cloneFloat64(p.Mass)
	}
}

// leapfrog performs nsteps velocity-Verlet (leapfrog) steps.
func (h *hmcEngine) leapfrog(theta, momentum []float64, nsteps int) (
	newTheta, newMomentum, grad []float64, lp float64, finite bool) {
	q := cloneFloat64(theta)
	p := cloneFloat64(momentum)
	g, err := h.gradient(q)
	if err != nil || g == nil {
		return nil, nil, nil, math.Inf(-1), false
	}
	// First half momentum update.
	for i := 0; i < h.dim; i++ {
		p[i] += 0.5 * h.stepSize * g[i]
	}
	for step := 0; step < nsteps; step++ {
		for i := 0; i < h.dim; i++ {
			q[i] += h.stepSize * p[i] / h.mass[i]
			if math.IsNaN(q[i]) || math.IsInf(q[i], 0) {
				return nil, nil, nil, math.Inf(-1), false
			}
		}
		lpNew, gNew, errG := h.evalGradAt(q)
		if errG != nil || gNew == nil || !allFinite(gNew) {
			return nil, nil, nil, math.Inf(-1), false
		}
		g = gNew
		lp = lpNew
		if step+1 < nsteps {
			for i := 0; i < h.dim; i++ {
				p[i] += h.stepSize * g[i]
			}
		}
	}
	// Final half momentum update; negate to exploit the H(-p|q)=H(p|q)
	// symmetry in the acceptance ratio.
	for i := 0; i < h.dim; i++ {
		p[i] = -(p[i] + 0.5*h.stepSize*g[i])
	}
	if !allFinite(p) || !isFinite(lp) {
		return nil, nil, nil, math.Inf(-1), false
	}
	return q, p, g, lp, true
}

// evalGradAt returns the log density and gradient, treating illegal points
// as -infinity with nil gradient.
func (h *hmcEngine) evalGradAt(theta []float64) (float64, []float64, error) {
	lp, g, err := h.target.EvalGrad(theta)
	if err != nil {
		return 0, nil, err
	}
	return lp, g, nil
}

// gradient evaluates the analytic gradient or a forward finite difference.
func (h *hmcEngine) gradient(theta []float64) ([]float64, error) {
	if h.target.GradLogProb != nil {
		_, g, err := h.target.EvalGrad(theta)
		if err != nil {
			return nil, err
		}
		return g, nil
	}
	lp0, err := h.target.Eval(theta)
	if err != nil {
		return nil, err
	}
	if math.IsInf(lp0, -1) {
		return nil, nil
	}
	work := cloneFloat64(theta)
	grad := make([]float64, h.dim)
	for i := 0; i < h.dim; i++ {
		eps := h.fdEps
		if abs64(theta[i]) > 1 {
			eps *= abs64(theta[i])
		}
		work[i] = theta[i] + eps
		lp1, err := h.target.Eval(work)
		if err != nil {
			return nil, err
		}
		work[i] = theta[i]
		if math.IsInf(lp1, -1) {
			return nil, nil
		}
		grad[i] = (lp1 - lp0) / eps
		if !isFinite(grad[i]) {
			return nil, nil
		}
	}
	return grad, nil
}

// kinetic evaluates K(p) = 0.5 sum p_i^2 / m_i for the diagonal mass.
func (h *hmcEngine) kinetic(momentum []float64) float64 {
	k := 0.0
	for i, pi := range momentum {
		k += pi * pi / h.mass[i]
	}
	return 0.5 * k
}

func allFinite(x []float64) bool {
	for _, v := range x {
		if !isFinite(v) {
			return false
		}
	}
	return true
}

func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
