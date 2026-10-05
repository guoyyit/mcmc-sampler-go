package diagnostics

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Thresholds configures the convergence decision.
type Thresholds struct {
	MaxRhat    float64 // maximum allowed split R-hat
	MinESS     int     // minimum allowed bulk ESS
	MinTailESS int     // minimum allowed tail ESS (0 disables)
}

// VariableResult holds diagnostics for one scalar variable.
type VariableResult struct {
	Index   int
	Rhat    float64
	BulkESS float64
	TailESS float64
}

// Report is the convergence verdict for a whole run.
type Report struct {
	Converged bool
	Variables []VariableResult
	Reasons   []string
}

var (
	// ErrInsufficientData is returned when diagnostics cannot be computed.
	ErrInsufficientData = errors.New("diagnostics: need at least 2 chains with at least 4 draws each")
)

// Default thresholds follow Vehtari et al. (2021): R-hat < 1.01 and ESS at
// least min(100, 4*numChains) per parameter.
func (t *Thresholds) Defaults(numChains int) {
	if t.MaxRhat == 0 {
		t.MaxRhat = 1.01
	}
	if t.MinESS == 0 {
		n := 4 * numChains
		if n > 100 {
			n = 100
		}
		t.MinESS = n
	}
}

// normInv is a rational approximation of the standard normal quantile
// function (Acklam's algorithm), accurate to ~1e-9.
func normInv(p float64) float64 {
	const (
		a1 = -3.969683028665376e+01
		a2 = 2.209460984245205e+02
		a3 = -2.759285104469687e+02
		a4 = 1.383577518672690e+02
		a5 = -3.066479806614716e+01
		a6 = 2.506628277459239e+00

		b1 = -5.447609879822406e+01
		b2 = 1.615858368580409e+02
		b3 = -1.556989798598866e+02
		b4 = 6.680131188771972e+01
		b5 = -1.328068155288572e+01

		c1 = -7.784894002430293e-03
		c2 = -3.223964580411365e-01
		c3 = -2.400758277161838e+00
		c4 = -2.549732539343734e+00
		c5 = 4.374664141464968e+00
		c6 = 2.938163982698783e+00

		d1 = 7.784695709041462e-03
		d2 = 3.224671290700398e-01
		d3 = 2.445134137142996e+00
		d4 = 3.754408661907416e+00
	)
	pLow, pHigh := 0.02425, 1-0.02425
	switch {
	case p <= 0:
		return math.Inf(-1)
	case p >= 1:
		return math.Inf(+1)
	case p < pLow:
		q := math.Sqrt(-2 * math.Log(p))
		return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) /
			((((d1*q+d2)*q+d3)*q+d4)*q + 1)
	case p <= pHigh:
		q := p - 0.5
		r := q * q
		return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r + a6) * q /
			(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r + 1)
	default:
		q := math.Sqrt(-2 * math.Log(1-p))
		return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) /
			((((d1*q+d2)*q+d3)*q+d4)*q + 1)
	}
}

// rankNormalize replaces every value by Phi^{-1}((rank - 3/8)/(n - 1/4))
// using average ranks for ties.
func rankNormalize(values []float64) []float64 {
	n := len(values)
	type item struct {
		v float64
		i int
	}
	items := make([]item, n)
	for i, v := range values {
		items[i] = item{v, i}
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].v != items[j].v {
			return items[i].v < items[j].v
		}
		return items[i].i < items[j].i
	})
	out := make([]float64, n)
	for i := 0; i < n; {
		j := i + 1
		for j < n && items[j].v == items[i].v {
			j++
		}
		avgRank := 0.5*float64(i+1+j) - 0.5
		z := normInv((avgRank - 0.375) / (float64(n) - 0.25))
		for k := i; k < j; k++ {
			out[items[k].i] = z
		}
		i = j
	}
	return out
}

// splitChains halves every chain, yielding 2*M pseudo-independent chains.
func splitChains(chains [][]float64) [][]float64 {
	split := make([][]float64, 0, 2*len(chains))
	for _, c := range chains {
		h := len(c) / 2
		if h < 2 {
			return nil
		}
		split = append(split,
			append([]float64(nil), c[:h]...),
			append([]float64(nil), c[h:2*h]...))
	}
	return split
}

// SplitRhat computes the rank-normalized split R-hat (Vehtari et al. 2021).
// Returns +Inf if it cannot be computed.
func SplitRhat(chains [][]float64) float64 {
	split := splitChains(chains)
	m := len(split)
	if m < 2 {
		return math.Inf(1)
	}
	n := len(split[0])
	if n < 2 {
		return math.Inf(1)
	}
	var all []float64
	for _, c := range split {
		if len(c) != n {
			return math.Inf(1)
		}
		all = append(all, c...)
	}
	z := rankNormalize(all)
	zChains := make([][]float64, m)
	for i := range zChains {
		zChains[i] = z[i*n : (i+1)*n]
	}

	means := make([]float64, m)
	vars := make([]float64, m)
	var grand float64
	for k, c := range zChains {
		var sum float64
		for _, v := range c {
			sum += v
		}
		means[k] = sum / float64(n)
		grand += means[k]
		var ss float64
		for _, v := range c {
			d := v - means[k]
			ss += d * d
		}
		vars[k] = ss / float64(n-1)
	}
	grand /= float64(m)
	var b float64
	for _, x := range means {
		d := x - grand
		b += d * d
	}
	b *= float64(n) / float64(m-1)
	var w float64
	for _, v := range vars {
		w += v
	}
	w /= float64(m)
	if w == 0 {
		if b == 0 {
			return 1
		}
		return math.Inf(1)
	}
	varPlus := (float64(n-1)/float64(n))*w + b/float64(n)
	return math.Sqrt(varPlus / w)
}

// autocorrESS computes ESS for equally sized chains via Geyer's initial
// positive sequence on per-chain autocorrelations (Vehtari et al. 2021).
// autocorrESS computes ESS for equally sized chains via Geyer's initial
// positive sequence on pooled per-chain autocorrelations (Vehtari et al.
// 2021, eq. 9-10).
func autocorrESS(chains [][]float64) float64 {
	m := len(chains)
	n := len(chains[0])
	means := make([]float64, m)
	vars := make([]float64, m)
	for k, c := range chains {
		var sum float64
		for _, v := range c {
			sum += v
		}
		means[k] = sum / float64(n)
		var ss float64
		for _, v := range c {
			d := v - means[k]
			ss += d * d
		}
		vars[k] = ss / float64(n)
	}

	// tau = -1 + 2 * sum_k P_k where P_k = rho_{2k} + rho_{2k+1}, summing
	// only while P_k is positive and enforcing monotone decrease.
	rhoEvenSum := 0.0
	prev := math.Inf(1)
	for k := 1; k < n; k += 2 {
		rho0 := chainAutocorr(chains, means, vars, k)
		rho1 := chainAutocorr(chains, means, vars, k+1)
		pair := rho0 + rho1
		if pair <= 0 {
			break
		}
		if pair > prev {
			pair = prev
		}
		prev = pair
		rhoEvenSum += pair
	}
	tau := -1 + 2*rhoEvenSum
	if tau <= 0 {
		tau = 1
	}
	ess := float64(m*n) / tau
	if ess > float64(m*n) {
		ess = float64(m * n)
	}
	if ess < 0 {
		ess = 0
	}
	return ess
}

// chainAutocorr returns the pooled per-chain autocorrelation estimate at lag.
func chainAutocorr(chains [][]float64, means, vars []float64, lag int) float64 {
	m := len(chains)
	n := len(chains[0])
	if lag >= n {
		return 0
	}
	var rho float64
	for k, c := range chains {
		if vars[k] <= 0 {
			continue
		}
		var cov float64
		for t := 0; t+lag < n; t++ {
			cov += (c[t] - means[k]) * (c[t+lag] - means[k])
		}
		rho += cov / float64(n-lag) / vars[k]
	}
	return rho / float64(m)
}

// BulkESS computes rank-normalized bulk effective sample size.
func BulkESS(chains [][]float64) float64 {
	split := splitChains(chains)
	if len(split) < 2 || len(split[0]) < 4 {
		return 0
	}
	n := len(split[0])
	var all []float64
	for _, c := range split {
		if len(c) != n {
			return 0
		}
		all = append(all, c...)
	}
	z := rankNormalize(all)
	m := len(split)
	zChains := make([][]float64, m)
	for i := range zChains {
		zChains[i] = z[i*n : (i+1)*n]
	}
	return autocorrESS(zChains)
}

// TailESS computes the minimum of the 5th/95th percentile tail ESS using
// indicator functions (Vehtari et al. 2021).
func TailESS(chains [][]float64) float64 {
	split := splitChains(chains)
	if len(split) < 2 || len(split[0]) < 4 {
		return 0
	}
	m := len(split)
	n := len(split[0])

	var all []float64
	for _, c := range split {
		if len(c) != n {
			return 0
		}
		all = append(all, c...)
	}
	sorted := append([]float64(nil), all...)
	sort.Float64s(sorted)
	q05 := sorted[int(0.05*float64(len(sorted)-1))]
	q95 := sorted[int(0.95*float64(len(sorted)-1))]

	essAt := func(threshold float64, upper bool) float64 {
		ind := make([][]float64, m)
		for i, c := range split {
			ind[i] = make([]float64, n)
			for t, v := range c {
				if (!upper && v <= threshold) || (upper && v >= threshold) {
					ind[i][t] = 1
				}
			}
		}
		return autocorrESS(ind)
	}
	return math.Min(essAt(q05, false), essAt(q95, true))
}

// Evaluate computes diagnostics for every dimension and applies the
// thresholds. The verdict is always explicit: any failure lists a reason in
// Report.Reasons and Converged is false.
func Evaluate(samplesPerChain [][][]float64, t Thresholds) Report {
	rep := Report{Converged: true}
	t.Defaults(len(samplesPerChain))
	if len(samplesPerChain) < 2 {
		rep.Converged = false
		rep.Reasons = append(rep.Reasons, "need at least 2 chains to assess convergence")
		return rep
	}
	dim := len(samplesPerChain[0][0])
	for d := 0; d < dim; d++ {
		chains := make([][]float64, len(samplesPerChain))
		for c, draws := range samplesPerChain {
			if len(draws) < 4 {
				rep.Converged = false
				rep.Reasons = append(rep.Reasons, fmt.Sprintf("dimension %d: chain %d has fewer than 4 draws", d, c))
			}
			chains[c] = make([]float64, len(draws))
			for i, x := range draws {
				chains[c][i] = x[d]
			}
		}
		vr := VariableResult{
			Index:   d,
			Rhat:    SplitRhat(chains),
			BulkESS: BulkESS(chains),
			TailESS: TailESS(chains),
		}
		if vr.Rhat > t.MaxRhat {
			rep.Converged = false
			rep.Reasons = append(rep.Reasons, fmt.Sprintf("dimension %d: R-hat %.4f exceeds threshold %.4f", d, vr.Rhat, t.MaxRhat))
		}
		if int(vr.BulkESS) < t.MinESS {
			rep.Converged = false
			rep.Reasons = append(rep.Reasons, fmt.Sprintf("dimension %d: bulk ESS %.1f below minimum %d", d, vr.BulkESS, t.MinESS))
		}
		if t.MinTailESS > 0 && int(vr.TailESS) < t.MinTailESS {
			rep.Converged = false
			rep.Reasons = append(rep.Reasons, fmt.Sprintf("dimension %d: tail ESS %.1f below minimum %d", d, vr.TailESS, t.MinTailESS))
		}
		rep.Variables = append(rep.Variables, vr)
	}
	return rep
}
