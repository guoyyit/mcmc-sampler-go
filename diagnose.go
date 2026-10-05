package mcmc

import (
	"math"
	"sort"
)

// Diagnostics holds per-chain and pooled convergence quantities.
type Diagnostics struct {
	// RHat is the maximum over coordinates of the rank-normalized
	// (split) R-hat (Vehtari et al. 2021). 1.0 means perfect mixing.
	RHat []float64
	// ESS is the per-coordinate bulk effective sample size.
	ESS []float64
	// ESSTail is the per-coordinate tail (5%/95%) effective sample size.
	ESSTail []float64
	// N is the total number of retained post-warmup draws across chains.
	N int
	// Mean is the pooled per-coordinate posterior mean.
	Mean []float64
	// Std is the pooled per-coordinate posterior standard deviation.
	Std []float64
}

// Converged reports whether the thresholds are met.
func (d *Diagnostics) Converged(rhatThreshold, essTarget float64, chainCount int) bool {
	if d == nil {
		return false
	}
	for _, r := range d.RHat {
		if math.IsNaN(r) || r > rhatThreshold {
			return false
		}
	}
	needed := essTarget
	if 5*float64(chainCount) > needed {
		needed = 5 * float64(chainCount)
	}
	for _, e := range d.ESS {
		if math.IsNaN(e) || e < needed {
			return false
		}
	}
	return true
}

// assess computes diagnostics from one History per chain.
func assess(histories []*History, dim int) (*Diagnostics, error) {
	if len(histories) == 0 {
		return nil, ErrNoSamples
	}
	n := 0
	for _, h := range histories {
		if len(h.Draws()) == 0 {
			return nil, ErrNoSamples
		}
		n += len(h.Draws())
	}
	d := &Diagnostics{
		RHat:    make([]float64, dim),
		ESS:     make([]float64, dim),
		ESSTail: make([]float64, dim),
		N:       n,
		Mean:    make([]float64, dim),
		Std:     make([]float64, dim),
	}
	for j := 0; j < dim; j++ {
		vals := make([][]float64, len(histories))
		var all []float64
		for c, h := range histories {
			draws := h.Draws()
			vals[c] = make([]float64, len(draws))
			for i, dr := range draws {
				vals[c][i] = dr.Theta[j]
				all = append(all, dr.Theta[j])
			}
		}
		d.RHat[j] = splitRhat(vals)
		d.ESS[j] = essBulk(vals)
		d.ESSTail[j] = essTail(vals)
		mean, std := meanStd(all)
		d.Mean[j] = mean
		d.Std[j] = std
	}
	return d, nil
}

// splitRhat computes the rank-normalized split R-hat for one coordinate.
func splitRhat(chainVals [][]float64) float64 {
	return rhatFromRanks(splitChains(chainVals))
}

// essBulk computes the autocorrelation-based bulk ESS for one coordinate.
func essBulk(chainVals [][]float64) float64 {
	split := splitChains(chainVals)
	m, n := len(split), len(split[0])
	allRanks := rankNormalize(flatten(split))
	z := make([][]float64, m)
	idx := 0
	for c := 0; c < m; c++ {
		z[c] = make([]float64, n)
		for i := 0; i < n; i++ {
			z[c][i] = invNormalCDF((allRanks[idx] - 0.5) / float64(m*n))
			idx++
		}
	}
	return essFromChains(z)
}

// essTail computes the minimum tail ESS for one coordinate at the 5% and
// 95% quantiles (Gelman et al. 2013).
func essTail(chainVals [][]float64) float64 {
	all := flatten(chainVals)
	sorted := append([]float64(nil), all...)
	sort.Float64s(sorted)
	q05 := quantileSorted(sorted, 0.05)
	q95 := quantileSorted(sorted, 0.95)
	indicator := func(thresh float64, lowTail bool) [][]float64 {
		out := make([][]float64, len(chainVals))
		for c := range chainVals {
			out[c] = make([]float64, len(chainVals[c]))
			for i, x := range chainVals[c] {
				if lowTail && x <= thresh {
					out[c][i] = 1
				}
				if !lowTail && x >= thresh {
					out[c][i] = 1
				}
			}
		}
		return out
	}
	essLow := essFromChains(indicator(q05, true))
	essHigh := essFromChains(indicator(q95, false))
	return math.Min(essLow, essHigh)
}

// splitChains halves each chain so R-hat/ESS also detect non-stationarity
// within a chain.
func splitChains(chains [][]float64) [][]float64 {
	out := make([][]float64, 0, 2*len(chains))
	for _, c := range chains {
		half := len(c) / 2
		if half < 1 {
			out = append(out, append([]float64(nil), c...))
			continue
		}
		out = append(out, append([]float64(nil), c[:half]...))
		out = append(out, append([]float64(nil), c[half:2*half]...))
	}
	return out
}

func flatten(chains [][]float64) []float64 {
	var out []float64
	for _, c := range chains {
		out = append(out, c...)
	}
	return out
}

// rankNormalize returns 1-based average ranks with fractional ranks for ties.
func rankNormalize(x []float64) []float64 {
	idx := make([]int, len(x))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return x[idx[a]] < x[idx[b]] })
	ranks := make([]float64, len(x))
	for i := 0; i < len(x); {
		j := i + 1
		for j < len(x) && x[idx[j]] == x[idx[i]] {
			j++
		}
		avg := float64(i+j+1) / 2
		for k := i; k < j; k++ {
			ranks[idx[k]] = avg
		}
		i = j
	}
	return ranks
}

// rhatFromRanks applies the rank-normalized split R-hat of Vehtari et al.
func rhatFromRanks(chains [][]float64) float64 {
	m, n := len(chains), len(chains[0])
	if m < 2 || n < 2 {
		return math.NaN()
	}
	allRanks := rankNormalize(flatten(chains))
	z := make([][]float64, m)
	idx := 0
	for c := 0; c < m; c++ {
		z[c] = make([]float64, n)
		for i := 0; i < n; i++ {
			z[c][i] = invNormalCDF((allRanks[idx] - 0.5) / float64(m*n))
			idx++
		}
	}
	means := make([]float64, m)
	vars := make([]float64, m)
	grand := 0.0
	for c := 0; c < m; c++ {
		mean, std := meanStd(z[c])
		means[c] = mean
		vars[c] = std * std
		grand += mean
	}
	grand /= float64(m)
	b := 0.0
	for c := 0; c < m; c++ {
		b += (means[c] - grand) * (means[c] - grand)
	}
	b *= float64(n) / float64(m-1)
	w := 0.0
	for c := 0; c < m; c++ {
		w += vars[c]
	}
	w /= float64(m)
	varPlus := (float64(n-1)/float64(n))*w + b/float64(n)
	if w <= 0 || varPlus <= 0 {
		return math.NaN()
	}
	return math.Sqrt(varPlus / w)
}

// essFromChains implements the multi-chain positive-sequence ESS estimator
// (Gelman et al. 2013).
func essFromChains(chains [][]float64) float64 {
	m, n := len(chains), len(chains[0])
	if m < 2 || n < 4 {
		return math.NaN()
	}
	chainMean := make([]float64, m)
	chainVar := make([]float64, m)
	for c := 0; c < m; c++ {
		mean, std := meanStd(chains[c])
		chainMean[c] = mean
		chainVar[c] = std * std
	}
	var0 := 0.0
	for c := 0; c < m; c++ {
		for i := 0; i < n; i++ {
			d := chains[c][i] - chainMean[c]
			var0 += d * d
		}
	}
	var0 /= float64(m * (n - 1))
	if !(var0 > 0) {
		return math.NaN()
	}
	rhoSum := 0.0
	prevNeg := false
	maxLag := n - 1
	for lag := 1; lag <= maxLag; lag++ {
		chat := 0.0
		for c := 0; c < m; c++ {
			for i := lag; i < n; i++ {
				chat += (chains[c][i] - chainMean[c]) * (chains[c][i-lag] - chainMean[c])
			}
		}
		chat /= float64(m * (n - lag))
		rho := 1 - (var0-chat)/var0
		if rho < 0 {
			if prevNeg {
				break
			}
			prevNeg = true
		} else {
			prevNeg = false
		}
		rhoSum += rho
	}
	ess := float64(m*n) / (1 + 2*rhoSum)
	if !(ess > 0) || math.IsNaN(ess) {
		return 0
	}
	if ess > float64(m*n) {
		ess = float64(m * n)
	}
	return ess
}

func meanStd(x []float64) (float64, float64) {
	if len(x) == 0 {
		return math.NaN(), math.NaN()
	}
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))
	if len(x) < 2 {
		return mean, 0
	}
	ss := 0.0
	for _, v := range x {
		d := v - mean
		ss += d * d
	}
	return mean, math.Sqrt(ss / float64(len(x)-1))
}

func quantileSorted(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return math.NaN()
	}
	pos := q * float64(len(sorted)+1)
	idx := int(pos)
	if idx < 1 {
		return sorted[0]
	}
	if idx >= len(sorted) {
		return sorted[len(sorted)-1]
	}
	frac := pos - float64(idx)
	return sorted[idx-1] + frac*(sorted[idx]-sorted[idx-1])
}

// invNormalCDF is Acklam's rational approximation of the standard normal
// quantile function. Input must be in (0,1).
func invNormalCDF(p float64) float64 {
	const (
		pLow  = 0.02425
		pHigh = 1 - pLow
	)
	a := []float64{-3.969683028665376e+01, 2.209460984245205e+02,
		-2.759285104469687e+02, 1.383577518672690e+02,
		-3.066479806614716e+01, 2.506628277459239e+00}
	b := []float64{-5.447609879822406e+01, 1.615858368580409e+02,
		-1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01}
	c := []float64{-7.784894002430293e-03, -3.223964580411365e-01,
		-2.400758277161838e+00, -2.549732539343734e+00,
		4.374664141464968e+00, 2.938163982698783e+00}
	d := []float64{7.784695709041462e-03, 3.224671290700398e-01,
		2.445134137142996e+00, 3.754408661907416e+00}
	switch {
	case p < pLow:
		q := math.Sqrt(-2 * math.Log(p))
		return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	case p <= pHigh:
		q := p - 0.5
		r := q * q
		return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r + a[5]) * q /
			(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r + 1)
	default:
		q := math.Sqrt(-2 * math.Log(1-p))
		return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	}
}
