package diagnostics

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Independent chains from the same distribution: R-hat ~ 1 and ESS close to
// the nominal sample size.
func TestConvergedIndependentChains(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	m, n := 4, 2000
	chains := make([][]float64, m)
	for i := range chains {
		chains[i] = make([]float64, n)
		for j := range chains[i] {
			chains[i][j] = r.NormFloat64()
		}
	}
	rh := SplitRhat(chains)
	if math.IsNaN(rh) || math.IsInf(rh, 0) || rh < 0.99 || rh > 1.01 {
		t.Fatalf("R-hat for iid chains = %.4f, want ~1", rh)
	}
	ess := BulkESS(chains)
	if ess < float64(m*n)*0.8 {
		t.Fatalf("bulk ESS for iid chains = %.0f, want >= %d", ess, int(float64(m*n)*0.8))
	}
	if ess > float64(m*n)*1.05 {
		t.Fatalf("bulk ESS exceeds nominal draws: %.0f > %d", ess, m*n)
	}
	te := TailESS(chains)
	if te < float64(m*n)*0.7 {
		t.Fatalf("tail ESS for iid chains = %.0f, suspiciously low", te)
	}
}

// One chain offset from the others: R-hat must be large.
func TestOffsetChainsHighRhat(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	chains := make([][]float64, 4)
	for i := range chains {
		chains[i] = make([]float64, 1000)
		offset := 0.0
		if i == 0 {
			offset = 5 // chain 0 stuck far away
		}
		for j := range chains[i] {
			chains[i][j] = offset + r.NormFloat64()
		}
	}
	if rh := SplitRhat(chains); rh <= 1.2 {
		t.Fatalf("R-hat for offset chains = %.4f, want > 1.2", rh)
	}
}

// Highly autocorrelated chains: ESS must be far below nominal size.
func TestAutocorrelatedLowESS(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	m, n := 4, 2000
	chains := make([][]float64, m)
	for i := range chains {
		chains[i] = make([]float64, n)
		x := 0.0
		for j := range chains[i] {
			x = 0.995*x + r.NormFloat64()
			chains[i][j] = x
		}
	}
	ess := BulkESS(chains)
	if ess > float64(m*n)*0.2 {
		t.Fatalf("bulk ESS for AR(1) rho=0.995 = %.0f, expected much less than %d", ess, m*n)
	}
}

func TestEvaluateVerdict(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	good := func() [][][]float64 {
		chains := make([][][]float64, 4)
		for i := range chains {
			chains[i] = make([][]float64, 1000)
			for j := range chains[i] {
				chains[i][j] = []float64{r.NormFloat64(), r.NormFloat64()}
			}
		}
		return chains
	}()
	rep := Evaluate(good, Thresholds{MaxRhat: 1.01, MinESS: 100})
	if !rep.Converged {
		t.Fatalf("expected convergence, reasons: %v", rep.Reasons)
	}

	// Same data but an impossible threshold: must explicitly fail.
	bad := Evaluate(good, Thresholds{MaxRhat: 1.0001, MinESS: 100000})
	if bad.Converged || len(bad.Reasons) == 0 {
		t.Fatal("expected explicit non-convergence with reasons")
	}
}

func TestConstantChainsRhatIsOne(t *testing.T) {
	chains := [][]float64{
		make([]float64, 1000),
		make([]float64, 1000),
	}
	for i := range chains[0] {
		chains[0][i] = 3
		chains[1][i] = 3
	}
	if rh := SplitRhat(chains); math.Abs(rh-1) > 0.02 {
		t.Fatalf("R-hat for identical constant chains = %.4f, want ~1", rh)
	}
}

func TestRankNormalizeMonotone(t *testing.T) {
	z := rankNormalize([]float64{-2, -1, 0, 1, 2})
	for i := 1; i < len(z); i++ {
		if !(z[i] > z[i-1]) {
			t.Fatalf("rank-normalized values not strictly increasing: %v", z)
		}
	}
}
