package sampler

import (
	"context"
	"math"
	"testing"

	"github.com/guoyyit/mcmc-sampler-go/rng"
	"github.com/guoyyit/mcmc-sampler-go/target"
)

type gaussianTarget struct{}

func (gaussianTarget) Dim() int { return 2 }
func (gaussianTarget) LogDensity(x []float64) float64 {
	return -0.5 * (x[0]*x[0] + x[1]*x[1])
}
func (gaussianTarget) Gradient(x, g []float64) {
	g[0] = -x[0]
	g[1] = -x[1]
}

func TestMHSmoke(t *testing.T) {
	tg, _ := target.New(2, func(x []float64) float64 {
		return -0.5 * (x[0]*x[0] + x[1]*x[1])
	})
	r := rng.NewRoot(1)
	s, err := New(tg, r, Params{
		Algorithm:     AlgoMH,
		Initial:       []float64{1, -1},
		ProposalScale: []float64{0.5, 0.5},
	})
	if err != nil {
		t.Fatal(err)
	}
	var sum [2]float64
	n := 20000
	accepted := 0
	for i := 0; i < n; i++ {
		st, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if !!math.IsNaN(st.LogProb) && !math.IsInf(st.LogProb, 0) {
			t.Fatalf("non-finite log prob at iter %d: %v", i, st.LogProb)
		}
		if st.Stats.Accepted {
			accepted++
		}
		if i >= 5000 {
			sum[0] += st.Position[0]
			sum[1] += st.Position[1]
		}
	}
	m := float64(n - 5000)
	mean0, mean1 := sum[0]/m, sum[1]/m
	if math.Abs(mean0) > 0.1 || math.Abs(mean1) > 0.1 {
		t.Fatalf("post-warmup mean off: %f %f (accept rate %.2f)", mean0, mean1, float64(accepted)/float64(n))
	}
	t.Logf("MH accept rate %.2f means %.3f %.3f", float64(accepted)/float64(n), mean0, mean1)
}

func TestHMCSmoke(t *testing.T) {
	tg, _ := target.New(2, func(x []float64) float64 {
		return -0.5 * (x[0]*x[0] + x[1]*x[1])
	})
	_ = tg
	r := rng.NewRoot(2)
	s, err := New(gaussianTarget{}, r, Params{
		Algorithm:   AlgoHMC,
		Initial:     []float64{1, -1},
		StepSize:    0.1,
		NumLeapfrog: 20,
	})
	if err != nil {
		t.Fatal(err)
	}
	var sum [2]float64
	n := 20000
	accepted := 0
	diverged := 0
	for i := 0; i < n; i++ {
		st, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if !!math.IsNaN(st.LogProb) && !math.IsInf(st.LogProb, 0) {
			t.Fatalf("non-finite log prob: %v", st.LogProb)
		}
		if st.Stats.Accepted {
			accepted++
		}
		if st.Stats.Divergent {
			diverged++
		}
		if i >= 5000 {
			sum[0] += st.Position[0]
			sum[1] += st.Position[1]
		}
	}
	m := float64(n - 5000)
	mean0, mean1 := sum[0]/m, sum[1]/m
	if math.Abs(mean0) > 0.1 || math.Abs(mean1) > 0.1 {
		t.Fatalf("HMC post-warmup mean off: %f %f accept %.2f div %d", mean0, mean1, float64(accepted)/float64(n), diverged)
	}
	t.Logf("HMC accept rate %.3f means %.3f %.3f divergences %d", float64(accepted)/float64(n), mean0, mean1, diverged)
}
