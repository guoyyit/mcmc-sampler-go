package service

import (
	"context"
	"testing"

	"github.com/guoyyit/mcmc-sampler-go/adapt"
	"github.com/guoyyit/mcmc-sampler-go/diagnostics"
	"github.com/guoyyit/mcmc-sampler-go/history"
	"github.com/guoyyit/mcmc-sampler-go/sampler"
)

// anisotropic is a diagonal Gaussian with variances 1 and 9, where mass
// matrix adaptation demonstrably matters.
type anisotropic struct{}

func (anisotropic) Dim() int { return 2 }
func (anisotropic) LogDensity(x []float64) float64 {
	return -0.5 * (x[0]*x[0] + x[1]*x[1]/9)
}
func (anisotropic) Gradient(x, g []float64) {
	g[0] = -x[0]
	g[1] = -x[1] / 9
}

func TestHMCEndToEndAdaptationAndReproducibility(t *testing.T) {
	cfg := Config{
		JobID:     "hmc",
		NumChains: 4,
		Draws:     1500,
		Warmup:    500,
		Seed:      888,
		Sampler: sampler.Params{
			Algorithm:   sampler.AlgoHMC,
			Initial:     []float64{1, 1},
			InitJitter:  []float64{1, 3},
			StepSize:    0.01, // deliberately small; adaptation must grow it
			NumLeapfrog: 25,
		},
		Adapt: adapt.Config{
			Warmup:          500,
			TargetAcceptMin: 0.7,
			TargetAcceptMax: 0.9,
			InitStepSize:    0.01,
			AdaptMass:       true,
		},
		History:         history.Policy{},
		Thresholds:      diagnostics.Thresholds{MaxRhat: 1.05, MinESS: 50},
		CheckpointEvery: 0,
	}
	r1, err := Run(context.Background(), cfg, anisotropic{})
	if err != nil {
		t.Fatalf("HMC run failed: %v", err)
	}
	r2, err := Run(context.Background(), cfg, anisotropic{})
	if err != nil {
		t.Fatal(err)
	}
	for c := range r1.Draws {
		if len(r1.Draws[c]) != len(r2.Draws[c]) {
			t.Fatalf("chain %d draw count differs", c)
		}
		for i := range r1.Draws[c] {
			for d := 0; d < 2; d++ {
				if r1.Draws[c][i][d] != r2.Draws[c][i][d] {
					t.Fatalf("HMC not reproducible at chain %d draw %d dim %d", c, i, d)
				}
			}
		}
	}

	// Empirical variance of the second coordinate should approach 9.
	var sum, sumSq float64
	var n float64
	for _, chain := range r1.Draws {
		for _, x := range chain {
			sum += x[1]
			sumSq += x[1] * x[1]
			n++
		}
	}
	mean := sum / n
	variance := sumSq/n - mean*mean
	if variance < 5 || variance > 15 {
		t.Fatalf("HMC variance of coord 1 = %.2f, want ~9", variance)
	}
}
