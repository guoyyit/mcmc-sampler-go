package target

import (
	"errors"
	"math"
	"testing"
)

func TestNewRejectsBadInput(t *testing.T) {
	if _, err := New(0, func(x []float64) float64 { return 0 }); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("dim 0: %v", err)
	}
	if _, err := New(-1, func(x []float64) float64 { return 0 }); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("dim -1: %v", err)
	}
	if _, err := New(2, nil); !errors.Is(err, ErrNilFunc) {
		t.Fatalf("nil func: %v", err)
	}
	tg, err := New(2, func(x []float64) float64 { return -x[0] * x[0] })
	if err != nil {
		t.Fatal(err)
	}
	if tg.Dim() != 2 {
		t.Fatalf("dim %d", tg.Dim())
	}
}

func TestValidatePosition(t *testing.T) {
	if err := ValidatePosition(2, []float64{1, 2}); err != nil {
		t.Fatalf("valid position rejected: %v", err)
	}
	if !errors.Is(ValidatePosition(2, []float64{1}), ErrWrongDimension) {
		t.Fatal("wrong dimension not rejected")
	}
	if !errors.Is(ValidatePosition(2, []float64{1, math.NaN()}), ErrNonFinitePosition) {
		t.Fatal("NaN position not rejected")
	}
	if !errors.Is(ValidatePosition(2, []float64{math.Inf(1), 1}), ErrNonFinitePosition) {
		t.Fatal("+Inf position not rejected")
	}
	if !errors.Is(ValidatePosition(2, []float64{math.Inf(-1), 1}), ErrNonFinitePosition) {
		t.Fatal("-Inf position not rejected")
	}
}

func TestValidateLogProb(t *testing.T) {
	if err := ValidateLogProb(0); err != nil {
		t.Fatalf("finite log prob rejected: %v", err)
	}
	if err := ValidateLogProb(math.Inf(-1)); err != nil {
		t.Fatalf("-Inf log prob must be legal: %v", err)
	}
	if err := ValidateLogProb(math.NaN()); !errors.Is(err, ErrNonFiniteLogProb) {
		t.Fatalf("NaN log prob: %v", err)
	}
	if err := ValidateLogProb(math.Inf(1)); !errors.Is(err, ErrNonFiniteLogProb) {
		t.Fatalf("+Inf log prob: %v", err)
	}
}
