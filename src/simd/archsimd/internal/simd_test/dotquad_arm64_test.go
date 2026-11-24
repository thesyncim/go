// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

// addDotProductQuadrupleSlice is the reference for AddDotProductQuadruple and
// AddDotProductQuadrupleMixed.
func addDotProductQuadrupleSlice[A, Y, Z int32 | uint32 | int8 | uint8](x []A, y []Y, z []Z) []A {
	w := make([]A, len(x))
	for i := range w {
		sum := x[i]
		for j := 4 * i; j < 4*i+4; j++ {
			sum += A(y[j]) * A(z[j])
		}
		w[i] = sum
	}
	return w
}

func TestAddDotProductQuadrupleUint(t *testing.T) {
	if !archsimd.ARM64.DotProd() {
		t.Skip("DotProd not supported")
	}
	acc := []uint32{0, 1, 0xffffffff, 0x80000000}
	forSlicePair(t, uint8s, 16, func(y, z []uint8) bool {
		t.Helper()
		x := archsimd.LoadUint32x4(acc)
		got := make([]uint32, 4)
		x.AddDotProductQuadruple(archsimd.LoadUint8x16(y), archsimd.LoadUint8x16(z)).Store(got)
		return checkSlicesLogInput(t, got, addDotProductQuadrupleSlice(acc, y, z), 0.0, func() {
			t.Helper()
			t.Logf("y=%v", y)
			t.Logf("z=%v", z)
		})
	})
}

func TestAddDotProductQuadrupleInt(t *testing.T) {
	if !archsimd.ARM64.DotProd() {
		t.Skip("DotProd not supported")
	}
	acc := []int32{0, -1, 0x7fffffff, -0x80000000}
	forSlicePair(t, int8s, 16, func(y, z []int8) bool {
		t.Helper()
		x := archsimd.LoadInt32x4(acc)
		got := make([]int32, 4)
		x.AddDotProductQuadruple(archsimd.LoadInt8x16(y), archsimd.LoadInt8x16(z)).Store(got)
		return checkSlicesLogInput(t, got, addDotProductQuadrupleSlice(acc, y, z), 0.0, func() {
			t.Helper()
			t.Logf("y=%v", y)
			t.Logf("z=%v", z)
		})
	})
}

func TestAddDotProductQuadrupleMixed(t *testing.T) {
	if !archsimd.ARM64.I8MM() {
		t.Skip("I8MM not supported")
	}
	acc := []int32{0, -1, 0x7fffffff, -0x80000000}
	forSliceMixed(t, uint8s, int8s, 16, func(y []uint8, z []int8) bool {
		t.Helper()
		x := archsimd.LoadInt32x4(acc)
		got := make([]int32, 4)
		x.AddDotProductQuadrupleMixed(archsimd.LoadUint8x16(y), archsimd.LoadInt8x16(z)).Store(got)
		return checkSlicesLogInput(t, got, addDotProductQuadrupleSlice(acc, y, z), 0.0, func() {
			t.Helper()
			t.Logf("y=%v", y)
			t.Logf("z=%v", z)
		})
	})
}

// TestAddDotProductQuadrupleChain checks that a chain of accumulating dot
// products, as in a convolution, keeps the accumulator in place.
func TestAddDotProductQuadrupleChain(t *testing.T) {
	if !archsimd.ARM64.DotProd() {
		t.Skip("DotProd not supported")
	}
	y0, y1 := uint8s[:16], uint8s[16:32]
	z0, z1 := uint8s[3:19], uint8s[7:23]
	want := addDotProductQuadrupleSlice(addDotProductQuadrupleSlice([]uint32{1, 2, 3, 4}, y0, z0), y1, z1)
	got := make([]uint32, 4)
	archsimd.LoadUint32x4([]uint32{1, 2, 3, 4}).
		AddDotProductQuadruple(archsimd.LoadUint8x16(y0), archsimd.LoadUint8x16(z0)).
		AddDotProductQuadruple(archsimd.LoadUint8x16(y1), archsimd.LoadUint8x16(z1)).
		Store(got)
	checkSlices(t, got, want)
}
