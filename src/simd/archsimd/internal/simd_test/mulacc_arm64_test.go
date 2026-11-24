// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"math"
	"simd/archsimd"
	"testing"
)

// TestMulWidenAccumulate tests acc.Add(x.MulWiden(y)) and
// acc.Add(x.HiToLo().MulWiden(y.HiToLo())), which compile to single SMLAL,
// SMLAL2, UMLAL and UMLAL2 instructions.
func TestMulWidenAccumulate(t *testing.T) {
	testMulWidenAccumulate(t, int8s, 16, archsimd.LoadInt8x16, archsimd.LoadInt16x8, archsimd.Int16x8.Store,
		func(a archsimd.Int16x8, x, y archsimd.Int8x16) archsimd.Int16x8 { return a.Add(x.MulWiden(y)) },
		func(a archsimd.Int16x8, x, y archsimd.Int8x16) archsimd.Int16x8 { return a.Add(x.HiToLo().MulWiden(y.HiToLo())) })
	testMulWidenAccumulate(t, int16s, 8, archsimd.LoadInt16x8, archsimd.LoadInt32x4, archsimd.Int32x4.Store,
		func(a archsimd.Int32x4, x, y archsimd.Int16x8) archsimd.Int32x4 { return a.Add(x.MulWiden(y)) },
		func(a archsimd.Int32x4, x, y archsimd.Int16x8) archsimd.Int32x4 { return a.Add(x.HiToLo().MulWiden(y.HiToLo())) })
	testMulWidenAccumulate(t, int32s, 4, archsimd.LoadInt32x4, archsimd.LoadInt64x2, archsimd.Int64x2.Store,
		func(a archsimd.Int64x2, x, y archsimd.Int32x4) archsimd.Int64x2 { return a.Add(x.MulWiden(y)) },
		func(a archsimd.Int64x2, x, y archsimd.Int32x4) archsimd.Int64x2 { return a.Add(x.HiToLo().MulWiden(y.HiToLo())) })
	testMulWidenAccumulate(t, uint8s, 16, archsimd.LoadUint8x16, archsimd.LoadUint16x8, archsimd.Uint16x8.Store,
		func(a archsimd.Uint16x8, x, y archsimd.Uint8x16) archsimd.Uint16x8 { return a.Add(x.MulWiden(y)) },
		func(a archsimd.Uint16x8, x, y archsimd.Uint8x16) archsimd.Uint16x8 { return a.Add(x.HiToLo().MulWiden(y.HiToLo())) })
	testMulWidenAccumulate(t, uint16s, 8, archsimd.LoadUint16x8, archsimd.LoadUint32x4, archsimd.Uint32x4.Store,
		func(a archsimd.Uint32x4, x, y archsimd.Uint16x8) archsimd.Uint32x4 { return a.Add(x.MulWiden(y)) },
		func(a archsimd.Uint32x4, x, y archsimd.Uint16x8) archsimd.Uint32x4 { return a.Add(x.HiToLo().MulWiden(y.HiToLo())) })
	testMulWidenAccumulate(t, uint32s, 4, archsimd.LoadUint32x4, archsimd.LoadUint64x2, archsimd.Uint64x2.Store,
		func(a archsimd.Uint64x2, x, y archsimd.Uint32x4) archsimd.Uint64x2 { return a.Add(x.MulWiden(y)) },
		func(a archsimd.Uint64x2, x, y archsimd.Uint32x4) archsimd.Uint64x2 { return a.Add(x.HiToLo().MulWiden(y.HiToLo())) })
}

func testMulWidenAccumulate[X, Z integer, VX, VZ any](t *testing.T, s []X, n int,
	load func([]X) VX, loadZ func([]Z) VZ, store func(VZ, []Z), mulAcc, mulAccHi func(VZ, VX, VX) VZ) {
	t.Helper()
	acc := make([]Z, n/2)
	for i := range acc {
		acc[i] = Z(i*12345 + 7)
	}
	forSlicePair(t, s, n, func(x, y []X) bool {
		t.Helper()
		logInput := func() { t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y) }
		for _, off := range []int{0, n / 2} {
			f := mulAcc
			if off != 0 {
				f = mulAccHi
			}
			want := make([]Z, n/2)
			for i := range want {
				want[i] = acc[i] + Z(x[off+i])*Z(y[off+i])
			}
			got := make([]Z, n/2)
			store(f(loadZ(acc), load(x), load(y)), got)
			if !checkSlicesLogInput(t, got, want, 0.0, logInput) {
				return false
			}
		}
		return true
	})
}

func mulFixedPointRoundedRef[E int16 | int32](x, y E, n uint) E {
	p := (int64(x)*int64(y) + 1<<(n-2)) >> (n - 1)
	maxv := int64(1)<<(n-1) - 1
	if p > maxv {
		p = maxv
	}
	return E(p)
}

func TestMulFixedPointRounded(t *testing.T) {
	forSlicePair(t, int16s, 8, func(x, y []int16) bool {
		t.Helper()
		got := make([]int16, 8)
		archsimd.LoadInt16x8(x).MulFixedPointRounded(archsimd.LoadInt16x8(y)).Store(got)
		want := make([]int16, 8)
		for i := range want {
			want[i] = mulFixedPointRoundedRef(x[i], y[i], 16)
		}
		return checkSlicesLogInput(t, got, want, 0.0, func() { t.Helper(); t.Logf("x=%v y=%v", x, y) })
	})
	forSlicePair(t, int32s, 4, func(x, y []int32) bool {
		t.Helper()
		got := make([]int32, 4)
		archsimd.LoadInt32x4(x).MulFixedPointRounded(archsimd.LoadInt32x4(y)).Store(got)
		want := make([]int32, 4)
		for i := range want {
			want[i] = mulFixedPointRoundedRef(x[i], y[i], 32)
		}
		return checkSlicesLogInput(t, got, want, 0.0, func() { t.Helper(); t.Logf("x=%v y=%v", x, y) })
	})

	// Saturation and rounding edge cases.
	got := make([]int16, 8)
	archsimd.LoadInt16x8([]int16{math.MinInt16, math.MinInt16, math.MaxInt16, 1 << 14, -1 << 14, 1, -1, 3}).
		MulFixedPointRounded(archsimd.LoadInt16x8([]int16{math.MinInt16, math.MaxInt16, math.MaxInt16, 1, 1, 1 << 14, 1 << 14, 1 << 14})).
		Store(got)
	checkSlices(t, got, []int16{math.MaxInt16, -32767, 32766, 1, 0, 1, 0, 2})
	got32 := make([]int32, 4)
	archsimd.LoadInt32x4([]int32{math.MinInt32, math.MinInt32, 1 << 30, -1 << 30}).
		MulFixedPointRounded(archsimd.LoadInt32x4([]int32{math.MinInt32, math.MaxInt32, 1, 1})).
		Store(got32)
	checkSlices(t, got32, []int32{math.MaxInt32, -math.MaxInt32, 1, 0})
}
