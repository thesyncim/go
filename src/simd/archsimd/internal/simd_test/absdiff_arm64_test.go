// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

// absDiffRef returns |x - y| as an unsigned Z.
func absDiffRef[X integer, Z integer](x, y X) Z {
	if x > y {
		return Z(x) - Z(y)
	}
	return Z(y) - Z(x)
}

// TestAbsDiff tests x.AbsDiff(y) and its fused accumulating form,
// acc.Add(x.AbsDiff(y)), which compiles to a single UABA or SABA.
func TestAbsDiff(t *testing.T) {
	testAbsDiff(t, int8s, 16, archsimd.LoadInt8x16, archsimd.LoadUint8x16, archsimd.Uint8x16.Store,
		archsimd.Int8x16.AbsDiff,
		func(a archsimd.Uint8x16, x, y archsimd.Int8x16) archsimd.Uint8x16 { return a.Add(x.AbsDiff(y)) })
	testAbsDiff(t, int16s, 8, archsimd.LoadInt16x8, archsimd.LoadUint16x8, archsimd.Uint16x8.Store,
		archsimd.Int16x8.AbsDiff,
		func(a archsimd.Uint16x8, x, y archsimd.Int16x8) archsimd.Uint16x8 { return a.Add(x.AbsDiff(y)) })
	testAbsDiff(t, int32s, 4, archsimd.LoadInt32x4, archsimd.LoadUint32x4, archsimd.Uint32x4.Store,
		archsimd.Int32x4.AbsDiff,
		func(a archsimd.Uint32x4, x, y archsimd.Int32x4) archsimd.Uint32x4 { return a.Add(x.AbsDiff(y)) })
	testAbsDiff(t, uint8s, 16, archsimd.LoadUint8x16, archsimd.LoadUint8x16, archsimd.Uint8x16.Store,
		archsimd.Uint8x16.AbsDiff,
		func(a, x, y archsimd.Uint8x16) archsimd.Uint8x16 { return a.Add(x.AbsDiff(y)) })
	testAbsDiff(t, uint16s, 8, archsimd.LoadUint16x8, archsimd.LoadUint16x8, archsimd.Uint16x8.Store,
		archsimd.Uint16x8.AbsDiff,
		func(a, x, y archsimd.Uint16x8) archsimd.Uint16x8 { return a.Add(x.AbsDiff(y)) })
	testAbsDiff(t, uint32s, 4, archsimd.LoadUint32x4, archsimd.LoadUint32x4, archsimd.Uint32x4.Store,
		archsimd.Uint32x4.AbsDiff,
		func(a, x, y archsimd.Uint32x4) archsimd.Uint32x4 { return a.Add(x.AbsDiff(y)) })
}

func testAbsDiff[X, Z integer, VX, VZ any](t *testing.T, s []X, n int,
	load func([]X) VX, loadZ func([]Z) VZ, store func(VZ, []Z),
	absDiff func(VX, VX) VZ, absDiffAcc func(VZ, VX, VX) VZ) {
	t.Helper()
	acc := make([]Z, n)
	for i := range acc {
		acc[i] = Z(i*37 + 250)
	}
	forSlicePair(t, s, n, func(x, y []X) bool {
		t.Helper()
		logInput := func() { t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y) }
		want := make([]Z, n)
		wantAcc := make([]Z, n)
		for i := range want {
			want[i] = absDiffRef[X, Z](x[i], y[i])
			wantAcc[i] = acc[i] + want[i]
		}
		got := make([]Z, n)
		store(absDiff(load(x), load(y)), got)
		if !checkSlicesLogInput(t, got, want, 0.0, logInput) {
			return false
		}
		store(absDiffAcc(loadZ(acc), load(x), load(y)), got)
		return checkSlicesLogInput(t, got, wantAcc, 0.0, logInput)
	})
}

// TestAbsDiffWiden tests x.AbsDiffWiden(y), x.HiToLo().AbsDiffWiden(y.HiToLo()),
// and their fused accumulating forms, which compile to single UABDL, UABDL2,
// UABAL and UABAL2 instructions (or their signed counterparts).
func TestAbsDiffWiden(t *testing.T) {
	testAbsDiffWiden(t, int8s, 16, archsimd.LoadInt8x16, archsimd.LoadUint16x8, archsimd.Uint16x8.Store,
		archsimd.Int8x16.AbsDiffWiden,
		func(x, y archsimd.Int8x16) archsimd.Uint16x8 { return x.HiToLo().AbsDiffWiden(y.HiToLo()) },
		func(a archsimd.Uint16x8, x, y archsimd.Int8x16) archsimd.Uint16x8 { return a.Add(x.AbsDiffWiden(y)) },
		func(a archsimd.Uint16x8, x, y archsimd.Int8x16) archsimd.Uint16x8 {
			return a.Add(x.HiToLo().AbsDiffWiden(y.HiToLo()))
		})
	testAbsDiffWiden(t, int16s, 8, archsimd.LoadInt16x8, archsimd.LoadUint32x4, archsimd.Uint32x4.Store,
		archsimd.Int16x8.AbsDiffWiden,
		func(x, y archsimd.Int16x8) archsimd.Uint32x4 { return x.HiToLo().AbsDiffWiden(y.HiToLo()) },
		func(a archsimd.Uint32x4, x, y archsimd.Int16x8) archsimd.Uint32x4 { return a.Add(x.AbsDiffWiden(y)) },
		func(a archsimd.Uint32x4, x, y archsimd.Int16x8) archsimd.Uint32x4 {
			return a.Add(x.HiToLo().AbsDiffWiden(y.HiToLo()))
		})
	testAbsDiffWiden(t, int32s, 4, archsimd.LoadInt32x4, archsimd.LoadUint64x2, archsimd.Uint64x2.Store,
		archsimd.Int32x4.AbsDiffWiden,
		func(x, y archsimd.Int32x4) archsimd.Uint64x2 { return x.HiToLo().AbsDiffWiden(y.HiToLo()) },
		func(a archsimd.Uint64x2, x, y archsimd.Int32x4) archsimd.Uint64x2 { return a.Add(x.AbsDiffWiden(y)) },
		func(a archsimd.Uint64x2, x, y archsimd.Int32x4) archsimd.Uint64x2 {
			return a.Add(x.HiToLo().AbsDiffWiden(y.HiToLo()))
		})
	testAbsDiffWiden(t, uint8s, 16, archsimd.LoadUint8x16, archsimd.LoadUint16x8, archsimd.Uint16x8.Store,
		archsimd.Uint8x16.AbsDiffWiden,
		func(x, y archsimd.Uint8x16) archsimd.Uint16x8 { return x.HiToLo().AbsDiffWiden(y.HiToLo()) },
		func(a archsimd.Uint16x8, x, y archsimd.Uint8x16) archsimd.Uint16x8 { return a.Add(x.AbsDiffWiden(y)) },
		func(a archsimd.Uint16x8, x, y archsimd.Uint8x16) archsimd.Uint16x8 {
			return a.Add(x.HiToLo().AbsDiffWiden(y.HiToLo()))
		})
	testAbsDiffWiden(t, uint16s, 8, archsimd.LoadUint16x8, archsimd.LoadUint32x4, archsimd.Uint32x4.Store,
		archsimd.Uint16x8.AbsDiffWiden,
		func(x, y archsimd.Uint16x8) archsimd.Uint32x4 { return x.HiToLo().AbsDiffWiden(y.HiToLo()) },
		func(a archsimd.Uint32x4, x, y archsimd.Uint16x8) archsimd.Uint32x4 { return a.Add(x.AbsDiffWiden(y)) },
		func(a archsimd.Uint32x4, x, y archsimd.Uint16x8) archsimd.Uint32x4 {
			return a.Add(x.HiToLo().AbsDiffWiden(y.HiToLo()))
		})
	testAbsDiffWiden(t, uint32s, 4, archsimd.LoadUint32x4, archsimd.LoadUint64x2, archsimd.Uint64x2.Store,
		archsimd.Uint32x4.AbsDiffWiden,
		func(x, y archsimd.Uint32x4) archsimd.Uint64x2 { return x.HiToLo().AbsDiffWiden(y.HiToLo()) },
		func(a archsimd.Uint64x2, x, y archsimd.Uint32x4) archsimd.Uint64x2 { return a.Add(x.AbsDiffWiden(y)) },
		func(a archsimd.Uint64x2, x, y archsimd.Uint32x4) archsimd.Uint64x2 {
			return a.Add(x.HiToLo().AbsDiffWiden(y.HiToLo()))
		})
}

func testAbsDiffWiden[X, Z integer, VX, VZ any](t *testing.T, s []X, n int,
	load func([]X) VX, loadZ func([]Z) VZ, store func(VZ, []Z),
	absDiff, absDiffHi func(VX, VX) VZ, absDiffAcc, absDiffAccHi func(VZ, VX, VX) VZ) {
	t.Helper()
	acc := make([]Z, n/2)
	for i := range acc {
		acc[i] = ^Z(i * 1000)
	}
	forSlicePair(t, s, n, func(x, y []X) bool {
		t.Helper()
		logInput := func() { t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y) }
		for _, hi := range []bool{false, true} {
			off := 0
			f, fAcc := absDiff, absDiffAcc
			if hi {
				off = n / 2
				f, fAcc = absDiffHi, absDiffAccHi
			}
			want := make([]Z, n/2)
			wantAcc := make([]Z, n/2)
			for i := range want {
				want[i] = absDiffRef[X, Z](x[off+i], y[off+i])
				wantAcc[i] = acc[i] + want[i]
			}
			got := make([]Z, n/2)
			store(f(load(x), load(y)), got)
			if !checkSlicesLogInput(t, got, want, 0.0, logInput) {
				return false
			}
			store(fAcc(loadZ(acc), load(x), load(y)), got)
			if !checkSlicesLogInput(t, got, wantAcc, 0.0, logInput) {
				return false
			}
		}
		return true
	})
}

// TestAddPairsWiden tests x.AddPairsWiden() and its fused accumulating form,
// acc.Add(x.AddPairsWiden()), which compiles to a single UADALP or SADALP.
func TestAddPairsWiden(t *testing.T) {
	testAddPairsWiden(t, int8s, 16, archsimd.LoadInt8x16, archsimd.LoadInt16x8, archsimd.Int16x8.Store,
		archsimd.Int8x16.AddPairsWiden,
		func(a archsimd.Int16x8, x archsimd.Int8x16) archsimd.Int16x8 { return a.Add(x.AddPairsWiden()) })
	testAddPairsWiden(t, int16s, 8, archsimd.LoadInt16x8, archsimd.LoadInt32x4, archsimd.Int32x4.Store,
		archsimd.Int16x8.AddPairsWiden,
		func(a archsimd.Int32x4, x archsimd.Int16x8) archsimd.Int32x4 { return a.Add(x.AddPairsWiden()) })
	testAddPairsWiden(t, int32s, 4, archsimd.LoadInt32x4, archsimd.LoadInt64x2, archsimd.Int64x2.Store,
		archsimd.Int32x4.AddPairsWiden,
		func(a archsimd.Int64x2, x archsimd.Int32x4) archsimd.Int64x2 { return a.Add(x.AddPairsWiden()) })
	testAddPairsWiden(t, uint8s, 16, archsimd.LoadUint8x16, archsimd.LoadUint16x8, archsimd.Uint16x8.Store,
		archsimd.Uint8x16.AddPairsWiden,
		func(a archsimd.Uint16x8, x archsimd.Uint8x16) archsimd.Uint16x8 { return a.Add(x.AddPairsWiden()) })
	testAddPairsWiden(t, uint16s, 8, archsimd.LoadUint16x8, archsimd.LoadUint32x4, archsimd.Uint32x4.Store,
		archsimd.Uint16x8.AddPairsWiden,
		func(a archsimd.Uint32x4, x archsimd.Uint16x8) archsimd.Uint32x4 { return a.Add(x.AddPairsWiden()) })
	testAddPairsWiden(t, uint32s, 4, archsimd.LoadUint32x4, archsimd.LoadUint64x2, archsimd.Uint64x2.Store,
		archsimd.Uint32x4.AddPairsWiden,
		func(a archsimd.Uint64x2, x archsimd.Uint32x4) archsimd.Uint64x2 { return a.Add(x.AddPairsWiden()) })
}

func testAddPairsWiden[X, Z integer, VX, VZ any](t *testing.T, s []X, n int,
	load func([]X) VX, loadZ func([]Z) VZ, store func(VZ, []Z),
	addPairs func(VX) VZ, addPairsAcc func(VZ, VX) VZ) {
	t.Helper()
	acc := make([]Z, n/2)
	for i := range acc {
		acc[i] = Z(i*7 - 3)
	}
	forSlice(t, s, n, func(x []X) bool {
		t.Helper()
		logInput := func() { t.Helper(); t.Logf("x=%v", x) }
		want := make([]Z, n/2)
		wantAcc := make([]Z, n/2)
		for i := range want {
			want[i] = Z(x[2*i]) + Z(x[2*i+1])
			wantAcc[i] = acc[i] + want[i]
		}
		got := make([]Z, n/2)
		store(addPairs(load(x)), got)
		if !checkSlicesLogInput(t, got, want, 0.0, logInput) {
			return false
		}
		store(addPairsAcc(loadZ(acc), load(x)), got)
		return checkSlicesLogInput(t, got, wantAcc, 0.0, logInput)
	})
}
