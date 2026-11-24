// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"math"
	"simd/archsimd"
	"testing"
	"unsafe"
)

// shiftRightRoundedRef returns (x + 1<<(s-1)) >> s computed exactly.
func shiftRightRoundedRef[E integer](x E, s uint64) E {
	if s == 0 {
		return x
	}
	if s > uint64(unsafe.Sizeof(x))*8 {
		return 0
	}
	h := x >> (s - 1)
	return h>>1 + h&1
}

// scaleRoundedRef is the reference for ScaleRounded with a shift of s.
func scaleRoundedRef[E integer](x E, s int8) E {
	if s >= 0 {
		return x << uint(s)
	}
	return shiftRightRoundedRef(x, uint64(-int(s)))
}

// shifts is a variable so that shifts by its elements aren't constants.
var shifts = []uint64{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 1000}

func TestShiftAllRightRounded(t *testing.T) {
	testShiftAllRightRounded(t, int8s, 16, archsimd.LoadInt8x16, archsimd.Int8x16.Store, archsimd.Int8x16.ShiftAllRightRounded)
	testShiftAllRightRounded(t, int16s, 8, archsimd.LoadInt16x8, archsimd.Int16x8.Store, archsimd.Int16x8.ShiftAllRightRounded)
	testShiftAllRightRounded(t, int32s, 4, archsimd.LoadInt32x4, archsimd.Int32x4.Store, archsimd.Int32x4.ShiftAllRightRounded)
	testShiftAllRightRounded(t, int64s, 2, archsimd.LoadInt64x2, archsimd.Int64x2.Store, archsimd.Int64x2.ShiftAllRightRounded)
	testShiftAllRightRounded(t, uint8s, 16, archsimd.LoadUint8x16, archsimd.Uint8x16.Store, archsimd.Uint8x16.ShiftAllRightRounded)
	testShiftAllRightRounded(t, uint16s, 8, archsimd.LoadUint16x8, archsimd.Uint16x8.Store, archsimd.Uint16x8.ShiftAllRightRounded)
	testShiftAllRightRounded(t, uint32s, 4, archsimd.LoadUint32x4, archsimd.Uint32x4.Store, archsimd.Uint32x4.ShiftAllRightRounded)
	testShiftAllRightRounded(t, uint64s, 2, archsimd.LoadUint64x2, archsimd.Uint64x2.Store, archsimd.Uint64x2.ShiftAllRightRounded)

	// Constant shifts use the immediate form.
	got := make([]uint16, 8)
	x := archsimd.LoadUint16x8([]uint16{0, 1, 2, 3, 0x7fff, 0x8000, 0xfffe, 0xffff})
	x.ShiftAllRightRounded(1).Store(got)
	checkSlices(t, got, []uint16{0, 1, 1, 2, 0x4000, 0x4000, 0x7fff, 0x8000})
	x.ShiftAllRightRounded(16).Store(got)
	checkSlices(t, got, []uint16{0, 0, 0, 0, 0, 1, 1, 1})
	x.ShiftAllRightRounded(17).Store(got)
	checkSlices(t, got, []uint16{0, 0, 0, 0, 0, 0, 0, 0})
	x.ShiftAllRightRounded(0).Store(got)
	checkSlices(t, got, []uint16{0, 1, 2, 3, 0x7fff, 0x8000, 0xfffe, 0xffff})
	goti := make([]int16, 8)
	y := archsimd.LoadInt16x8([]int16{-1, -2, -3, -4, math.MinInt16, math.MaxInt16, 5, -5})
	y.ShiftAllRightRounded(1).Store(goti)
	checkSlices(t, goti, []int16{0, -1, -1, -2, -16384, 16384, 3, -2})
	y.ShiftAllRightRounded(16).Store(goti)
	checkSlices(t, goti, []int16{0, 0, 0, 0, 0, 0, 0, 0})
}

func testShiftAllRightRounded[E integer, V any](t *testing.T, s []E, n int, load func([]E) V, store func(V, []E), f func(V, uint64) V) {
	t.Helper()
	forSlice(t, s, n, func(x []E) bool {
		t.Helper()
		for _, sh := range shifts {
			want := make([]E, n)
			for i := range want {
				want[i] = shiftRightRoundedRef(x[i], sh)
			}
			got := make([]E, n)
			store(f(load(x), sh), got)
			if !checkSlicesLogInput(t, got, want, 0.0, func() { t.Helper(); t.Logf("x=%v shift=%d", x, sh) }) {
				return false
			}
		}
		return true
	})
}

func TestScaleRounded(t *testing.T) {
	testScaleRounded(t, int16s, 8, archsimd.LoadInt16x8, archsimd.LoadInt16x8, archsimd.Int16x8.Store, archsimd.Int16x8.ScaleRounded)
	testScaleRounded(t, uint16s, 8, archsimd.LoadUint16x8, archsimd.LoadInt16x8, archsimd.Uint16x8.Store, archsimd.Uint16x8.ScaleRounded)
	testScaleRounded(t, int32s, 4, archsimd.LoadInt32x4, archsimd.LoadInt32x4, archsimd.Int32x4.Store, archsimd.Int32x4.ScaleRounded)
	testScaleRounded(t, uint32s, 4, archsimd.LoadUint32x4, archsimd.LoadInt32x4, archsimd.Uint32x4.Store, archsimd.Uint32x4.ScaleRounded)
}

func testScaleRounded[E integer, S int16 | int32, V, VS any](t *testing.T, s []E, n int, load func([]E) V, loadS func([]S) VS, store func(V, []E), f func(V, VS) V) {
	t.Helper()
	nbits := int(unsafe.Sizeof(s[0])) * 8
	var amounts []S
	for a := -nbits - 1; a < nbits; a++ {
		amounts = append(amounts, S(a))
	}
	forSlice(t, s, n, func(x []E) bool {
		t.Helper()
		for i := 0; i+n <= len(amounts); i++ {
			sh := amounts[i : i+n]
			want := make([]E, n)
			for j := range want {
				want[j] = scaleRoundedRef(x[j], int8(sh[j]))
			}
			got := make([]E, n)
			store(f(load(x), loadS(sh)), got)
			if !checkSlicesLogInput(t, got, want, 0.0, func() { t.Helper(); t.Logf("x=%v shift=%v", x, sh) }) {
				return false
			}
		}
		return true
	})
}

// TestShiftRightNarrow tests the narrowing conversions of right shifts by a
// constant, which compile to single narrowing shift instructions.
func TestShiftRightNarrow(t *testing.T) {
	sat := func(v, lo, hi int64) int64 { return min(max(v, lo), hi) }
	forSlice(t, int16s, 8, func(x []int16) bool {
		t.Helper()
		v := archsimd.LoadInt16x8(x)
		for _, tc := range []struct {
			name string
			got  archsimd.Int8x16
			want func(int16) int64
		}{
			{"SQSHRN", v.ShiftAllRight(3).SaturateToInt8(), func(e int16) int64 { return sat(int64(e>>3), -128, 127) }},
			{"SQRSHRN", v.ShiftAllRightRounded(3).SaturateToInt8(), func(e int16) int64 { return sat(int64(shiftRightRoundedRef(e, 3)), -128, 127) }},
			{"RSHRN", v.ShiftAllRightRounded(5).TruncToInt8(), func(e int16) int64 { return int64(int8(shiftRightRoundedRef(e, 5))) }},
		} {
			got := make([]int8, 16)
			tc.got.Store(got)
			for i := range 8 {
				if int64(got[i]) != tc.want(x[i]) {
					t.Errorf("%s: x=%v got %v want %v at %d", tc.name, x, got[:8], tc.want(x[i]), i)
					return false
				}
			}
		}
		for _, tc := range []struct {
			name string
			got  archsimd.Uint8x16
			want func(int16) int64
		}{
			{"SQSHRUN", v.ShiftAllRight(2).SaturateToUint8(), func(e int16) int64 { return sat(int64(e>>2), 0, 255) }},
			{"SQRSHRUN", v.ShiftAllRightRounded(4).SaturateToUint8(), func(e int16) int64 { return sat(int64(shiftRightRoundedRef(e, 4)), 0, 255) }},
		} {
			got := make([]uint8, 16)
			tc.got.Store(got)
			for i := range 8 {
				if int64(got[i]) != tc.want(x[i]) {
					t.Errorf("%s: x=%v got %v want %v at %d", tc.name, x, got[:8], tc.want(x[i]), i)
					return false
				}
			}
		}
		return true
	})
	forSlice(t, uint16s, 8, func(x []uint16) bool {
		t.Helper()
		v := archsimd.LoadUint16x8(x)
		for _, tc := range []struct {
			name string
			got  archsimd.Uint8x16
			want func(uint16) int64
		}{
			{"UQSHRN", v.ShiftAllRight(1).SaturateToUint8(), func(e uint16) int64 { return sat(int64(e>>1), 0, 255) }},
			{"UQRSHRN", v.ShiftAllRightRounded(6).SaturateToUint8(), func(e uint16) int64 { return sat(int64(shiftRightRoundedRef(e, 6)), 0, 255) }},
			{"RSHRN", v.ShiftAllRightRounded(8).TruncToUint8(), func(e uint16) int64 { return int64(uint8(shiftRightRoundedRef(e, 8))) }},
		} {
			got := make([]uint8, 16)
			tc.got.Store(got)
			for i := range 8 {
				if int64(got[i]) != tc.want(x[i]) {
					t.Errorf("%s: x=%v got %v want %v at %d", tc.name, x, got[:8], tc.want(x[i]), i)
					return false
				}
			}
		}
		return true
	})
}
