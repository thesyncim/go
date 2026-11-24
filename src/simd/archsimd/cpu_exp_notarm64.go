// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && !arm64

package archsimd

// DotProd returns whether the CPU supports the Advanced SIMD dot product
// instructions (FEAT_DotProd), such as UDOT and SDOT.
//
// DotProd is defined on all GOARCHes, but will only return true on
// GOARCH arm64.
func (ARM64Features) DotProd() bool {
	return false
}

// I8MM returns whether the CPU supports the Advanced SIMD int8 matrix
// multiplication instructions (FEAT_I8MM), such as USDOT.
//
// I8MM is defined on all GOARCHes, but will only return true on
// GOARCH arm64.
func (ARM64Features) I8MM() bool {
	return false
}
