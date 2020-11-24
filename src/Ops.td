//===- Ops.td - JSONiq dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the JSONiq dialect.
//
//===----------------------------------------------------------------------===//

#ifndef JSONIQ_OPS
#define JSONIQ_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'jsoniq' dialect in the ODS framework so that we
// can define our operations.
def JSONiq_Dialect : Dialect {
  let name = "jsoniq";
  let cppNamespace = "jsoniq";
}

// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class JSONiq_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<JSONiq_Dialect, mnemonic, traits>;

    // Provide a definition for the JSONiq SequenceType for use in ODS. This allows for 
    // using SequenceType in a similiar way to Tensor or MemRef.
    def JSONiq_SequenceType : 
    	Type<CPred<"$_self.isa<SequenceType>()">, "jsoniq Sequence Type ">;

// TODO : define the operations for jsoniq

def UnaryExpr : JSONiq_Op<"neg", [NoSideEffect]> {
	let summary = "Unary Expression";
	let arguments = (ins JSONiq_SequenceType);
    let results = (outs JSONiq_SequenceType);


	let hasCanonicalizer = 1;
}

def LitExpr : JSONiq_Op<"lit", [NoSideEffect]> {
	let summary = "Literal Expression";
	let arguments = (ins AnyAttr:$value);
    let results = (outs JSONiq_SequenceType);


}

def ArrayConstExpr : JSONiq_Op<"arrayconstructor", [NoSideEffect]> {
	let summary = "ArrayConstructor Expression";
	let arguments = (ins JSONiq_SequenceType);
    let results = (outs JSONiq_SequenceType);


	let hasCanonicalizer = 1;
}


def ArrayUnbExpr : JSONiq_Op<"arrayunboxing", [NoSideEffect]> {
	let summary = "Array Unboxing Expression";
	let arguments = (ins JSONiq_SequenceType);
    let results = (outs JSONiq_SequenceType);


	let hasCanonicalizer = 1;
}


#endif  // JSONIQ_OPS