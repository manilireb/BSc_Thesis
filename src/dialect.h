#ifndef MLIR_TUTORIAL_TOY_DIALECT_H_
#define MLIR_TUTORIAL_TOY_DIALECT_H_

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Function.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace mlir {
namespace jsoniq {

/// This is the definition of the jsoniq dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override some general behavior exposed via virtual
/// methods.
class JSONiqDialect : public mlir::Dialect {
public:
  explicit JSONiqDialect(mlir::MLIRContext *ctx);


  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static llvm::StringRef getDialectNamespace() { return "jsoniq"; }

  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;

};

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "/home/manuel/6-Semester/Thesis/bsc-reberma/cpp/build/Ops.h.inc"


//===----------------------------------------------------------------------===//
// JSOniq Types
//===----------------------------------------------------------------------===//

namespace JSONiqTypes {
enum Types{
	sequence = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
};
} // end namespace JSONiqTypes


class SequenceType : public Type::TypeBase<SequenceType, Type>{
public:
	/// Inherit some necessary constructors from 'TypeBase'
	using Base::Base;

	/// This static method is used to support type inquiry through isa, cast,
	/// and dyn_cast.
	static bool kindof(unsigned kind) { return kind == JSONiqTypes::sequence; }

	/// This method is used to get an instance of the 'SequenceType'. Given that this
	/// is a parameterless type, it just needs to take the context for
	// uniquing purposes.
	static SequenceType get(MLIRContext *context){

		// Call into a helper 'get' method in 'TypeBase' to get a uniqued instance 
		// of this type.
		return Base::get(context, JSONiqTypes::sequence);
	}


};





} // end namespace jsoniq
} // end namespace mlir

#endif 