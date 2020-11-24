#include "dialect.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Transforms/InliningUtils.h>

using namespace mlir;
using namespace mlir::jsoniq;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
JSONiqDialect::JSONiqDialect(mlir::MLIRContext *ctx) : mlir::Dialect("jsoniq", ctx)
{
    addOperations <
#define GET_OP_LIST
#include "/home/manuel/6-Semester/Thesis/bsc-reberma/cpp/build/Ops.cpp.inc"
    > ();
    allowUnknownTypes();
    allowUnknownOperations();
    addTypes<SequenceType>();
}

mlir::Type JSONiqDialect::parseType(mlir::DialectAsmParser &parser) const
{
    auto const name_space = Identifier::get(getNamespace(), getContext());
    auto const type_name = parser.getFullSymbolSpec();
    if (type_name == "sequence")
    {
        return SequenceType::get(getContext());
    }

    return OpaqueType::get(name_space, type_name, getContext());
}



void JSONiqDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const
{

    SequenceType sequenceType = type.cast<SequenceType>();
    printer << "sequence";

    // Print the type according to the parser format.

}

#define GET_OP_CLASSES
#include "/home/manuel/6-Semester/Thesis/bsc-reberma/cpp/build/Ops.cpp.inc"