#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>

#include "dialect.h"

using namespace mlir::jsoniq;
using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "/home/manuel/6-Semester/Thesis/bsc-reberma/cpp/build/Combine.inc"
} // end anonymous namespace

void UnaryExpr::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<NegNeqOptPattern>(context);
}

void ArrayUnbExpr::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                               MLIRContext *context){
	results.insert<ArrayArrayUnboxPattern>(context);
}

void ArrayConstExpr::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                               MLIRContext *context){
	results.insert<ArrayArrayUnboxPattern>(context);
}


