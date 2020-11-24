/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
	/home/manuel/6-Semester/Thesis/bsc-reberma/cpp/src/Combine.td:12
*/
struct ArrayArrayUnboxPattern : public RewritePattern {
  ArrayArrayUnboxPattern(MLIRContext *context)
      : RewritePattern("jsoniq.arrayunboxing", {}, 2, context) {}

  LogicalResult matchAndRewrite(Operation *op0,
                                     PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    Operation::operand_range arg(op0->getOperands());
    Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = dyn_cast_or_null<jsoniq::ArrayUnbExpr>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = dyn_cast_or_null<jsoniq::ArrayConstExpr>(op1); (void)castedOp1;
      if (!castedOp1) return failure();
      arg = castedOp1.getODSOperands(0);
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    SmallVector<Value, 4> tblgen_repl_values;
    
    for (auto v : SmallVector<Value, 4>{ arg }) { tblgen_repl_values.push_back(v); }    
    rewriter.replaceOp(op0, tblgen_repl_values);
    return success();
  };
};
/* Generated from:
	/home/manuel/6-Semester/Thesis/bsc-reberma/cpp/src/Combine.td:8
*/
struct NegNeqOptPattern : public RewritePattern {
  NegNeqOptPattern(MLIRContext *context)
      : RewritePattern("jsoniq.neg", {}, 2, context) {}

  LogicalResult matchAndRewrite(Operation *op0,
                                     PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    Operation::operand_range arg(op0->getOperands());
    Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = dyn_cast_or_null<jsoniq::UnaryExpr>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = dyn_cast_or_null<jsoniq::UnaryExpr>(op1); (void)castedOp1;
      if (!castedOp1) return failure();
      arg = castedOp1.getODSOperands(0);
      tblgen_ops[1] = op1;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    SmallVector<Value, 4> tblgen_repl_values;
    
    for (auto v : SmallVector<Value, 4>{ arg }) { tblgen_repl_values.push_back(v); }    
    rewriter.replaceOp(op0, tblgen_repl_values);
    return success();
  };
};
void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(MLIRContext *context, OwningRewritePatternList *patterns) {
  patterns->insert<ArrayArrayUnboxPattern>(context);
  patterns->insert<NegNeqOptPattern>(context);
}
