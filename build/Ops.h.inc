/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

class ArrayConstExpr;
class ArrayUnbExpr;
class LitExpr;
class UnaryExpr;

//===----------------------------------------------------------------------===//
// jsoniq::ArrayConstExpr declarations
//===----------------------------------------------------------------------===//

class ArrayConstExprOperandAdaptor {
public:
  ArrayConstExprOperandAdaptor(ArrayRef<Value> values, DictionaryAttr attrs = nullptr);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);

private:
  ArrayRef<Value> odsOperands;
  DictionaryAttr odsAttrs;
};
class ArrayConstExpr : public Op<ArrayConstExpr, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = ArrayConstExprOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value odsArg_0);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value odsArg_0);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

//===----------------------------------------------------------------------===//
// jsoniq::ArrayUnbExpr declarations
//===----------------------------------------------------------------------===//

class ArrayUnbExprOperandAdaptor {
public:
  ArrayUnbExprOperandAdaptor(ArrayRef<Value> values, DictionaryAttr attrs = nullptr);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);

private:
  ArrayRef<Value> odsOperands;
  DictionaryAttr odsAttrs;
};
class ArrayUnbExpr : public Op<ArrayUnbExpr, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = ArrayUnbExprOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value odsArg_0);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value odsArg_0);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

//===----------------------------------------------------------------------===//
// jsoniq::LitExpr declarations
//===----------------------------------------------------------------------===//

class LitExprOperandAdaptor {
public:
  LitExprOperandAdaptor(ArrayRef<Value> values, DictionaryAttr attrs = nullptr);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);
  Attribute value();

private:
  ArrayRef<Value> odsOperands;
  DictionaryAttr odsAttrs;
};
class LitExpr : public Op<LitExpr, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::ZeroOperands, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = LitExprOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  Attribute valueAttr();
  Attribute value();
  void valueAttr(Attribute attr);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Attribute value);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Attribute value);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

//===----------------------------------------------------------------------===//
// jsoniq::UnaryExpr declarations
//===----------------------------------------------------------------------===//

class UnaryExprOperandAdaptor {
public:
  UnaryExprOperandAdaptor(ArrayRef<Value> values, DictionaryAttr attrs = nullptr);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ArrayRef<Value> getODSOperands(unsigned index);

private:
  ArrayRef<Value> odsOperands;
  DictionaryAttr odsAttrs;
};
class UnaryExpr : public Op<UnaryExpr, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand, MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = UnaryExprOperandAdaptor;
  static StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value odsArg_0);
  static void build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value odsArg_0);
  static void build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  static void getCanonicalizationPatterns(OwningRewritePatternList &results, MLIRContext *context);
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

#endif  // GET_OP_CLASSES

