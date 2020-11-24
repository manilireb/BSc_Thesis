/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

jsoniq::ArrayConstExpr,
jsoniq::ArrayUnbExpr,
jsoniq::LitExpr,
jsoniq::UnaryExpr
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// jsoniq::ArrayConstExpr definitions
//===----------------------------------------------------------------------===//

ArrayConstExprOperandAdaptor::ArrayConstExprOperandAdaptor(ArrayRef<Value> values, DictionaryAttr attrs ) {
  odsOperands = values;
  odsAttrs = attrs;
}

std::pair<unsigned, unsigned> ArrayConstExprOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

ArrayRef<Value> ArrayConstExprOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

StringRef ArrayConstExpr::getOperationName() {
  return "jsoniq.arrayconstructor";
}

std::pair<unsigned, unsigned> ArrayConstExpr::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::operand_range ArrayConstExpr::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ArrayConstExpr::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range ArrayConstExpr::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ArrayConstExpr::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(resultType0);
}

void ArrayConstExpr::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArrayConstExpr::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult ArrayConstExpr::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<SequenceType>()))) {
        return emitOpError("operand #") << index << " must be jsoniq Sequence Type , but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<SequenceType>()))) {
        return emitOpError("result #") << index << " must be jsoniq Sequence Type , but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void ArrayConstExpr::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// jsoniq::ArrayUnbExpr definitions
//===----------------------------------------------------------------------===//

ArrayUnbExprOperandAdaptor::ArrayUnbExprOperandAdaptor(ArrayRef<Value> values, DictionaryAttr attrs ) {
  odsOperands = values;
  odsAttrs = attrs;
}

std::pair<unsigned, unsigned> ArrayUnbExprOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

ArrayRef<Value> ArrayUnbExprOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

StringRef ArrayUnbExpr::getOperationName() {
  return "jsoniq.arrayunboxing";
}

std::pair<unsigned, unsigned> ArrayUnbExpr::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::operand_range ArrayUnbExpr::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ArrayUnbExpr::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range ArrayUnbExpr::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ArrayUnbExpr::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(resultType0);
}

void ArrayUnbExpr::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ArrayUnbExpr::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult ArrayUnbExpr::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<SequenceType>()))) {
        return emitOpError("operand #") << index << " must be jsoniq Sequence Type , but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<SequenceType>()))) {
        return emitOpError("result #") << index << " must be jsoniq Sequence Type , but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void ArrayUnbExpr::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// jsoniq::LitExpr definitions
//===----------------------------------------------------------------------===//

LitExprOperandAdaptor::LitExprOperandAdaptor(ArrayRef<Value> values, DictionaryAttr attrs ) {
  odsOperands = values;
  odsAttrs = attrs;
}

std::pair<unsigned, unsigned> LitExprOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

ArrayRef<Value> LitExprOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

Attribute LitExprOperandAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  Attribute attr = odsAttrs.get("value").cast<Attribute>();
  return attr;
}

StringRef LitExpr::getOperationName() {
  return "jsoniq.lit";
}

std::pair<unsigned, unsigned> LitExpr::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::operand_range LitExpr::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> LitExpr::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range LitExpr::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

Attribute LitExpr::valueAttr() {
  return this->getAttr("value").cast<Attribute>();
}

Attribute LitExpr::value() {
  auto attr = valueAttr();
  return attr;
}

void LitExpr::valueAttr(Attribute attr) {
  this->getOperation()->setAttr("value", attr);
}

void LitExpr::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Attribute value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(resultType0);
}

void LitExpr::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Attribute value) {
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LitExpr::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult LitExpr::verify() {
  auto tblgen_value = this->getAttr("value");
  if (!tblgen_value) return emitOpError("requires attribute 'value'");
  {
    if (!((true))) return emitOpError("attribute 'value' failed to satisfy constraint: any attribute");
  }
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<SequenceType>()))) {
        return emitOpError("result #") << index << " must be jsoniq Sequence Type , but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void LitExpr::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// jsoniq::UnaryExpr definitions
//===----------------------------------------------------------------------===//

UnaryExprOperandAdaptor::UnaryExprOperandAdaptor(ArrayRef<Value> values, DictionaryAttr attrs ) {
  odsOperands = values;
  odsAttrs = attrs;
}

std::pair<unsigned, unsigned> UnaryExprOperandAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

ArrayRef<Value> UnaryExprOperandAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

StringRef UnaryExpr::getOperationName() {
  return "jsoniq.neg";
}

std::pair<unsigned, unsigned> UnaryExpr::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::operand_range UnaryExpr::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UnaryExpr::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

Operation::result_range UnaryExpr::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void UnaryExpr::build(OpBuilder &odsBuilder, OperationState &odsState, Type resultType0, Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(resultType0);
}

void UnaryExpr::build(OpBuilder &odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnaryExpr::build(OpBuilder &, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult UnaryExpr::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<SequenceType>()))) {
        return emitOpError("operand #") << index << " must be jsoniq Sequence Type , but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<SequenceType>()))) {
        return emitOpError("result #") << index << " must be jsoniq Sequence Type , but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void UnaryExpr::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


#endif  // GET_OP_CLASSES

