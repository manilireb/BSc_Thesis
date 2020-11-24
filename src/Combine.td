#ifndef JSONIQ_COMBINE
#define JSONIQ_COMBINE

include "Ops.td"



def NegNeqOptPattern : Pat<(UnaryExpr(UnaryExpr $arg)),
                           (replaceWithValue $arg)>;


def ArrayArrayUnboxPattern : Pat<(ArrayUnbExpr(ArrayConstExpr $arg)), (replaceWithValue $arg)>;



#endif 
