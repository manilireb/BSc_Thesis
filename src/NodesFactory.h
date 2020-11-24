#include <memory>
#include <string>
#include <sstream>

#include <mlir/IR/Module.h>

#include <llvm/IR/Module.h>




class NodesFactory{

public:

	stack<std::string> terminatorstack;

	std::string getOperandRegisterAsString(mlir::Operation *op, int index)
	{
    	std::stringstream stream;
    	stream << op->getOperand(index).getDefiningOp();
    	std::string opregister = stream.str();
    	return opregister;
	}

	std::string convertPointerToString(mlir::Operation *op)
	{
    	std::stringstream ss;
   		ss << op;
    	std::string name = ss.str();
    	return name;
	}


	Vertex* makeTreeNode(mlir::Operation * op, map<std::string, Vertex *> &graph){

		std::string opName = op->getName().getStringRef().str(); 

		if(opName == "jsoniq.lit")
            {
            	Lit* node;
                std::string var;
                int val;
                double decimal;

                auto valOfDictionary = op->getAttr("value");
                if(valOfDictionary.getKind() == mlir::StandardAttributes::String)
                {
                    var = valOfDictionary.cast<mlir::StringAttr>().getValue().str();
                    node = Lit::StringLit(var);
                }
                else if (valOfDictionary.getKind() == mlir::StandardAttributes::Integer)
                {
                    val = valOfDictionary.cast<mlir::IntegerAttr>().getInt();
                    node = Lit::IntegerLit(val);
                }
                else
                {
                    decimal = valOfDictionary.cast<mlir::FloatAttr>().getValueAsDouble();
                    node = Lit::DecimalLit(decimal);
                }

                return node;
            }

        else if (opName == "jsoniq.treat")
            {
                std::string type = op->getAttr("type").cast<mlir::StringAttr>().getValue().str();
                Typing* node  = Typing::Treat(type);
                std::string child = getOperandRegisterAsString(op, 0);

                node->appendToNeighbors(graph[child]);
                return node;

            }

            else if (opName == "jsoniq.instanceof")
            {
                std::string type = op->getAttr("type").cast<mlir::StringAttr>().getValue().str();
                Typing* node = Typing::InstanceOf(type);
                std::string child = getOperandRegisterAsString(op, 0);

                node->appendToNeighbors(graph[child]);
                return node;

            }

            else if (opName == "jsoniq.castable")
            {
                std::string type = op->getAttr("type").cast<mlir::StringAttr>().getValue().str();
                Typing* node = Typing::Castable(type);
                std::string child = getOperandRegisterAsString(op, 0);

                node->appendToNeighbors(graph[child]);
                return node;

            }


            else if (opName == "jsoniq.cast")
            {
                std::string type = op->getAttr("type").cast<mlir::StringAttr>().getValue().str();
                Typing* node = Typing::Cast(type);
                std::string child = getOperandRegisterAsString(op, 0);

                node->appendToNeighbors(graph[child]);
                return node;

            }    


		else if (opName == "jsoniq.to")
            	{
                	Range* node = new Range();
                	std::string child1 = getOperandRegisterAsString(op, 0);
                	std::string child2 = getOperandRegisterAsString(op, 1);

                	node->appendToNeighbors(graph[child1]);
                	node->appendToNeighbors(graph[child2]);
                	return node;

            	}

        else if(opName == "jsoniq.[[]]")
            {
                ArrayLookup* node = new ArrayLookup();
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;

            }

            else if (opName == "jsoniq.objectlookup")
            {
                ObjectLookup* node = new ObjectLookup();
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }


            else if (opName == "jsoniq.||")
            {
                RegularBinary* node = new RegularBinary("||");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.constructobject")
            {
                ObjectConstructor* node = new ObjectConstructor();
                std::string child1 = getOperandRegisterAsString(op, 0);

                if(op->getOperands().size() == 1)
                {
                    node->appendToNeighbors(graph[child1]);
                }
                else
                {
                    std::string child2 = getOperandRegisterAsString(op, 1);

                    node->appendToNeighbors(graph[child1]);
                    node->appendToNeighbors(graph[child2]);
                }
                return node;
            }

            else if (opName == "jsoniq.mergeobjects")
            {
                ObjectMerge* node = new ObjectMerge();
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);

                return node;
            }

            else if (opName == "jsoniq.>")
            {
                RegularBinary* node = new RegularBinary(">");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.comma")
            {
                Comma* node = new Comma();
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.terminator")
            {
            	std::string reg = convertPointerToString(op); 
                Terminator* node = new Terminator();
                std::string child = getOperandRegisterAsString(op, 0);
                terminatorstack.push(reg);

                node->appendToNeighbors(graph[child]);
                return node;


            }

            else if (opName == "jsoniq.[]")
            {
                Predicate* node = new Predicate();
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = terminatorstack.top();
                terminatorstack.pop();
                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.let")
            {
                std::string val = op->getAttr("var").cast<mlir::StringAttr>().getValue().str();
                Let* node = new Let(val);
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = terminatorstack.top();
                terminatorstack.pop();
                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;

            }

            else if (opName == "jsoniq.func")
            {
                std::string val = op->getAttr("funcname").cast<mlir::StringAttr>().getValue().str();

                FunctionCall* node = new FunctionCall(val);

                for (int i = 0; i < op->getNumOperands(); i++)
                {
                    std::string child = getOperandRegisterAsString(op, i);
                    node->appendToNeighbors(graph[child]);
                }
                return node;
            }

            else if (opName == "jsoniq.return")
            {
                Return* node = new Return();
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = terminatorstack.top();
                terminatorstack.pop();
                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.where")
            {
                Where* node = new Where();
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = terminatorstack.top();
                terminatorstack.pop();
                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.count")
            {
                std::string val = op->getAttr("var").cast<mlir::StringAttr>().getValue().str();
                Count* node = new Count(val);
                std::string child1 = getOperandRegisterAsString(op, 0);
                node->appendToNeighbors(graph[child1]);
                return node;
            }


            else if (opName == "jsoniq.varref")
            {
                std::string val = op->getAttr("var").cast<mlir::StringAttr>().getValue().str();
                Varref* node = new Varref(val);
                return node;
            }

            else if (opName == "jsoniq.for")
            {
                std::string val = op->getAttr("var").cast<mlir::StringAttr>().getValue().str();
                For* node = new For(val);
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = terminatorstack.top();
                terminatorstack.pop();
                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.orderby")
            {
                std::string val = op->getAttr("rule").cast<mlir::StringAttr>().getValue().str();

                Orderby* node = new Orderby(val);
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = terminatorstack.top();
                terminatorstack.pop();
                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.groupby")
            {
                std::string val = op->getAttr("var").cast<mlir::StringAttr>().getValue().str();
                Groupby* node = new Groupby(val);
                std::string child = getOperandRegisterAsString(op, 0);

                node->appendToNeighbors(graph[child]);
                return node;

            }

             else if (opName == "jsoniq.arrayconstructor")
            {
                ArrayConstructor* node = new ArrayConstructor();
                std::string child = getOperandRegisterAsString(op, 0);

                node->appendToNeighbors(graph[child]);
                return node;

            }

            else if (opName == "jsoniq.arrayunboxing")
            {
                ArrayUnboxing* node = new ArrayUnboxing();
                std::string child = getOperandRegisterAsString(op, 0);

                node->appendToNeighbors(graph[child]);
                return node;
            }

            else if (opName == "jsoniq.neg")
            {
                RegularUnary* node = new RegularUnary("-");
                std::string child = getOperandRegisterAsString(op, 0);

                node->appendToNeighbors(graph[child]);
                return node;
            }

            else if (opName == "jsoniq.not")
            {
                RegularUnary* node = new RegularUnary("not");
                std::string child = getOperandRegisterAsString(op, 0);

                node->appendToNeighbors(graph[child]);
                return node;
            }

            else if (opName == "jsoniq.+")
            {
                RegularBinary* node = new RegularBinary("+");
                std::stringstream tmp;
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.mod")
            {
                RegularBinary* node = new RegularBinary("mod");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);


                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;

            }

            else if (opName == "jsoniq.eq")
            {
                RegularBinary* node = new RegularBinary("eq");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.ge")
            {
                RegularBinary* node = new RegularBinary("ge");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);

                return node;
            }

            else if (opName == "jsoniq.ne")
            {
                RegularBinary* node = new RegularBinary("ne");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);

                return node;
            }

             else if (opName == "jsoniq.lt")
            {
                RegularBinary* node  = new RegularBinary("lt");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);


                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.le")
            {
                RegularBinary* node = new RegularBinary("le");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);


                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.gt")
            {
                RegularBinary* node = new RegularBinary("gt");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);


                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.=")
            {
                RegularBinary* node  = new RegularBinary("=");
                std::stringstream tmp;
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);


                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.!=")
            {
                RegularBinary* node = new RegularBinary("!=");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);


                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.<")
            {
                RegularBinary* node = new RegularBinary("<");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);


                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

             else if (opName == "jsoniq.<=")
            {
                RegularBinary* node  = new RegularBinary("<=");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);


                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.>=")
            {
                RegularBinary* node  = new RegularBinary(">=");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);


                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.*")
            {
                RegularBinary* node = new RegularBinary("*");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

             else if (opName == "jsoniq.and")
            {
                RegularBinary* node = new RegularBinary("and");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.conditional")
            {
                Conditional* node = new Conditional();
                std::stringstream tmp;
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);
                std::string child3 = getOperandRegisterAsString(op, 2);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                node->appendToNeighbors(graph[child3]);
                return node;
            }


            else if(opName == "jsoniq.or")
            {
                RegularBinary* node = new RegularBinary("or");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);


                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.div")
            {
                RegularBinary* node = new RegularBinary("div");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }


            else if (opName == "jsoniq.idiv")
            {
                RegularBinary* node = new RegularBinary("idiv");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);

                return node;
            }




            else if (opName == "jsoniq.-")
            {
                RegularBinary* node = new RegularBinary("-");
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq.emptyobject")
            {
                EmptyObject* node = new EmptyObject();
                return node;
            }


    	return new Vertex();
	}


    Node* makeGraphNode(mlir::Operation * op, map<std::string, Node *> &graph){

        std::string opName = op->getName().getStringRef().str(); 
        

        if(opName == "jsoniq2.IntegerLiteralExpression")
            {
                auto valinDictionary = op->getAttr("value");
                int val = valinDictionary.cast<mlir::IntegerAttr>().getInt();
                IntegerLiteralExpression* node = new IntegerLiteralExpression(val);
                return node;
            }

            else if (opName == "jsoniq2.AdditiveExpression")
            {
                auto operatorInDictionary = op->getAttr("operator");
                string operatorsign = operatorInDictionary.cast<mlir::StringAttr>().getValue().str();

                AdditiveExpression* node = new AdditiveExpression(operatorsign);
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);

                return node;
            }

            else if (opName == "jsoniq2.RangeExpression")
            {
                RangeExpression* node = new RangeExpression();
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);

                return node;

            }

            else if (opName == "jsoniq2.ForClause")
            {
                auto variableInDictionary = op->getAttr("var");
                string var = variableInDictionary.cast<mlir::StringAttr>().getValue().str();

                ForClause* node = new ForClause(var);

                for (int i = 0; i < op->getNumOperands(); i++)
                {
                    std::string child = getOperandRegisterAsString(op, i);
                    node->appendToNeighbors(graph[child]);
                }

                return node;
            }

            else if (opName == "jsoniq2.ReturnClause")
            {
                ReturnClause* node = new ReturnClause();
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);

                return node;

            }

            else if (opName == "jsoniq2.VariableReferenceExpression")
            {
                auto variableInDictionary = op->getAttr("variable");
                string var = variableInDictionary.cast<mlir::StringAttr>().getValue().str();

                VariableReferenceExpression* node = new VariableReferenceExpression(var);
                return node;

            }


            else if (opName == "jsoniq2.ContextItemExpression")
            {
                ContextItemExpression* node  = new ContextItemExpression();
                return node;
            }


            else if (opName == "jsoniq2.ComparisonExpression")
            {
                auto operatorInDictionary = op->getAttr("operator");
                string operatorsign = operatorInDictionary.cast<mlir::StringAttr>().getValue().str();

                ComparisonExpression* node = new ComparisonExpression(operatorsign);
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq2.PredicateExpression")
            {
                PredicateExpression* node = new PredicateExpression();
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);
                return node;
            }

            else if (opName == "jsoniq2.LetClause")
            {
                auto variableInDictionary = op->getAttr("var");
                string var = variableInDictionary.cast<mlir::StringAttr>().getValue().str();

                LetClause* node = new LetClause(var);
                for (int i = 0; i < op->getNumOperands(); i++)
                {
                    std::string child = getOperandRegisterAsString(op, i);
                    node->appendToNeighbors(graph[child]);
                }

                return node;
            }

            else if (opName == "jsoniq2.WhereClause")
            {
                WhereClause* node = new WhereClause();
                std::string child1 = getOperandRegisterAsString(op, 0);
                std::string child2 = getOperandRegisterAsString(op, 1);

                node->appendToNeighbors(graph[child1]);
                node->appendToNeighbors(graph[child2]);

                return node;
            }

             Node* node= new Node();
                for (int i = 0; i < op->getNumOperands(); i++)
                {
                    std::string child = getOperandRegisterAsString(op, i);
                    node->appendToNeighbors(graph[child]);
                }

                return node;

        }




};