#include<iostream>
#include<vector>
#include<string>
#include<assert.h>


using namespace std;


class Vertex
{
    vector<Vertex *> neighbors;
    bool mergeflag = false;
    bool commaflag = false;

public:

    virtual string accept()
    {
        return "";
    }

    virtual bool isGroupby()
    {
        return false;
    }

    virtual bool isOrderby()
    {
        return false;
    }


    void appendToNeighbors(Vertex *vertex)
    {
        neighbors.push_back(vertex);
    }

    vector<Vertex *> getNeighbors()
    {
        return this->neighbors;
    }

    bool getMergeFlag()
    {
        return this->mergeflag;
    }

    void setMergeFlag(bool b)
    {
        this->mergeflag = b;
    }

    bool getCommaFlag()
    {
        return this->commaflag;
    }

    void setCommaFlag(bool b)
    {
        this->commaflag = b;
    }

};


 // sort of a hack, 
 // the ContextItemExpression is not represented by an Operation 
 // in this dialect, but by a block argument. 
 // Since block arguemnts do not store operations 
 // they receive no adress space and therefore their pointer is NULL. 
 // So whenever one of the child pointers is NULL we know that this 
 // represents the ContextItemExpression
 


class RegularBinary : public Vertex
{
    string op;
public:
    RegularBinary(string op) : op(op) {}

    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "Regular Binaries must have two neighbors");
        string child1;
        string child2;
        if(neighbors.at(0) == NULL && neighbors.at(1) == NULL)
        {
            child1 = "$$";
            child2 = "$$";
        }
        else if(neighbors.at(0) == NULL)
        {
            child1 = "$$";
            child2 = neighbors.at(1)->accept();
        }
        else if(neighbors.at(1) == NULL)
        {
            child2 = "$$";
            child1 = neighbors.at(0)->accept();
        }
        else
        {

            child1 = neighbors.at(0)->accept();
            child2 = neighbors.at(1)->accept();

        }
        return "(" +child1 + " " + this->op + " " + child2 +")";

    }
};


class RegularUnary : public Vertex
{
    string op;
public:
    RegularUnary(string op) : op(op) {}

    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 1 && "Unary Expressions must have one neighbor");
        if(neighbors.at(0) == NULL)
        {
            return this->op + " $$";
        }

        string child = neighbors.at(0)->accept();
        return this->op + "(" + child + ")";
    }
};


class ArrayUnboxing : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 1 && "ArrayUnboxing must have one child");
        if(neighbors.at(0) == NULL)
        {
            return "$$[]";
        }
        string child = neighbors.at(0)->accept();
        return child + "[]";
    }
};


class Lit : public Vertex
{
    int val;
    string var;
    double decimal;
public:
    // static factory method for IntegerLiteral
    static Lit *IntegerLit(int val)
    {
        Lit *integerLit = new Lit();
        integerLit->val = val;
        integerLit->var = "Integer";
        return integerLit;
    }

    // static factory method for StringLiteral, NullLiteral, BooleanLiteral
    static Lit *StringLit(string var)
    {
        Lit *stringLit = new Lit();
        stringLit->var = var;
        return stringLit;
    }

    // static factory method for DoubleLiteral
    static Lit *DecimalLit(double decimal)
    {
        Lit *decimalLit = new Lit();
        decimalLit->decimal = decimal;
        decimalLit->var = "Decimal";
        return decimalLit;
    }

    string accept()
    {
        if(var == "Integer")
        {
            return to_string(this->val);
        }
        if(var == "Decimal")
        {
            return to_string(decimal);
        }
        if(var == "true" || var == "false" || var == "null")
        {
            return this->var;
        }

        return +"\"" + this->var + "\"";
    }
};


class Typing : public Vertex
{
    string type;
    string expression;
public:
    // static factory method for instance-of
    static Typing *InstanceOf(string type)
    {
        Typing *instaceofExpression = new Typing();
        instaceofExpression->type = type;
        instaceofExpression->expression = "instance of";
        return instaceofExpression;
    }

    // static factory method for treat
    static Typing *Treat(string type)
    {
        Typing *treatExpression = new Typing();
        treatExpression->type = type;
        treatExpression->expression = "treat as";
        return treatExpression;
    }

    // static factory method for castable
    static Typing *Castable(string type)
    {
        Typing *castableExpression = new Typing();
        castableExpression->type = type;
        castableExpression->expression = "castable as";
        return castableExpression;
    }

    // static factory method for cast
    static Typing *Cast(string type)
    {
        Typing *castExpression = new Typing();
        castExpression->type = type;
        castExpression->expression = "cast as";
        return castExpression;
    }

    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 1 && "Typing Expression must have one neighbor");
        string child;
        if(neighbors.at(0) == NULL)
        {
            child = "$$";
        }
        else
        {
            child = neighbors.at(0)->accept();
        }

        return child + " " + this->expression + " " + this->type;
    }
};




class Terminator : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 1 && "Terminator must have exactly one neighbor!");

        if(neighbors.at(0) == NULL)
        {
            return "$$";
        }


        return neighbors.at(0)->accept();
    }
};


class Let : public Vertex
{
    string var;
public:
    Let(string var) : var(var) {}

    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "Let must have 2 neighbors");
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();
        return child1 + " let $" + this->var + ":= " + child2;

    }
};


class Varref : public Vertex
{
    string var;
public:
    Varref(string var) : var(var) {}

    string accept()
    {
        return "$" + this->var;
    }

};


class Range : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "RangeExpression must have two neighbors");
        if(neighbors.at(0) == NULL && neighbors.at(1) == NULL)
        {
            return "($$ to $$)";
        }
        if(neighbors.at(0) == NULL)
        {
            string child = neighbors.at(1)->accept();
            return "($$ to " + child + ")";
        }
        if(neighbors.at(1) == NULL)
        {
            string child = neighbors.at(0)->accept();
            return "(" + child + " to $$)";
        }
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();

        return "(" + child1 + " to " + child2 + ")";
    }
};


class Conditional : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 3 && "Conditional Expression must have 3 neighbors!");

        string child1;
        string child2;
        string child3;

        if(neighbors.at(0) == NULL)
        {
            child1 = "$$";
        }
        else
        {
            child1 = neighbors.at(0)->accept();
        }

        if(neighbors.at(1) == NULL)
        {
            child2 = "$$";
        }
        else
        {
            child2 = neighbors.at(1)->accept();
        }

        if(neighbors.at(2) == NULL)
        {
            child3 = "$$";
        }
        else
        {
            child3 = neighbors.at(2)->accept();
        }

        return "if (" + child1 + ") then " + child2 + " else " + child3;
    }
};


class FunctionCall : public Vertex
{
    string funcname;
public:
    FunctionCall(string funcname) : funcname(funcname) {}


    string getParameterList(vector<Vertex *> list)
    {
        string paramlist = "(";
        for(int i = 0; i < list.size() - 1; i++)
        {
            paramlist += list.at(i)->accept() + ", ";
        }
        if(list.size() > 0)
        {
            paramlist += list.at(list.size() - 1)->accept() + ")";
        }
        else
        {
            paramlist = "()";
        }

        return paramlist;
    }

    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        string parameterList = getParameterList(neighbors);

        return this->funcname + parameterList;
    }
};

class ArrayLookup : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "ArrayLookup must have two neighbors");
        if(neighbors.at(0) == NULL && neighbors.at(1) == NULL)
        {
            return "$$[[$$]]";
        }
        if(neighbors.at(0) == NULL)
        {
            string child = neighbors.at(1)->accept();
            return "$$[[" + child + "]]";
        }
        if(neighbors.at(1) == NULL)
        {
            string child = neighbors.at(0)->accept();
            return child + "[[" + child + "]]";
        }
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();

        return child1 + "[[" + child2 + "]]";
    }

};


class ObjectLookup : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "ObjectLookup must have two neighbors");
        if(neighbors.at(0) == NULL && neighbors.at(1) == NULL)
        {
            return "$$.$$";
        }
        if(neighbors.at(0) == NULL)
        {
            string child = neighbors.at(1)->accept();
            return "$$." + child;
        }
        if(neighbors.at(1) == NULL)
        {
            string child = neighbors.at(0)->accept();
            return child + ".$$";
        }
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();

        return child1 + "." + child2;
    }

};


class Predicate : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "Predicate Expression must have two neighbors");
        if(neighbors.at(0) == NULL && neighbors.at(1) == NULL)
        {
            return "$$[$$]";
        }
        if(neighbors.at(0) == NULL)
        {
            string child = neighbors.at(1)->accept();
            return "$$[" + child + "]";
        }
        if(neighbors.at(1) == NULL)
        {
            string child = neighbors.at(0)->accept();
            return child + "[$$]";
        }
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();

        return child1 + "[" + child2 + "]";
    }
};


class Return : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "ReturnClause must have two neighbors");
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();
        return "(" + child1 + " return " + child2 + " )";
    }
};

class Where : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "Where Clause must have two neighbors");
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();
        return child1 + " where " + child2;
    }
};

class Count : public Vertex
{
    string var;
public:
    Count(string var) : var(var) {}

    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 1 && "Count must have one neighbor!");
        string child = neighbors.at(0)->accept();
        return child + " count $" + this->var;
    }
};


class For : public Vertex
{
    string var;
public:
    For(string var) : var(var) {}

    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "For must have 2 neighbors");
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();
        return child1 + " for $" + this->var + " in " + child2;
    }
};

class Groupby : public Vertex
{
    string var;
public:
    Groupby(string var) : var(var) {}

    bool isGroupby()
    {
        return true;
    }

    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 1 && "Groupby must have one neighbor");
        Vertex *let_child = neighbors.at(0);
        vector<Vertex *> let_child_neighbors = let_child->getNeighbors();
        assert(let_child_neighbors.size() == 2 && "Let Clause must have two children");
        if(let_child_neighbors.at(0)->isGroupby())
        {
            string child1 = let_child_neighbors.at(0)->accept();
            string child2 = let_child_neighbors.at(1)->accept();

            return child1 + ", $" + this->var + " := " + child2;

        }
        string child1 = let_child_neighbors.at(0)->accept();
        string child2 = let_child_neighbors.at(1)->accept();

        return child1 + " group by $" + this->var + " := " + child2;
    }
};

class Orderby : public Vertex
{
    string var;
public:
    Orderby(string var) : var(var) {}

    bool isOrderby()
    {
        return true;
    }

    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "Order by must have two children");
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();
        if(neighbors.at(0)->isOrderby())
        {
            return child1 + ", " + child2 + " " + this->var ;
        }

        return child1 + " order by " + child2 + " " + this->var ;

    }
};


class ArrayConstructor : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 1 && "Array Constructor must have one child!");
        if(neighbors.at(0) == NULL)
        {
            return "[$$]";
        }
        string child = neighbors.at(0)->accept();
        return "[" + child + "]";
    }
};


class ObjectConstructor : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() >= 1 && "Object Constructor must at least one child!");
        if(neighbors.size() == 1)
        {
            if(neighbors.at(0) == NULL)
            {
                return "{| $$ |}";
            }
            string child = neighbors.at(0)->accept();
            return "{|" + child + "|}";
        }

        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();

        if(this->getMergeFlag())
        {
            this->setMergeFlag(false);
            return child1 + " : " + child2;
        }

        return "{" + child1 + " : " + child2 + "}";
    }
};

class ObjectMerge : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "Object Merger must have two children");
        neighbors.at(0)->setMergeFlag(true);
        neighbors.at(1)->setMergeFlag(true);
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();

        if(this->getMergeFlag())
        {
            this->setMergeFlag(false);
            return child1 + ", " + child2;
        }

        return "{" + child1 + ", " + child2 + "}";
    }
};


class Comma : public Vertex
{
public:
    string accept()
    {
        vector<Vertex *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "Comma Expression must have two children");
        neighbors.at(0)->setCommaFlag(true);
        neighbors.at(1)->setCommaFlag(true);
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();

        if(this->getCommaFlag())
        {
            this->setCommaFlag(false);
            return child1 + ", " + child2;
        }

        return "(" + child1 + ", " + child2 + ")";
    }
};


class EmptyObject : public Vertex
{
public:
	string accept()
	{
		return "{}";
	}
};



