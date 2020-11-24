#include<iostream>
#include<vector>
#include<string>
#include<assert.h>


using namespace std;



class Node
{
    vector<Node *> neighbors;
public:

    virtual string accept()
    {
        return neighbors.at(0)->accept();
    }

    void appendToNeighbors(Node *node)
    {
        neighbors.push_back(node);
    }

    vector<Node *> getNeighbors()
    {
        return this->neighbors;
    }

};


class IntegerLiteralExpression : public Node
{
    int value;
public:
    IntegerLiteralExpression(int value) : value(value){}
   
    string accept()
    {
        return to_string(this->value);
    }
};


class VariableReferenceExpression : public Node
{
    string variable;
public:
    VariableReferenceExpression(string variable) : variable(variable){}
  
    string accept()
    {
        return "$" + this->variable;
    }
};



class ContextItemExpression : public Node
{
public:
    string accept()
    {
        return "$$";
    }
};


class AdditiveExpression : public Node
{
    string op;
public:
    AdditiveExpression(string op) : op(op){}
    
    string accept()
    {
        vector<Node *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "AdditiveExpression must have  two neighbors!");  
        string s1 = neighbors.at(0)->accept();
        string s2 = neighbors.at(1)->accept();
        return "("+s1 +" " +this->op +" "  + s2 +")";
    }
};


class RangeExpression : public Node
{
public:

    string accept()
    {
        vector<Node *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "RangeExpression must have two neighbors!");
        string s1 = neighbors.at(0)->accept();
        string s2 = neighbors.at(1)->accept();
        return "(" +s1 + " to " + s2 +")";
    }
};



class ComparisonExpression : public Node
{
    string op;
public:
    ComparisonExpression(string op) : op(op){}
    
    string accept()
    {
        vector<Node *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "ComparisonExpression must have two neighbors!");
        string s1 = neighbors.at(0)->accept();
        string s2 = neighbors.at(1)->accept();
        return "(" +s1 + " " + this->op + " " + s2 +")";
    }

};


class PredicateExpression : public Node
{
public:

    string accept()
    {
        vector<Node *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "PredicateExpression must have two neighbors!");
        string s1 = neighbors.at(0)->accept();
        string s2 = neighbors.at(1)->accept();
        return s1 + "[" + s2 + "]";
    }
};


class LetClause : public Node
{
    string variable;
public:
    LetClause(string variable) : variable(variable){}

    string accept()
    {
        vector<Node *> neighbors = this->getNeighbors();
        assert(neighbors.size() > 0 && "LetClause must have  neighbors! ");

        if(neighbors.size() == 1){
            string child = neighbors.at(0)->accept();
            return "let $" +this->variable + " := " +child;
        }

        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();
        return child1 +" let $" + this->variable +" := " +child2;
    }
};


class ReturnClause : public Node
{
public:
    
    string accept()
    {
        vector<Node *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "ReturnClause must have two neighbors!") ;
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();
        return "(" +child1 + " return " +child2 +")";
    }
};


class ForClause : public Node
{
    string variable;
public:
    ForClause(string variable) : variable(variable){}
  
    string accept()
    {
        vector<Node *> neighbors = this->getNeighbors();
        assert(neighbors.size() >0 && "ForClause must have neighbors!");

         if(neighbors.size() == 1){
            string child = neighbors.at(0)->accept();
            return "for $" +this->variable + " in " +child;
        }

        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();
        return child1 +" for $" + this->variable +" in " +child2;
    }
};


class WhereClause : public Node
{
public:

    string accept()
    {
        vector<Node *> neighbors = this->getNeighbors();
        assert(neighbors.size() == 2 && "WhereClause must have at least one neighbor!");
        string child1 = neighbors.at(0)->accept();
        string child2 = neighbors.at(1)->accept();
        return child1 + " where " +child2;
    }
};