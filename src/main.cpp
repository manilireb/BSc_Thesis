#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include "GraphNodes.h"
#include "TreeNodes.h"
#include "dialect.h"
#include "NodesFactory.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/Parser.h>
#include "mlir/Pass/Pass.h"
#include <mlir/Pass/PassManager.h>
#include "mlir/Transforms/Passes.h"

#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>



namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
        cl::desc("<input mlir file>"),
        cl::init("-"),
        cl::value_desc("filename"));



static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
static cl::opt<bool> isApproach1("approach1", cl::desc("Enable Approach1"));
static cl::opt<bool> isApproach2("approach2", cl::desc("Enable Approach2"));
static cl::opt<bool> isToFile("asFile", cl::desc("Write to file"));



std::string convertPointerToString(mlir::Operation *op)
{
    std::stringstream ss;
    ss << op;
    std::string name = ss.str();
    return name;
}

bool isNecessaryOperation(std::string opName)
{
    if(opName != "std.return" && opName != "func" && opName != "module_terminator" && opName != "module")
    {
        return true;
    }

    return false;
}


std::string getNameOfLastNodeOfGraph(mlir::OwningModuleRef &module)
{
    string nameOfLastNode;
    module->walk([&nameOfLastNode](mlir::Operation * op)
    {
        std::string nameOfRegister = convertPointerToString(op); 
        std::string opName = op->getName().getStringRef().str(); 
        if(isNecessaryOperation(opName))
        {
            nameOfLastNode = nameOfRegister;
        }
    });
    return nameOfLastNode;
}



llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> openMLIRFile()
{
    // Open '.mlir' file.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
                llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code EC = fileOrErr.getError())
    {
        llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        exit(1);
    }
    return fileOrErr;
}


map<std::string, Vertex *> getTheGraph1(mlir::OwningModuleRef &module)
{
    map<std::string, Vertex *> graph;
    NodesFactory nodesFactory;
    module->walk([&graph, &nodesFactory](mlir::Operation * op)
    {
        std::string reg = nodesFactory.convertPointerToString(op);  
        std::string opName = op->getName().getStringRef().str(); 


        if(isNecessaryOperation(opName))
        {
            graph[reg] = nodesFactory.makeTreeNode(op, graph);

        }

    });

    return graph;

}

map<std::string, Node *> getTheGraph2(mlir::OwningModuleRef &module)
{
    map<std::string, Node *> graph;
    NodesFactory nodesFactory;
    module->walk([&graph, &nodesFactory](mlir::Operation * op)
    {
        std::string reg = nodesFactory.convertPointerToString(op);  
        std::string opName = op->getName().getStringRef().str(); 


        if(isNecessaryOperation(opName))
        {

            graph[reg] = nodesFactory.makeGraphNode(op, graph);
            
        }

    });

    return graph;
}


std::string getJSONiq2(mlir::MLIRContext &context, mlir::OwningModuleRef &module)
{

    map<std::string, Node *> graph = getTheGraph2(module);
    std::string nameOfLastNode = getNameOfLastNodeOfGraph(module);
    std::string query = graph[nameOfLastNode]->accept().c_str();

    return query;
}


std::string getJSONiq1(mlir::MLIRContext &context, mlir::OwningModuleRef &module)
{

    map<std::string, Vertex *> graph = getTheGraph1(module);
    std::string nameOfLastNode = getNameOfLastNodeOfGraph(module);
    std::string query = graph[nameOfLastNode]->accept().c_str();

    return query;
}


int dumpMLIR1()
{


    mlir::registerDialect<mlir::StandardOpsDialect>();
    mlir::registerDialect<mlir::jsoniq::JSONiqDialect>();

    mlir::MLIRContext context;
    mlir::OwningModuleRef module;

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = openMLIRFile();

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module)
    {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }

    if(isToFile){
        ofstream myfile;
        myfile.open ("/home/manuel/6-Semester/Thesis/testquery.jq");
        myfile << getJSONiq1(context,module)<<std::endl;
        myfile.close();
        return 0;
    }

    std::cout << "Input MLIR:" << std::endl;
    module->dump();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "BackTransformation:" << std::endl;
    std::cout << std::endl;
    std::cout<<getJSONiq1(context, module)<<std::endl;


    std::cout << std::endl;
    std::cout << std::endl;

    if(enableOpt)
    {
        mlir::PassManager pm(&context);

        applyPassManagerCLOptions(pm);
        pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());


        if(mlir::failed(pm.run(*module)))
        {
            return 4;
        }
        std::cout << "Optimized MLIR:" << std::endl;
        module->dump();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "BackTransformation:" << std::endl;
        std::cout << std::endl;

        std::cout<<getJSONiq1(context, module)<<std::endl;
        

        std::cout << std::endl;
        std::cout << std::endl;
    }


    return 0;
}


int dumpMLIR2()
{

    mlir::registerDialect<mlir::StandardOpsDialect>();
    mlir::registerDialect<mlir::jsoniq::JSONiqDialect>();

    mlir::MLIRContext context;
    mlir::OwningModuleRef module;

    context.allowUnregisteredDialects();

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = openMLIRFile();

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module)
    {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }

    if(isToFile){
        ofstream myfile;
        myfile.open ("/home/manuel/6-Semester/Thesis/testquery.jq");
        myfile << getJSONiq2(context,module)<<std::endl;
        myfile.close();
        return 0;
    }

    std::cout << "Input MLIR:" << std::endl;
    module->dump();
    printf("\n");
    printf("\n");
    std::cout << "Backtransformation:" << std::endl;
    printf("\n");
    printf("\n");
    std::cout<<getJSONiq2(context, module)<<std::endl;

    return 0;
}




int main(int argc, char **argv)
{

    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "jsoniq compiler\n");


    printf("\n");
    printf("\n");

    if(isApproach1)
    {
        return dumpMLIR1();
    }

    else if(isApproach2)
    {
        return dumpMLIR2();
    }

    else
    {
        std::cout << "You need to define your Approach!" << std::endl;
        return 0;
    }
}
