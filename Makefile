CXX=mpicxx
INC=-I/usr/local/include/blaze -I/software/boost -I/usr/local/include/eigen3
SRC=main_mpi.cpp GLP/GLP-Libraries/GLP_libraries.cpp GLP/GLP-Functions/GLP_functions.cpp GLP/GLP-Data/GLP_Data.cpp GLP/GLP-Libraries/GLP_splines.cpp GLP/GLP-DisUtensils/GLP_DisUtensils.cpp GLP/GLP-Recipes/GLP_Recipes.cpp
OBJ=*.o
CXXFLAGS=-std=c++14 -O3 -DNDEBUG -mavx -pthread -fopenmp
ASK = -lgomp -lgfortran -fopenmp -fPIC -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core
MKLPATH = /software/MKL
MKLINK= -Wl,--start-group $(MKLPATH)/libmkl_intel_lp64.a $(MKLPATH)/libmkl_gnu_thread.a $(MKLPATH)/libmkl_core.a -Wl,--end-group
LDFLAGS=-L/software/boost/stage/lib
LIBS=-lgomp -lpthread -lm -ldl -lmpi -lstdc++ -lboost_serialization -lboost_mpi -llapack
EXE=output
all: inlaplus.o inlaplus.exe

inlaplus.exe:
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(EXE) $(MKLINK) $(LDFLAGS) $(ASK) $(LIBS)
inlaplus.o: 
	$(CXX) $(CXXFLAGS) -c $(SRC) $(INC)
clean:
	rm $(OBJ) $(EXE)
