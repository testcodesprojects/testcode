CXX=mpicxx
MKLPATH=/usr/local/MKL
INC=-I/usr/local/include/blaze -I/software/boost -I/usr/local/include/eigen3 -I$(MKLPATH)
SRC=main_mpi.cpp GLP/GLP-Libraries/GLP_libraries.cpp GLP/GLP-Functions/GLP_functions.cpp GLP/GLP-Data/GLP_Data.cpp GLP/GLP-Libraries/GLP_splines.cpp GLP/GLP-DisUtensils/GLP_DisUtensils.cpp GLP/GLP-Recipes/GLP_Recipes.cpp
OBJ=*.o
CXXFLAGS=-std=c++14 -O3 -DNDEBUG -mavx -pthread -fopenmp
ASK = 
MKLINK= -Wl,--start-group $(MKLPATH)/libmkl_intel_lp64.a $(MKLPATH)/libmkl_gnu_thread.a $(MKLPATH)/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl
LDFLAGS=-L/software/boost/stage/lib
LIBS= -lmpi -lstdc++ -lboost_serialization -lboost_mpi 
EXE=output
all: inlaplus.o inlaplus.exe

inlaplus.exe:
	$(CXX) $(MKLPATH) $(CXXFLAGS) $(OBJ) -o $(EXE) $(LDFLAGS) $(LIBS) $(MKLLINK) 
inlaplus.o: 
	$(CXX) $(CXXFLAGS) -c $(SRC) $(INC)
clean:
	rm $(OBJ) $(EXE)
