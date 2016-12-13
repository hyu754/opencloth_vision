#include <iostream>
#include "AFEM_tools.hpp"

int main(void){
	AFEM::Geometry geo;
	geo.set_dim(AFEM::THREE_DIMENSION);
	geo.read_nodes("FEM_Nodes.txt");
	geo.read_elem("FEM_Elem.txt");
	//"FEM_Elem.txt"
	//"FEM_Nodes.txt"
	return 0;
}