#include <iostream>
#include "AFEM_geometry.hpp"
#include "AFEM_simulation.hpp"
//#include "AFEM_cuda.cuh"

int main(void){
	
	AFEM::Geometry geo;
	geo.set_dim(AFEM::THREE_DIMENSION);
	geo.read_nodes("FEM_Nodes.txt");
	geo.read_elem("FEM_Elem.txt");
	//geo.make_K_matrix();

	AFEM::Simulation sim(geo);
	sim.element_std_to_array();
	sim.run();
	/*cuda_tools cc;
	cc.hello();*/
	//"FEM_Elem.txt"
	//"FEM_Nodes.txt"
	return 0;
}