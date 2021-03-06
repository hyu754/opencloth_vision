#include <iostream>
#include "AFEM_geometry.hpp"
#include "AFEM_simulation.hpp"
#include "AFEM_visualization.hpp"
//#include "AFEM_cuda.cuh"
//#include "AFEM_kinect_grabber.h"

int main(void){
	
	/*AFEM_KINECT kinect_class;

	kinect_class.initialize_kinect();
	while (1){
		kinect_class.acquire_color_frame_kinect(true);
	}
*/



	AFEM::Geometry geo;
	geo.set_dim(AFEM::THREE_DIMENSION);
	geo.read_nodes("FEM_Nodes.txt");
	geo.read_elem("FEM_Elem.txt");
	geo.read_stationary("FEM_Stationary.txt");
	//geo.make_K_matrix();

	AFEM::Simulation sim(geo);
	sim.element_std_to_array();

	AFEM::Visualization viz(&sim);

	viz.run_visualization();
	
	/*cuda_tools cc;
	cc.hello();*/
	//"FEM_Elem.txt"
	//"FEM_Nodes.txt"
	return 0;
}