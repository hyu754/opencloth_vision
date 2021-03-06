#include <iostream>
#include "AFEM_simulation.hpp"
#include <ctime>





AFEM::Simulation::Simulation(AFEM::Geometry geo_in){
	std::cout << "Geometry added to simulation" << std::endl;
	afem_geometry = geo_in;
	pos_vec = afem_geometry.return_position3D();
	element_vec = afem_geometry.return_element_vector();
	stationary_vec = afem_geometry.return_stationary_vector();
	std::cout << "With " << element_vec.size() << "elements" << " and with " << pos_vec.size() << " nodes" << std::endl;


	cuda_tools_class.initialize_cholesky_variables(pos_vec.size(), element_vec.size(), 3);
	std::cout << sizeof(AFEM::Geometry) << std::endl;

}

AFEM::Simulation::~Simulation(){

	afem_geometry.~Geometry();



}


void AFEM::Simulation::element_std_to_array(){
	element_array = new element[element_vec.size()];
	stationary_array = new stationary[stationary_vec.size()];
	pos_array = new position_3D[pos_vec.size()];

	//Populating the element array with AFEM::element structures
	for (int i = 0; i < element_vec.size(); i++){
		element_array[i] = element_vec.at(i);
	}

	for (int i = 0; i < stationary_vec.size(); i++){
		stationary_array[i] = stationary_vec.at(i);
	}

	for (int i = 0; i < pos_vec.size(); i++){
		pos_array[i] = pos_vec.at(i);
	}
	std::cout << "Converted std vectors to arrays " << std::endl;
	//	cuda_tools_class.allocate_CUDA_geometry_data((void**)&element_array, element_vec.size());
	if (afem_geometry.get_dim() == AFEM::dimension::THREE_DIMENSION){
		cuda_tools_class.allocate_copy_CUDA_geometry_data(element_array, stationary_array, pos_array, stationary_vec.size(), element_vec.size(), pos_vec.size(), 3);
	}
	else{
		cuda_tools_class.allocate_copy_CUDA_geometry_data(element_array, stationary_array, pos_array, stationary_vec.size(), element_vec.size(), pos_vec.size(), 2);
	}


	std::cout << "allocated data to GPU memory " << std::endl;

}



void AFEM::Simulation::run(){


	double start = std::clock();
	cuda_tools_class.make_K(element_vec.size(), pos_vec.size());
	//cuda_tools_class.make_f(pos_vec.size(), 3);
	
	cuda_tools_class.dynamic();
	//cuda_tools_class.set_RHS_LHS();
	cuda_tools_class.cg();



	if (afem_geometry.get_dim() == AFEM::dimension::THREE_DIMENSION){
		cuda_tools_class.copy_data_from_cuda(element_array, pos_array);

	}
	else{
		cuda_tools_class.copy_data_from_cuda(element_array, pos_array);

	}
	//cuda_tools_class.copy_data_from_cuda();

	cuda_tools_class.reset_K(element_vec.size(), pos_vec.size());
	//std::cout <<  1.0 / ((std::clock() - start) / CLOCKS_PER_SEC) << std::endl;


}















