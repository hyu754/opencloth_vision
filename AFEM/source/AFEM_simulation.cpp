#include <iostream>
#include "AFEM_simulation.hpp"






AFEM::Simulation::Simulation(AFEM::Geometry geo_in){
	std::cout << "Geometry added to simulation" << std::endl;
	afem_geometry =  geo_in;
	pos_vec = afem_geometry.return_position3D();
	element_vec = afem_geometry.return_element_vector();
	std::cout << "With " << element_vec.size() <<"elements"<< " and with " << pos_vec.size() <<" nodes"<<std::endl;
	std::cout << sizeof(AFEM::Geometry) << std::endl;

}

AFEM::Simulation::~Simulation(){
	
	afem_geometry.~Geometry();



}


void AFEM::Simulation::element_std_to_array(){
	element_array = new element[element_vec.size()];


	for (int i = 0; i < element_vec.size(); i++){
		element_array[i] = element_vec.at(i);
	}

	std::cout << "Converted std vector to array " << std::endl;


}
















