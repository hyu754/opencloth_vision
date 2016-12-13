#include <iostream>
#include "AFEM_simulation.hpp"






AFEM::Simulation::Simulation(AFEM::Geometry geo_in){
	std::cout << "Geometry added to simulation" << std::endl;
	afem_geometry =  geo_in;
	pos_vec = afem_geometry.return_position3D();
	element_vec = afem_geometry.return_element_vector();
	std::cout << "With " << element_vec.size() <<"elements"<< " and with " << pos_vec.size() <<" nodes"<<std::endl;

}

AFEM::Simulation::~Simulation(){
	
	afem_geometry.~Geometry();



}
















