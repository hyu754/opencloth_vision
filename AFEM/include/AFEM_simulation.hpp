#include "AFEM_geometry.hpp"
#ifndef AFEM_SIMULATION_H
#define AFEM_SIMULATION_H

namespace AFEM{
	class Simulation;
}

class AFEM::Simulation
{
public:
	Simulation(AFEM::Geometry geo_in);
	~Simulation();
	
	//This function will change the element std::vector to an array form for CUDA
	void element_std_to_array(void);
	

	
	

private:
	cuda_tools cuda_tools_class;
	AFEM::Geometry afem_geometry;

	element *element_array;

	//These variables will be populated when the class is initialized
	std::vector<AFEM::position_3D> pos_vec;
	std::vector<AFEM::element> element_vec;

};



#endif AFEM_SIMULATION_H