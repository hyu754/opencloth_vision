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
	std::vector<AFEM::position_3D> pos_vec;
	std::vector<AFEM::element> element_vec;

	void host_to_device(void);

private:
	cuda_tools cuda_tools_class;
	AFEM::Geometry afem_geometry;

};



#endif AFEM_SIMULATION_H