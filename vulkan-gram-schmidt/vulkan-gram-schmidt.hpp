/**
 * @file common.hpp
 * @author JointPoints, 2021, github.com/jointpoints
 */
#ifndef __VGS_COMMON_HPP__
#define __VGS_COMMON_HPP__





#include <vulkan/vulkan.hpp>





/**
 * @class GPUGramSchmidt
 * @brief Tools to execute Gram-Schmidt process on GPU.
 *
 * This class provides interface for calculation of orthonormal or orthogonal basis
 * on GPU given the initial set of \f$n\f$ linearly independent vectors from \f$\mathbb{R}^n\f$.
 * Vulkan 1.2 support is required.
 */
class GPUGramSchmidt final
{



private:

	VkInstance vk_instance;



public:

	/**
	 * @brief Creates new wrapper interface
	 *
	 * Load jobs to this object after its creation.
	 */
	GPUGramSchmidt(bool const enable_debug = false);

	~GPUGramSchmidt(void);



};





#endif // __VGS_COMMON_HPP__