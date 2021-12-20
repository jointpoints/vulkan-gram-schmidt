/**
 * @file common.hpp
 * @author JointPoints, 2021, github.com/jointpoints
 */
#ifndef __VGS_COMMON_HPP__
#define __VGS_COMMON_HPP__





#include <vulkan/vulkan.hpp>
#include <vector>
#include <map>
#include <mutex>





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

	VkInstance            vk_instance;
	VkPhysicalDevice      vk_physical_device;
	VkDevice              vk_device;
	std::vector<VkQueue>  vk_queues;
	VkShaderModule        vk_compute_shader;
	VkDescriptorSetLayout vk_descriptor_set_0_layout;
	VkPipelineLayout      vk_compute_pipeline_layout;
	VkPipeline            vk_compute_pipeline;
	VkCommandPool         vk_command_pool;
	VkCommandBuffer       vk_command_buffer;

	uint32_t vk_selected_gpu_i;
	uint32_t vk_selected_queue_family_i;
	uint32_t vk_selected_queues_count;

	static std::map<std::pair<uint32_t, uint32_t>, uint32_t> vk_busy_queues;

	static std::mutex constructor;



public:

	using Matrix = std::vector<std::vector<double>>;

	/// @name Static parameters
	/// @{
	
	/**
	 * Path to a folder containing "vulkan-gram-schmidt.spv"
	 */
	static std::string shader_folder;

	/// @}

	/// @name Constructors & destructors
	/// @{

	/**
	 * @brief Creates new wrapper interface
	 *
	 * Load jobs to this object after its creation.
	 */
	GPUGramSchmidt(bool const enable_debug = false);

	~GPUGramSchmidt(void);

	/// @}



	/// @name Computations
	/// @{
	
	double run(GPUGramSchmidt::Matrix &matrix);

	/// @}



};





#endif // __VGS_COMMON_HPP__
