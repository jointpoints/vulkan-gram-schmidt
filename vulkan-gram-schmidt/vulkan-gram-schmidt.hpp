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
 * This class provides interface for calculation of orthonormal basis
 * on GPU given the initial set of \f$n\f$ linearly independent vectors from \f$\mathbb{R}^n\f$.
 * 
 * The following requirements are needed to be explicitly satisfied by the end user:
 * * GPU is requitred to be able to perform compute operations.
 * * GPU is required to have a host coherent part of memory.
 * * Vulkan 1.2 (or newer) is required to be supported by the GPU driver.
 * * Matrices passed to the GPUGramSchmidt::run function are required to be non-singular; otherwise,
 *   no guarantees are given about the behaviour of the program.
 * 
 * Instances of this class are generally expected to be thread-secure, however, this was not
 * heavily tested.
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
	VkDescriptorPool      vk_descriptor_pool;
	VkDescriptorSet       vk_descriptor_set_0;
	VkFence               vk_fence;

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
	 * @brief Creates a new solver
	 *
	 * Sets up a Vulkan communication environment with the GPU.
	 * 
	 * @param enable_debug Send Vulkan debug information to the output.
	 * 
	 * @warning `enable_debug = true` will require the presence of the @c VK_LAYER_KHRONOS_validation
	 * Vulkan layer and the @c VK_EXT_debug_utils Vulkan extension.
	 */
	GPUGramSchmidt(bool const enable_debug = false);

	/**
	 * @brief Destroys the solver
	 *
	 * Destructs the communication environment with the GPU.
	 */
	~GPUGramSchmidt(void);

	/// @}



	/// @name Computations
	/// @{
	
	/**
	 * @brief Run Gram-Schmidt process on GPU
	 *
	 * Perform orthonormalisation of vectors with the help of GPU.
	 * 
	 * @param matrix Square matrix with the coordinates of the original vectors.
	 * @param vectors_as_columns Indicates whether vectors are packed into @c matrix
	 *                           as columns or as rows.
	 * 
	 * @warning Keep in mind, that the non-singularity of @c matrix must be guaranteed
	 * by you.
	 * 
	 * @return Nothing; the answer is written directly into @c matrix. If `vectors_as_columns == true`,
	 * the answer will also be written in columns.
	 */
	void run(GPUGramSchmidt::Matrix &matrix, bool const vectors_as_columns=false);

	/// @}



};





#endif // __VGS_COMMON_HPP__
