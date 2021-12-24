/**
 * @file vulkan-gram-schmidt.cpp
 * @author JointPoints, 2021, github.com/jointpoints
 */
#include "vulkan-gram-schmidt.hpp"
#include <exception>
#include <cstdlib>
#include <fstream>





#define VK_VALIDATE(func, error_message, unlock_constructor)                \
	try                                                                     \
	{                                                                       \
		VkResult vk_result = func;                                          \
		if (vk_result != VK_SUCCESS)                                        \
		{                                                                   \
			if (unlock_constructor) GPUGramSchmidt::constructor.unlock();   \
			throw std::runtime_error(std::string("Execution of ") + #func + " has failed with exitcode " + std::to_string(vk_result) + " and the following message:\n\t" + error_message); \
		}                                                                   \
	}                                                                       \
	catch (...)                                                             \
	{                                                                       \
		if (unlock_constructor) GPUGramSchmidt::constructor.unlock();       \
		throw std::runtime_error(std::string("Execution of ") + #func + " has failed with exception and the following message:\n\t" + error_message); \
	}





// Initialisation of static members
std::map<std::pair<uint32_t, uint32_t>, uint32_t> GPUGramSchmidt::vk_busy_queues;
std::mutex GPUGramSchmidt::constructor;
std::string GPUGramSchmidt::shader_folder = ".";





// Constructors & destructors





GPUGramSchmidt::GPUGramSchmidt(bool const enable_debug)
{
	// 1. Lock the constructor mutex so that no two GPUGramSchmidt objects are constructed at the
	//    same time.
	GPUGramSchmidt::constructor.lock();

	// 2. Create Vulkan Instance
	//   2.1. Define necessary metadata for Vulkan Instance
	uint32_t const    vk_api_req_version        = VK_MAKE_API_VERSION(0, 1, 2, 0);
	uint32_t const    vk_debug_layers_count     = 1;
	char const *const vk_debug_layers[]         = {"VK_LAYER_KHRONOS_validation"};
	uint32_t const    vk_debug_extensions_count = 1;
	char const *const vk_debug_extensions[]     = {"VK_EXT_debug_utils"};
	VkApplicationInfo const vk_application_info =
	{
		.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pNext              = nullptr,
		.pApplicationName   = "Vulkan Gram-Schmidt",
		.applicationVersion = 1,
		.pEngineName        = "Vulkan Gram-Schmidt",
		.engineVersion      = 1,
		.apiVersion         = vk_api_req_version
	};
	VkInstanceCreateInfo const vk_instance_info =
	{
		.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext                   = nullptr,
		.flags                   = 0,
		.pApplicationInfo        = &vk_application_info,
		.enabledLayerCount       = (enable_debug) ? (vk_debug_layers_count) : (0U),
		.ppEnabledLayerNames     = (enable_debug) ? (vk_debug_layers) : (nullptr),
		.enabledExtensionCount   = (enable_debug) ? (vk_debug_extensions_count) : (0U),
		.ppEnabledExtensionNames = (enable_debug) ? (vk_debug_extensions) : (nullptr)
	};
	//   2.2. Check current version of Vulkan Instance before creation of instance
	//     2.2.1. If vkEnumerateInstanceVersion is not available, this is Vulkan 1.0
	if (vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion") == nullptr)
		throw std::runtime_error("Vulkan 1.2 is not supported by this machine.");
	//     2.2.2. If vkEnumerateInstanceVersion is available, we may call it and check the version
	uint32_t vk_api_version = 0;
	VK_VALIDATE(  vkEnumerateInstanceVersion(&vk_api_version), "Unable to identify available Vulkan version.", true  );
	if (vk_api_version < vk_api_req_version)
		throw std::runtime_error("Vulkan 1.2 is not supported by this machine.");
	//   2.3. If debugging is required, check availability of debug layers
	if (enable_debug)
	{
		uint32_t vk_layers_count = 0;
		vkEnumerateInstanceLayerProperties(&vk_layers_count, nullptr);
		VkLayerProperties vk_layers[vk_layers_count];
		vkEnumerateInstanceLayerProperties(&vk_layers_count, vk_layers);
		for (const char *vk_debug_layer : vk_debug_layers)
		{
			bool found = false;
			for (VkLayerProperties &vk_layer : vk_layers)
				if (strcmp(vk_layer.layerName, vk_debug_layer) == 0)
				{
					found = true;
					break;
				}
			if (!found)
				throw std::runtime_error(std::string("Debug layer ") + vk_debug_layer + " was not found. Debugging impossible.");
		}
	}
	//   2.4. If all explicit checks are passed, we may proceed to the creation of Instance itself
	VK_VALIDATE(  vkCreateInstance(&vk_instance_info, nullptr, &this->vk_instance), "Vulkan Instance creation failed.", true  );

	// 3. Find suitable physical device
	//   3.1. Enumerate all physical devices (GPUs) available to the Vulkan Instance
	uint32_t vk_gpus_count = 0;
	VK_VALIDATE(  vkEnumeratePhysicalDevices(this->vk_instance, &vk_gpus_count, nullptr), "Physical device enumeration failed.", true  );
	VkPhysicalDevice vk_gpus[vk_gpus_count];
	VK_VALIDATE(  vkEnumeratePhysicalDevices(this->vk_instance, &vk_gpus_count, vk_gpus), "Physical device enumeration failed.", true  );
	//   3.2. Analyse queues of each GPU. We're looking for queues that can exclusively do
	//        computations. If we can't find such queues, we select queues that can at least do
	//        computations.
	std::vector<VkQueueFamilyProperties> vk_queue_properties[vk_gpus_count];
	this->vk_selected_gpu_i          = 0U - 1;
	this->vk_selected_queue_family_i = 0U - 1;
	VkPhysicalDeviceFeatures vk_gpu_features;
	for (uint32_t gpu_i = 0; gpu_i < vk_gpus_count; ++gpu_i)
	{
		//     3.2.1. Check GPU features to certify that it supports double precision calculations
		vkGetPhysicalDeviceFeatures(vk_gpus[gpu_i], &vk_gpu_features);
		if (vk_gpu_features.shaderFloat64 == false)
			continue;
		//     3.2.2. For each GPU get information about its queue families
		uint32_t vk_queue_families_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(vk_gpus[gpu_i], &vk_queue_families_count, nullptr);
		vk_queue_properties[gpu_i].resize(vk_queue_families_count);
		vkGetPhysicalDeviceQueueFamilyProperties(vk_gpus[gpu_i], &vk_queue_families_count, vk_queue_properties[gpu_i].data());
		//     3.2.3. Check these properties
		for (uint32_t queue_family_i = 0; queue_family_i < vk_queue_families_count; ++queue_family_i)
			if (vk_queue_properties[gpu_i][queue_family_i].queueFlags & VK_QUEUE_COMPUTE_BIT != 0)
				if (GPUGramSchmidt::vk_busy_queues[std::make_pair(gpu_i, queue_family_i)] < vk_queue_properties[gpu_i][queue_family_i].queueCount)
				{
					this->vk_selected_gpu_i = gpu_i;
					this->vk_selected_queue_family_i = queue_family_i;
					if (vk_queue_properties[gpu_i][queue_family_i].queueFlags & VK_QUEUE_GRAPHICS_BIT == 0)
						break;
				}
		//     3.2.4. If suitable queue family was found, remember this by marking them as occupied
		if (vk_selected_gpu_i != 0U - 1)
		{
			this->vk_selected_queues_count = 1; // vk_queue_properties[this->vk_selected_gpu_i][this->vk_selected_queue_family_i].queueCount;
			GPUGramSchmidt::vk_busy_queues[std::make_pair(this->vk_selected_gpu_i, this->vk_selected_queue_family_i)] += this->vk_selected_queues_count;
			break;
		}
	}
	if (this->vk_selected_gpu_i == 0U - 1)
		throw std::runtime_error("This computer does not support GPU calculations or all available queues are occupied.");

	// 4. Create Vulkan Device for selected GPU
	std::vector<float> const vk_queue_priorities(this->vk_selected_queues_count, 1.F);
	VkDeviceQueueCreateInfo vk_device_queue_info =
	{
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.queueFamilyIndex = this->vk_selected_queue_family_i,
		.queueCount = this->vk_selected_queues_count,
		.pQueuePriorities = vk_queue_priorities.data()
	};
	VkDeviceCreateInfo vk_device_info =
	{
		.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext                   = nullptr,
		.flags                   = 0,
		.queueCreateInfoCount    = 1,
		.pQueueCreateInfos       = &vk_device_queue_info,
		.enabledLayerCount       = 0, // deprecated
		.ppEnabledLayerNames     = nullptr, // deprecated
		.enabledExtensionCount   = 0,
		.ppEnabledExtensionNames = nullptr,
		.pEnabledFeatures        = &vk_gpu_features // nullptr
	};
	this->vk_physical_device = vk_gpus[this->vk_selected_gpu_i];
	VK_VALIDATE(  vkCreateDevice(this->vk_physical_device, &vk_device_info, nullptr, &this->vk_device), "Logical device creation failed.", true  );

	// 5. Get Vulkan Queues associated with this Vulkan Device
	this->vk_queues.resize(this->vk_selected_queues_count);
	for (uint32_t queue_i = 0; queue_i < this->vk_selected_queues_count; ++queue_i)
		vkGetDeviceQueue(this->vk_device, this->vk_selected_queue_family_i, queue_i, this->vk_queues.data() + queue_i);
	
	// 6. Load the precompiled compute shader
	//   6.1. Open the file and fetch the bytes 
	std::fstream compute_shader_loader(GPUGramSchmidt::shader_folder + "/vulkan-gram-schmidt.spv", std::ios_base::binary | std::ios_base::in | std::ios_base::ate);
	if (compute_shader_loader.fail())
		throw std::runtime_error("File '" + GPUGramSchmidt::shader_folder + "/vulkan-gram-schmidt.spv' was not found.");
	//compute_shader_loader.seekg(0, compute_shader_loader.end);
	size_t compute_shader_byte_count = compute_shader_loader.tellg();
	compute_shader_loader.seekg(0, compute_shader_loader.beg);
	std::vector<char> compute_shader_bytes(compute_shader_byte_count + (4 - compute_shader_byte_count % 4) % 4, 0);
	compute_shader_loader.read(compute_shader_bytes.data(), compute_shader_byte_count);
	compute_shader_loader.close();
	//   6.2. Make a shader module
	VkShaderModuleCreateInfo const vk_compute_shader_info =
	{
		.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext    = nullptr,
		.flags    = 0, // reserved
		.codeSize = compute_shader_bytes.size(),
		.pCode    = reinterpret_cast<uint32_t const *>(compute_shader_bytes.data())
	};
	VK_VALIDATE(  vkCreateShaderModule(this->vk_device, &vk_compute_shader_info, nullptr, &this->vk_compute_shader), "Compute shader module creation failed.", true  );

	// 7. Prepare metadata for computations
	//   7.1. Describe the binding for the matrix (descriptor set 0, binding 0)
	VkDescriptorSetLayoutBinding const vk_descriptor_set_0_binding_0 =
	{
		.binding            = 0,
		.descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount    = 1,
		.stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT,
		.pImmutableSamplers = nullptr
	};
	//   7.2. Create descriptor set layout
	VkDescriptorSetLayoutCreateInfo const vk_descriptor_set_0_layout_info =
	{
		.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext        = nullptr,
		.flags        = 0,
		.bindingCount = 1,
		.pBindings    = &vk_descriptor_set_0_binding_0
	};
	VK_VALIDATE(  vkCreateDescriptorSetLayout(this->vk_device, &vk_descriptor_set_0_layout_info, nullptr, &this->vk_descriptor_set_0_layout), "Descriptor set 0 layout creation failed.", true  );
	//   7.3. Describe push constants ranges (dim, vector_count, start_dim_i)
	VkPushConstantRange const vk_push_constant_range =
	{
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset     = 0,
		.size       = 4 * 3 // 3 integer numbers, 4 bytes each
	};
	//   7.4. Specify layout for the compute pipeline
	VkPipelineLayoutCreateInfo const vk_compute_pipeline_layout_info =
	{
		.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext                  = nullptr,
		.flags                  = 0, // reserved
		.setLayoutCount         = 1,
		.pSetLayouts            = &this->vk_descriptor_set_0_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges    = &vk_push_constant_range
	};
	VK_VALIDATE(  vkCreatePipelineLayout(this->vk_device, &vk_compute_pipeline_layout_info, nullptr, &this->vk_compute_pipeline_layout), "Compute pipeline layout creation failed.", true  );

	// 8. Create compute pipeline
	VkPipelineShaderStageCreateInfo const vk_shader_stage_info =
	{
		.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.pNext               = nullptr,
		.flags               = 0,
		.stage               = VK_SHADER_STAGE_COMPUTE_BIT,
		.module              = this->vk_compute_shader,
		.pName               = "main",
		.pSpecializationInfo = nullptr
	};
	VkComputePipelineCreateInfo const vk_compute_pipeline_info =
	{
		.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.pNext              = nullptr,
		.flags              = 0, // VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT,
		.stage              = vk_shader_stage_info,
		.layout             = this->vk_compute_pipeline_layout,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex  = -1
	};
	VK_VALIDATE(  vkCreateComputePipelines(this->vk_device, VK_NULL_HANDLE, 1, &vk_compute_pipeline_info, nullptr, &this->vk_compute_pipeline), "Compute pipeline creation failed.", true  );
	
	// 9. Create command pool from where buffers will be allocated
	VkCommandPoolCreateInfo const vk_command_pool_info =
	{
		.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext            = nullptr,
		.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = this->vk_selected_queue_family_i
	};
	VK_VALIDATE(  vkCreateCommandPool(this->vk_device, &vk_command_pool_info, nullptr, &this->vk_command_pool), "Command pool creation failed.", true  );

	// 10. Create a command buffer
	VkCommandBufferAllocateInfo const vk_command_buffer_info =
	{
		.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext              = nullptr,
		.commandPool        = this->vk_command_pool,
		.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};
	VK_VALIDATE(  vkAllocateCommandBuffers(this->vk_device, &vk_command_buffer_info, &this->vk_command_buffer), "Command buffer was not allocated.", true  );
	
	// 11. Create descriptor pool from where descriptor sets will be allocated
	VkDescriptorPoolSize const vk_descriptor_pool_size_storage_buffers =
	{
		.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1
	};
	VkDescriptorPoolCreateInfo const vk_descriptor_pool_info =
	{
		.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext         = nullptr,
		.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets       = 1,
		.poolSizeCount = 1,
		.pPoolSizes    = &vk_descriptor_pool_size_storage_buffers
	};
	VK_VALIDATE(  vkCreateDescriptorPool(this->vk_device, &vk_descriptor_pool_info, nullptr, &this->vk_descriptor_pool), "Descriptor pool creation failed.", true  );

	// 12. Create a descriptor set (set = 0, binding = 0)
	VkDescriptorSetAllocateInfo const vk_descriptor_set_0_info =
	{
		.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext              = nullptr,
		.descriptorPool     = this->vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts        = &vk_descriptor_set_0_layout
	};
	VK_VALIDATE(  vkAllocateDescriptorSets(this->vk_device, &vk_descriptor_set_0_info, &this->vk_descriptor_set_0), "Descriptor set 0 allocation failed.", true  );

	// 13. Create a fence to signal after each workload
	VkFenceCreateInfo const vk_fence_info =
	{
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0
	};
	VK_VALIDATE(  vkCreateFence(this->vk_device, &vk_fence_info, nullptr, &this->vk_fence), "Fence creation failed.", true  );

	// 14. Unlock constructor mutex
	GPUGramSchmidt::constructor.unlock();
}





GPUGramSchmidt::~GPUGramSchmidt(void)
{
	GPUGramSchmidt::vk_busy_queues[std::make_pair(this->vk_selected_gpu_i, this->vk_selected_queue_family_i)] -= this->vk_selected_queues_count;
	vkDestroyFence(this->vk_device, this->vk_fence, nullptr);
	vkFreeDescriptorSets(this->vk_device, this->vk_descriptor_pool, 1, &this->vk_descriptor_set_0);
	vkDestroyDescriptorPool(this->vk_device, this->vk_descriptor_pool, nullptr);
	vkFreeCommandBuffers(this->vk_device, this->vk_command_pool, 1, &this->vk_command_buffer);
	vkDestroyCommandPool(this->vk_device, this->vk_command_pool, nullptr);
	vkDestroyPipeline(this->vk_device, this->vk_compute_pipeline, nullptr);
	vkDestroyPipelineLayout(this->vk_device, this->vk_compute_pipeline_layout, nullptr);
	vkDestroyDescriptorSetLayout(this->vk_device, this->vk_descriptor_set_0_layout, nullptr);
	vkDestroyShaderModule(this->vk_device, this->vk_compute_shader, nullptr);
	vkDestroyDevice(this->vk_device, nullptr);
	vkDestroyInstance(this->vk_instance, nullptr);
}





// Computations





void GPUGramSchmidt::run(GPUGramSchmidt::Matrix &matrix, bool const vectors_as_columns)
{
	// 1. Create buffer for the matrix
	//   1.1. Create handle for the storage buffer
	VkBuffer vk_matrix_buffer;
	VkBufferCreateInfo const vk_matrix_buffer_info =
	{
		.sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext                 = nullptr,
		.flags                 = 0,
		.size                  = matrix.size() * matrix.size() * 8,
		.usage                 = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		.sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices   = &this->vk_selected_queue_family_i // ignored due to VK_SHARING_MODE_EXCLUSIVE
	};
	VK_VALIDATE(  vkCreateBuffer(this->vk_device, &vk_matrix_buffer_info, nullptr, &vk_matrix_buffer), "Matrix buffer creation failed.", false  );
	//   1.2. Get the device memory requirements for the buffer
	VkMemoryRequirements vk_matrix_buffer_memory_reqs;
	vkGetBufferMemoryRequirements(this->vk_device, vk_matrix_buffer, &vk_matrix_buffer_memory_reqs);

	// 2. Allocate device memory for computations
	//   2.1. Find a suitable memory type
	VkPhysicalDeviceMemoryProperties vk_device_memory_properties;
	vkGetPhysicalDeviceMemoryProperties(this->vk_physical_device, &vk_device_memory_properties);
	//   2.2. Try to find memory type with needed properties and enough free space
	VkDeviceMemory vk_matrix_memory;
	bool allocation_success = false;
	for (uint32_t memory_type_i = 0; memory_type_i < vk_device_memory_properties.memoryTypeCount; ++memory_type_i)
	{
		if (((vk_device_memory_properties.memoryTypes[memory_type_i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) ||
		    ((vk_device_memory_properties.memoryTypes[memory_type_i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) ||
		    ((vk_device_memory_properties.memoryTypes[memory_type_i].propertyFlags & vk_matrix_buffer_memory_reqs.memoryTypeBits) == 0) ||
		    (vk_device_memory_properties.memoryHeaps[vk_device_memory_properties.memoryTypes[memory_type_i].heapIndex].size < vk_matrix_buffer_memory_reqs.size))
			continue;
		VkMemoryAllocateInfo const vk_memory_info =
		{
			.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.pNext           = nullptr,
			.allocationSize  = vk_matrix_buffer_memory_reqs.size, // matrix.size() * matrix.size() * 8,
			.memoryTypeIndex = memory_type_i
		};
		if (vkAllocateMemory(this->vk_device, &vk_memory_info, nullptr, &vk_matrix_memory) == VK_SUCCESS)
		{
			allocation_success = true;
			break;
		}
	}
	if (allocation_success == false)
		throw std::runtime_error("Unable to allocate memory on your GPU.");
	
	// 3. Bind memory with the buffer
	VK_VALIDATE(  vkBindBufferMemory(this->vk_device, vk_matrix_buffer, vk_matrix_memory, 0), "Device memory association with the matrix buffer failed.", false  );

	// 4. Fill the buffer with the matrix data
	double *payload = nullptr;
	vkMapMemory(this->vk_device, vk_matrix_memory, 0, matrix.size() * matrix.size() * 8, 0, reinterpret_cast<void **>(&payload));
	for (uint32_t i = 0; i < matrix.size(); ++i)
		for (uint32_t j = 0; j < matrix.size(); ++j)
			payload[i * matrix.size() + j] = vectors_as_columns ? matrix[j][i] : matrix[i][j];
	vkUnmapMemory(this->vk_device, vk_matrix_memory);

	// 5. Associate the buffer with the descriptor set binding
	VkDescriptorBufferInfo const vk_matrix_buffer_descriptor_info =
	{
		.buffer = vk_matrix_buffer,
		.offset = 0,
		.range  = VK_WHOLE_SIZE
	};
	VkWriteDescriptorSet const vk_write_descriptor_set_0 =
	{
		.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext            = nullptr,
		.dstSet           = this->vk_descriptor_set_0,
		.dstBinding       = 0,
		.dstArrayElement  = 0,
		.descriptorCount  = 1,
		.descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pImageInfo       = nullptr,
		.pBufferInfo      = &vk_matrix_buffer_descriptor_info,
		.pTexelBufferView = nullptr
	};
	vkUpdateDescriptorSets(this->vk_device, 1, &vk_write_descriptor_set_0, 0, nullptr);

	// 6. Record and submit commands into the command buffer
	//   6.1. Start buffer recording
	VkCommandBufferBeginInfo const vk_command_buffer_begin_info =
	{
		.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext            = nullptr,
		.flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr // ignored for the primary buffers
	};
	VkSubmitInfo const vk_submit_info =
	{
		.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext                = nullptr,
		.waitSemaphoreCount   = 0,
		.pWaitSemaphores      = nullptr,
		.pWaitDstStageMask    = nullptr,
		.commandBufferCount   = 1,
		.pCommandBuffers      = &this->vk_command_buffer,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores    = nullptr
	};
	uint32_t push_constants[] = {(uint32_t)matrix.size(), (uint32_t)matrix.size(), 0};
	for (uint32_t start_vec_i = 0; start_vec_i < matrix.size(); ++start_vec_i)
	{
		VK_VALIDATE(  vkBeginCommandBuffer(this->vk_command_buffer, &vk_command_buffer_begin_info), "Command buffer recording failed to start.", false  );
		//   6.2. Bind the compute pipeline with the buffer
		vkCmdBindPipeline(this->vk_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->vk_compute_pipeline);
		//   6.3. Bind the descriptor set with the buffer
		vkCmdBindDescriptorSets(this->vk_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->vk_compute_pipeline_layout, 0, 1, &this->vk_descriptor_set_0, 0, nullptr);
		//   6.4. Push constants
		push_constants[2] = start_vec_i;
		vkCmdPushConstants(this->vk_command_buffer, this->vk_compute_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * 3, push_constants);
		vkCmdDispatch(this->vk_command_buffer, (matrix.size() - start_vec_i) / 32 + ((matrix.size() - start_vec_i) % 32 > 0), 1, 1);
		//   6.5. Finish buffer recording
		VK_VALIDATE(  vkEndCommandBuffer(this->vk_command_buffer), "Command buffer recording failed to end.", false  );
		//   6.6. Submit the command buffer to the GPU queue
		VK_VALIDATE(  vkQueueSubmit(this->vk_queues[0], 1, &vk_submit_info, this->vk_fence), "Queue submission failed.", false  );
		//   6.7. Wait for the fence before continuing execution
		VK_VALIDATE(  vkWaitForFences(this->vk_device, 1, &this->vk_fence, VK_TRUE, 10000000), "Waiting for the fence failed.", false  );
		VK_VALIDATE(  vkResetFences(this->vk_device, 1, &this->vk_fence), "Fence reset failed.", false  );
	}

	// 7. Read the result into the original matrix
	VK_VALIDATE(  vkMapMemory(this->vk_device, vk_matrix_memory, 0, matrix.size() * matrix.size() * 8, 0, reinterpret_cast<void **>(&payload)), "Memory mapping after calculations failed.", false  );
	for (uint32_t i = 0; i < matrix.size(); ++i)
		for (uint32_t j = 0; j < matrix.size(); ++j)
			matrix[vectors_as_columns ? j : i][vectors_as_columns? i : j] = payload[i * matrix.size() + j];
	vkUnmapMemory(this->vk_device, vk_matrix_memory);

	vkDestroyBuffer(this->vk_device, vk_matrix_buffer, nullptr);
	vkFreeMemory(this->vk_device, vk_matrix_memory, nullptr);
	
	return;
}
