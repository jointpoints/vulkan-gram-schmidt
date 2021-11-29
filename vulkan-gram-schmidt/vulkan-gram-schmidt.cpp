/**
 * @file vulkan-gram-schmidt.cpp
 * @author JointPoints, 2021, github.com/jointpoints
 */
#include "vulkan-gram-schmidt.hpp"
#include <exception>
#include <cstdlib>





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
	for (uint32_t gpu_i = 0; gpu_i < vk_gpus_count; ++gpu_i)
	{
		//     3.2.1. For each GPU get information about its queue families
		uint32_t vk_queue_families_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(vk_gpus[gpu_i], &vk_queue_families_count, nullptr);
		vk_queue_properties[gpu_i].resize(vk_queue_families_count);
		vkGetPhysicalDeviceQueueFamilyProperties(vk_gpus[gpu_i], &vk_queue_families_count, vk_queue_properties[gpu_i].data());
		//     3.2.2. Check these properties
		for (uint32_t queue_family_i = 0; queue_family_i < vk_queue_families_count; ++queue_family_i)
			if (vk_queue_properties[gpu_i][queue_family_i].queueFlags & VK_QUEUE_COMPUTE_BIT != 0)
				if (GPUGramSchmidt::vk_busy_queues[std::make_pair(gpu_i, queue_family_i)] < vk_queue_properties[gpu_i][queue_family_i].queueCount)
				{
					this->vk_selected_gpu_i = gpu_i;
					this->vk_selected_queue_family_i = queue_family_i;
					if (vk_queue_properties[gpu_i][queue_family_i].queueFlags & VK_QUEUE_GRAPHICS_BIT == 0)
						break;
				}
		//     3.2.3. If suitable queue family was found, remember this by marking them as occupied
		if (vk_selected_gpu_i != 0U - 1)
		{
			this->vk_selected_queues_count = vk_queue_properties[this->vk_selected_gpu_i][this->vk_selected_queue_family_i].queueCount;
			GPUGramSchmidt::vk_busy_queues[std::make_pair(this->vk_selected_gpu_i, this->vk_selected_queue_family_i)] += this->vk_selected_queues_count;
		}
		else
			throw std::runtime_error("This computer does not support GPU calculations or all available queues are occupied.");
	}

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
		.pEnabledFeatures        = nullptr
	};
	VK_VALIDATE(  vkCreateDevice(vk_gpus[this->vk_selected_gpu_i], &vk_device_info, nullptr, &this->vk_device), "Logical device creation failed.", true  );

	// 5. Get Vulkan Queues associated with this Vulkan Device
	this->vk_queues.resize(this->vk_selected_queues_count);
	for (uint32_t queue_i = 0; queue_i < this->vk_selected_queues_count; ++queue_i)
		vkGetDeviceQueue(this->vk_device, this->vk_selected_queue_family_i, queue_i, this->vk_queues.data() + queue_i);
	
	// 6. Create command pool from where buffers will be allocated
	VkCommandPoolCreateInfo const vk_command_pool_info =
	{
		.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext            = nullptr,
		.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = this->vk_selected_queue_family_i
	};
	VK_VALIDATE(  vkCreateCommandPool(this->vk_device, &vk_command_pool_info, nullptr, &this->vk_command_pool), "Command pool creation failed.", true  );
	
	// 7. Unlock constructor mutex
	GPUGramSchmidt::constructor.unlock();
}





GPUGramSchmidt::~GPUGramSchmidt(void)
{
	GPUGramSchmidt::vk_busy_queues[std::make_pair(this->vk_selected_gpu_i, this->vk_selected_queue_family_i)] -= this->vk_selected_queues_count;
	vkDestroyCommandPool(this->vk_device, this->vk_command_pool, nullptr);
	vkDestroyDevice(this->vk_device, nullptr);
	vkDestroyInstance(this->vk_instance, nullptr);
}





// Computations





double GPUGramSchmidt::run(void)
{
	// ??. Create compute pipeline
	

	// ??. Create command buffer
	VkCommandBuffer vk_command_buffer;
	VkCommandBufferAllocateInfo const vk_command_buffer_info =
	{
		.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext              = nullptr,
		.commandPool        = this->vk_command_pool,
		.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};
	VK_VALIDATE(  vkAllocateCommandBuffers(this->vk_device, &vk_command_buffer_info, &vk_command_buffer), "Command buffer was not allocated.", false  );
	
	vkFreeCommandBuffers(this->vk_device, this->vk_command_pool, 1, &vk_command_buffer);
	
	return 1.0;
}
