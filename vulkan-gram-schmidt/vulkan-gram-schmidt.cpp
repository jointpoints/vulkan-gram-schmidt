/**
 * @file vulkan-gram-schmidt.cpp
 * @author JointPoints, 2021, github.com/jointpoints
 */
#include "vulkan-gram-schmidt.hpp"
#include <exception>
#include <vector>
#include <cstdlib>





#define VK_VALIDATE(func, error_message) do {VkResult vk_result = func; \
                                             if (vk_result != VK_SUCCESS) \
                                                 throw std::runtime_error(std::string("Execution of ") + #func + " has failed with exitcode " + std::to_string(vk_result) + " and the following message:\n\t" + error_message); \
                                            } while (false)





GPUGramSchmidt::GPUGramSchmidt(bool const enable_debug)
{
	// 1. Create Vulkan Instance
	//   1.1. Define necessary metadata for Vulkan Instance
	uint32_t const  vk_api_req_version    = VK_MAKE_API_VERSION(0, 1, 2, 0);
	uint32_t const  vk_debug_layers_count = 1;
	char const     *vk_debug_layers[]     = {"VK_LAYER_KHRONOS_validation"};
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
		.enabledExtensionCount   = 0,
		.ppEnabledExtensionNames = nullptr
	};
	//   1.2. Check current version of Vulkan Instance before creation of instance
	//     1.2.1. If vkEnumerateInstanceVersion is not available, this is Vulkan 1.0
	if (vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion") == nullptr)
		throw std::runtime_error("Vulkan 1.2 is not supported by this machine.");
	//     1.2.2. If vkEnumerateInstanceVersion is available, we may call it and check the version
	uint32_t vk_api_version = 0;
	VK_VALIDATE(  vkEnumerateInstanceVersion(&vk_api_version), "Unable to identify available Vulkan version."  );
	if (vk_api_version < vk_api_req_version)
		throw std::runtime_error("Vulkan 1.2 is not supported by this machine.");
	//   1.3. If debugging is required, check availability of debug layers
	if (enable_debug)
	{
		uint32_t vk_layers_count = 0;
		vkEnumerateInstanceLayerProperties(&vk_layers_count, nullptr);
		VkLayerProperties vk_layers[vk_layers_count];
		vkEnumerateInstanceLayerProperties(&vk_layers_count, vk_layers);
		for (uint32_t debug_layer_i = 0; debug_layer_i < vk_debug_layers_count; ++debug_layer_i)
		{
			bool found = false;
			for (uint32_t layer_i = 0; layer_i < vk_layers_count; ++layer_i)
				if (strcmp(vk_layers[layer_i].layerName, vk_debug_layers[debug_layer_i]) == 0)
				{
					found = true;
					break;
				}
			if (!found)
				throw std::runtime_error(std::string("Debug layer ") + vk_debug_layers[debug_layer_i] + " was not found.");
		}
	}
	//   1.4. If all explicit checks are passed, we may proceed to the creation of Instance itself
	VK_VALIDATE(  vkCreateInstance(&vk_instance_info, nullptr, &this->vk_instance), "Vulkan Instance creation failed."  );
}





GPUGramSchmidt::~GPUGramSchmidt(void)
{
	vkDestroyInstance(this->vk_instance, nullptr);
}
