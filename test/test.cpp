#include <chrono>
#include <vector>
#include <iostream>
#include <onnxruntime_cxx_api.h>

void test()
{
	// Allocate ONNXRuntime session
	auto ps = Ort::GetAvailableProviders();
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Env env;
	Ort::SessionOptions sessionOptions;
	OrtCUDAProviderOptions options;
	//options.device_id = 0;
	//options.arena_extend_strategy = 0;
	////options.cuda_mem_limit = (size_t)1 * 1024 * 1024 * 1024;//onnxruntime1.7.0
	//options.gpu_mem_limit = (size_t)1 * 1024 * 1024 * 1024; //onnxruntime1.8.1, onnxruntime1.9.0
	//options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
	//options.do_copy_in_default_stream = 1;
	sessionOptions.AppendExecutionProvider_CUDA(options);
	Ort::Session session{ env, ORT_TSTR("G:/model/test.onnx"), sessionOptions };
	Ort::Allocator alloctor(session, memory_info);
	//session.get_inputs()[0].name

	const int batch_size =1;
	// Allocate model inputs: fill in shape and size

	// Allocate model outputs: fill in shape and size
	std::array<float, batch_size * 11 * 11> policy{};
	std::array<int64_t, 2> policy_shape{ batch_size, 11 * 11 };

	std::array<float, batch_size> value{};
	std::array<int64_t, 2> value_shape{ batch_size, 1 };

	Ort::Value policy_tensor = Ort::Value::CreateTensor<float>(memory_info, policy.data(), policy.size(), policy_shape.data(), policy_shape.size());
	Ort::Value value_tensor = Ort::Value::CreateTensor<float>(memory_info, value.data(), value.size(), value_shape.data(), value_shape.size());

	const char* output_names[] = { "policy","value" };
	Ort::Value output_tensor[2] = { std::move(policy_tensor), std::move(value_tensor) };//{ Ort::Value(nullptr),Ort::Value(nullptr) };// 
	// Run the model
	auto xx = session.GetOutputNameAllocated(1, alloctor);
	auto typeInfo = session.GetOutputTypeInfo(0);
	std::string name = xx.get();

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for (uint32_t i = 0; i < 1; ++i)
	{
		std::array<float, batch_size * 3 * 11 * 11> input{};
		std::array<int64_t, 4> input_shape{ batch_size,3,11,11 };
		//memset(&input, 0, sizeof(input));

		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), input_shape.data(), input_shape.size());
		const char* input_names[] = { "input" };
		session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, output_tensor, 2);
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::duration startDuration = end - start;
	double seconds = startDuration.count() * 0.000000001;

	printf("%f", seconds);
}

int main() 
{
	try
	{
		test();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
	}

}