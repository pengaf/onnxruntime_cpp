set(ProjectName test)

set(AllFiles 
"test.cpp"
)

add_executable(${ProjectName} ${AllFiles})
target_link_libraries(${ProjectName} "onnxruntime.lib")
