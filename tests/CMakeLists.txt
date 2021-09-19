add_executable(test_ssg_index test_ssg_index.cpp)
target_link_libraries(test_ssg_index PRIVATE ${PROJECT_NAME})

add_executable(test_ssg_optimized_search test_ssg_optimized_search.cpp)
target_link_libraries(test_ssg_optimized_search PRIVATE ${PROJECT_NAME})
