cmake_minimum_required(VERSION 3.17)
project(Calypso)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

add_compile_definitions(ROOT_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(${PROJECT_NAME}
        src/main.cpp
        src/hello_triangle.cpp
        src/hello_triangle.h
        src/util/file.cpp
        src/util/file.h src/thirdparty.cpp)

### glfw ###############################################################################################################

# don't build GLFW tests/docs
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(lib/glfw)
target_link_libraries(${PROJECT_NAME} PUBLIC glfw)

########################################################################################################################
### glm ################################################################################################################

include_directories(lib/glm)

########################################################################################################################
### vulkan #############################################################################################################

find_package(Vulkan REQUIRED FATAL_ERROR)

if(WIN32)
    target_compile_definitions(Vulkan::Vulkan INTERFACE VK_USE_PLATFORM_WIN32_KHR)
elseif(APPLE)
    target_compile_definitions(Vulkan::Vulkan INTERFACE VK_USE_PLATFORM_MACOS_MVK)
elseif(UNIX)
    # See whether X11 is available. If not, fall back to direct-to-display mode.
    find_package(X11 QUIET)
    if (X11_FOUND)
        target_compile_definitions(Vulkan::Vulkan INTERFACE VK_USE_PLATFORM_XCB_KHR)
    else()
        set(DIRECT_TO_DISPLAY TRUE)
        set(DIRECT_TO_DISPLAY TRUE PARENT_SCOPE)
        target_compile_definitions(Vulkan::Vulkan INTERFACE VK_USE_PLATFORM_DISPLAY_KHR)
    endif()
endif()

target_compile_definitions(Vulkan::Vulkan INTERFACE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_FLAGS_MASK_TYPE_AS_PUBLIC)
target_link_libraries(${PROJECT_NAME} PUBLIC Vulkan::Vulkan)

########################################################################################################################
### spdlog #############################################################################################################

include_directories(lib/spdlog/include)

########################################################################################################################
### vma ################################################################################################################

include_directories(lib/VulkanMemoryAllocator/src)

########################################################################################################################
### stb ################################################################################################################

include_directories(lib/stb)

########################################################################################################################
