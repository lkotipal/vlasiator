/*
 * This file is part of Vlasiator.
 * Copyright 2010-2025 Finnish Meteorological Institute and University of Helsinki
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef GPU_BASE_H
#define GPU_BASE_H

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "arch_device_api.h"

#include <stdio.h>
#include <mutex>
#include "include/splitvector/splitvec.h"
#include "include/hashinator/hashinator.h"
#include "../definitions.h"
#include "../vlasovsolver/vec.h"
#include "../velocity_mesh_parameters.h"
#include <phiprof.hpp>

#ifndef THREADS_PER_MP
#define THREADS_PER_MP 2048
#endif
#ifndef REGISTERS_PER_MP
#define REGISTERS_PER_MP 65536
#endif

// Device properties
extern int gpuMultiProcessorCount;
extern int blocksPerMP;
extern int threadsPerMP;

// Magic multipliers used to make educated guesses for initial allocations
// and for managing dynamic increases in allocation sizes. Some of these are
// scaled based on WID value for better guesses,
static const uint VLASOV_BUFFER_MINBLOCKS = 32768/WID3;
static const uint VLASOV_BUFFER_MINCOLUMNS = 2000/WID;
static const uint INIT_VMESH_SIZE (32768/WID3);
static const uint INIT_MAP_SIZE (16 - WID);
static const double BLOCK_ALLOCATION_PADDING = 1.2;
static const double BLOCK_ALLOCATION_FACTOR = 1.1;

// Used in acceleration column construction. The flattened version of the
// probe cube must store (5) counters / offsets, see vlasovsolver/gpu_acc_map.cpp for details.
static const int GPU_PROBEFLAT_N = 5;

// buffers need to be larger for translation to allow proper parallelism
// GPUTODO: Get rid of this multiplier and consolidate buffer allocations.
// WARNING: Simply removing this factor led to diffs in Flowthrough_trans_periodic, indicating that
// there is somethign wrong with the evaluation of buffers! To be investigated.
static const int TRANSLATION_BUFFER_ALLOCATION_FACTOR = 5;

#define MAXCPUTHREADS 512 // hypothetical max size for some allocation arrays

void gpu_init_device();
void gpu_clear_device();
gpuStream_t gpu_getStream();
gpuStream_t gpu_getPriorityStream();
uint gpu_getThread();
uint gpu_getMaxThreads();
int gpu_getDevice();
uint gpu_getAllocationCount();
int gpu_reportMemory(const size_t local_cap=0, const size_t ghost_cap=0, const size_t local_size=0, const size_t ghost_size=0);

unsigned int nextPowerOfTwo(unsigned int n);

void gpu_vlasov_allocate(uint maxBlockCount);
void gpu_calculateProbeAllocation(uint maxBlockCount);
void gpu_vlasov_deallocate();
void gpu_vlasov_allocate_perthread(uint cpuThreadID, uint maxBlockCount);
uint gpu_vlasov_getSmallestAllocation();

void gpu_batch_allocate(uint nCells=0, uint maxNeighbours=0);

void gpu_acc_allocate(uint maxBlockCount);
void gpu_acc_allocate_perthread(uint cpuThreadID, uint firstAllocationCount, uint columnSetAllocationCount=0);
void gpu_acc_deallocate();

void gpu_trans_allocate(cuint nAllCells=0,
                        cuint largestVmesh=0,
                        cuint unionSetSize=0);
void gpu_trans_deallocate();

extern gpuStream_t gpuStreamList[];
extern gpuStream_t gpuPriorityStreamList[];

// Struct used by Vlasov Acceleration semi-Lagrangian solver
struct ColumnOffsets {
   split::SplitVector<uint> setColumnOffsets; // index from columnBlockOffsets where new set of columns starts (length nColumnSets)
   split::SplitVector<uint> setNumColumns; // how many columns in set of columns (length nColumnSets)

   split::SplitVector<uint> columnBlockOffsets; // indexes where columns start (in blocks, length totalColumns)
   split::SplitVector<uint> columnNumBlocks; // length of column (in blocks, length totalColumns)
   split::SplitVector<int> minBlockK,maxBlockK;
   split::SplitVector<int> kBegin;
   split::SplitVector<int> i,j;
   uint colSize = 0;
   uint colSetSize = 0;
   uint colCapacity = 0;
   uint colSetCapacity = 0;

   ColumnOffsets(uint nColumns=1, uint nColumnSets=1) {
      gpuStream_t stream = gpu_getStream();
      setColumnOffsets.resize(nColumnSets);
      setNumColumns.resize(nColumnSets);
      columnBlockOffsets.resize(nColumns);
      columnNumBlocks.resize(nColumns);
      minBlockK.resize(nColumns);
      maxBlockK.resize(nColumns);
      kBegin.resize(nColumns);
      i.resize(nColumns);
      j.resize(nColumns);
      // These vectors themselves are not in unified memory, just their content data
      setColumnOffsets.optimizeGPU(stream);
      setNumColumns.optimizeGPU(stream);
      columnBlockOffsets.optimizeGPU(stream);
      columnNumBlocks.optimizeGPU(stream);
      minBlockK.optimizeGPU(stream);
      maxBlockK.optimizeGPU(stream);
      kBegin.optimizeGPU(stream);
      i.optimizeGPU(stream);
      j.optimizeGPU(stream);
      // Cached values
      colSize = nColumns;
      colSetSize = nColumnSets;
      colCapacity = columnBlockOffsets.capacity(); // Uses this as an example
      colSetCapacity = setNumColumns.capacity(); // Uses this as an example
   }
   void prefetchDevice(gpuStream_t stream) {
      setColumnOffsets.optimizeGPU(stream);
      setNumColumns.optimizeGPU(stream);
      columnBlockOffsets.optimizeGPU(stream);
      columnNumBlocks.optimizeGPU(stream);
      minBlockK.optimizeGPU(stream);
      maxBlockK.optimizeGPU(stream);
      kBegin.optimizeGPU(stream);
      i.optimizeGPU(stream);
      j.optimizeGPU(stream);
   }
   __host__ size_t sizeCols() const {
      return colSize;
   }
   __host__ size_t capacityCols() const {
      return colCapacity;
   }
   __host__ size_t capacityColSets() const {
      return colSetCapacity;
   }
   __device__ size_t dev_sizeCols() const {
      return columnBlockOffsets.size(); // Uses this as an example
   }
   __device__ size_t dev_sizeColSets() const {
      return setNumColumns.size(); // Uses this as an example
   }
   __device__ size_t dev_capacityCols() const {
      return columnBlockOffsets.capacity(); // Uses this as an example
   }
   __device__ size_t dev_capacityColSets() const {
      return setNumColumns.capacity(); // Uses this as an example
   }
   size_t capacityInBytes() const {
      return colCapacity * (2*sizeof(uint)+5*sizeof(int))
         + colSetCapacity * (2*sizeof(uint))
         + 4 * sizeof(split::SplitVector<uint>)
         + 5 * sizeof(split::SplitVector<int>);
   }
   void setSizes(size_t nCols=0, size_t nColSets=0) {
      // Ensure capacities are handled with cached values
      setCapacities(nCols,nColSets);
      // Only then resize
      setColumnOffsets.resize(nColSets,true);
      setNumColumns.resize(nColSets,true);
      columnBlockOffsets.resize(nCols,true);
      columnNumBlocks.resize(nCols,true);
      minBlockK.resize(nCols,true);
      maxBlockK.resize(nCols,true);
      kBegin.resize(nCols,true);
      i.resize(nCols,true);
      j.resize(nCols,true);
      colSize = nCols;
      colSetSize = nColSets;
   }
   __device__ void device_setSizes(size_t nCols=0, size_t nColSets=0) {
      // Cannot recapacitate
      setColumnOffsets.device_resize(nColSets);
      setNumColumns.device_resize(nColSets);
      columnBlockOffsets.device_resize(nCols);
      columnNumBlocks.device_resize(nCols);
      minBlockK.device_resize(nCols);
      maxBlockK.device_resize(nCols);
      kBegin.device_resize(nCols);
      i.device_resize(nCols);
      j.device_resize(nCols);
      colSize = nCols;
      colSetSize = nColSets;
   }
   void setCapacities(size_t nCols=0, size_t nColSets=0) {
      // check cached capacities to prevent page faults if not necessary
      if (nCols > colCapacity) {
         // Recapacitate column vectors
         colCapacity = nCols * BLOCK_ALLOCATION_PADDING;
         columnBlockOffsets.reallocate(colCapacity);
         columnNumBlocks.reallocate(colCapacity);
         minBlockK.reallocate(colCapacity);
         maxBlockK.reallocate(colCapacity);
         kBegin.reallocate(colCapacity);
         i.reallocate(colCapacity);
         j.reallocate(colCapacity);
      }
      if (nColSets > colSetCapacity) {
         // Recapacitate columnSet vectors
         colSetCapacity = nColSets * BLOCK_ALLOCATION_PADDING;
         setColumnOffsets.reallocate(colSetCapacity);
         setNumColumns.reallocate(colSetCapacity);
      }
   }
};

struct GPUMemoryManager {
   // Store pointers and their allocation sizes
   std::unordered_map<std::string, void*> gpuMemoryPointers;
   std::unordered_map<std::string, size_t> allocationSizes;
   std::unordered_map<std::string, int> nameCounters;
   std::unordered_map<std::string, std::string> pointerDevice;
   std::unordered_map<std::string, void*> sessionPointers;
   std::unordered_map<std::string, size_t> sessionPointerOffset;
   std::mutex memoryMutex;
   bool sessionOn = false;
   size_t dev_sessionSize = 0;
   size_t host_sessionSize = 0;
   size_t dev_sessionAllocationSize = 0;
   size_t host_sessionAllocationSize = 0;
   size_t dev_previousSessionSize = 0;
   size_t host_previousSessionSize = 0;

   // Create a new pointer with a base name, ensure unique name
   bool createPointer(const std::string& baseName, std::string &uniqueName) {
      if (uniqueName != "null"){
         return false;
      }

      std::lock_guard<std::mutex> lock(memoryMutex);

      // If a pointer with the given name already exists, we create a new name
      if (gpuMemoryPointers.count(baseName)) {
         int& counter = nameCounters[baseName];
         uniqueName = baseName + "_" + std::to_string(++counter);
      } else {
         nameCounters[baseName] = 0;
         uniqueName = baseName;
      }

      gpuMemoryPointers[uniqueName] = nullptr;
      allocationSizes[uniqueName] = (size_t)(0);
      pointerDevice[uniqueName] = "None";
      return true;
   }

   // Create a new pointer, ensure that there aren't duplicates
   bool createUniquePointer(const std::string& name) {
      std::lock_guard<std::mutex> lock(memoryMutex);

      if (gpuMemoryPointers.count(name)) {
         return false;
      }

      nameCounters[name] = 0;
      gpuMemoryPointers[name] = nullptr;
      allocationSizes[name] = (size_t)(0);
      pointerDevice[name] = "None";
      return true;
   }

   // Create a new pointer with a base name and an index
   bool createSubPointer(const std::string& basePointerName, const uint index) {
      // Ensure the base pointer exists
      if (!gpuMemoryPointers.count(basePointerName)){
         std::cerr << "Error: Pointer name '" << basePointerName << "' not found.\n";
         return false;
      }

      std::string pointerName = basePointerName + "#" + std::to_string(index);

      if (gpuMemoryPointers.count(pointerName)){
         return false;
      }

      std::string uniqueName = "null";

      bool success = createPointer(pointerName, uniqueName);
      
      if(pointerName != uniqueName){
         std::cerr << "Error: Pointer '" << pointerName << "' already exists.\n";
         return false;
      }

      return success;
   }

   // Start a session, where we have one big session pointer which can be split into multiple smaller pointers
   bool startSession(size_t dev_bytes, size_t host_bytes){
      // Ensure that the session pointers are at least as big as the largest session so far
      size_t host_requiredSessionSize = max(host_previousSessionSize, host_bytes);
      size_t dev_requiredSessionSize = max(dev_previousSessionSize, dev_bytes);

      if (gpuMemoryPointers.count("dev_sessionPointer") == 0) {
         createUniquePointer("dev_sessionPointer");
      }
      if (gpuMemoryPointers.count("host_sessionPointer") == 0) {
         createUniquePointer("host_sessionPointer");
      }

      if(sessionOn){
         std::cerr << "Concurrent sessions not supported. Please end previous session before starting a new one.\n";
         return false;
      }
      sessionOn = true;

      // Reallocate session pointers if the required size increases
      if(dev_requiredSessionSize > dev_sessionAllocationSize){
         allocate("dev_sessionPointer", dev_requiredSessionSize);
         dev_sessionAllocationSize = dev_requiredSessionSize;
      }
      dev_sessionSize = 0;
      pointerDevice["dev_sessionPointer"] = "dev";

      if(host_requiredSessionSize > host_sessionAllocationSize){
         hostAllocate("host_sessionPointer", host_requiredSessionSize);
         host_sessionAllocationSize = host_requiredSessionSize;
      }
      host_sessionSize = 0;
      pointerDevice["host_sessionPointer"] = "host";

      return true;
   }

   // Ending a session wipes the divisions of the session pointer
   bool endSession(){
      if(!sessionOn){
         std::cerr << "No session is currently on. Please start a session before ending it.\n";
         return false;
      }
      sessionOn = false;
      dev_previousSessionSize = dev_sessionSize;
      dev_sessionSize = 0;
      host_previousSessionSize = host_sessionSize;
      host_sessionSize = 0;

      // Free the pointers that did not fit into the session pointer
      freeSessionPointers();

      return true;
   }

   // Allocate memory to a pointer by name
   bool allocate(const std::string& name, size_t bytes) {
      std::lock_guard<std::mutex> lock(memoryMutex);
      if (gpuMemoryPointers.count(name) == 0) {
         std::cerr << "Error: Pointer name '" << name << "' not found.\n";
         return false;
      }

      if (allocationSizes[name] >= bytes) {
         //No need to reallocate
         return false;
      }
      
      if (gpuMemoryPointers[name] != nullptr) {
         CHK_ERR( gpuFree(gpuMemoryPointers[name]) );
      }

      CHK_ERR( gpuMalloc(&gpuMemoryPointers[name], bytes) );
      allocationSizes[name] = bytes;
      pointerDevice[name] = "dev";
      return true;
   }

   // Allocate memory to a pointer by name with an extra buffer added
   bool allocateWithBuffer(const std::string& name, size_t bytes, size_t buffer) {
      std::lock_guard<std::mutex> lock(memoryMutex);
      if (gpuMemoryPointers.count(name) == 0) {
         std::cerr << "Error: Pointer name '" << name << "' not found.\n";
         return false;
      }

      if (allocationSizes[name] >= bytes) {
         //No need to reallocate
         return false;
      }
      
      if (gpuMemoryPointers[name] != nullptr) {
         CHK_ERR( gpuFree(gpuMemoryPointers[name]) );
      }

      CHK_ERR( gpuMalloc(&gpuMemoryPointers[name], bytes*buffer) );
      allocationSizes[name] = bytes*buffer;
      pointerDevice[name] = "dev";
      return true;
   }

   // Allocate pinned host memory to a pointer by name
   bool hostAllocate(const std::string& name, size_t bytes) {
      std::lock_guard<std::mutex> lock(memoryMutex);
      if (gpuMemoryPointers.count(name) == 0) {
         std::cerr << "Error: Pointer name '" << name << "' not found.\n";
         return false;
      }

      if (allocationSizes[name] >= bytes) {
         //No need to reallocate
         return false;
      }

      if (gpuMemoryPointers[name] != nullptr) {
         CHK_ERR( gpuFreeHost(gpuMemoryPointers[name]) );
      }

      CHK_ERR( gpuMallocHost(&gpuMemoryPointers[name], bytes) );
      allocationSizes[name] = bytes;
      pointerDevice[name] = "host";
      return true;
   }

   // Allocate pinned host memory to a pointer by name with an extra buffer
   bool hostAllocateWithBuffer(const std::string& name, size_t bytes, size_t buffer) {
      std::lock_guard<std::mutex> lock(memoryMutex);
      if (gpuMemoryPointers.count(name) == 0) {
         std::cerr << "Error: Pointer name '" << name << "' not found.\n";
         return false;
      }

      if (allocationSizes[name] >= bytes) {
         //No need to reallocate
         return false;
      }

      if (gpuMemoryPointers[name] != nullptr) {
         CHK_ERR( gpuFreeHost(gpuMemoryPointers[name]) );
      }

      CHK_ERR( gpuMallocHost(&gpuMemoryPointers[name], bytes*buffer) );
      allocationSizes[name] = bytes*buffer;
      pointerDevice[name] = "host";
      return true;
   }

   // Allocate memory to a sub pointer by name and index
   bool subPointerAllocate(const std::string& basePointerName, const uint index, size_t bytes) {
      if (gpuMemoryPointers.count(basePointerName) == 0) {
         std::cerr << "Error: Pointer name '" << basePointerName << "' not found.\n";
         return false;
      }

      std::string pointerName = basePointerName + "#" + std::to_string(index);

      if (!gpuMemoryPointers.count(pointerName)){
         std::cerr << "Error: Pointer name '" << pointerName << "' not found.\n";
         return false;
      }

      return allocate(pointerName, bytes);
   }

   // Allocate memory to a sub pointer by name and index and copy data from old pointer
   bool subPointerCopyAndReallocate(const std::string& basePointerName, const uint index, size_t bytes) {
      if (gpuMemoryPointers.count(basePointerName) == 0) {
         std::cerr << "Error: Pointer name '" << basePointerName << "' not found.\n";
         return false;
      }

      std::string pointerName = basePointerName + "#" + std::to_string(index);

      if (!gpuMemoryPointers.count(pointerName)){
         std::cerr << "Error: Pointer name '" << pointerName << "' not found.\n";
         return false;
      }

      if (allocationSizes[pointerName] == 0) {
         return allocate(pointerName, bytes);
      }

      if (allocationSizes[pointerName] >= bytes) {
         return false;
      }

      void *newPointer;
      CHK_ERR( gpuMalloc((void**)&newPointer, bytes) );
      CHK_ERR( gpuMemcpy(newPointer, gpuMemoryPointers[pointerName], allocationSizes[pointerName], gpuMemcpyDeviceToDevice) );
      CHK_ERR( gpuFree(gpuMemoryPointers[pointerName]) );
      updatePointer(pointerName, newPointer);

      allocationSizes[pointerName] = bytes;

      return true;
   }

   // Allocate memory to a sub pointer by name and index with an extra buffer
   bool subPointerAllocateWithBuffer(const std::string& basePointerName, const uint index, size_t bytes, size_t buffer) {
      if (gpuMemoryPointers.count(basePointerName) == 0) {
         std::cerr << "Error: Pointer name '" << basePointerName << "' not found.\n";
         return false;
      }

      std::string pointerName = basePointerName + "#" + std::to_string(index);

      if (!gpuMemoryPointers.count(pointerName)){
         std::cerr << "Error: Pointer name '" << pointerName << "' not found.\n";
         return false;
      }

      return allocateWithBuffer(pointerName, bytes, buffer);
   }

   // Allocate host memory to a sub pointer by name and index
   bool subPointerHostAllocate(const std::string& basePointerName, const uint index, size_t bytes) {
      if (gpuMemoryPointers.count(basePointerName) == 0) {
         std::cerr << "Error: Pointer name '" << basePointerName << "' not found.\n";
         return false;
      }

      std::string pointerName = basePointerName + "#" + std::to_string(index);

      if (!gpuMemoryPointers.count(pointerName)){
         std::cerr << "Error: Pointer name '" << pointerName << "' not found.\n";
         return false;
      }

      return hostAllocate(pointerName, bytes);
   }

   // Calculate the offset required to align a pointer
   template<typename T>
   size_t alignOffset(void* base, size_t offset) {
      uintptr_t fullAddress = reinterpret_cast<uintptr_t>(base) + offset;
      size_t alignment = alignof(T);
      size_t alignedAddress = (fullAddress + alignment - 1) & ~(alignment - 1);
      return alignedAddress - reinterpret_cast<uintptr_t>(base);
   }

   // Create a new pointer by partitioning the session pointer
   template<typename T>
   bool sessionAllocate(const std::string& name, size_t bytes){
      if(!sessionOn){
         std::cerr << "No session is currently on. Please start a session before allocating to it.\n";
         return false;
      }
      void *sessionPointer = getPointer<void>("dev_sessionPointer");
      size_t offset = alignOffset<T>(sessionPointer, dev_sessionSize);
      sessionPointerOffset[name] = offset;

      int padding = offset - dev_sessionSize;
      dev_sessionSize += bytes + padding;

      // If the pointer does not fit into the session pointer, create a new pointer
      if (dev_sessionSize > dev_sessionAllocationSize){
         std::lock_guard<std::mutex> lock(memoryMutex);
         sessionPointerOffset[name] = dev_sessionSize;
         sessionPointers[name] = nullptr;
         pointerDevice[name] = "dev";
         allocationSizes[name] = bytes;
         CHK_ERR( gpuMalloc(&sessionPointers[name], bytes) );
      }

      return true;
   }

   // Create a new pointer by partitioning the host session pointer
   template<typename T>
   bool sessionHostAllocate(const std::string& name, size_t bytes){
      void *sessionPointer = getPointer<void>("host_sessionPointer");
      size_t offset = alignOffset<T>(sessionPointer, host_sessionSize);
      sessionPointerOffset[name] = offset;

      int padding = offset - host_sessionSize;
      host_sessionSize += bytes + padding;

      // If the pointer does not fit into the session pointer, create a new pointer
      if (host_sessionSize > host_sessionAllocationSize){
         std::lock_guard<std::mutex> lock(memoryMutex);
         sessionPointerOffset[name] = host_sessionSize;
         sessionPointers[name] = nullptr;
         pointerDevice[name] = "host";
         allocationSizes[name] = bytes;
         CHK_ERR( gpuMallocHost(&sessionPointers[name], bytes) );
      }

      return true;
   }

   // Get allocated size for a pointer
   size_t getSize(const std::string& name) const {
      if (allocationSizes.count(name)){
         return allocationSizes.at(name);
      }
      return 0;
   }

   // Get the total amount of GPU memory allocated with the memory manager
   size_t totalGpuAllocation(){
      size_t total = 0;

      for (auto& pair : gpuMemoryPointers) {
         if (pair.second != nullptr) {
            std::string name = pair.first;
            if (pointerDevice[name] == "dev"){
               total += allocationSizes[name];
            }
         }
      }

      for (auto& pair : sessionPointers) {
         if (pair.second != nullptr) {
            std::string name = pair.first;
            if (pointerDevice[name] == "dev"){
               total += allocationSizes[name];
            }
         }
      }

      return total;
   }

   std::vector<std::tuple<std::string, size_t>> largestGpuAllocations(size_t numberOfOutputPointers) {
      std::unordered_map<std::string, size_t> aggregatedSizes;

      // Sizes of subpointers are added to the size of the basepointer
      auto addPointers = [&](auto& pointerMap) {
         for (auto& pair : pointerMap) {
            if (pair.second != nullptr) {
               const std::string& name = pair.first;
               if (pointerDevice[name] == "dev") {
                  std::string baseName = name;
                  size_t hashtagPosition = name.find('#');
                  if (hashtagPosition != std::string::npos) {
                     baseName = name.substr(0, hashtagPosition);
                  }
                  aggregatedSizes[baseName] += allocationSizes[name];
               }
            }
         }
      };

      addPointers(gpuMemoryPointers);
      addPointers(sessionPointers);

      // Convert map to vector of tuples
      std::vector<std::tuple<std::string, size_t>> allocations;
      for (auto& pair : aggregatedSizes) {
         allocations.emplace_back(pair.first, pair.second);
      }

      // Sort descending by size
      std::sort(allocations.begin(), allocations.end(), [](auto& a, auto& b) {
         return std::get<1>(a) > std::get<1>(b);
      });

      // Extract the largest pointers
      if (allocations.size() > numberOfOutputPointers) {
         allocations.resize(numberOfOutputPointers);
      }

      return allocations;
   }

   // get the total amount of host memory allocated with the memory manager
   size_t totalCpuAllocation(){
      size_t total = 0;

      for (auto& pair : gpuMemoryPointers) {
         if (pair.second != nullptr) {
            std::string name = pair.first;
            if (pointerDevice[name] == "host"){
               total += allocationSizes[name];
            }
         }
      }

      for (auto& pair : sessionPointers) {
         if (pair.second != nullptr) {
            std::string name = pair.first;
            if (pointerDevice[name] == "host"){
               total += allocationSizes[name];
            }
         }
      }

      return total;
   }

   // Free all allocated pointers from session, besides the sessionpointer itself
   void freeSessionPointers() {
      for (auto& pair : sessionPointers) {
         if (pair.second != nullptr) {
            std::string name = pair.first;
            if (pointerDevice[name] == "dev"){
               CHK_ERR( gpuFree(pair.second) );
            }else if (pointerDevice[name] == "host"){
               CHK_ERR( gpuFreeHost(pair.second) );
            }
            allocationSizes[name] = 0;
            pair.second = nullptr;
         }
      }
      sessionPointers.clear();
   }

   // Free all allocated GPU memory
   void freeAll() {
      for (auto& pair : gpuMemoryPointers) {
         if (pair.second != nullptr) {
            std::string name = pair.first;
            if (allocationSizes[name] > 0){
               if (pointerDevice[name] == "dev"){
                  CHK_ERR( gpuFree(pair.second) );
               }else if (pointerDevice[name] == "host"){
                  CHK_ERR( gpuFreeHost(pair.second) );
               }
            }
            pair.second = nullptr;
         }
      }
      freeSessionPointers();

      gpuMemoryPointers.clear();
      allocationSizes.clear();
      nameCounters.clear();
      pointerDevice.clear();
      sessionPointerOffset.clear();
      sessionOn = false;
      dev_sessionSize = 0;
      host_sessionSize = 0;
      dev_sessionAllocationSize = 0;
      host_sessionAllocationSize = 0;
      dev_previousSessionSize = 0;
      host_previousSessionSize = 0;
   }

   // Get typed pointer
   template <typename T>
   T* getPointer(const std::string& name) const {
      if (!gpuMemoryPointers.count(name)){
         throw std::runtime_error("Unknown pointer name: " + name);
      }
      return static_cast<T*>(gpuMemoryPointers.at(name));
   }

   // Get typed subpointer with an index
   template <typename T>
   T* getSubPointer(const std::string& basePointerName, const uint index) const {
      if (!gpuMemoryPointers.count(basePointerName)){
         throw std::runtime_error("Unknown base pointer name: " + basePointerName);
      }
      std::string pointerName = basePointerName + "#" + std::to_string(index);
      return getPointer<T>(pointerName);
   }

   // Get pointer in a session
   template <typename T>
   T* getSessionPointer(const std::string& name) const {
      if (!sessionPointerOffset.count(name)){
         throw std::runtime_error("Unknown pointer name: " + name);
      }

      // The pointer is usually contained in the session pointer and distinguished by an offset
      char *sessionPointer = static_cast<char*>(gpuMemoryPointers.at("dev_sessionPointer"));
      size_t offset = sessionPointerOffset.at(name);

      // If the pointer did not fit inside the session pointer, retrieve it from the separate map
      if (offset > dev_sessionAllocationSize){
         if (!sessionPointers.count(name)){
            throw std::runtime_error("Unknown pointer name: " + name);
         }
         return static_cast<T*>(sessionPointers.at(name));
      }
      
      return reinterpret_cast<T*>(sessionPointer + offset);
   }

   // Get host pointer in a session
   template <typename T>
   T* getSessionHostPointer(const std::string& name) const {
      if (!sessionPointerOffset.count(name)){
         throw std::runtime_error("Unknown pointer name: " + name);
      }

      // The pointer is usually contained in the session pointer and distinguished by an offset
      char *sessionPointer = static_cast<char*>(gpuMemoryPointers.at("host_sessionPointer"));
      size_t offset = sessionPointerOffset.at(name);

      // If the pointer did not fit inside the session pointer, retrieve it from the separate map
      if (offset > host_sessionAllocationSize){
         if (!sessionPointers.count(name)){
            throw std::runtime_error("Unknown pointer name: " + name);
         }
         return static_cast<T*>(sessionPointers.at(name));
      }
      
      return reinterpret_cast<T*>(sessionPointer + offset);
   }

   // Update a new pointer to the memory manager
   void updatePointer(const std::string& name, void* newPtr) {
      std::lock_guard<std::mutex> lock(memoryMutex);
      gpuMemoryPointers[name] = newPtr;
   }

   // Set the value of the base pointer at a given index to be the corresponding sub pointer
   template <typename T>
   void setSubPointer(const std::string& basePointerName, const uint index){
      if (!gpuMemoryPointers.count(basePointerName)){
         throw std::runtime_error("Unknown base pointer name: " + basePointerName);
      }

      std::string pointerName = basePointerName + "#" + std::to_string(index);

      if (!gpuMemoryPointers.count(pointerName)){
         throw std::runtime_error("Unknown pointer name: " + pointerName);
      }

      T** basePointer = static_cast<T**>(gpuMemoryPointers[basePointerName]);
      T* subPointer  = static_cast<T*>(gpuMemoryPointers[pointerName]);
      basePointer[index] = subPointer;
   }
};

extern GPUMemoryManager gpuMemoryManager;

extern ColumnOffsets *host_columnOffsetData;
extern uint gpu_largest_columnCount;
extern size_t gpu_probeFullSize, gpu_probeFlattenedSize, gpu_probeStride;

// Hash map and splitvectors buffers used in block adjustment are declared in block_adjust_gpu.hpp
// Vector and set for use in translation are declared in vlasovsolver/gpu_trans_map_amr.hpp

// Counters used in allocations
extern std::vector<uint> gpu_vlasov_allocatedSize;
extern uint gpu_acc_allocatedColumns;
extern uint gpu_acc_foundColumnsCount;

#endif
