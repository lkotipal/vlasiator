/*
 * This file is part of Vlasiator.
 * Copyright 2010-2024 Finnish Meteorological Institute and University of Helsinki
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

#include "block_adjust_gpu.hpp"
#include "block_adjust_gpu_kernels.hpp"
#include "../arch/gpu_base.hpp"
#include "../object_wrapper.h"
#include "../velocity_mesh_parameters.h"

using namespace std;

namespace spatial_cell {

   /*!\brief spatial_cell::update_velocity_block_content_lists Finds blocks above the sparsity threshold
    *
    * Bulk call over listed cells of spatial grid
    * Prepares the content / no-content velocity block lists
    * for all requested cells, for the requested popID
    */
void update_velocity_block_content_lists(
   dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
   const vector<CellID>& cells,
   const uint popID) {

   const uint nCells = cells.size();
   if (nCells == 0) {
      return;
   }

   // Consider mass loss evaluation?
   const bool gatherMass = getObjectWrapper().particleSpecies[popID].sparse_conserve_mass;

   const gpuStream_t baseStream = gpu_getStream();
   // Allocate buffers for GPU operations
   phiprof::Timer mallocTimer {"allocate buffers for content list analysis"};
   gpu_batch_allocate(nCells,0);
   mallocTimer.stop();

   phiprof::Timer sparsityTimer {"update Sparsity values, apply memory reservations"};
   size_t largestSizePower = 0;
   size_t largestVelMesh = 0;
#pragma omp parallel
   {
      size_t threadLargestVelMesh = 0;
      size_t threadLargestSizePower = 0;
      SpatialCell *SC;
#pragma omp for
      for (uint i=0; i<nCells; ++i) {
         SC = mpiGrid[cells[i]];
         SC->velocity_block_with_content_list_size = 0;
         SC->updateSparseMinValue(popID);

         vmesh::VelocityMesh* vmesh = SC->get_velocity_mesh(popID);
         // Make sure local vectors are large enough
         const size_t mySize = vmesh->size();
         SC->setReservation(popID,mySize);
         SC->applyReservation(popID);

         // might be better to apply reservation *after* clearing maps, but pointers might change.
         // Store values and pointers
         host_vmeshes[i] = SC->dev_get_velocity_mesh(popID);
         host_VBCs[i] = SC->dev_get_velocity_blocks(popID);
         host_minValues[i] = SC->getVelocityBlockMinValue(popID);
         host_allMaps[i] = SC->dev_velocity_block_with_content_map;
         host_allMaps[nCells+i] = SC->dev_velocity_block_with_no_content_map;
         host_vbwcl_vec[i] = SC->dev_velocity_block_with_content_list;

         // Gather largest values
         threadLargestVelMesh = threadLargestVelMesh > mySize ? threadLargestVelMesh : mySize;
         threadLargestSizePower = threadLargestSizePower > SC->vbwcl_sizePower ? threadLargestSizePower : SC->vbwcl_sizePower;
         threadLargestSizePower = threadLargestSizePower > SC->vbwncl_sizePower ? threadLargestSizePower : SC->vbwncl_sizePower;
      }
#pragma omp critical
      {
         largestVelMesh = threadLargestVelMesh > largestVelMesh ? threadLargestVelMesh : largestVelMesh;
         largestSizePower = threadLargestSizePower > largestSizePower ? threadLargestSizePower : largestSizePower;
      }
   }
   sparsityTimer.stop();

   phiprof::Timer copyTimer {"copy values to device"};
   // Copy pointers and counters over to device
   CHK_ERR( gpuMemcpyAsync(dev_allMaps, host_allMaps, 2*nCells*sizeof(Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_vbwcl_vec, host_vbwcl_vec, nCells*sizeof(split::SplitVector<vmesh::GlobalID>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_minValues, host_minValues, nCells*sizeof(Real), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_vmeshes, host_vmeshes, nCells*sizeof(vmesh::VelocityMesh*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_VBCs, host_VBCs, nCells*sizeof(vmesh::VelocityBlockContainer*), gpuMemcpyHostToDevice, baseStream) );
   if (gatherMass) {
      CHK_ERR( gpuMemsetAsync(dev_mass, 0, nCells*sizeof(Real), baseStream) );
   }
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   copyTimer.stop();

   // Batch clear all hash maps
   phiprof::Timer clearTimer {"clear all content maps"};
   const size_t largestMapSize = std::pow(2,largestSizePower);
   // fast ceil for positive ints
   //const size_t blocksNeeded = 1 + ((largestMapSize - 1) / Hashinator::defaults::MAX_BLOCKSIZE);
   size_t blocksNeeded = 1 + floor(sqrt(largestMapSize / Hashinator::defaults::MAX_BLOCKSIZE)-1);
   blocksNeeded = blocksNeeded < 1 ? 1 : blocksNeeded;
   dim3 grid1(blocksNeeded,2*nCells,1);
   batch_reset_all_to_empty<<<grid1, Hashinator::defaults::MAX_BLOCKSIZE, 0, baseStream>>>(
      dev_allMaps
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   clearTimer.stop();

   // Batch gather GID-LID-pairs into two maps (one with content, one without)
   phiprof::Timer blockKernelTimer {"update content lists kernel"};
   const uint vlasiBlocksPerWorkUnit = 1;
   // ceil int division
   const uint launchBlocks = 1 + ((largestVelMesh - 1) / vlasiBlocksPerWorkUnit);
   dim3 grid2(launchBlocks,nCells,1);
   batch_update_velocity_block_content_lists_kernel<<<grid2, (2 * vlasiBlocksPerWorkUnit * WID3), 0, baseStream>>> (
      dev_vmeshes,
      dev_VBCs,
      dev_allMaps,
      dev_minValues,
      gatherMass, // Also gathers total mass?
      dev_mass
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   blockKernelTimer.stop();

   // Extract all keys from content maps into content list
   phiprof::Timer extractKeysTimer {"extract content keys"};
   auto rule = []
      __device__(const Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *map,
                 const Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>& kval,
                 const vmesh::LocalID threshold) -> bool {
                  // This rule does not use the threshold value
                  const vmesh::GlobalID emptybucket = map->get_emptybucket();
                  const vmesh::GlobalID tombstone   = map->get_tombstone();
                  return kval.first != emptybucket && kval.first != tombstone;
               };
   // Go via launcher due to templating
   extract_GIDs_kernel_launcher<decltype(rule),vmesh::GlobalID,true>(
      dev_allMaps, // points to has_content maps
      dev_vbwcl_vec, // content list vectors, output value
      dev_contentSizes, // content list vector sizes, output value
      rule,
      dev_vmeshes, // rule_meshes, not used in this call
      dev_allMaps, // rule_maps, not used in this call
      dev_vbwcl_vec, // rule_vectors, not used in this call
      nCells,
      baseStream
      );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   extractKeysTimer.stop();

   // Update host-side size values
   phiprof::Timer blocklistTimer {"update content lists extract"};
   CHK_ERR( gpuMemcpy(host_contentSizes, dev_contentSizes, nCells*sizeof(vmesh::LocalID), gpuMemcpyDeviceToHost) );
   CHK_ERR( gpuMemcpy(host_mass, dev_mass, nCells*sizeof(Real), gpuMemcpyDeviceToHost) );
   #pragma omp parallel for
   for (uint i=0; i<nCells; ++i) {
      mpiGrid[cells[i]]->velocity_block_with_content_list_size = host_contentSizes[i];
      mpiGrid[cells[i]]->density_pre_adjust = host_mass[i]; // Only one counter per cell, but both this and adjustment are done per-pop before moving to next population.
   }
   blocklistTimer.stop();
}

void adjust_velocity_blocks_in_cells(
   dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
   const vector<CellID>& cellsToAdjust,
   const uint popID,
   bool includeNeighbors
   ) {

   int adjustPreId {phiprof::initializeTimer("Adjusting blocks Pre")};
   int adjustId {phiprof::initializeTimer("Adjusting blocks")};
   int cleanupId {phiprof::initializeTimer("Hashmap cleanup")};
   int adjustPostId {phiprof::initializeTimer("Adjusting blocks Post")};
   const gpuStream_t baseStream = gpu_getStream();
   const gpuStream_t priorityStream = gpu_getPriorityStream();
   const uint nCells = cellsToAdjust.size();
            // If we are within an acceleration substep prior to the last one,
            // it's enough to adjust blocks based on local data only, and in
            // that case we simply pass an empty list of pointers.

   //GPUTODO: make nCells last dimension of grid in dim3(*,*,nCells)?
   // Allocate buffers for GPU operations
   phiprof::Timer mallocTimer {"allocate buffers for content list analysis"};
   gpu_batch_allocate(nCells,0);

   size_t maxNeighbors = 0;
   size_t largestContentList = 0;
   size_t largestContentListNeighbors = 0;
   // Count maximum number of neighbors, largest size of content blocks
#pragma omp parallel
   {
      size_t threadMaxNeighbors = 0;
      size_t threadLargestContentList = 0;
      size_t threadLargestContentListNeighbors = 0;
#pragma omp for
      for (size_t i=0; i<nCells; ++i) {
         CellID cell_id = cellsToAdjust[i];
         SpatialCell* SC = mpiGrid[cell_id];
         if (SC->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
            continue;
         }
         threadLargestContentList = threadLargestContentList > SC->velocity_block_with_content_list_size
            ? threadLargestContentList : SC->velocity_block_with_content_list_size ;
         size_t cellLargestContentListNeighbors = 0;
         std::unordered_set<CellID> uniqueNeighbors;
         if (includeNeighbors) {
            const auto* neighbors = mpiGrid.get_neighbors_of(cell_id, Neighborhoods::NEAREST);
            // find only unique neighbor cells
            for ( const auto& [neighbor_id, dir] : *neighbors) {
               cellLargestContentListNeighbors = cellLargestContentListNeighbors > mpiGrid[neighbor_id]->velocity_block_with_content_list_size
                  ? cellLargestContentListNeighbors : mpiGrid[neighbor_id]->velocity_block_with_content_list_size;
               if (neighbor_id != cell_id) {
                  uniqueNeighbors.insert(neighbor_id);
               }
            }
         }
         uint reservationSize = SC->getReservation(popID);
         reservationSize = cellLargestContentListNeighbors > reservationSize ? cellLargestContentListNeighbors : reservationSize;
         SC->setReservation(popID,reservationSize);
         SC->applyReservation(popID);
         uint nNeighbors = uniqueNeighbors.size();
         threadMaxNeighbors = threadMaxNeighbors > nNeighbors ? threadMaxNeighbors : nNeighbors;
         threadLargestContentListNeighbors = threadLargestContentListNeighbors > cellLargestContentListNeighbors
            ? threadLargestContentListNeighbors : cellLargestContentListNeighbors;
      }
#pragma omp critical
      {
         maxNeighbors = maxNeighbors > threadMaxNeighbors ? maxNeighbors : threadMaxNeighbors;
         largestContentList = threadLargestContentList > largestContentList ? threadLargestContentList : largestContentList;
         largestContentListNeighbors = threadLargestContentListNeighbors > largestContentListNeighbors ? threadLargestContentListNeighbors : largestContentListNeighbors;
      }
   } // end parallel region

   // Early return if empty region for this population (
   // GPUTODO FIX: BREAKS VLASOV SUBSTEPPING
   // if (largestContentList==largestContentListNeighbors==0) {
   //    return;
   // }
   gpu_batch_allocate(nCells,maxNeighbors);
   mallocTimer.stop();

   size_t largestVelMesh = 0;
#pragma omp parallel
   {
      phiprof::Timer timer {adjustPreId};
      size_t threadLargestVelMesh = 0;
#pragma omp for schedule(dynamic,1)
      for (size_t i=0; i<nCells; ++i) {
         CellID cell_id=cellsToAdjust[i];
         SpatialCell* SC = mpiGrid[cell_id];
         if (SC->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
            host_vmeshes[i]=0;
            host_allMaps[i]=0;
            host_allMaps[nCells+i]=0;
            host_vbwcl_vec[i]=0;
            host_lists_with_replace_new[i]=0;
            continue;
         }

         // Gather largest mesh size for launch parameters
         vmesh::VelocityMesh* vmesh = SC->get_velocity_mesh(popID);
         threadLargestVelMesh = threadLargestVelMesh > vmesh->size() ? threadLargestVelMesh : vmesh->size();

         if (includeNeighbors) {
            // gather vector with pointers to spatial neighbor lists
            const auto* neighbors = mpiGrid.get_neighbors_of(cell_id, Neighborhoods::NEAREST);
            // Note: at AMR refinement boundaries this can cause blocks to propagate further
            // than absolutely required. Face neighbors, however, are not enough as we must
            // account for diagonal propagation.
            // This will be fixed with DCCRG new neighborhoods

            // find only unique neighbor cells
            std::unordered_set<CellID> uniqueNeighbors;
            for ( const auto& [neighbor_id, dir] : *neighbors) {
               if (neighbor_id != cell_id) {
                  uniqueNeighbors.insert(neighbor_id);
               }
            }
            std::vector<CellID> reducedNeighbors;
            reducedNeighbors.insert(reducedNeighbors.end(), uniqueNeighbors.begin(), uniqueNeighbors.end());
            const uint nNeighbors = reducedNeighbors.size();
            for (uint iN = 0; iN < maxNeighbors; ++iN) {
               if (iN >= nNeighbors) {
                  host_vbwcl_neigh[i*maxNeighbors + iN] = 0; // no neighbor at this index
                  continue;
               }
               CellID neighbor_id = reducedNeighbors.at(iN);
               // store pointer to neighbor content list
               SpatialCell* NC = mpiGrid[neighbor_id];
               if (NC->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
                  host_vbwcl_neigh[i*maxNeighbors + iN] = 0;
               } else {
                  host_vbwcl_neigh[i*maxNeighbors + iN] = mpiGrid[neighbor_id]->dev_velocity_block_with_content_list;
               }
            }
         }

         // Store values and pointers
         host_vmeshes[i] = SC->dev_get_velocity_mesh(popID);
         host_VBCs[i] = SC->dev_get_velocity_blocks(popID);
         host_allMaps[i] = SC->dev_velocity_block_with_content_map;
         host_allMaps[nCells+i] = SC->dev_velocity_block_with_no_content_map;
         host_vbwcl_vec[i] = SC->dev_velocity_block_with_content_list;
         host_lists_with_replace_new[i] = SC->dev_list_with_replace_new;
         host_lists_delete[i] = SC->dev_list_delete;
         host_lists_to_replace[i] = SC->dev_list_to_replace;
         host_lists_with_replace_old[i] = SC->dev_list_with_replace_old;
      }
      timer.stop();
#pragma omp critical
      {
         largestVelMesh = threadLargestVelMesh > largestVelMesh ? threadLargestVelMesh : largestVelMesh;
      }
   } // end parallel region

   /*
    * Perform block adjustment via batch operations
    * */
   phiprof::Timer copyTimer {"copy values to device"};
   // Copy pointers and counters over to device
   CHK_ERR( gpuMemcpyAsync(dev_allMaps, host_allMaps, 2*nCells*sizeof(Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_vbwcl_vec, host_vbwcl_vec, nCells*sizeof(split::SplitVector<vmesh::GlobalID>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_vmeshes, host_vmeshes, nCells*sizeof(vmesh::VelocityMesh*), gpuMemcpyHostToDevice, baseStream) );
   if (includeNeighbors && maxNeighbors>0) {
      CHK_ERR( gpuMemcpyAsync(dev_vbwcl_neigh, host_vbwcl_neigh, nCells*maxNeighbors*sizeof(split::SplitVector<vmesh::GlobalID>*), gpuMemcpyHostToDevice, baseStream) );
   }
   CHK_ERR( gpuMemsetAsync(dev_contentSizes, 0, 5*nCells*sizeof(vmesh::LocalID), baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_lists_with_replace_new, host_lists_with_replace_new, nCells*sizeof(split::SplitVector<vmesh::GlobalID>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_lists_delete, host_lists_delete, nCells*sizeof(split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_lists_to_replace, host_lists_to_replace, nCells*sizeof(split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_lists_with_replace_old, host_lists_with_replace_old, nCells*sizeof(split::SplitVector<Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>>*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuMemcpyAsync(dev_VBCs, host_VBCs, nCells*sizeof(vmesh::VelocityBlockContainer*), gpuMemcpyHostToDevice, baseStream) );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   copyTimer.stop();

   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> **dev_has_content_maps = dev_allMaps;
   Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> **dev_has_no_content_maps = dev_allMaps + nCells;

   // Note: Velocity halo and spatial neighbor halo can both be evaluated simultaneously.
   // Thus, we launch one into the prioritystream, the other into baseStream.

   // Evaluate velocity halo for local content blocks
   phiprof::Timer blockHaloTimer {"Block halo batch kernels"};
   const int addWidthV = getObjectWrapper().particleSpecies[popID].sparseBlockAddWidthV;
   if (addWidthV!=1) {
      std::cerr<<"Warning! "<<__FILE__<<":"<<__LINE__<<" Halo extent is not 1, unsupported size."<<std::endl;
   }
   // Halo of 1 in each direction adds up to 26 velocity neighbors.
   // For NVIDIA/CUDA, we can do 26 neighbors and 32 threads per warp in a single block.
   // For AMD/HIP, we can do 13 neighbors and 64 threads per warp in a single block, meaning two loops per cell.
   // In either case, we launch blocks equal to largest found velocity_block_with_content_list_size, which was stored
   // into largestContentList
   //const size_t blocksNeeded_velhalo = 1+floor(sqrt(largestContentList)-1);
   const size_t blocksNeeded_velhalo = largestContentList;
   dim3 grid_vel_halo(blocksNeeded_velhalo,nCells,1);
   batch_update_velocity_halo_kernel<<<grid_vel_halo, 26*32, 0, priorityStream>>> (
      dev_vmeshes,
      dev_vbwcl_vec,
      dev_allMaps // Needs both content and no content maps
      );
   CHK_ERR( gpuPeekAtLastError() );
   // CHK_ERR( gpuStreamSynchronize(priorityStream) );
   if (includeNeighbors && maxNeighbors>1) {
      // ceil int division
      // largestContentListNeighbors accounts for remote (ghost neighbor) content list sizes as well
      //const size_t NeighLaunchBlocks = 1 + floor(sqrt(largestContentListNeighbors / WARPSPERBLOCK)-1);
      const size_t NeighLaunchBlocks = 1 + ((largestContentListNeighbors - 1) / WARPSPERBLOCK);
      dim3 grid_neigh_halo(NeighLaunchBlocks,nCells,maxNeighbors);
      // For NVIDIA/CUDA, we can do 32 neighbor GIDs and 32 threads per warp in a single block.
      // For AMD/HIP, we can do 16 neighbor GIDs and 64 threads per warp in a single block
      // This is handled in-kernel.
      batch_update_neighbour_halo_kernel<<<grid_neigh_halo, WARPSPERBLOCK*GPUTHREADS, 0, baseStream>>> (
         dev_vmeshes,
         dev_allMaps, // Needs both has_content and has_no_content maps
         dev_vbwcl_neigh
         );
      CHK_ERR( gpuPeekAtLastError() );
   }
   // Sync both streams
   CHK_ERR( gpuDeviceSynchronize() );
   blockHaloTimer.stop();

   /**
       Now extract vectors to be used in actual block adjustment.
       Previous kernels may have added dummy (to be added) entries to
       velocity_block_with_content_map with LID=vmesh->invalidLocalID()
       Or non-content blocks which should be retained (with correct LIDs).

       Note: These batch operations always include deletion of not-needed blocks.

       Rules used in extracting keys or elements from hashmaps:
       These are provided with the value of nBlocksAfterAdjust as the
       threshold argument by the kernel. To this end, rule_meshes, rule_maps,
       and rule_vectors pointer buffers are provided to the kernels.
   */
   phiprof::Timer extractKeysTimer {"extract content keys"};
   const vmesh::GlobalID invalidGID  = host_vmeshes[0]->invalidGlobalID();
   const vmesh::LocalID  invalidLID  = host_vmeshes[0]->invalidLocalID();

   auto rule_add = [invalidGID, invalidLID]
      __device__(const Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *map,
                 const Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>& kval,
                 const vmesh::LocalID threshold) -> bool {
                      // This rule does not use the threshold value
                      const vmesh::GlobalID emptybucket = map->get_emptybucket();
                      const vmesh::GlobalID tombstone   = map->get_tombstone();
                      return kval.first != emptybucket &&
                         kval.first != tombstone &&
                         kval.first != invalidGID &&
                         // Required GIDs which do not yet exist in vmesh were stored in
                         // velocity_block_with_content_map with kval.second==invalidLID
                         kval.second == invalidLID;
                   };
   auto rule_delete_move = [invalidGID, invalidLID]
      __device__(const Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *map,
                 const Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>& kval,
                 const vmesh::LocalID threshold) -> bool {
                              const vmesh::GlobalID emptybucket = map->get_emptybucket();
                              const vmesh::GlobalID tombstone   = map->get_tombstone();
                              return kval.first != emptybucket &&
                                 kval.first != tombstone &&
                                 kval.first != invalidGID &&
                                 kval.second >= threshold &&
                                 kval.second != invalidLID;
                           };
   auto rule_to_replace = [invalidGID, invalidLID]
      __device__(const Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *map,
                 const Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>& kval,
                 const vmesh::LocalID threshold) -> bool {
                             const vmesh::GlobalID emptybucket = map->get_emptybucket();
                             const vmesh::GlobalID tombstone   = map->get_tombstone();
                             return kval.first != emptybucket &&
                                kval.first != tombstone &&
                                kval.first != invalidGID &&
                                kval.second < threshold &&
                                              kval.second != invalidGID;
                          };

   // Go via launcher due to templating. Templating manages rule lambda type,
   // output vector type, as well as a flag whether the output vector should take the whole
   // element from the map, or just the first of the pair.

   // Finds new Blocks (GID,LID) needing to be added
   // Note:list_with_replace_new then contains both new GIDs to use for replacements and new GIDs to place at end of vmesh
   extract_GIDs_kernel_launcher<decltype(rule_add),vmesh::GlobalID,true>(
      dev_has_content_maps, // input maps
      dev_lists_with_replace_new, // output vecs
      dev_contentSizes, // GPUTODO: add flag which either sets, adds, or subtracts the final size from this buffer.
      rule_add,
      dev_vmeshes, // rule_meshes, not used in this call
      dev_has_no_content_maps, // rule_maps, not used in this call
      dev_vbwcl_vec, // rule_vectors, not used in this call
      nCells,
      baseStream
      ); // This needs to complete before the next 3 extractions
   // Finds Blocks (GID,LID) to be rescued from end of v-space
   extract_GIDs_kernel_launcher<decltype(rule_delete_move),Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>,false>(
      dev_has_content_maps, // input maps
      dev_lists_with_replace_old, // output vecs
      dev_contentSizes, // GPUTODO: add flag which either sets, adds, or subtracts the final size from this buffer.
      rule_delete_move,
      dev_vmeshes, // rule_meshes
      dev_has_no_content_maps, // rule_maps
      dev_lists_with_replace_new, // rule_vectors
      nCells,
      baseStream
      );
   // Find Blocks (GID,LID) to be outright deleted
   extract_GIDs_kernel_launcher<decltype(rule_delete_move),Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>,false>(
      dev_has_no_content_maps, // input maps
      dev_lists_delete, // output vecs
      dev_contentSizes, // GPUTODO: add flag which either sets, adds, or subtracts the final size from this buffer.
      rule_delete_move,
      dev_vmeshes, // rule_meshes
      dev_has_no_content_maps, // rule_maps
      dev_lists_with_replace_new, // rule_vectors
      nCells,
      baseStream
      );
   // Find Blocks (GID,LID) to be replaced with new ones
   extract_GIDs_kernel_launcher<decltype(rule_to_replace),Hashinator::hash_pair<vmesh::GlobalID,vmesh::LocalID>,false>(
      dev_has_no_content_maps, // input maps
      dev_lists_to_replace, // output vecs
      dev_contentSizes, // GPUTODO: add flag which either sets, adds, or subtracts the final size from this buffer.
      rule_to_replace,
      dev_vmeshes, // rule_meshes
      dev_has_no_content_maps, // rule_maps
      dev_lists_with_replace_new, // rule_vectors
      nCells,
      baseStream
      );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   extractKeysTimer.stop();
   // Resizes are faster this way with larger grid and single thread
   phiprof::Timer deviceResizeTimer {"GPU resize mesh on-device"};
   batch_resize_vbc_kernel_pre<<<nCells, 1, 0, baseStream>>> (
      dev_vmeshes,
      dev_VBCs,
      dev_lists_with_replace_new,
      dev_lists_delete,
      dev_lists_to_replace,
      dev_lists_with_replace_old,
      dev_contentSizes, // content list vector sizes, output value
      dev_massLoss // mass loss, set to zero
      );
   CHK_ERR( gpuPeekAtLastError() );
   CHK_ERR( gpuMemcpyAsync(host_contentSizes, dev_contentSizes, nCells*4*sizeof(vmesh::LocalID), gpuMemcpyDeviceToHost, baseStream) );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   deviceResizeTimer.stop();

   phiprof::Timer hostResizeTimer {"GPU resize mesh from host "};
   uint largestBlocksToChange = 0;
   uint largestBlocksBeforeOrAfter = 0;
#pragma omp parallel
   {
      uint thread_largestBlocksToChange = 0;
      uint thread_largestBlocksBeforeOrAfter = 0;
#pragma omp for schedule(dynamic,1)
      for (size_t i=0; i<nCells; ++i) {
         SpatialCell* SC = mpiGrid[cellsToAdjust[i]];
         if (SC->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
            continue;
         }
         // Grow mesh if necessary and on-device resize did not work??
         const vmesh::LocalID nBlocksBeforeAdjust = host_contentSizes[i*4 + 0];
         const vmesh::LocalID nBlocksAfterAdjust  = host_contentSizes[i*4 + 1];
         const vmesh::LocalID nBlocksToChange     = host_contentSizes[i*4 + 2];
         const vmesh::LocalID resizeDevSuccess    = host_contentSizes[i*4 + 3];
         thread_largestBlocksToChange = thread_largestBlocksToChange > nBlocksToChange ? thread_largestBlocksToChange : nBlocksToChange;
         // This is gathered for mass loss correction: for each cell, we want the smaller of either blocks before or after. Then,
         // we want to gather the largest of those values.
         const vmesh::LocalID lowBlocks = nBlocksBeforeAdjust < nBlocksAfterAdjust ? nBlocksBeforeAdjust : nBlocksAfterAdjust;
         thread_largestBlocksBeforeOrAfter = thread_largestBlocksBeforeOrAfter > lowBlocks ? thread_largestBlocksBeforeOrAfter : lowBlocks;
         if ( (nBlocksAfterAdjust > nBlocksBeforeAdjust) && (resizeDevSuccess == 0)) {
            //GPUTODO is _FACTOR enough instead of _PADDING?
            SC->get_velocity_mesh(popID)->setNewCapacity(nBlocksAfterAdjust*BLOCK_ALLOCATION_PADDING);
            SC->get_velocity_mesh(popID)->setNewSize(nBlocksAfterAdjust);
            SC->get_velocity_blocks(popID)->setNewCapacity(nBlocksAfterAdjust*BLOCK_ALLOCATION_PADDING);
            SC->get_velocity_blocks(popID)->setNewSize(nBlocksAfterAdjust);
            SC->dev_upload_population(popID);
         }
      } // end cell loop
#pragma omp critical
      {
         largestBlocksToChange = thread_largestBlocksToChange > largestBlocksToChange ? thread_largestBlocksToChange : largestBlocksToChange;
         largestBlocksBeforeOrAfter = largestBlocksBeforeOrAfter > thread_largestBlocksBeforeOrAfter ? largestBlocksBeforeOrAfter : thread_largestBlocksBeforeOrAfter;
      }
   } // end parallel region
   CHK_ERR( gpuDeviceSynchronize() );
   hostResizeTimer.stop();

   // Do we actually have any changes to perform?
   if (largestBlocksToChange > 0) {
      phiprof::Timer addRemoveKernelTimer {"GPU batch add and remove blocks kernel"};
      // Each GPU block / workunit could manage several Vlasiator velocity blocks at once.
      // However, thread syncs inside the kernel prevent this.
      // const uint vlasiBlocksPerWorkUnit = WARPSPERBLOCK * GPUTHREADS / WID3;
      const uint vlasiBlocksPerWorkUnit = 1;
      // ceil int division
      const uint blocksNeeded = 1 + ((largestBlocksToChange - 1) / vlasiBlocksPerWorkUnit);
      // Third argument specifies the number of bytes in *shared memory* that is
      // dynamically allocated per block for this call in addition to the statically allocated memory.
      dim3 grid_addremove(blocksNeeded,nCells,1);
      batch_update_velocity_blocks_kernel<<<grid_addremove, vlasiBlocksPerWorkUnit * WID3, 0, baseStream>>> (
         dev_vmeshes,
         dev_VBCs,
         dev_lists_with_replace_new,
         dev_lists_delete,
         dev_lists_to_replace,
         dev_lists_with_replace_old,
         dev_contentSizes, // return values: nbefore, nafter, nblockstochange, (resize success)
         dev_massLoss
         );
      CHK_ERR( gpuPeekAtLastError() );
      // Pull mass loss values to host
      CHK_ERR( gpuMemcpyAsync(host_massLoss, dev_massLoss, nCells*sizeof(Real), gpuMemcpyDeviceToHost, baseStream) );
      CHK_ERR( gpuStreamSynchronize(baseStream) );
      addRemoveKernelTimer.stop();

      // Should not re-allocate on shrinking, so do on-device
      phiprof::Timer deviceResizePostTimer {"GPU resize mesh on-device post"};
      // Resizes are faster this way with larger grid and single thread
      batch_resize_vbc_kernel_post<<<nCells, 1, 0, baseStream>>> (
         dev_vmeshes,
         dev_VBCs,
         dev_contentSizes
         );
      CHK_ERR( gpuPeekAtLastError() );
      CHK_ERR( gpuStreamSynchronize(baseStream) );
      deviceResizePostTimer.stop();
   }

   /* Batch tombstone cleaning
    * Extract all entries (GID,LID) which are overflown (see Hashinator for further details). At same time,
    * remove tombstones and overflown elements.
    * 
    * By calling a few kernels which operate over all spatial cells at once instead of launching a few kernels per cell,
    * we reduce operational time by circa 10x.
    */
   phiprof::Timer tombstoneTimer {"GPU batch clean tombstones"};
   auto rule_overflown = []
      __device__(Hashinator::Hashmap<vmesh::GlobalID,vmesh::LocalID> *map,
                 Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID>& kval) -> bool {
                            const vmesh::GlobalID emptybucket = map->get_emptybucket();
                            const vmesh::GlobalID tombstone   = map->get_tombstone();
                            if (kval.first == emptybucket) {
                               return false;
                            }
                            if (kval.first == tombstone) {
                               // Note: tombstones preceding overflown are deleted, so
                               // resetting overflown elements after this cannot rely on
                               // tombstones.
                               kval.first = emptybucket;
                               return false;
                            }
                            const size_t currentSizePower = map->getSizePower();
                            Hashinator::hash_pair<vmesh::GlobalID, vmesh::LocalID> *bck_ptr = map->expose_bucketdata<false>();
                            //const size_t hashIndex = Hashinator::HashFunction::_hash(kval.first, currentSizePower);
                            const size_t hashIndex = map->hash(kval.first);
                            const int bitMask = (1 << (currentSizePower)) - 1;
                            const bool isOverflown = (bck_ptr[hashIndex & bitMask].first != kval.first);
                            return isOverflown;
                         };
   clean_tombstones_launcher<decltype(rule_overflown)>(
      dev_vmeshes, // velocity meshes which include the hash maps to clean
      dev_lists_with_replace_old, // use this for storing overflown elements
      dev_contentSizes + 4*nCells, // return values: n_overflown_elements
      rule_overflown,
      nCells,
      baseStream
      );
   // Re-insert overflown elements back in vmeshes. First calculate
   // Launch parameters after using blocking memcpy to get overflow counts
   CHK_ERR( gpuMemcpyAsync(host_contentSizes+4*nCells, dev_contentSizes+4*nCells, nCells*sizeof(vmesh::LocalID), gpuMemcpyDeviceToHost, baseStream) );
   CHK_ERR( gpuStreamSynchronize(baseStream) );
   uint largestOverflow = 0;
#pragma omp parallel
   {
      uint thread_largestOverflow = 0;
#pragma omp for
      for (size_t i=0; i<nCells; ++i) {
         thread_largestOverflow = thread_largestOverflow > host_contentSizes[4*nCells+i] ? thread_largestOverflow : host_contentSizes[4*nCells+i];
      }
#pragma omp critical
      {
         largestOverflow = thread_largestOverflow > largestOverflow ? thread_largestOverflow : largestOverflow;
      }
   } // end parallel region
   if (largestOverflow > 0) {
      dim3 grid_reinsert(largestOverflow,nCells,1);
      batch_insert_kernel<<<grid_reinsert, GPUTHREADS, 0, baseStream>>>(
         dev_vmeshes, // velocity meshes which include the hash maps to clean
         dev_lists_with_replace_old // use this for storing overflown elements
         );
      CHK_ERR( gpuPeekAtLastError() );
      CHK_ERR( gpuStreamSynchronize(baseStream) );
   }
   tombstoneTimer.stop();

#pragma omp parallel
   {
#pragma omp for schedule(dynamic,1)
      for (size_t i=0; i<nCells; ++i) {
         SpatialCell* SC = mpiGrid[cellsToAdjust[i]];
         if (SC->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
            SC->get_velocity_mesh(popID)->setNewCachedSize(0);
            SC->get_velocity_blocks(popID)->setNewCachedSize(0);
            continue;
         }
         // Update vmesh cached size and mass Loss
         SC->get_velocity_mesh(popID)->setNewCachedSize(host_contentSizes[i*4 + 1]);
         SC->get_velocity_blocks(popID)->setNewCachedSize(host_contentSizes[i*4 + 1]);
         SC->increment_mass_loss(popID, host_massLoss[i]);

         // Perform hashmap cleanup here (instead of at acceleration mid-steps)
         phiprof::Timer cleanupTimer {cleanupId};
         //SC->get_velocity_mesh(popID)->gpu_cleanHashMap(gpu_getStream());
         //SC->dev_upload_population(popID);
         cleanupTimer.stop();

         phiprof::Timer postTimer {adjustPostId};
         #ifdef DEBUG_SPATIAL_CELL
         // Not re-doing old debug here, this should be enough
         SC->checkSizes(popID);
         #endif
         #ifdef DEBUG_VLASIATOR
         // This is a bit extreme
         SC->checkMesh(popID);
         #endif

         if (getObjectWrapper().particleSpecies[popID].sparse_conserve_mass) {
            // Block adjustment can only add empty blocks or delete existing blocks,
            // So post_adjust density must be equal to pre_adjust density minus mass loss.
            SC->density_post_adjust = SC->density_pre_adjust - host_massLoss[i];
            if ( (SC->density_post_adjust > 0.0) && (host_massLoss[i] != 0) ) {
               //SC->scale_population(SC->density_pre_adjust/SC->density_post_adjust, popID);
               // Now use the massloss buffer for the scaling value
               const Real mass_scaling = SC->density_pre_adjust/SC->density_post_adjust;
               host_massLoss[i] = mass_scaling;
            } else {
               // Skip scaling this cell
               host_massLoss[i] = 0;
            }
         } // end if conserve mass
         postTimer.stop();
      } // end cell loop
   } // end parallel region

   if ( (getObjectWrapper().particleSpecies[popID].sparse_conserve_mass)
        && (largestBlocksToChange > 0) ) {
      phiprof::Timer massConservationTimer {"GPU batch conserve mass"};
      CHK_ERR( gpuMemcpyAsync(dev_massLoss, host_massLoss, nCells*sizeof(Real), gpuMemcpyHostToDevice, baseStream) );
      // Each GPU block / workunit could manage several Vlasiator velocity blocks at once.
      // const uint vlasiBlocksPerWorkUnit = WARPSPERBLOCK * GPUTHREADS / WID3;
      const uint vlasiBlocksPerWorkUnit = 1;
      // Launch parameters: Although post-adjustment, some VBCs can have more blocks than when entering
      // block adjustment, any new blocks will be empty and thus do not need to be scaled. Thus, we can use
      // The count which is the gathered max value over all cells of a counter which is either blocksBeforeAdjust
      // or BlocksAfterAdjust, whichever is smaller.
      // ceil int division
      const uint blocksNeeded = 1 + ((largestBlocksBeforeOrAfter - 1) / vlasiBlocksPerWorkUnit);
      // Third argument specifies the number of bytes in *shared memory* that is
      // dynamically allocated per block for this call in addition to the statically allocated memory.
      dim3 grid_mass_conservation(blocksNeeded,nCells,1);
      batch_population_scale_kernel<<<grid_mass_conservation, 0, 0, baseStream>>> (
         dev_VBCs,
         dev_massLoss // used now for scaling parameter
         );
      CHK_ERR( gpuPeekAtLastError() );
      CHK_ERR( gpuStreamSynchronize(baseStream) );
   }
}

} // namespace
