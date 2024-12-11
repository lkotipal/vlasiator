/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
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

/*!\file sysboundarycondition.cpp
 * \brief Implementation of the base class SysBoundaryCondition to handle system boundary cells.
 * 
 * \sa donotcompute.cpp ionosphere.cpp outflow.cpp setbyuser.cpp setmaxwellian.cpp
 * 
 */

#include <cstdlib>
#include <iostream>

#include "../parameters.h"
#include "../vlasovsolver/vlasovmover.h"
#include "sysboundarycondition.h"
#include "../projects/projects_common.h"
#include "../object_wrapper.h"

using namespace std;

namespace SBC {
   // ************************************************************
   // ***** DEFINITIONS FOR BOUNDARYCONDITION BASE CLASS *****
   // ************************************************************
   
   /*!\brief Function used to determine on which face(s) if any the cell at given coordinates is.
    * 
    * This function is used by some of the classes inheriting from this base class.
    * 
    * Depth is hard-coded to be 2 as other parts of the code (field solver especially) rely on that.
    * 
    * \param isThisCellOnAFace Pointer to an array of 6 bool returning of each face whether the cell is on that face. Order: 0 x+; 1 x-; 2 y+; 3 y-; 4 z+; 5 z-
    * \param x Cell x coordinate
    * \param y Cell y coordinate
    * \param z Cell z coordinate
    * \param dx Cell dx size
    * \param dy Cell dy size
    * \param dz Cell dz size
    * \param excludeSlicesAndPeriodicDimensions If true, do not consider a cell to be part of the face if that face has a depth of 1 only (single-cell thick slices/columns) or if that direciton is periodic..
    */
   void SysBoundaryCondition::determineFace(
      bool* isThisCellOnAFace,
      creal x, creal y, creal z,
      creal dx, creal dy, creal dz,
      const bool excludeSlicesAndPeriodicDimensions //=false (default)
   ) {
      for(uint i=0; i<6; i++) {
         isThisCellOnAFace[i] = false;
      }
      if(x > Parameters::xmax - dx * 2) {
         isThisCellOnAFace[0] = true;
      }
      if(x < Parameters::xmin + dx * 2) {
         isThisCellOnAFace[1] = true;
      }
      if(y > Parameters::ymax - dy * 2) {
         isThisCellOnAFace[2] = true;
      }
      if(y < Parameters::ymin + dy * 2) {
         isThisCellOnAFace[3] = true;
      }
      if(z > Parameters::zmax - dz * 2) {
         isThisCellOnAFace[4] = true;
      }
      if(z < Parameters::zmin + dz * 2) {
         isThisCellOnAFace[5] = true;
      }
      if(excludeSlicesAndPeriodicDimensions == true) {
         if (Parameters::xcells_ini == 1 || this->periodic[0]) {
            isThisCellOnAFace[0] = false;
            isThisCellOnAFace[1] = false;
         }
         if (Parameters::ycells_ini == 1 || this->periodic[1]) {
            isThisCellOnAFace[2] = false;
            isThisCellOnAFace[3] = false;
         }
         if (Parameters::zcells_ini == 1 || this->periodic[2]) {
            isThisCellOnAFace[4] = false;
            isThisCellOnAFace[5] = false;
         }
      }
   }

   /*!\brief Function used to determine on which face(s) if any the cell with the given MPI id is on
    * 
    * This function is used by some of the classes inheriting from this base class.
    * 
    * Depth is hard-coded to be 2 as other parts of the code (field solver especially) rely on that.
    * So if a cell has less than two unique neighbors in some direction, it is considered to be on that face.
    * 
    * \param isThisCellOnAFace Referemce to an std::array of 6 bool returning of each face whether the cell is on that face. Order: 0 x+; 1 x-; 2 y+; 3 y-; 4 z+; 5 z-
    * \param mpiGrid Reference to grid
    * \param id ID of cell to check
    * \param excludeSlicesAndPeriodicDimensions If true, do not consider a cell to be part of the face if that face has a depth of 1 only (single-cell thick slices/columns) or if that direciton is periodic..
    */
   void SysBoundaryCondition::determineFace(
      std::array<bool, 6> &isThisCellOnAFace,
      const dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      CellID id,
      const bool excludeSlicesAndPeriodicDimensions //=false (default)
   ) {
      isThisCellOnAFace.fill(false);

      std::array<std::set<CellID>, 6> dirNeighbors;
      auto* p = mpiGrid.get_neighbors_of(id, VLASOV_SOLVER_NEIGHBORHOOD_ID);
      if (!p) {
         std::cerr << "No neighbors found for " << id << std::endl;
         return;
      }
      int nbrs {0};
      for (auto pair : *p) {
         if (!pair.first) {
            continue;   // Error cells should obviously not be counted
         } else if (pair.second[0] > 0) {
            dirNeighbors[0].insert(pair.first);
         } else if (pair.second[0] < 0) {
            dirNeighbors[1].insert(pair.first);
         } else if (pair.second[1] > 0) {
            dirNeighbors[2].insert(pair.first);
         } else if (pair.second[1] < 0) {
            dirNeighbors[3].insert(pair.first);
         } else if (pair.second[2] > 0) {
            dirNeighbors[4].insert(pair.first);
         } else if (pair.second[2] < 0) {
            dirNeighbors[5].insert(pair.first);
         }
         ++nbrs;
         //std::cerr << pair.second[0] << " " << pair.second[1] << " " << pair.second[2] << std::endl;
      }

      for (int i = 0; i < 6; ++i) {
         if(dirNeighbors[i].size() < 2) {
            isThisCellOnAFace[i] = true;
         }
      }

      if(excludeSlicesAndPeriodicDimensions == true) {
         if(Parameters::xcells_ini == 1 || this->periodic[0]) {
            isThisCellOnAFace[0] = false;
            isThisCellOnAFace[1] = false;
         }
         if(Parameters::ycells_ini == 1 || this->periodic[1]) {
            isThisCellOnAFace[2] = false;
            isThisCellOnAFace[3] = false;
         }
         if(Parameters::zcells_ini == 1 || this->periodic[2]) {
            isThisCellOnAFace[4] = false;
            isThisCellOnAFace[5] = false;
         }
      }
      return;
   }
   
   /*! SysBoundaryCondition base class constructor. The constructor is empty.*/
   SysBoundaryCondition::SysBoundaryCondition() { }
   
   /*! SysBoundaryCondition base class virtual destructor. The destructor is empty.*/
   SysBoundaryCondition::~SysBoundaryCondition() { }
   
   /*! SysBoundaryCondition base class instance of the addParameters function. Should not be used, each derived class should have its own.*/
   void SysBoundaryCondition::addParameters() {
      cerr << "ERROR: SysBoundaryCondition::addParameters called instead of derived class function!" << endl;
   }
   
   /*! Function used to set the system boundary condition cell's derivatives to 0.
    * \param mpiGrid Grid
    * \param cellID The cell's ID.
    * \param component 0: x-derivatives, 1: y-derivatives, 2: z-derivatives, 3: xy-derivatives, 4: xz-derivatives, 5: yz-derivatives.
    */
   void SysBoundaryCondition::setCellDerivativesToZero(
      FsGrid< array<Real, fsgrids::dperb::N_DPERB>, FS_STENCIL_WIDTH> & dPerBGrid,
      FsGrid< array<Real, fsgrids::dmoments::N_DMOMENTS>, FS_STENCIL_WIDTH> & dMomentsGrid,
      cint i,
      cint j,
      cint k,
      cuint component
   ) {
      array<Real, fsgrids::dperb::N_DPERB> * dPerBGrid0 = dPerBGrid.get(i,j,k);
      array<Real, fsgrids::dmoments::N_DMOMENTS> * dMomentsGrid0 = dMomentsGrid.get(i,j,k);
      switch(component) {
         case 0: // x, xx
            dMomentsGrid0->at(fsgrids::dmoments::drhomdx) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::drhoqdx) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dp11dx) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dp22dx) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dp33dx) = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBydx)  = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBzdx)  = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dVxdx)  = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dVydx)  = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dVzdx)  = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBydxx) = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBzdxx) = 0.0;
            break;
         case 1: // y, yy
            dMomentsGrid0->at(fsgrids::dmoments::drhomdy) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::drhoqdy) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dp11dy) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dp22dy) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dp33dy) = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBxdy)  = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBzdy)  = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dVxdy)  = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dVydy)  = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dVzdy)  = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBxdyy) = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBzdyy) = 0.0;
            break;
         case 2: // z, zz
            dMomentsGrid0->at(fsgrids::dmoments::drhomdz) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::drhoqdz) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dp11dz) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dp22dz) = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dp33dz) = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBxdz)  = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBydz)  = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dVxdz)  = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dVydz)  = 0.0;
            dMomentsGrid0->at(fsgrids::dmoments::dVzdz)  = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBxdzz) = 0.0;
            dPerBGrid0->at(fsgrids::dperb::dPERBydzz) = 0.0;
            break;
         case 3: // xy
            dPerBGrid0->at(fsgrids::dperb::dPERBzdxy) = 0.0;
            break;
         case 4: // xz
            dPerBGrid0->at(fsgrids::dperb::dPERBydxz) = 0.0;
            break;
         case 5: // yz
            dPerBGrid0->at(fsgrids::dperb::dPERBxdyz) = 0.0;
            break;
         default:
            cerr << __FILE__ << ":" << __LINE__ << ":" << " Invalid component" << endl;
            abort_mpi("Invalid component", 1);
      }
   }
   
   /*! Function used to set the system boundary condition cell's BVOL derivatives to 0.
    * \param mpiGrid Grid
    * \param cellID The cell's ID.
    * \param component 0: x-derivatives, 1: y-derivatives, 2: z-derivatives.
    */
   void SysBoundaryCondition::setCellBVOLDerivativesToZero(
      FsGrid< array<Real, fsgrids::volfields::N_VOL>, FS_STENCIL_WIDTH> & volGrid,
      cint i,
      cint j,
      cint k,
      cuint component
   ) {
      array<Real, fsgrids::volfields::N_VOL> * volGrid0 = volGrid.get(i,j,k);
      switch(component) {
         case 0:
            volGrid0->at(fsgrids::volfields::dPERBXVOLdx) = 0.0;
            volGrid0->at(fsgrids::volfields::dPERBYVOLdx) = 0.0;
            volGrid0->at(fsgrids::volfields::dPERBZVOLdx) = 0.0;
            break;
         case 1:
            volGrid0->at(fsgrids::volfields::dPERBXVOLdy) = 0.0;
            volGrid0->at(fsgrids::volfields::dPERBYVOLdy) = 0.0;
            volGrid0->at(fsgrids::volfields::dPERBZVOLdy) = 0.0;
            break;
         case 2:
            volGrid0->at(fsgrids::volfields::dPERBXVOLdz) = 0.0;
            volGrid0->at(fsgrids::volfields::dPERBYVOLdz) = 0.0;
            volGrid0->at(fsgrids::volfields::dPERBZVOLdz) = 0.0;
            break;
         default:
         abort_mpi("Invalid component", 1);
      }
   }
   
   /*! Function used to copy the distribution and moments from (one of) the closest sysboundarytype::NOT_SYSBOUNDARY cell.
    * \param mpiGrid Grid
    * \param cellID The cell's ID.
    * \param copyMomentsOnly If true, do not touch velocity space.
    * \param popID ID of the particle species.
    * \param copy_V_moments which set of moments to copy (_V or _R)
    */
   void SysBoundaryCondition::vlasovBoundaryCopyFromTheClosestNbr(
         dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
         const CellID& cellID,
         const bool& copyMomentsOnly,
         const uint popID,
         const bool copy_V_moments
   ) {
      const CellID closestCell = getTheClosestNonsysboundaryCell(cellID);
      
      if(closestCell == INVALID_CELLID) {
         abort_mpi("No closest cell found!", 1);
      }
      phiprof::Timer boundaryTimer {"vlasovBoundaryCopyFromTheClosestNbr"};
      copyCellData(mpiGrid[closestCell],mpiGrid[cellID], copyMomentsOnly, popID, copy_V_moments);
      boundaryTimer.stop();
   }
   
   /*! Function used to average and copy the distribution and moments from all the closest sysboundarytype::NOT_SYSBOUNDARY cells.
    * \param mpiGrid Grid
    * \param cellID The cell's ID.
    * \param popID ID of the particle species.
    * \param copy_V_moments which set of moments to copy (_V or _R)
    */
   void SysBoundaryCondition::vlasovBoundaryCopyFromAllClosestNbrs(
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const CellID& cellID,const uint popID, const bool copy_V_moments
   ) {
      const vector<CellID>& closestCells = getAllClosestNonsysboundaryCells(cellID);
      
      if(closestCells[0] == INVALID_CELLID) {
         abort_mpi("No closest cell found!", 1);
      }
      phiprof::Timer boundaryTimer {"vlasovBoundaryCopyFromAllClosestNbrs"};
      averageCellData(mpiGrid, closestCells, mpiGrid[cellID], popID);
      boundaryTimer.stop();
   }
   
   /*! Function used to average and copy the distribution and moments from all the close sysboundarytype::NOT_SYSBOUNDARY cells.
    * \param mpiGrid Grid
    * \param cellID The cell's ID.
    * \param popID ID of the particle species.
    * \param copy_V_moments which set of moments to copy (_V or _R)
    * \patam fluffiness (parameter from 0 to 1; value 0 maintains original value, 1 replaces it completely with average)
    */
   void SysBoundaryCondition::vlasovBoundaryFluffyCopyFromAllCloseNbrs(
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const CellID& cellID,const uint popID,const bool copy_V_moments, creal fluffiness
   ) {
      const vector<CellID>& closeCells = getAllCloseNonsysboundaryCells(cellID);
      
      if(closeCells[0] == INVALID_CELLID) {
         abort_mpi("No close cell found!", 1);
      }
      phiprof::Timer boundaryTimer {"vlasovBoundaryFluffyCopyFromAllCloseNbrs"};
      averageCellData(mpiGrid, closeCells, mpiGrid[cellID], popID, fluffiness);
      boundaryTimer.stop();
   }
   
   /*! Function used to copy the distribution and moments from one cell to another.
    * \param from Pointer to parent cell to copy from.
    * \param to Pointer to destination cell.
    * \param copyMomentsOnly skip copying of VDF
    * \param popID ID of the particle species.
    * \param copy_V_moments which set of moments to copy (_V or _R)
    */
   void SysBoundaryCondition::copyCellData(
            SpatialCell* from,
            SpatialCell* to,
            const bool copyMomentsOnly,
            const uint popID,
            const bool copy_V_moments
   ) {
      if (popID == 0) {
         if (copy_V_moments) {
            to->parameters[CellParams::RHOM_V] = from->parameters[CellParams::RHOM_V];
            to->parameters[CellParams::VX_V] = from->parameters[CellParams::VX_V];
            to->parameters[CellParams::VY_V] = from->parameters[CellParams::VY_V];
            to->parameters[CellParams::VZ_V] = from->parameters[CellParams::VZ_V];
            to->parameters[CellParams::RHOQ_V] = from->parameters[CellParams::RHOQ_V];
            to->parameters[CellParams::P_11_V] = from->parameters[CellParams::P_11_V];
            to->parameters[CellParams::P_22_V] = from->parameters[CellParams::P_22_V];
            to->parameters[CellParams::P_33_V] = from->parameters[CellParams::P_33_V];
         } else {
            to->parameters[CellParams::RHOM_R] = from->parameters[CellParams::RHOM_R];
            to->parameters[CellParams::VX_R] = from->parameters[CellParams::VX_R];
            to->parameters[CellParams::VY_R] = from->parameters[CellParams::VY_R];
            to->parameters[CellParams::VZ_R] = from->parameters[CellParams::VZ_R];
            to->parameters[CellParams::RHOQ_R] = from->parameters[CellParams::RHOQ_R];
            to->parameters[CellParams::P_11_R] = from->parameters[CellParams::P_11_R];
            to->parameters[CellParams::P_22_R] = from->parameters[CellParams::P_22_R];
            to->parameters[CellParams::P_33_R] = from->parameters[CellParams::P_33_R];
         }
      }
      
      if (!copyMomentsOnly) { // Do this only if copyMomentsOnly is false.
         to->set_population(from->get_population(popID), popID);
      } else {
         if (copy_V_moments) {
            to->get_population(popID).RHO_V = from->get_population(popID).RHO_V;
         } else {
            to->get_population(popID).RHO_R = from->get_population(popID).RHO_R;
         }
         
         for (uint i=0; i<3; i++) {
            if (copy_V_moments) {
               to->get_population(popID).V_V[i] = from->get_population(popID).V_V[i];
               to->get_population(popID).P_V[i] = from->get_population(popID).P_V[i];
            } else {
               to->get_population(popID).V_R[i] = from->get_population(popID).V_R[i];
               to->get_population(popID).P_R[i] = from->get_population(popID).P_R[i];
            }
         }
      }
   }
   
   /*! Take a list of cells and set the destination cell distribution function to the average of the list's cells'.
    * \param mpiGrid Grid
    * \param cellList Vector of cells to copy from.
    * \param to Pointer to cell in which to set the averaged distribution.
    * \param popID ID of population to average the distribution of
    * \param fluffiness Factor to replace data with from 0.0 (default, do nothing) to 1.0 (replace all destination data with source data)
    */
   void averageCellData(
         dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
         const vector<CellID> cellList,
         SpatialCell *to,
         const uint popID,
         creal fluffiness /* default =0.0*/
   ) {
      const size_t numberOfCells = cellList.size();
      creal factor = fluffiness / convert<Real>(numberOfCells);

      // Rescale own vspace
      if (fluffiness != 0.0) {
         to->scale_population(1.0 - fluffiness, popID);
      }
      // Add other cell vspaces
      for (size_t i=0; i<numberOfCells; i++) {
         const SpatialCell* from = mpiGrid[cellList[i]];
         to->increment_population(from->get_population(popID), factor, popID);
      }
   }

   /*! Updates the system boundary conditions after load balancing. This is called from e.g. the class SysBoundary.
    * \param mpiGrid Grid
    * \param local_cells_on_boundary Cells within this process
    * \retval success Returns true if the operation is successful
    */
   bool SysBoundaryCondition::updateSysBoundaryConditionsAfterLoadBalance(
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const vector<CellID> & local_cells_on_boundary
   ) {
      // Loop over cellids
      for(vector<CellID>::const_iterator it = local_cells_on_boundary.begin(); it != local_cells_on_boundary.end(); ++it) {
         const CellID cellId = *it;
         vector<CellID> & closestCells = allClosestNonsysboundaryCells[cellId];
         closestCells.clear();
         vector<CellID> & closeCells = allCloseNonsysboundaryCells[cellId];
         closeCells.clear();
         array<SpatialCell*,27> & flowtoCells = allFlowtoCells[cellId];
         flowtoCells.fill(NULL);
         uint dist = numeric_limits<uint>::max();

         uint d2 = numeric_limits<uint>::max();
         // int indexstep = pow(2,P::amrMaxSpatialRefLevel - mpiGrid[cellId]->SpatialCell::parameters[CellParams::REFINEMENT_LEVEL]);
         // Note this must be int, not uint, for latter calculations

         // This is broken, but also obsolete.
         // Find flowto cells (note, L2 cells do not have flowto cells)
         // auto* nearNbrs = mpiGrid.get_neighbors_of(cellId, NEAREST_NEIGHBORHOOD_ID);
         // for (auto nbrPair : *nearNbrs) {
         //    if(nbrPair.first != INVALID_CELLID) {
         //       if(mpiGrid[nbrPair.first]->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
         //          flowtoCells.at((int)(nbrPair.second[0]/indexstep) + 3*(int)(nbrPair.second[1]/indexstep) + 9*(int)(nbrPair.second[2]/indexstep) + 13) = mpiGrid[nbrPair.first];
         //          //flowtoCells.at(i + 3*j + 9*k + 13) = mpiGrid[cell];
         //       }
         //    }
         // }

         // Only closer neighborhood for layer 1
         if(mpiGrid[cellId]->sysBoundaryLayer == 1) {		      
            for (auto nbrPair : *mpiGrid.get_neighbors_of(cellId, SYSBOUNDARIES_NEIGHBORHOOD_ID)) {
               if(nbrPair.first != INVALID_CELLID) {
                  CellID neighbor = nbrPair.first;
                  if(mpiGrid[neighbor]->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
                     // Find distance and update closestCells
                     d2 = nbrPair.second[0]*nbrPair.second[0] + nbrPair.second[1]*nbrPair.second[1] + nbrPair.second[2]*nbrPair.second[2];
                     for (auto i : *mpiGrid.get_neighbors_to(cellId, SYSBOUNDARIES_NEIGHBORHOOD_ID)) {
                        if (i.first == neighbor) {
                           closeCells.push_back(neighbor);
                           if(d2 == dist) {
                              closestCells.push_back(neighbor);
                           } else if (d2 < dist) {
                              closestCells.clear();
                              closestCells.push_back(neighbor);
                              dist = d2;
                           }
                        }
                     }
                  } 
               }
            }
         }	 

         // search further for L2
         if (mpiGrid[cellId]->sysBoundaryLayer == 2) {
            for (auto nbrPair : *mpiGrid.get_neighbors_of(cellId, SYSBOUNDARIES_EXTENDED_NEIGHBORHOOD_ID)) {
               if(nbrPair.first != INVALID_CELLID) {
                  CellID neighbor = nbrPair.first;
                  if(mpiGrid[neighbor]->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
                     // Find distance and update closestCells
                     d2 = nbrPair.second[0]*nbrPair.second[0] + nbrPair.second[1]*nbrPair.second[1] + nbrPair.second[2]*nbrPair.second[2];
                     closeCells.push_back(neighbor);
                     if(d2 == dist) {
                        closestCells.push_back(neighbor);
                     } else if (d2 < dist) {
                        closestCells.clear();
                        closestCells.push_back(neighbor);
                        dist = d2;
                     }
                  }
               }
            }
         }	 

         if(closestCells.size() == 0) {
            closestCells.push_back(INVALID_CELLID);
         }
         if(closeCells.size() == 0) {
            closeCells.push_back(INVALID_CELLID);
         }
      }
      return true;
   }
   
   /*! Get the cellID of the first closest cell of type NOT_SYSBOUNDARY found.
    * \param i,j,k Coordinates of the cell to start looking from
    * \return The cell index of that cell
    * \sa getAllClosestNonsysboundaryCells
    */
   array<int, 3> SysBoundaryCondition::getTheClosestNonsysboundaryCell(
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      cint i,
      cint j,
      cint k
   ) {
      const vector< array<int, 3> > closestCells = getAllClosestNonsysboundaryCells(technicalGrid, i, j, k);
      return closestCells.at(0);
   }
   
   /*! Get the cellIDs of all the closest cells of type NOT_SYSBOUNDARY.
    * \param i,j,k Coordinates of the cell to start looking from
    * \return The vector of cell indices of those cells
    * \sa getTheClosestNonsysboundaryCell
    */
   vector< array<int, 3> > SysBoundaryCondition::getAllClosestNonsysboundaryCells(
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      cint i,
      cint j,
      cint k
   ) {
      int distance = numeric_limits<int>::max();
      vector< array<int,3> > closestCells;
      
      for (int kk=-2; kk<3; kk++) {
         for (int jj=-2; jj<3; jj++) {
            for (int ii=-2; ii<3 ; ii++) {
               if( technicalGrid.get(i+ii,j+jj,k+kk) && technicalGrid.get(i+ii,j+jj,k+kk)->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
                  distance = min(distance, ii*ii + jj*jj + kk*kk);
               }
            }
         }
      }
      
      for (int kk=-2; kk<3; kk++) {
         for (int jj=-2; jj<3; jj++) {
            for (int ii=-2; ii<3 ; ii++) {
               if( technicalGrid.get(i+ii,j+jj,k+kk) && technicalGrid.get(i+ii,j+jj,k+kk)->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
                  int d = ii*ii + jj*jj + kk*kk;
                  if( d == distance ) {
                     array<int, 3> cell = {i+ii, j+jj, k+kk};
                     closestCells.push_back(cell);
                  }
               }
            }
         }
      }
      
      if(closestCells.size() == 0) {
         array<int, 3> dummy  = {numeric_limits<int>::min()};
         closestCells.push_back(dummy);
      }
      
      return closestCells;
   }
   
   /*! Get the cellID of the first closest cell of type NOT_SYSBOUNDARY found.
    * \param cellID ID of the cell to start look from.
    * \return The cell index of that cell
    * \sa getAllClosestNonsysboundaryCells
    */
   CellID & SysBoundaryCondition::getTheClosestNonsysboundaryCell(
      const CellID& cellID
   ) {
      vector<CellID> & closestCells = allClosestNonsysboundaryCells.at(cellID);
      return closestCells.at(0);
   }
   
   /*! Get the cellIDs of all the closest cells of type NOT_SYSBOUNDARY.
    * \param cellID ID of the cell to start look from.
    * \return The vector of cell indices of those cells
    * \sa getTheClosestNonsysboundaryCell
    */
   vector<CellID> & SysBoundaryCondition::getAllClosestNonsysboundaryCells(
      const CellID& cellID
   ) {
      vector<CellID> & closestCells = allClosestNonsysboundaryCells.at(cellID);
      return closestCells;
   }
   
   /*! Get the cellIDs of all the close cells of type NOT_SYSBOUNDARY.
    * \param cellID ID of the cell to start look from.
    * \return The vector of cell indices of those cells
    */
   vector<CellID> & SysBoundaryCondition::getAllCloseNonsysboundaryCells(
      const CellID& cellID
   ) {
      vector<CellID> & closeCells = allCloseNonsysboundaryCells.at(cellID);
      return closeCells;
   }
   
   /*! Get the cellIDs of all flowto cells (cells into which the velocity distribution can flow and which is of type NOT_SYSBOUNDARY).
    * \param cellID ID of the cell to start look from.
    * \return The vector of cell indices of those cells
    */
   array<SpatialCell*,27> & SysBoundaryCondition::getFlowtoCells(
      const CellID& cellID
   ) {
      phiprof::Timer timer {"getFlowtoCells"};
      array<SpatialCell*,27> & flowtoCells = allFlowtoCells.at(cellID);
      return flowtoCells;
   }
   
   array<Realf*,27> SysBoundaryCondition::getFlowtoCellsBlock(
      const array<SpatialCell*,27> flowtoCells,
      const vmesh::GlobalID blockGID,
      const uint popID
   ) {
      phiprof::Timer timer {"getFlowtoCellsBlock"};
      array<Realf*,27> flowtoCellsBlock;
      flowtoCellsBlock.fill(NULL);
      for (uint i=0; i<27; i++) {
         if(flowtoCells.at(i)) {
            flowtoCellsBlock.at(i) = flowtoCells.at(i)->get_data(flowtoCells.at(i)->get_velocity_block_local_id(blockGID,popID), popID);
         }
      }
      return flowtoCellsBlock;
   }
   
   Real SysBoundaryCondition::fieldBoundaryCopyFromSolvingNbrMagneticField(
      FsGrid< array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & bGrid,
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      cint i,
      cint j,
      cint k,
      cuint component,
      cuint mask
   ) {

      int distance = numeric_limits<int>::max();
      vector< array<int,3> > closestCells;

      for (int kk=-2; kk<3; kk++) {
         for (int jj=-2; jj<3; jj++) {
            for (int ii=-2; ii<3 ; ii++) {
               if( technicalGrid.get(i+ii,j+jj,k+kk) // skip invalid cells returning NULL
                   && (technicalGrid.get(i+ii,j+jj,k+kk)->SOLVE & mask) == mask // Did that guy solve this component?
                   && technicalGrid.get(i+ii,j+jj,k+kk)->sysBoundaryFlag != sysboundarytype::DO_NOT_COMPUTE // Do not copy from there
                   && technicalGrid.get(i+ii,j+jj,k+kk)->sysBoundaryFlag != sysboundarytype::OUTER_BOUNDARY_PADDING // Do not copy from there either
               ) {
                  distance = min(distance, ii*ii + jj*jj + kk*kk);
               }
            }
         }
      }

      for (int kk=-2; kk<3; kk++) {
         for (int jj=-2; jj<3; jj++) {
            for (int ii=-2; ii<3 ; ii++) {
               if( technicalGrid.get(i+ii,j+jj,k+kk) // skip invalid cells returning NULL
                   && (technicalGrid.get(i+ii,j+jj,k+kk)->SOLVE & mask) == mask // Did that guy solve this component?
                   && technicalGrid.get(i+ii,j+jj,k+kk)->sysBoundaryFlag != sysboundarytype::DO_NOT_COMPUTE // Do not copy from there
                   && technicalGrid.get(i+ii,j+jj,k+kk)->sysBoundaryFlag != sysboundarytype::OUTER_BOUNDARY_PADDING // Do not copy from there either
               ) {
                  int d = ii*ii + jj*jj + kk*kk;
                  if( d == distance ) {
                     array<int, 3> cell = {i+ii, j+jj, k+kk};
                     closestCells.push_back(cell);
                  }
               }
            }
         }
      }

      if (closestCells.size() == 0) {
         abort_mpi("No closest cell found!", 1);
      }

      return bGrid.get(closestCells[0][0], closestCells[0][1], closestCells[0][2])->at(fsgrids::bfield::PERBX+component);
   }
   
   /*! Function used in some cases to know which faces the system boundary condition is being applied to.
    * \param faces Pointer to array of 6 bool in which the values are returned whether the corresponding face is of that type. Order: 0 x+; 1 x-; 2 y+; 3 y-; 4 z+; 5 z-
    */
   void SysBoundaryCondition::getFaces(bool* faces) {
      cerr << "ERROR: SysBoundaryCondition::getFaces called instead of derived class function!" << endl;
      for(int i=0; i<6; i++) {
        faces[i]=false;
      }
   }
   
   /*! Get the precedence value of the system boundary condition.
    * \return The precedence value of the system boundary condition as set by parameter.
    */
   uint SysBoundaryCondition::getPrecedence() const {return precedence;}
   
   /*! Returns whether the boundary condition is dynamic in time.
    * \return Boolean value.
    */
   bool SysBoundaryCondition::isDynamic() const { return dynamic; }
   
   void SysBoundaryCondition::setPeriodicity(
      bool isFacePeriodic[3]
   ) {
      for (uint i=0; i<3; i++) {
         this->periodic[i] = isFacePeriodic[i];
      }
   }
   
   /*! Get a bool telling whether to call again applyInitialState upon restarting the simulation. */
   bool SysBoundaryCondition::doApplyUponRestart() const {return this->applyUponRestart;}

   void OuterBoundaryCondition::assignSysBoundary(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid, FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid) {
      array<bool,6> isThisCellOnAFace;
      
      // Assign boundary flags to local DCCRG cells
      for(const auto& id : getLocalCells()) {
         if (mpiGrid[id]->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) 
            continue;

         determineFace(isThisCellOnAFace, mpiGrid, id);
         for (int i = 0; i < 6; ++i) {
            if(facesToProcess[i] && isThisCellOnAFace[i]) {
               mpiGrid[id]->sysBoundaryFlag = this->getIndex();
            }
         }
      }
   }

   void SysBoundaryCondition::mapCellPotentialAndGetEXBDrift(
      std::array<Real, CellParams::N_SPATIAL_CELL_PARAMS>& cellParams
   ) {
      std::cerr << "Error: SysBoundaryCondition::mapCellPotentialAndGetEXBDrift called!\n";
   }

   vmesh::LocalID findMaxwellianBlocksToInitialize(
      const uint popID,
      spatial_cell::SpatialCell& cell,
      creal& rho,
      creal& T,
      creal& VX0,
      creal& VY0,
      creal& VZ0) {

      bool search = true;
      uint counter = 0;
      vmesh::VelocityMesh *vmesh = cell.get_velocity_mesh(popID);
      const Real mass = getObjectWrapper().particleSpecies[popID].mass;

      vmesh::GlobalID *GIDbuffer;
      #ifdef USE_GPU
      // Host-pinned memory buffer, max possible size
      const vmesh::LocalID* vblocks_ini = cell.get_velocity_grid_length(popID);
      const uint blocksCount = vblocks_ini[0]*vblocks_ini[1]*vblocks_ini[2];
      CHK_ERR( gpuMallocHost((void**)&GIDbuffer,blocksCount*sizeof(vmesh::GlobalID)) );
      #endif
      // Non-GPU: insert directly into vmesh

      Real V_crds[3];
      Real dV[3];
      dV[0] = cell.get_velocity_grid_block_size(popID)[0];
      dV[1] = cell.get_velocity_grid_block_size(popID)[1];
      dV[2] = cell.get_velocity_grid_block_size(popID)[2];
      creal minValue = cell.getVelocityBlockMinValue(popID);
      // Single cell, not block
      const Real dvx=cell.get_velocity_grid_cell_size(popID)[0];
      const Real dvy=cell.get_velocity_grid_cell_size(popID)[1];
      const Real dvz=cell.get_velocity_grid_cell_size(popID)[2];

      while (search) {
         if (0.1 * minValue > projects::MaxwellianPhaseSpaceDensity(counter*dV[0]+0.5*dvx, 0.5*dvy, 0.5*dvz, T, rho, mass) || counter > vblocks_ini[0]) {
            search = false;
         }
         counter++;
      }
      counter+=2;

      Real vRadiusSquared = (Real)counter * (Real)counter * dV[0] * dV[0];

      #ifndef USE_GPU
      // sphere volume is 4/3 pi r^3, approximate that 5*counterX*counterY*counterZ is enough.
      vmesh::LocalID currentMaxSize = LID + 5*counter*counter*counter;
      vmesh->setNewSize(currentMaxSize);
      GIDbuffer = vmesh->getGrid()->data();
      #endif

      vmesh::LocalID LID = 0;
      for (uint kv=0; kv<vblocks_ini[2]; ++kv) {
         for (uint jv=0; jv<vblocks_ini[1]; ++jv) {
            for (uint iv=0; iv<vblocks_ini[0]; ++iv) {
               const vmesh::GlobalID GID = vmesh->getGlobalID(iv,jv,kv);

               cell.get_velocity_block_coordinates(popID,GID,V_crds);
               V_crds[0] += 0.5*dV[0] - VX0;
               V_crds[1] += 0.5*dV[1] - VY0;
               V_crds[2] += 0.5*dV[2] - VZ0;
               Real R2 = ((V_crds[0])*(V_crds[0])
                          + (V_crds[1])*(V_crds[1])
                          + (V_crds[2])*(V_crds[2]));

               #ifndef USE_GPU
               if (LID >= currentMaxSize) {
                  currentMaxSize = LID + counter*counter*counter;
                  vmesh->setNewSize(currentMaxSize);
                  GIDbuffer = vmesh->getGrid()->data();
               }
               #endif
               if (R2 < vRadiusSquared) {
                  GIDbuffer[LID] = GID;
                  LID++;
               }
            }
         }
      }

      // Set final size of vmesh
      cell.get_population(popID).N_blocks = LID;

      #ifdef USE_GPU
      // Copy data into place
      cell.dev_resize_vmesh(popID,LID);
      vmesh::GlobalID *GIDtarget = vmesh->getGrid()->data();
      gpuStream_t stream = gpu_getStream();
      CHK_ERR( gpuMemcpyAsync(GIDtarget, GIDbuffer, LID*sizeof(vmesh::GlobalID), gpuMemcpyHostToDevice, stream));
      CHK_ERR( gpuStreamSynchronize(stream) );
      CHK_ERR( gpuFreeHost(GIDbuffer));
      #else
      vmesh->setNewSize(LID);
      #endif

      return LID;
   }

} // namespace SBC
