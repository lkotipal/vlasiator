/*
 * This file is part of Vlasiator.
 * 
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

/*!\file fieldtracing.cpp
 * \brief Implementation of the field tracing algorithms used in Magnetosphere runs and in magnetosphere-ionosphere coupling.
 */

#include "../fieldsolver/fs_common.h"
#include "fieldtracing.h"
#include "bulirschStoer.h"
#include "dormandPrince.h"
#include "euler.h"
#include "eulerAdaptive.h"

#include <Eigen/Dense>

#define Vec3d Eigen::Vector3d
#define cross_product(av,bv) (av).cross(bv)
#define dot_product(av,bv) (av).dot(bv)
#define vector_length(v) (v).norm()
#define normalize_vector(v) (v).normalized()

namespace FieldTracing {
   FieldTracingParameters fieldTracingParameters;

   /* Call the heavier operations for DROs to be called only if needed, before an IO.
    */
   void reduceData(
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & perBGrid,
      FsGrid< std::array<Real, fsgrids::dperb::N_DPERB>, FS_STENCIL_WIDTH> & dPerBGrid,
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry> & mpiGrid,
      std::vector<SBC::SphericalTriGrid::Node> & nodes
   ) {
      if(fieldTracingParameters.doTraceOpenClosed) {
         traceOpenClosedConnection(technicalGrid, perBGrid, dPerBGrid, nodes);
      }
      if(fieldTracingParameters.doTraceFullBox) {
         traceFullBoxConnectionAndFluxRopes(technicalGrid, perBGrid, dPerBGrid, mpiGrid);
      }
   }
   
   /*! Take a step along the field line*/
   void stepFieldLine(
      std::array<Real, 3>& x,
      std::array<Real, 3>& v,
      Real& stepsize,
      creal minStepSize,
      creal maxStepSize,
      TracingMethod method,
      TracingFieldFunction& BFieldFunction,
      const bool outwards
   ) {
      bool reTrace;
      uint32_t attempts=0;
      switch(method) {
         case Euler:
            eulerStep(x, v,stepsize, BFieldFunction, outwards);
            break;
         case ADPT_Euler:
            do {
               reTrace=!adaptiveEulerStep(x, v, stepsize, minStepSize, maxStepSize, BFieldFunction, outwards);
               attempts++;
            } while (reTrace && attempts<= fieldTracingParameters.max_field_tracer_attempts);
            if (reTrace) {
               logFile << "(fieldtracing) Warning: Adaptive Euler field line tracer exhausted all available attempts and still did not converge." << std::endl;
            }
            break;
         case BS:
            do{
               reTrace=!bulirschStoerStep(x, v, stepsize, minStepSize, maxStepSize, BFieldFunction, outwards);
               attempts++;
            } while (reTrace && attempts<= fieldTracingParameters.max_field_tracer_attempts);
            break;
         case DPrince:
            do {
               reTrace=!dormandPrinceStep(x, v, stepsize, minStepSize, maxStepSize, BFieldFunction, outwards);
               attempts++;
            } while (reTrace && attempts<= fieldTracingParameters.max_field_tracer_attempts);
            if (reTrace) {
               logFile << "(fieldtracing) Warning: Dormand Prince field line tracer exhausted all available attempts and still did not converge..." << std::endl;
            }
            break;
         default:
            std::cerr << "(fieldtracing) Error: No field line tracing method defined."<<std::endl;
            abort();
            break;
      }
   }//stepFieldLine
   
   bool traceFullFieldFunction(
      FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & perBGrid,
      FsGrid< std::array<Real, fsgrids::dperb::N_DPERB>, FS_STENCIL_WIDTH> & dPerBGrid,
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      std::array<Real,3>& r,
      const bool alongB,
      std::array<Real,3>& b
   ) {
      if(   r[0] > P::xmax - 2*P::dx_ini
         || r[0] < P::xmin + 2*P::dx_ini
         || r[1] > P::ymax - 2*P::dy_ini
         || r[1] < P::ymin + 2*P::dy_ini
         || r[2] > P::zmax - 2*P::dz_ini
         || r[2] < P::zmin + 2*P::dz_ini
      ) {
         cerr << (string)("(fieldtracing) Error: fsgrid coupling trying to step outside of the global domain?\n");
         return false;
      }
      
      // Get field direction
      b[0] = SBC::ionosphereGrid.dipoleField(r[0],r[1],r[2],X,0,X) + SBC::ionosphereGrid.BGB[0];
      b[1] = SBC::ionosphereGrid.dipoleField(r[0],r[1],r[2],Y,0,Y) + SBC::ionosphereGrid.BGB[1];
      b[2] = SBC::ionosphereGrid.dipoleField(r[0],r[1],r[2],Z,0,Z) + SBC::ionosphereGrid.BGB[2];
      
      std::array<int32_t, 3> fsgridCell = getGlobalFsGridCellIndexForCoord(technicalGrid,r);
      const std::array<int32_t, 3> localStart = technicalGrid.getLocalStart();
      const std::array<int32_t, 3> localSize = technicalGrid.getLocalSize();
      // Make the global index a local one, bypass the fsgrid function that yields (-1,-1,-1) also for ghost cells.
      fsgridCell[0] -= localStart[0];
      fsgridCell[1] -= localStart[1];
      fsgridCell[2] -= localStart[2];
      
      if(fsgridCell[0] > localSize[0] || fsgridCell[1] > localSize[1] || fsgridCell[2] > localSize[2]
         || fsgridCell[0] < -1 || fsgridCell[1] < -1 || fsgridCell[2] < -1) {
         cerr << (string)("(fieldtracing) Error: fsgrid coupling trying to access local ID " + to_string(fsgridCell[0]) + " " + to_string(fsgridCell[1]) + " " + to_string(fsgridCell[2])
         + " for local domain size " + to_string(localSize[0]) + " " + to_string(localSize[1]) + " " + to_string(localSize[2])
         + " at position " + to_string(r[0]) + " " + to_string(r[1]) + " " + to_string(r[2]) + " radius " + to_string(sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]))
         + "\n");
         return false;
      } else {
         if(technicalGrid.get(fsgridCell[0],fsgridCell[1],fsgridCell[2])->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
            const std::array<Real, 3> perB = interpolatePerturbedB(
               perBGrid,
               dPerBGrid,
               technicalGrid,
               fieldTracingParameters.reconstructionCoefficientsCache,
               fsgridCell[0],fsgridCell[1],fsgridCell[2],
               r
            );
            b[0] += perB[0];
            b[1] += perB[1];
            b[2] += perB[2];
         }
      }
      
      // Normalize
      Real  norm = 1. / sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
      for(int c=0; c<3; c++) {
         b[c] = b[c] * norm;
      }
      
      // Make sure motion is outwards. Flip b if dot(r,b) < 0
      if(std::isnan(b[0]) || std::isnan(b[1]) || std::isnan(b[2])) {
         cerr << "(fieldtracing) Error: magnetic field is nan in getRadialBfieldDirection at location "
         << r[0] << ", " << r[1] << ", " << r[2] << ", with B = " << b[0] << ", " << b[1] << ", " << b[2] << endl;
         b[0] = 0;
         b[1] = 0;
         b[2] = 0;
      }
      if(!alongB) { // In this function, outwards indicates whether we trace along (true) or against (false) the field direction
         b[0] *= -1;
         b[1] *= -1;
         b[2] *= -1;
      }
      return true;
   }
   
   /*! Calculate mapping between ionospheric nodes and fsGrid cells.
   * To do so, the magnetic field lines are traced from all mesh nodes
   * outwards until a non-boundary cell is encountered. Their proportional
   * coupling values are recorded in the grid nodes.
   */
   void calculateIonosphereFsgridCoupling(
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & perBGrid,
      FsGrid< std::array<Real, fsgrids::dperb::N_DPERB>, FS_STENCIL_WIDTH> & dPerBGrid,
      std::vector<SBC::SphericalTriGrid::Node> & nodes,
      creal couplingRadius
   ) {
      
      // we don't need to do anything if we have no nodes
      if(nodes.size() == 0) {
         return;
      }
      
      phiprof::start("fieldtracing-ionosphere-fsgridCoupling");
      // Pick an initial stepsize
      creal stepSize = min(100e3, technicalGrid.DX / 2.);
      std::vector<Real> nodeTracingStepSize(nodes.size(), stepSize); // In-flight storage of step size, needed when crossing into next MPI domain
      std::vector<Real> reducedNodeTracingStepSize(nodes.size());
      
      std::vector<Real> nodeDistance(nodes.size(), std::numeric_limits<Real>::max()); // For reduction of node coordinate in case of multiple hits
      std::vector<int> nodeNeedsContinuedTracing(nodes.size(), 1);                    // Flag, whether tracing needs to continue on another task
      std::vector<std::array<Real, 3>> nodeTracingCoordinates(nodes.size());          // In-flight node upmapping coordinates (for global reduction)
      for(uint n=0; n<nodes.size(); n++) {
         nodeTracingCoordinates.at(n) = nodes.at(n).x;
         nodes.at(n).haveCouplingData = 0;
         for (uint c=0; c<3; c++) {
            nodes.at(n).xMapped.at(c) = 0;
            nodes.at(n).parameters.at(ionosphereParameters::UPMAPPED_BX+c) = 0;
         }
      }
      bool anyNodeNeedsTracing;
      
      TracingFieldFunction tracingFullField = [&perBGrid, &dPerBGrid, &technicalGrid](std::array<Real,3>& r, const bool alongB, std::array<Real,3>& b)->bool{
         return traceFullFieldFunction(perBGrid, dPerBGrid, technicalGrid, r, alongB, b);
      };
      
      int itCount = 0;
      do {
         itCount++;
         anyNodeNeedsTracing = false;
         
         #pragma omp parallel
         {
            // Trace node coordinates outwards until a non-sysboundary cell is encountered or the local fsgrid domain has been left.
            #pragma omp for schedule(dynamic)
            for(uint n=0; n<nodes.size(); n++) {
               
               if(!nodeNeedsContinuedTracing[n]) {
                  // This node has already found its target, no need for us to do anything about it.
                  continue;
               }
               SBC::SphericalTriGrid::Node& no = nodes[n];
               
               std::array<Real, 3> x = nodeTracingCoordinates[n];
               std::array<Real, 3> v({0,0,0});
               
               while( true ) {
                  
                  // Check if the current coordinates (pre-step) are in our own domain.
                  std::array<int, 3> fsgridCell = getLocalFsGridCellIndexForCoord(technicalGrid,x);
                  // If it is not in our domain, somebody else takes care of it.
                  if(fsgridCell[0] == -1) {
                     nodeNeedsContinuedTracing[n] = 0;
                     nodeTracingCoordinates[n] = {0,0,0};
                     nodeTracingStepSize[n]=0;
                     break;
                  }
                  
                  
                  // Make one step along the fieldline
                  stepFieldLine(x,v, nodeTracingStepSize[n],fieldTracingParameters.min_tracer_dx,technicalGrid.DX/2,fieldTracingParameters.tracingMethod,tracingFullField,(no.x[2] < 0));
                  
                  // Look up the fsgrid cell belonging to these coordinates
                  fsgridCell = getLocalFsGridCellIndexForCoord(technicalGrid,x);
                  std::array<Real, 3> interpolationFactor=getFractionalFsGridCellForCoord(technicalGrid,x);
                  
                  creal distance = sqrt((x[0]-no.x[0])*(x[0]-no.x[0])+(x[1]-no.x[1])*(x[1]-no.x[1])+(x[2]-no.x[2])*(x[2]-no.x[2]));
                  
                  // TODO I simplified by just looking when we change hemispheres now.
                  // This WILL fail as soon as there is a dipole tilt.
                  // But do we need it beyond debugging? Tracing back for closed/non-mapping lines is perfectly legit (once the tracer is debugged).
                  if(sign(x[2]) != sign(no.x[2])) {
                     nodeNeedsContinuedTracing.at(n) = 0;
                     nodeTracingCoordinates.at(n) = {0,0,0};
                     break;
                  }
                  
                  // If we somehow still map into the ionosphere, we missed the 88 degree criterion but shouldn't couple there.
                  if(sqrt(x.at(0)*x.at(0) + x.at(1)*x.at(1) + x.at(2)*x.at(2)) < SBC::Ionosphere::innerRadius) {
                     // TODO drop this warning if it never occurs? To be followed.
                     cerr << (string)("(fieldtracing) Warning: Triggered mapping back into Earth from node " + to_string(n) + " at z " + to_string(no.x[2]) + "\n");
                     nodeNeedsContinuedTracing.at(n) = 0;
                     nodeTracingCoordinates.at(n) = {0,0,0};
                     break;
                  }
                  
                  // Now, after stepping, if it is no longer in our domain, another MPI rank will pick up later.
                  if(fsgridCell[0] == -1) {
                     nodeNeedsContinuedTracing[n] = 1;
                     nodeTracingCoordinates[n] = x;
                     break;
                  }
                  
                  if(
                     technicalGrid.get( fsgridCell[0], fsgridCell[1], fsgridCell[2])->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY
                     && x[0]*x[0]+x[1]*x[1]+x[2]*x[2] > SBC::Ionosphere::downmapRadius*SBC::Ionosphere::downmapRadius
                  ) {
                     
                     // Store the cells mapped coordinates and upmapped magnetic field
                     no.xMapped = x;
                     no.haveCouplingData = 1;
                     nodeDistance[n] = distance;
                     const std::array<Real, 3> perB = interpolatePerturbedB(
                        perBGrid,
                        dPerBGrid,
                        technicalGrid,
                        fieldTracingParameters.reconstructionCoefficientsCache,
                        fsgridCell[0], fsgridCell[1], fsgridCell[2],
                        x
                     );
                     no.parameters[ionosphereParameters::UPMAPPED_BX] = SBC::ionosphereGrid.dipoleField(x[0],x[1],x[2],X,0,X) + perB[0];
                     no.parameters[ionosphereParameters::UPMAPPED_BY] = SBC::ionosphereGrid.dipoleField(x[0],x[1],x[2],Y,0,Y) + perB[1];
                     no.parameters[ionosphereParameters::UPMAPPED_BZ] = SBC::ionosphereGrid.dipoleField(x[0],x[1],x[2],Z,0,Z) + perB[2];
                     
                     nodeNeedsContinuedTracing[n] = 0;
                     nodeTracingCoordinates[n] = {0,0,0};
                     break;
                  }
               } // while(true)
            } // for
         } // pragma omp parallel
         
         // Globally reduce whether any node still needs to be picked up and traced onwards
         std::vector<int> sumNodeNeedsContinuedTracing(nodes.size(), 0);
         std::vector<std::array<Real, 3>> sumNodeTracingCoordinates(nodes.size());
         MPI_Allreduce(nodeNeedsContinuedTracing.data(), sumNodeNeedsContinuedTracing.data(), nodes.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         if(sizeof(Real) == sizeof(double)) {
            MPI_Allreduce(nodeTracingCoordinates.data(), sumNodeTracingCoordinates.data(), 3*nodes.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(nodeTracingStepSize.data(), reducedNodeTracingStepSize.data(), nodes.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         } else {
            MPI_Allreduce(nodeTracingCoordinates.data(), sumNodeTracingCoordinates.data(), 3*nodes.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(nodeTracingStepSize.data(), reducedNodeTracingStepSize.data(), nodes.size(), MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
         }
         for(uint n=0; n<nodes.size(); n++) {
            if(sumNodeNeedsContinuedTracing[n] > 0) {
               anyNodeNeedsTracing=true;
               nodeNeedsContinuedTracing[n] = 1;
               
               // Update that nodes' tracing coordinates
               nodeTracingCoordinates[n][0] = sumNodeTracingCoordinates[n][0] / sumNodeNeedsContinuedTracing[n];
               nodeTracingCoordinates[n][1] = sumNodeTracingCoordinates[n][1] / sumNodeNeedsContinuedTracing[n];
               nodeTracingCoordinates[n][2] = sumNodeTracingCoordinates[n][2] / sumNodeNeedsContinuedTracing[n];
            }
            nodeTracingStepSize[n] = reducedNodeTracingStepSize[n];
         }
         
      } while(anyNodeNeedsTracing);
      
      logFile << "(fieldtracing) fsgrid coupling traced in " << itCount << " iterations of the tracing loop." << endl;
      
      std::vector<Real> reducedNodeDistance(nodes.size());
      if(sizeof(Real) == sizeof(double)) {
         MPI_Allreduce(nodeDistance.data(), reducedNodeDistance.data(), nodes.size(), MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      } else {
         MPI_Allreduce(nodeDistance.data(), reducedNodeDistance.data(), nodes.size(), MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
      }
      
      // Reduce upmapped magnetic field to be consistent on all nodes
      std::vector<Real> sendUpmappedB(3 * nodes.size());
      std::vector<Real> reducedUpmappedB(3 * nodes.size());
      // Likewise, reduce upmapped coordinates
      std::vector<Real> sendxMapped(3 * nodes.size());
      std::vector<Real> reducedxMapped(3 * nodes.size());
      // And coupling rank number
      std::vector<int> sendCouplingNum(nodes.size());
      std::vector<int> reducedCouplingNum(nodes.size());
      
      for(uint n=0; n<nodes.size(); n++) {
         SBC::SphericalTriGrid::Node& no = nodes[n];
         // Discard false hits from cells that are further out from the node
         if(nodeDistance[n] > reducedNodeDistance[n]) {
            no.haveCouplingData = 0;
            for(int c=0; c<3; c++) {
               no.parameters[ionosphereParameters::UPMAPPED_BX+c] = 0;
               no.xMapped[c] = 0;
            }
         } else {
            // Cell found, add association.
            SBC::ionosphereGrid.isCouplingInwards = true;
         }
         
         
         sendUpmappedB[3*n] = no.parameters[ionosphereParameters::UPMAPPED_BX];
         sendUpmappedB[3*n+1] = no.parameters[ionosphereParameters::UPMAPPED_BY];
         sendUpmappedB[3*n+2] = no.parameters[ionosphereParameters::UPMAPPED_BZ];
         sendxMapped[3*n] = no.xMapped[0];
         sendxMapped[3*n+1] = no.xMapped[1];
         sendxMapped[3*n+2] = no.xMapped[2];
         sendCouplingNum[n] = no.haveCouplingData;
      }
      if(sizeof(Real) == sizeof(double)) { 
         MPI_Allreduce(sendUpmappedB.data(), reducedUpmappedB.data(), 3*nodes.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
         MPI_Allreduce(sendxMapped.data(), reducedxMapped.data(), 3*nodes.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      } else {
         MPI_Allreduce(sendUpmappedB.data(), reducedUpmappedB.data(), 3*nodes.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
         MPI_Allreduce(sendxMapped.data(), reducedxMapped.data(), 3*nodes.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      }
      MPI_Allreduce(sendCouplingNum.data(), reducedCouplingNum.data(), nodes.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      
      for(uint n=0; n<nodes.size(); n++) {
         SBC::SphericalTriGrid::Node& no = nodes[n];
         
         // We don't even care about nodes that couple nowhere.
         if(reducedCouplingNum[n] == 0) {
            continue;
         }
         no.parameters[ionosphereParameters::UPMAPPED_BX] = reducedUpmappedB[3*n] / reducedCouplingNum[n];
         no.parameters[ionosphereParameters::UPMAPPED_BY] = reducedUpmappedB[3*n+1] / reducedCouplingNum[n];
         no.parameters[ionosphereParameters::UPMAPPED_BZ] = reducedUpmappedB[3*n+2] / reducedCouplingNum[n];
         no.xMapped[0] = reducedxMapped[3*n] / reducedCouplingNum[n];
         no.xMapped[1] = reducedxMapped[3*n+1] / reducedCouplingNum[n];
         no.xMapped[2] = reducedxMapped[3*n+2] / reducedCouplingNum[n];
      }
      
      phiprof::stop("fieldtracing-ionosphere-fsgridCoupling");
   }

   /*! Calculate mapping between ionospheric nodes and Vlasov grid cells.
   * Input is the cell coordinate of the vlasov grid cell.
   * To do so, magnetic field lines are traced inwords from the Vlasov grid
   * IONOSPHERE boundary cells to the ionosphere shell.
   *
   * The return value is a pair of nodeID and coupling factor for the three
   * corners of the containing element.
   */
   std::array<std::pair<int, Real>, 3> calculateIonosphereVlasovGridCoupling(
      std::array<Real,3> x,
      std::vector<SBC::SphericalTriGrid::Node> & nodes,
      creal couplingRadius
   ) {
      
      std::array<std::pair<int, Real>, 3> coupling;
      
      Real stepSize = 100e3;
      std::array<Real,3> v;
      phiprof::start("fieldtracing-ionosphere-VlasovGridCoupling");
      
      // For tracing towards the vlasov boundary, we only require the dipole field.
      TracingFieldFunction dipoleFieldOnly = [](std::array<Real,3>& r, const bool outwards, std::array<Real,3>& b)->bool {
         
         // Get field direction
         b[0] = SBC::ionosphereGrid.dipoleField(r[0],r[1],r[2],X,0,X) + SBC::ionosphereGrid.BGB[0];
         b[1] = SBC::ionosphereGrid.dipoleField(r[0],r[1],r[2],Y,0,Y) + SBC::ionosphereGrid.BGB[1];
         b[2] = SBC::ionosphereGrid.dipoleField(r[0],r[1],r[2],Z,0,Z) + SBC::ionosphereGrid.BGB[2];
         
         // Normalize
         Real  norm = 1. / sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
         for(int c=0; c<3; c++) {
            b[c] = b[c] * norm;
         }
         
         // Make sure motion is outwards. Flip b if dot(r,b) < 0
         if(outwards) {
            if(b[0]*r[0] + b[1]*r[1] + b[2]*r[2] < 0) {
               b[0]*=-1;
               b[1]*=-1;
               b[2]*=-1;
            }
         } else {
            if(b[0]*r[0] + b[1]*r[1] + b[2]*r[2] > 0) {
               b[0]*=-1;
               b[1]*=-1;
               b[2]*=-1;
            }
         }
         return true;
      };
      
      while(sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) > SBC::Ionosphere::innerRadius) {
         
         // Make one step along the fieldline
         stepFieldLine(x,v, stepSize,50e3,100e3,fieldTracingParameters.tracingMethod,dipoleFieldOnly,false);
         
         // If the field lines is moving even further outwards, abort.
         // (this shouldn't happen under normal magnetospheric conditions, but who
         // knows what crazy driving this will be run with)
         if(sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) > 1.5*couplingRadius) {
            cerr << "(fieldtracing) Warning: coupling of Vlasov grid cell failed due to weird magnetic field topology." << endl;
            
            // Return a coupling that has 0 value and results in zero potential
            phiprof::stop("fieldtracing-ionosphere-VlasovGridCoupling");
            return coupling;
         }
      }
      
      // Determine the nearest ionosphere node to this point.
      uint32_t nearestNode = SBC::ionosphereGrid.findNodeAtCoordinates(x);
      int32_t elementIndex = nodes[nearestNode].touchingElements[0];
      int32_t oldElementIndex;
      
      std::unordered_set<int32_t> elementHistory;
      bool override=false;
      
      for (uint toto=0; toto<15; toto++) {
         const SBC::SphericalTriGrid::Element& el = SBC::ionosphereGrid.elements[elementIndex];
         oldElementIndex = elementIndex;
         
         if(elementHistory.find(elementIndex) == elementHistory.end()) {
            elementHistory.insert(elementIndex);
         } else {
            // This element was already seen, entering a loop, let's get out
            // It happens when the projection rx is left seen from the right triangle and right seen from the left triangle.
            cerr << "Entered a loop, taking the current element " << elementIndex << "." << endl;
            override=true;
         }
         
         // Calculate barycentric coordinates for x in this element.
         Vec3d r1(nodes[el.corners[0]].x.data());
         Vec3d r2(nodes[el.corners[1]].x.data());
         Vec3d r3(nodes[el.corners[2]].x.data());
         
         Vec3d rx(x[0],x[1],x[2]);
         
         cint handedness = sign(dot_product(cross_product(r2-r1, r3-r1), r1));
         
         creal kappa1 = handedness*sign(dot_product(cross_product(r1, r2-r1), rx-r1));
         creal kappa2 = handedness*sign(dot_product(cross_product(r2, r3-r2), rx-r2));
         creal kappa3 = handedness*sign(dot_product(cross_product(r3, r1-r3), rx-r3));
         
         if(override || (kappa1 > 0 && kappa2 > 0 && kappa3 > 0)) {
            // Total area
            Real A = vector_length(cross_product(r2-r1,r3-r1));
            
            // Project x into the plane of this triangle
            Vec3d normal = normalize_vector(cross_product(r2-r1, r3-r1));
            rx -= normal*dot_product(rx-r1, normal);
            
            // Area of the sub-triangles
            Real lambda1 = vector_length(cross_product(r2-rx, r3-rx)) / A;
            Real lambda2 = vector_length(cross_product(r1-rx, r3-rx)) / A;
            Real lambda3 = vector_length(cross_product(r1-rx, r2-rx)) / A;
            
            coupling[0] = {el.corners[0], lambda1};
            coupling[1] = {el.corners[1], lambda2};
            coupling[2] = {el.corners[2], lambda3};
            phiprof::stop("fieldtracing-ionosphere-VlasovGridCoupling");
            return coupling;
         } else if (kappa1 > 0 && kappa2 > 0 && kappa3 < 0) {
            elementIndex = SBC::ionosphereGrid.findElementNeighbour(elementIndex,0,2);
         } else if (kappa2 > 0 && kappa3 > 0 && kappa1 < 0) {
            elementIndex = SBC::ionosphereGrid.findElementNeighbour(elementIndex,0,1);
         } else if (kappa3 > 0 && kappa1 > 0 && kappa2 < 0) {
            elementIndex = SBC::ionosphereGrid.findElementNeighbour(elementIndex,1,2);
         } else if (kappa1 < 0 && kappa2 < 0 && kappa3 > 0) {
            if (handedness > 0) {
               elementIndex = SBC::ionosphereGrid.findElementNeighbour(elementIndex,0,1);
            } else {
               elementIndex = SBC::ionosphereGrid.findElementNeighbour(elementIndex,1,2);
            }
         } else if (kappa1 < 0 && kappa2 > 0 && kappa3 < 0) {
            if (handedness > 0) {
               elementIndex = SBC::ionosphereGrid.findElementNeighbour(elementIndex,0,2);
            } else {
               elementIndex = SBC::ionosphereGrid.findElementNeighbour(elementIndex,0,1);
            }
         } else if (kappa1 > 0 && kappa2 < 0 && kappa3 < 0) {
            if (handedness > 0) {
               elementIndex = SBC::ionosphereGrid.findElementNeighbour(elementIndex,1,2);
            } else {
               elementIndex = SBC::ionosphereGrid.findElementNeighbour(elementIndex,0,2);
            }
         }  else {
            cerr << "This fell through, strange."
            << " kappas " << kappa1 << " " << kappa2 << " " << kappa3
            << " handedness " << handedness
            << " r1 " << r1[0] << " " << r1[1] << " " << r1[2]
            << " r2 " << r2[0] << " " << r2[1] << " " << r2[2]
            << " r3 " << r3[0] << " " << r3[1] << " " << r3[2]
            << " rx " << rx[0] << " " << rx[1] << " " << rx[2]
            << endl;
         }
         if(elementIndex == -1) {
            cerr << __FILE__ << ":" << __LINE__ << ": invalid elementIndex returned for coordinate "
            << x[0] << " " << x[1] << " " << x[2] << " projected to rx " << rx[0] << " " << rx[1] << " " << rx[2]
            << ". Last valid elementIndex: " << oldElementIndex << "." << endl;
            phiprof::stop("ionosphere-VlasovGridCoupling");
            return coupling;
         }
      }
      
      // If we arrived here, we did not find an element to couple to (why?)
      // Return an empty coupling instead
      cerr << "(fieldtracing) Failed to find an ionosphere element to couple to for coordinate " <<
      x[0] << " " << x[1] << " " << x[2] << endl;
      phiprof::stop("fieldtracing-ionosphere-VlasovGridCoupling");
      return coupling;
   }

   /*! Trace magnetic field lines out from ionospheric nodes to record whether they are on an open or closed field line.
   */
   void traceOpenClosedConnection(
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & perBGrid,
      FsGrid< std::array<Real, fsgrids::dperb::N_DPERB>, FS_STENCIL_WIDTH> & dPerBGrid,
      std::vector<SBC::SphericalTriGrid::Node> & nodes
   ) {
      
      // we don't need to do anything if we have no nodes
      if(nodes.size() == 0) {
         return;
      }
      
      phiprof::start("fieldtracing-ionosphere-openclosedTracing");
      // Pick an initial stepsize
      creal stepSize = min(1000e3, technicalGrid.DX / 2.);
      std::vector<Real> nodeTracingStepSize(nodes.size(), stepSize); // In-flight storage of step size, needed when crossing into next MPI domain
      std::vector<Real> reducedNodeTracingStepSize(nodes.size());
      std::array<int, 3> gridSize = technicalGrid.getGlobalSize();
      uint64_t maxTracingSteps = 8 * (gridSize[0] * technicalGrid.DX + gridSize[1] * technicalGrid.DY + gridSize[2] * technicalGrid.DZ) / stepSize;
      
      std::vector<int> nodeMapping(nodes.size(), TracingLineEndType::UNPROCESSED);                                 /*!< For reduction of node coupling */
      std::vector<uint64_t> nodeStepCounter(nodes.size(), 0);                                 /*!< Count number of field line tracing steps */
      std::vector<int> nodeNeedsContinuedTracing(nodes.size(), 1);                    /*!< Flag, whether tracing needs to continue on another task */
      std::vector<std::array<Real, 3>> nodeTracingCoordinates(nodes.size());          /*!< In-flight node upmapping coordinates (for global reduction) */
      
      std::vector<int> nodeTracingStepCount(nodes.size());
      
      for(uint n=0; n<nodes.size(); n++) {
         nodeTracingCoordinates.at(n) = nodes.at(n).x;
      }
      bool anyNodeNeedsTracing;
      
      TracingFieldFunction tracingFullField = [&perBGrid, &dPerBGrid, &technicalGrid](std::array<Real,3>& r, const bool alongB, std::array<Real,3>& b)->bool{
         return traceFullFieldFunction(perBGrid, dPerBGrid, technicalGrid, r, alongB, b);
      };
      
      int itCount=0;
      bool warnMaxStepsExceeded = false;
      do {
         anyNodeNeedsTracing = false;
         itCount++;
         
         #pragma omp parallel
         {
            // Trace node coordinates outwards until a non-sysboundary cell is encountered or the local fsgrid domain has been left.
            #pragma omp for schedule(dynamic)
            for(uint n=0; n<nodes.size(); n++) {
               
               if(!nodeNeedsContinuedTracing[n]) {
                  // This node has already found its target, no need for us to do anything about it.
                  continue;
               }
               SBC::SphericalTriGrid::Node& no = nodes[n];
               
               std::array<Real, 3> x = nodeTracingCoordinates[n];
               std::array<Real, 3> v({0,0,0});
               
               while( true ) {
                  nodeStepCounter[n]++;
                  
                  // Check if the current coordinates (pre-step) are in our own domain.
                  std::array<int, 3> fsgridCell = getLocalFsGridCellIndexForCoord(technicalGrid,x);
                  // If it is not in our domain, somebody else takes care of it.
                  if(fsgridCell[0] == -1) {
                     nodeNeedsContinuedTracing[n] = 0;
                     nodeTracingCoordinates[n] = {0,0,0};
                     nodeTracingStepSize[n]=0;
                     break;
                  }
                  
                  if(nodeStepCounter[n] > maxTracingSteps) {
                     nodeNeedsContinuedTracing[n] = 0;
                     nodeTracingCoordinates[n] = {0,0,0};
                     #pragma omp critical
                     {
                        warnMaxStepsExceeded = true;
                     }
                     break;
                  }
                  
                  // Make one step along the fieldline
                  // If the node is in the North, trace along -B (false for last argument), in the South, trace along B
                  stepFieldLine(x,v, nodeTracingStepSize[n],fieldTracingParameters.min_tracer_dx,technicalGrid.DX/2,fieldTracingParameters.tracingMethod,tracingFullField,(no.x[2] < 0));
                  nodeTracingStepCount[n]++;
                  
                  // Look up the fsgrid cell belonging to these coordinates
                  fsgridCell = getLocalFsGridCellIndexForCoord(technicalGrid,x);
                  std::array<Real, 3> interpolationFactor=getFractionalFsGridCellForCoord(technicalGrid,x);
                  
                  // If we map into the ionosphere, this node is on a closed field line.
                  if(sqrt(x.at(0)*x.at(0) + x.at(1)*x.at(1) + x.at(2)*x.at(2)) < SBC::Ionosphere::innerRadius) {
                     nodeNeedsContinuedTracing[n] = 0;
                     nodeTracingCoordinates[n] = {0,0,0};
                     nodeMapping[n] = TracingLineEndType::CLOSED;
                     break;
                  }
                  
                  // If we map out of the box, this node is on an open field line.
                  if(   x[0] > P::xmax - 4*P::dx_ini
                     || x[0] < P::xmin + 4*P::dx_ini
                     || x[1] > P::ymax - 4*P::dy_ini
                     || x[1] < P::ymin + 4*P::dy_ini
                     || x[2] > P::zmax - 4*P::dz_ini
                     || x[2] < P::zmin + 4*P::dz_ini
                  ) {
                     nodeNeedsContinuedTracing[n] = 0;
                     nodeTracingCoordinates[n] = {0,0,0};
                     nodeMapping[n] = TracingLineEndType::OPEN;
                     break;
                  }
                  
                  // Now, after stepping, if it is no longer in our domain, another MPI rank will pick up later.
                  if(fsgridCell[0] == -1) {
                     nodeNeedsContinuedTracing[n] = 1;
                     nodeTracingCoordinates[n] = x;
                     break;
                  }
               }
            } // pragma omp parallel
         }
         
         // Globally reduce whether any node still needs to be picked up and traced onwards
         std::vector<int> sumNodeNeedsContinuedTracing(nodes.size(), 0);
         std::vector<std::array<Real, 3>> sumNodeTracingCoordinates(nodes.size());
         std::vector<uint64_t> maxNodeStepCounter(nodes.size(), 0);
         MPI_Allreduce(nodeNeedsContinuedTracing.data(), sumNodeNeedsContinuedTracing.data(), nodes.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
         MPI_Allreduce(nodeStepCounter.data(), maxNodeStepCounter.data(), nodes.size(), MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
         if(sizeof(Real) == sizeof(double)) {
            MPI_Allreduce(nodeTracingCoordinates.data(), sumNodeTracingCoordinates.data(), 3*nodes.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(nodeTracingStepSize.data(), reducedNodeTracingStepSize.data(), nodes.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
         } else {
            MPI_Allreduce(nodeTracingCoordinates.data(), sumNodeTracingCoordinates.data(), 3*nodes.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(nodeTracingStepSize.data(), reducedNodeTracingStepSize.data(), nodes.size(), MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
         }
         for(uint n=0; n<nodes.size(); n++) {
            if(sumNodeNeedsContinuedTracing[n] > 0) {
               anyNodeNeedsTracing=true;
               nodeNeedsContinuedTracing[n] = 1;
               
               // Update that nodes' tracing coordinates
               nodeTracingCoordinates[n][0] = sumNodeTracingCoordinates[n][0] / sumNodeNeedsContinuedTracing[n];
               nodeTracingCoordinates[n][1] = sumNodeTracingCoordinates[n][1] / sumNodeNeedsContinuedTracing[n];
               nodeTracingCoordinates[n][2] = sumNodeTracingCoordinates[n][2] / sumNodeNeedsContinuedTracing[n];
               
               nodeStepCounter[n] = maxNodeStepCounter[n];
            }
            nodeTracingStepSize[n] = reducedNodeTracingStepSize[n];
         }
      } while(anyNodeNeedsTracing);
      
      logFile << "(fieldtracing) open-closed tracing traced in " << itCount << " iterations of the tracing loop." << endl;
      
      bool redWarning = false;
      MPI_Allreduce(&warnMaxStepsExceeded, &redWarning, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if(redWarning && rank == MASTER_RANK) {
         logFile << "(fieldtracing) Warning: reached the maximum number of tracing steps " << maxTracingSteps << " allowed for open-closed ionosphere tracing." << endl;
      }
      
      std::vector<int> reducedNodeMapping(nodes.size());
      std::vector<int> reducedNodeTracingStepCount(nodes.size());
      MPI_Allreduce(nodeMapping.data(), reducedNodeMapping.data(), nodes.size(), MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(nodeTracingStepCount.data(), reducedNodeTracingStepCount.data(), nodes.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      
      for(uint n=0; n<nodes.size(); n++) {
         nodes[n].openFieldLine = reducedNodeMapping.at(n);
      }
      
      phiprof::stop("fieldtracing-ionosphere-openclosedTracing");
   }
   
   /*!< Inside the tracing loop for full box + flux rope tracing,
    * trace all field lines across this task's domain.
    * Beware this is inside a threaded region.
    * \sa traceFullBoxConnectionAndFluxRopes
    */
   void stepCellAcrossTaskDomain(
      cint n,
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      TracingFieldFunction & tracingFullField,
      const std::vector<std::array<Real,3>> & cellInitialCoordinates,
      const std::vector<Real> & cellCurvatureRadius,
      std::vector<signed char> & cellNeedsContinuedTracing,
      std::vector<std::array<Real, 3>> & cellTracingCoordinates,
      std::vector<Real> & cellTracingStepSize,
      std::vector<Real> & cellRunningDistance,
      std::vector<Real> & cellMaxExtension,
      std::vector<signed char> & cellConnection,
      bool & warnMaxDistanceExceeded,
      creal maxTracingDistance,
      cuint DIRECTION
   ) {
      std::array<Real, 3> x = cellTracingCoordinates[n];
      std::array<Real, 3> v({0,0,0});
      while( true ) {
         // Check if the current coordinates (pre-step) are in our own domain.
         std::array<int, 3> fsgridCell = getLocalFsGridCellIndexForCoord(technicalGrid,x);
         // If it is not in our domain, somebody else takes care of it.
         if(fsgridCell[0] == -1) {
            cellNeedsContinuedTracing[n] = 0;
            cellTracingCoordinates[n] = {0,0,0};
            cellTracingStepSize[n]=0;
            break;
         }
         
         // Make one step along the fieldline
         // Forward tracing means true for last argument
         stepFieldLine(x,v, cellTracingStepSize[n],100e3,technicalGrid.DX/2,fieldTracingParameters.tracingMethod,tracingFullField,(DIRECTION == Direction::FORWARD));
         cellRunningDistance[n] += cellTracingStepSize[n];
         
         // Look up the fsgrid cell belonging to these coordinates
         fsgridCell = getLocalFsGridCellIndexForCoord(technicalGrid,x);
         
         // If we map into the ionosphere, discard this field line.
         if(sqrt(x.at(0)*x.at(0) + x.at(1)*x.at(1) + x.at(2)*x.at(2)) < fieldTracingParameters.innerBoundaryRadius) {
            cellNeedsContinuedTracing[n] = 0;
            cellTracingCoordinates[n] = {0,0,0};
            cellConnection[n] += TracingLineEndType::CLOSED;
            break;
         }
         
         // If we map out of the box, discard this field line.
         if(
               x[0] > P::xmax - 4*P::dx_ini
            || x[0] < P::xmin + 4*P::dx_ini
            || x[1] > P::ymax - 4*P::dy_ini
            || x[1] < P::ymin + 4*P::dy_ini
            || x[2] > P::zmax - 4*P::dz_ini
            || x[2] < P::zmin + 4*P::dz_ini
         ) {
            cellNeedsContinuedTracing[n] = 0;
            cellTracingCoordinates[n] = {0,0,0};
            cellConnection[n] += TracingLineEndType::OPEN;
            break;
         }
         
         // If we exceed the max tracing distance we're probably looping
         if(cellRunningDistance[n] > maxTracingDistance) {
            cellNeedsContinuedTracing[n] = 0;
            cellTracingCoordinates[n] = {0,0,0};
            cellConnection[n] += TracingLineEndType::DANGLING;
            #pragma omp critical
            {
               warnMaxDistanceExceeded = true;
            }
            break;
         }
         
         // See the longer comment for the function traceFullBoxConnectionAndFluxRopes for details.
         // If we are still in the race for flux rope...
         if(cellConnection[n] < TracingLineEndType::N_TYPES) {
            creal extension = sqrt(
                 (x[0]-(cellInitialCoordinates[n])[0])*(x[0]-(cellInitialCoordinates[n])[0])
               + (x[1]-(cellInitialCoordinates[n])[1])*(x[1]-(cellInitialCoordinates[n])[1])
               + (x[2]-(cellInitialCoordinates[n])[2])*(x[2]-(cellInitialCoordinates[n])[2])
            );
            cellMaxExtension[n] = max(cellMaxExtension[n], extension);
            // ...and if we traced too far from the seed, this is not a flux rope candidate and we do a single +=
            if(extension > fieldTracingParameters.fluxrope_max_curvature_radii_extent*cellCurvatureRadius[n]
               || cellRunningDistance[n] > fieldTracingParameters.fluxrope_max_m_to_trace
            ) {
               cellConnection[n] += TracingLineEndType::N_TYPES;
            } else if(cellRunningDistance[n] > fieldTracingParameters.fluxrope_max_curvature_radii_to_trace*cellCurvatureRadius[n]) {
               // If we're still in the game and reach this limit we have a hit and we do a double +=
               cellConnection[n] += 2*TracingLineEndType::N_TYPES;
            }
         }
         
         // Now, after stepping, if it is no longer in our domain, another MPI rank will pick up later.
         if(fsgridCell[0] == -1) {
            cellNeedsContinuedTracing[n] = 1;
            cellTracingCoordinates[n] = x;
            break;
         }
      } // while true
   }
   
   /*!< \brief Trace magnetic field lines forward and backward from each DCCRG cell to record the connectivity and detect flux ropes.
    *
    * Full box connection and flux rope tracing
    *
    * We are doing two things in one function here as we save a *lot* especially in MPI reductions by combining those. And they are
    * doing very similar things anyway.
    *
    * Firstly we are interested in knowing the ultimate magnetic connection forward and backward from any given location. The
    * locations are the DCCRG cell centers, there's too many fsgrid cells for practical purposes.
    * Therefore we step along the magnetic field until we hit an outer box wall (OPEN) or the inner boundary
    * radius (CLOSED). In rare cases the line might not have reached either of these two termination conditions by the time the
    * field line has reached a length of maxTracingDistance. In that case we call it DANGLING, it likely ended up in a loop
    * somewhere (we don't call it "loop" to avoid confusion with the flux rope tracing). As long as we have not hit any of the above
    * termination conditions, the type is called UNPROCESSED. We allow fieldTracingParameters.fullbox_max_incomplete_lines to remain
    * UNPROCESSED when we exit the loop, that is a fraction of the total field lines we are tracing (number of cells x2 due to
    * forward and backward tracing), as this allows substantial shortening of the total time spent due to the MPI_Allreduce calls
    * occurring when we cross MPI domain boundaries. (NOTE: an algorithm doing direct task-to-task communication will likely be more
    * efficient!) The connection type of the field line is a member of the enum TracingLineEndType and stored in cellFWConnection,
    * cellBWConnection, cellConnection and the reduction arrays.
    * enum TracingLineEndType {
    *    UNPROCESSED,
    *    CLOSED,
    *    OPEN,
    *    DANGLING,
    *    N_TYPES
    * };
    * Once we have obtained a connection type for both directions, we parse the combinations and assign a value in the enum
    * TracingPointConnectionType below to CellParams::CONNECTION that will be written out by the vg_connection DRO.
    * enum TracingPointConnectionType {
    *    CLOSED_CLOSED,
    *    CLOSED_OPEN,
    *    OPEN_CLOSED,
    *    OPEN_OPEN,
    *    CLOSED_DANGLING,
    *    DANGLING_CLOSED,
    *    OPEN_DANGLING,
    *    DANGLING_OPEN,
    *    DANGLING_DANGLING,
    *    INVALID
    * };
    *
    * Secondly, we are interested in finding out whether the seed point/DCCRG cell centre coordinate (again, too many fsgrid cells
    * for sanity) are near/in a flux rope. As this relies also on tracing forward and backward along the field, we piggyback on the
    * full box connection algorithm explained above.
    * We trace along the field up to fieldTracingParameters.fluxrope_max_curvature_radii_to_trace*cellCurvatureRadius[n] and if
    * within that tracing distance we have't extended further than
    * fieldTracingParameters.fluxrope_max_curvature_radii_extent*cellCurvatureRadius[n] we are rolled up tightly enough to consider
    * being close to a flux rope. We'll store the max extension reached into cellMaxExtension[n] for later fine-grained analysis.
    * The arithmetic idea to avoid using yet more arrays:
    * We use the cellFWConnection[n]/cellBWConnection[n]/cellConnection[n] (that can only be UNPROCESSED, CLOSED, OPEN or DANGLING,
    * see TracingLineEndType enum above, ending with N_TYPES). As long as no full box tracing termination condition was reached
    * we are at UNPROCESSED. If we exceed fieldTracingParameters.fluxrope_max_curvature_radii_to_trace*cellCurvatureRadius[n] OR
    * fieldTracingParameters.fluxrope_max_m_to_trace we definitely are not near a flux rope and we mark this by adding N_TYPES to
    * cellConnection[n].
    * If we reach fieldTracingParameters.fluxrope_max_curvature_radii_to_trace*cellCurvatureRadius[n] without hitting the other
    * thresholds or the inner/outer domain limits we are near/at a flux rope and we mark this by adding 2*N_TYPES to
    * cellConnection[n].
    * Later in the tracing when we reach a full box tracing termination condition we add CLOSED, OPEN or DANGLING to
    * cellConnection[n]. If we reached such termination condition before the flux rope method reached a conclusion it's fine too,
    * nothing has been added to cellConnection[n].
    * At the very end, we check whether both cellFWConnection[n] and cellBWConnection[n] >= 2*TracingLineEndType::N_TYPES. If yes,
    * this cell is near a fluxrope. This means we store the value of cellMaxExtension[n] into CellParams::FLUXROPE. Otherwise we
    * store zero.
    * After that we apply % TracingLineEndType::N_TYPES to recover values UNPROCESSED, CLOSED, OPEN, DANGLING in
    * cellFWConnection[n] and cellBWConnection[n] so that we can assign the connection types for the full box connection described
    * above.
    *
    * As a freebie since we computed the curvature anyway for flux rope tracing we store that into CellParams::CURVATUREX/Y/Z.
    *
    * \sa stepCellAcrossTaskDomain
    */
   void traceFullBoxConnectionAndFluxRopes(
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & perBGrid,
      FsGrid< std::array<Real, fsgrids::dperb::N_DPERB>, FS_STENCIL_WIDTH> & dPerBGrid,
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid
   ) {
      phiprof::start("fieldtracing-fullAndFluxTracing");
      
      std::vector<CellID> localDccrgCells = getLocalCells();
      int localDccrgSize = localDccrgCells.size();
      int globalDccrgSize;
      MPI_Allreduce(&localDccrgSize, &globalDccrgSize, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      int commSize;
      MPI_Comm_size(MPI_COMM_WORLD, &commSize);
      std::vector<int> amounts(commSize);
      std::vector<int> displacements(commSize);
      std::vector<CellID> allDccrgCells(globalDccrgSize);
      MPI_Allgather(&localDccrgSize, 1, MPI_INT, amounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
      for(int i=1; i<commSize; i++) {
         displacements[i] = displacements[i-1] + amounts[i-1];
      }
      MPI_Allgatherv(localDccrgCells.data(), localDccrgSize, MPI_UINT64_T, allDccrgCells.data(), amounts.data(), displacements.data(), MPI_UINT64_T, MPI_COMM_WORLD);
      
      // Pick an initial stepsize
      creal stepSize = min(1000e3, technicalGrid.DX / 2.);
      std::vector<Real> cellFWTracingStepSize(globalDccrgSize, stepSize); // In-flight storage of step size, needed when crossing into next MPI domain
      std::vector<Real> cellBWTracingStepSize(globalDccrgSize, stepSize); // In-flight storage of step size, needed when crossing into next MPI domain
      
      std::array<int, 3> gridSize = technicalGrid.getGlobalSize();
      // This is a heuristic considering how far an IMF+dipole combo can sensibly stretch in the box before we're safe to assume it's rolled up more or less pathologically.
      creal maxTracingDistance = 4 * (gridSize[0] * technicalGrid.DX + gridSize[1] * technicalGrid.DY + gridSize[2] * technicalGrid.DZ);
      if(maxTracingDistance < fieldTracingParameters.fluxrope_max_m_to_trace) {
         int myRank;
         MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
         if(myRank == MASTER_RANK) {
            cerr << "WARNING: Field tracing maxTracingDistance is smaller than cfg fieldtracing.fluxrope_max_m_to_trace, consider changing the latter.\n";
         }
      }
      
      std::vector<Real> cellCurvatureRadius(globalDccrgSize, 0);
      std::vector<Real> reducedCellCurvatureRadius(globalDccrgSize);
      
      std::vector<signed char> cellFWConnection(globalDccrgSize, TracingLineEndType::UNPROCESSED); /*!< For reduction of node coupling */
      std::vector<signed char> cellBWConnection(globalDccrgSize, TracingLineEndType::UNPROCESSED); /*!< For reduction of node coupling */
      
      std::vector<signed char> cellNeedsContinuedFWTracing(globalDccrgSize, 1); /*!< Flag, whether tracing needs to continue on another task */
      std::vector<signed char> cellNeedsContinuedBWTracing(globalDccrgSize, 1); /*!< Flag, whether tracing needs to continue on another task */
      std::vector<std::array<Real, 3>> cellFWTracingCoordinates(globalDccrgSize); /*!< In-flight node upmapping coordinates (for global reduction) */
      std::vector<std::array<Real, 3>> cellBWTracingCoordinates(globalDccrgSize); /*!< In-flight node upmapping coordinates (for global reduction) */
      std::vector<Real> cellFWRunningDistance(globalDccrgSize, 0);
      std::vector<Real> cellBWRunningDistance(globalDccrgSize, 0);
      
      // This we need only once and not forward and backward separately as we'll only record the max
      std::vector<Real> cellMaxExtension(globalDccrgSize, 0);
      
      // These guys are needed in the reductions
      std::vector<signed char> storedCellNeedsContinuedFWTracing(globalDccrgSize, 0);
      std::vector<signed char> storedCellNeedsContinuedBWTracing(globalDccrgSize, 0);
      std::vector<std::array<Real, 3>> sumCellFWTracingCoordinates(globalDccrgSize);
      std::vector<std::array<Real, 3>> sumCellBWTracingCoordinates(globalDccrgSize);
      std::vector<Real> reducedCellFWRunningDistance(globalDccrgSize, 0);
      std::vector<Real> reducedCellBWRunningDistance(globalDccrgSize, 0);
      std::vector<Real> reducedCellFWTracingStepSize(globalDccrgSize);
      std::vector<Real> reducedCellBWTracingStepSize(globalDccrgSize);
      std::vector<signed char> reducedCellFWConnection(globalDccrgSize);
      std::vector<signed char> reducedCellBWConnection(globalDccrgSize);
      
      phiprof::start("first-loop");
      for(int n=0; n<globalDccrgSize; n++) {
         const CellID id = allDccrgCells[n];
         cellFWTracingCoordinates.at(n) = mpiGrid.get_center(id);
         cellBWTracingCoordinates.at(n) = cellFWTracingCoordinates.at(n);
         if(mpiGrid.is_local(id)) {
            if((mpiGrid[id]->sysBoundaryFlag != sysboundarytype::NOT_SYSBOUNDARY)
               || cellFWTracingCoordinates[n][0] > P::xmax - 4*P::dx_ini
               || cellFWTracingCoordinates[n][0] < P::xmin + 4*P::dx_ini
               || cellFWTracingCoordinates[n][1] > P::ymax - 4*P::dy_ini
               || cellFWTracingCoordinates[n][1] < P::ymin + 4*P::dy_ini
               || cellFWTracingCoordinates[n][2] > P::zmax - 4*P::dz_ini
               || cellFWTracingCoordinates[n][2] < P::zmin + 4*P::dz_ini
            ) {
               cellNeedsContinuedFWTracing[n] = 0;
               cellNeedsContinuedBWTracing[n] = 0;
               cellFWTracingCoordinates[n] = {0,0,0};
               cellBWTracingCoordinates[n] = {0,0,0};
               cellFWTracingStepSize[n] = 0;
               cellBWTracingStepSize[n] = 0;
            } else {
               cellCurvatureRadius[n] = 1 / sqrt(mpiGrid[id]->parameters[CellParams::CURVATUREX]*mpiGrid[id]->parameters[CellParams::CURVATUREX] + mpiGrid[id]->parameters[CellParams::CURVATUREY]*mpiGrid[id]->parameters[CellParams::CURVATUREY] + mpiGrid[id]->parameters[CellParams::CURVATUREZ]*mpiGrid[id]->parameters[CellParams::CURVATUREZ]);
               if(fieldTracingParameters.fluxrope_max_curvature_radii_to_trace*cellCurvatureRadius[n] > fieldTracingParameters.fluxrope_max_m_to_trace) {
                  cellCurvatureRadius[n] = 0; // This will discard the field lines in the first iteration below.
               }
            }
         }
      }
      phiprof::stop("first-loop");
      
      const std::vector<std::array<Real,3>> cellInitialCoordinates = cellFWTracingCoordinates;
      
      MPI_Allreduce(cellNeedsContinuedFWTracing.data(), storedCellNeedsContinuedFWTracing.data(), globalDccrgSize, MPI_SIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(cellNeedsContinuedBWTracing.data(), storedCellNeedsContinuedBWTracing.data(), globalDccrgSize, MPI_SIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
      if(sizeof(Real) == sizeof(double)) {
         MPI_Allreduce(cellFWTracingStepSize.data(), reducedCellFWTracingStepSize.data(), globalDccrgSize, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
         MPI_Allreduce(cellBWTracingStepSize.data(), reducedCellBWTracingStepSize.data(), globalDccrgSize, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
         MPI_Allreduce(cellCurvatureRadius.data(), reducedCellCurvatureRadius.data(), globalDccrgSize, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      } else {
         MPI_Allreduce(cellFWTracingStepSize.data(), reducedCellFWTracingStepSize.data(), globalDccrgSize, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
         MPI_Allreduce(cellBWTracingStepSize.data(), reducedCellBWTracingStepSize.data(), globalDccrgSize, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
         MPI_Allreduce(cellCurvatureRadius.data(), reducedCellCurvatureRadius.data(), globalDccrgSize, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      }
      // Don't swap the first two as the stored guys are used below
      cellNeedsContinuedFWTracing = storedCellNeedsContinuedFWTracing;
      cellNeedsContinuedBWTracing = storedCellNeedsContinuedBWTracing;
      cellFWTracingStepSize.swap(reducedCellFWTracingStepSize);
      cellBWTracingStepSize.swap(reducedCellBWTracingStepSize);
      cellCurvatureRadius.swap(reducedCellCurvatureRadius);
      
      TracingFieldFunction tracingFullField = [&perBGrid, &dPerBGrid, &technicalGrid](std::array<Real,3>& r, const bool alongB, std::array<Real,3>& b)->bool{
         return traceFullFieldFunction(perBGrid, dPerBGrid, technicalGrid, r, alongB, b);
      };
      int itCount = 0;
      bool warnMaxDistanceExceeded = false;
      int cellsToDoFullBox, cellsToDoFluxRopes;
      
      phiprof::start("loop");
      #pragma omp parallel shared(cellsToDoFullBox,cellsToDoFluxRopes)
      {
         do { // while(either leftover fraction is not achieved
            #pragma omp single
            {
               itCount++;
            }
            // Trace node coordinates forward and backwards until a non-sysboundary cell is encountered or the local fsgrid domain has been left.
            #pragma omp for schedule(dynamic)
            for(int n=0; n<globalDccrgSize; n++) {
               if(cellNeedsContinuedFWTracing[n] > 0) {
                  stepCellAcrossTaskDomain(
                     n,
                     technicalGrid,
                     tracingFullField,
                     cellInitialCoordinates,
                     cellCurvatureRadius,
                     cellNeedsContinuedFWTracing,
                     cellFWTracingCoordinates,
                     cellFWTracingStepSize,
                     cellFWRunningDistance,
                     cellMaxExtension,
                     cellFWConnection,
                     warnMaxDistanceExceeded,
                     maxTracingDistance,
                     Direction::FORWARD
                  );
               }
               if(cellNeedsContinuedBWTracing[n] > 0) {
                  stepCellAcrossTaskDomain(
                     n,
                     technicalGrid,
                     tracingFullField,
                     cellInitialCoordinates,
                     cellCurvatureRadius,
                     cellNeedsContinuedBWTracing,
                     cellBWTracingCoordinates,
                     cellBWTracingStepSize,
                     cellBWRunningDistance,
                     cellMaxExtension,
                     cellBWConnection,
                     warnMaxDistanceExceeded,
                     maxTracingDistance,
                     Direction::BACKWARD
                  );
               }
            } // for
            
            // Globally reduce whether any node still needs to be picked up and traced onwards
            #pragma omp barrier
            phiprof::start("MPI-loop");
            #pragma omp master
            {
               std::vector<int> indicesToReduceFW, indicesToReduceBW;
               std::vector<signed char> smallCellNeedsContinuedFWTracing, smallCellNeedsContinuedBWTracing;
               std::vector<std::array<Real, 3>> smallCellFWTracingCoordinates, smallCellBWTracingCoordinates;
               std::vector<Real> smallCellFWRunningDistance, smallCellBWRunningDistance;
               std::vector<Real> smallCellFWTracingStepSize, smallCellBWTracingStepSize;
               std::vector<signed char> smallCellFWConnection, smallCellBWConnection;

               for(int n=0; n<globalDccrgSize; n++) {
                  if(storedCellNeedsContinuedFWTracing[n]) { // the old one has the previous round's data
                     indicesToReduceFW.push_back(n);
                     smallCellNeedsContinuedFWTracing.push_back(cellNeedsContinuedFWTracing[n]);
                     smallCellFWTracingCoordinates.push_back(cellFWTracingCoordinates[n]);
                     smallCellFWRunningDistance.push_back(cellFWRunningDistance[n]);
                     smallCellFWTracingStepSize.push_back(cellFWTracingStepSize[n]);
                     smallCellFWConnection.push_back(cellFWConnection[n]);
                  }
                  if(storedCellNeedsContinuedBWTracing[n]) { // the old one has the previous round's data
                     indicesToReduceBW.push_back(n);
                     smallCellNeedsContinuedBWTracing.push_back(cellNeedsContinuedBWTracing[n]);
                     smallCellBWTracingCoordinates.push_back(cellBWTracingCoordinates[n]);
                     smallCellBWRunningDistance.push_back(cellBWRunningDistance[n]);
                     smallCellBWTracingStepSize.push_back(cellBWTracingStepSize[n]);
                     smallCellBWConnection.push_back(cellBWConnection[n]);
                  }
               }
               int smallSizeFW = indicesToReduceFW.size();
               int smallSizeBW = indicesToReduceBW.size();
               
               std::vector<signed char> smallReducedCellNeedsContinuedFWTracing(smallSizeFW), smallReducedCellNeedsContinuedBWTracing(smallSizeBW);
               std::vector<std::array<Real, 3>> smallSumCellFWTracingCoordinates(smallSizeFW), smallSumCellBWTracingCoordinates(smallSizeBW);
               std::vector<Real> smallReducedCellFWRunningDistance(smallSizeFW), smallReducedCellBWRunningDistance(smallSizeBW);
               std::vector<Real> smallReducedCellFWTracingStepSize(smallSizeFW), smallReducedCellBWTracingStepSize(smallSizeBW);
               std::vector<signed char> smallReducedCellFWConnection(smallSizeFW), smallReducedCellBWConnection(smallSizeBW);
               
               MPI_Allreduce(smallCellNeedsContinuedFWTracing.data(), smallReducedCellNeedsContinuedFWTracing.data(), smallSizeFW, MPI_SIGNED_CHAR, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(smallCellNeedsContinuedBWTracing.data(), smallReducedCellNeedsContinuedBWTracing.data(), smallSizeBW, MPI_SIGNED_CHAR, MPI_SUM, MPI_COMM_WORLD);
               MPI_Allreduce(smallCellFWConnection.data(), smallReducedCellFWConnection.data(), smallSizeFW, MPI_SIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);
               MPI_Allreduce(smallCellBWConnection.data(), smallReducedCellBWConnection.data(), smallSizeBW, MPI_SIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);
               if(sizeof(Real) == sizeof(double)) {
                  MPI_Allreduce(smallCellFWTracingCoordinates.data(), smallSumCellFWTracingCoordinates.data(), 3*smallSizeFW, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                  MPI_Allreduce(smallCellBWTracingCoordinates.data(), smallSumCellBWTracingCoordinates.data(), 3*smallSizeBW, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                  MPI_Allreduce(smallCellFWTracingStepSize.data(), smallReducedCellFWTracingStepSize.data(), smallSizeFW, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                  MPI_Allreduce(smallCellBWTracingStepSize.data(), smallReducedCellBWTracingStepSize.data(), smallSizeBW, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                  MPI_Allreduce(smallCellFWRunningDistance.data(), smallReducedCellFWRunningDistance.data(), smallSizeFW, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                  MPI_Allreduce(smallCellBWRunningDistance.data(), smallReducedCellBWRunningDistance.data(), smallSizeBW, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
               } else {
                  MPI_Allreduce(smallCellFWTracingCoordinates.data(), smallSumCellFWTracingCoordinates.data(), 3*smallSizeFW, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                  MPI_Allreduce(smallCellBWTracingCoordinates.data(), smallSumCellBWTracingCoordinates.data(), 3*smallSizeBW, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                  MPI_Allreduce(smallCellFWTracingStepSize.data(), smallReducedCellFWTracingStepSize.data(), smallSizeFW, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
                  MPI_Allreduce(smallCellBWTracingStepSize.data(), smallReducedCellBWTracingStepSize.data(), smallSizeBW, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
                  MPI_Allreduce(smallCellFWRunningDistance.data(), smallReducedCellFWRunningDistance.data(), smallSizeFW, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
                  MPI_Allreduce(smallCellBWRunningDistance.data(), smallReducedCellBWRunningDistance.data(), smallSizeBW, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
               }
               for(int n=0; n<smallSizeFW; n++) {
                  cellFWTracingStepSize[indicesToReduceFW[n]] = smallReducedCellFWTracingStepSize[n];
                  cellFWRunningDistance[indicesToReduceFW[n]] = smallReducedCellFWRunningDistance[n];
                  cellNeedsContinuedFWTracing[indicesToReduceFW[n]] = smallReducedCellNeedsContinuedFWTracing[n];
                  cellFWConnection[indicesToReduceFW[n]] = smallReducedCellFWConnection[n];
                  cellFWTracingCoordinates[indicesToReduceFW[n]] = smallSumCellFWTracingCoordinates[n];
               }
               for(int n=0; n<smallSizeBW; n++) {
                  cellBWTracingStepSize[indicesToReduceBW[n]] = smallReducedCellBWTracingStepSize[n];
                  cellBWRunningDistance[indicesToReduceBW[n]] = smallReducedCellBWRunningDistance[n];
                  cellNeedsContinuedBWTracing[indicesToReduceBW[n]] = smallReducedCellNeedsContinuedBWTracing[n];
                  cellBWConnection[indicesToReduceBW[n]] = smallReducedCellBWConnection[n];
                  cellBWTracingCoordinates[indicesToReduceBW[n]] = smallSumCellBWTracingCoordinates[n];
               }
            }
            #pragma omp barrier
            phiprof::stop("MPI-loop");
            #pragma omp single
            {
               cellsToDoFullBox = 0;
               cellsToDoFluxRopes = 0;
               // These are used for the small arrays ~70 lines up.
               storedCellNeedsContinuedFWTracing = cellNeedsContinuedFWTracing;
               storedCellNeedsContinuedBWTracing = cellNeedsContinuedBWTracing;
            }
            #pragma omp for schedule(dynamic) reduction(+:cellsToDoFullBox) reduction(+:cellsToDoFluxRopes)
            for(int n=0; n<globalDccrgSize; n++) {
               if(cellNeedsContinuedFWTracing[n]) {
                  cellsToDoFullBox++;
                  if(cellFWConnection[n] == TracingLineEndType::UNPROCESSED) {
                     cellsToDoFluxRopes++;
                  }
               }
               if(cellNeedsContinuedBWTracing[n]) {
                  cellsToDoFullBox++;
                  if(cellBWConnection[n] == TracingLineEndType::UNPROCESSED) {
                     cellsToDoFluxRopes++;
                  }
               }
            }
            #pragma omp barrier
         } while(!(
            cellsToDoFullBox <= fieldTracingParameters.fullbox_max_incomplete_lines * 2 * globalDccrgSize
            && cellsToDoFluxRopes <= fieldTracingParameters.fluxrope_max_incomplete_lines * 2 * globalDccrgSize
         ));
      } // pragma omp parallel
      phiprof::stop("loop");
      
      logFile << "(fieldtracing) combined flux rope + full box tracing traced in " << itCount
         << " iterations of the tracing loop with flux rope " << cellsToDoFluxRopes
         << ", full box " << cellsToDoFullBox
         << " remaining incomplete field lines (total spatial cells " << globalDccrgSize
         << ")." << endl;
      
      bool redWarning = false;
      MPI_Allreduce(&warnMaxDistanceExceeded, &redWarning, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if(redWarning && rank == MASTER_RANK) {
         logFile << "(fieldtracing) Warning: reached the maximum tracing distance " << maxTracingDistance << " m allowed for combined flux rope + full box tracing." << endl;
      }
      
      // Now we're all done we want to reduce the max extension so we can store it
      std::vector<Real> reducedCellMaxExtension(globalDccrgSize);
      if(sizeof(Real) == sizeof(double)) {
         MPI_Allreduce(cellMaxExtension.data(), reducedCellMaxExtension.data(), globalDccrgSize, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      } else {
         MPI_Allreduce(cellMaxExtension.data(), reducedCellMaxExtension.data(), globalDccrgSize, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      }
      
      phiprof::start("final-loop");
      for(int n=0; n<globalDccrgSize; n++) {
         const CellID id = allDccrgCells.at(n);
         if(mpiGrid.is_local(id)) {
            // Handle flux ropes
            mpiGrid[id]->parameters[CellParams::FLUXROPE] = 0;
            const std::array<Real, 3> x = mpiGrid.get_center(id);
            // We use cellXWConnection as we swapped with reducedCellXWConnection.
            // Earlier, if we marked nothing (e.g. hit a wall or ionosphere before making a call) cellXWConnection[n] is less than N_TYPES.
            // If we went beyond the thresholds we did += N_TYPES, which is also not a positive hit.
            // If we identified a flux rope we did a double += by N_TYPES and we pick them out with this.
            if(   cellFWConnection[n] >= 2*TracingLineEndType::N_TYPES
               && cellBWConnection[n] >= 2*TracingLineEndType::N_TYPES
            ) {
               mpiGrid[id]->parameters[CellParams::FLUXROPE] = reducedCellMaxExtension[n] / cellCurvatureRadius[n];
            }
            
            // Now remove the flux rope mark so we're left with UNPROCESSED, OPEN, CLOSED, DANGLING.
            cellFWConnection[n] %= TracingLineEndType::N_TYPES;
            cellBWConnection[n] %= TracingLineEndType::N_TYPES;
            
            // Handle full box connection
            mpiGrid[id]->parameters[CellParams::CONNECTION] = TracingPointConnectionType::INVALID;
            if (cellFWConnection[n] == TracingLineEndType::CLOSED && cellBWConnection[n] == TracingLineEndType::CLOSED) {
               mpiGrid[id]->parameters[CellParams::CONNECTION] = TracingPointConnectionType::CLOSED_CLOSED;
            }
            if (cellFWConnection[n] == TracingLineEndType::CLOSED && cellBWConnection[n] == TracingLineEndType::OPEN) {
               mpiGrid[id]->parameters[CellParams::CONNECTION] = TracingPointConnectionType::CLOSED_OPEN;
            }
            if (cellFWConnection[n] == TracingLineEndType::OPEN && cellBWConnection[n] == TracingLineEndType::CLOSED) {
               mpiGrid[id]->parameters[CellParams::CONNECTION] = TracingPointConnectionType::OPEN_CLOSED;
            }
            if (cellFWConnection[n] == TracingLineEndType::OPEN && cellBWConnection[n] == TracingLineEndType::OPEN) {
               mpiGrid[id]->parameters[CellParams::CONNECTION] = TracingPointConnectionType::OPEN_OPEN;
            }
            if (cellFWConnection[n] == TracingLineEndType::CLOSED && cellBWConnection[n] == TracingLineEndType::DANGLING) {
               mpiGrid[id]->parameters[CellParams::CONNECTION] = TracingPointConnectionType::CLOSED_DANGLING;
            }
            if (cellFWConnection[n] == TracingLineEndType::DANGLING && cellBWConnection[n] == TracingLineEndType::CLOSED) {
               mpiGrid[id]->parameters[CellParams::CONNECTION] = TracingPointConnectionType::DANGLING_CLOSED;
            }
            if (cellFWConnection[n] == TracingLineEndType::OPEN && cellBWConnection[n] == TracingLineEndType::DANGLING) {
               mpiGrid[id]->parameters[CellParams::CONNECTION] = TracingPointConnectionType::OPEN_DANGLING;
            }
            if (cellFWConnection[n] == TracingLineEndType::DANGLING && cellBWConnection[n] == TracingLineEndType::OPEN) {
               mpiGrid[id]->parameters[CellParams::CONNECTION] = TracingPointConnectionType::DANGLING_OPEN;
            }
            if (cellFWConnection[n] == TracingLineEndType::DANGLING && cellBWConnection[n] == TracingLineEndType::DANGLING) {
               mpiGrid[id]->parameters[CellParams::CONNECTION] = TracingPointConnectionType::DANGLING_DANGLING;
            }
         }
      }
      phiprof::stop("final-loop");
      phiprof::stop("fieldtracing-fullAndFluxTracing");
   }
} // namespace FieldTracing
