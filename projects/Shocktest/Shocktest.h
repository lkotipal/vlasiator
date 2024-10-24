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

#ifndef SHOCKTEST_H
#define SHOCKTEST_H

#include <vector>

#include "../../definitions.h"
#include "../../spatial_cell_wrapper.hpp"
#include "../project.h"
#include "../projectTriAxisSearch.h"

namespace projects {
   class Shocktest: public TriAxisSearch {
    public:
      Shocktest(); // Constructor
      virtual ~Shocktest(); // Destructor
      
      virtual bool initialize(void);
      static void addParameters(void);
      virtual void getParameters(void);
      
      enum {
         LEFT,
         RIGHT
      };
      Real rho[2];
      Real T[2];
      Real Vx[2];
      Real Vy[2];
      Real Vz[2];
      Real Bx[2];
      Real By[2];
      Real Bz[2];
      
      virtual Realf fillPhaseSpace(spatial_cell::SpatialCell *cell,
                                  const uint popID,
                                  const uint nRequested,
                                  Realf* bufferData,
                                  vmesh::GlobalID *GIDlist) const override;
      virtual Realf probePhaseSpace(spatial_cell::SpatialCell *cell,
                                    const uint popID,
                                    Real vx_in, Real vy_in, Real vz_in) const override;

      virtual void setProjectBField(
         FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & perBGrid,
         FsGrid< std::array<Real, fsgrids::bgbfield::N_BGB>, FS_STENCIL_WIDTH> & BgBGrid,
         FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid
      );
      
      virtual void calcCellParameters(spatial_cell::SpatialCell* cell,creal& t);
         
      virtual std::vector<std::array<Real, 3> > getV0(
         creal x,
         creal y,
         creal z,
         const uint popID
         ) const;
         
   }; // Class Shocktest

} // Namespace projects
#endif

