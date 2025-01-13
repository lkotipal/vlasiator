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

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

//#pragma once

#ifndef IPSHOCK_H
#define IPSHOCK_H

#include "../../definitions.h"
#include "../projectTriAxisSearch.h"

namespace projects {

   struct IPShockSpeciesParameters {
      Real V0u[3];
      Real DENSITYu;
      Real TEMPERATUREu;
      Real V0utangential;
      Real Vucosphi;

      Real V0d[3];
      Real DENSITYd;
      Real TEMPERATUREd;
      Real V0dtangential;
      Real Vdcosphi;

      int Vyusign;
      int Vydsign;
      int Vzusign;
      int Vzdsign;

      Real maxwCutoff;
   };

   class IPShock: public TriAxisSearch {
   public:
      IPShock();
      virtual ~IPShock();

      virtual bool initialize(void) override;
      static void addParameters(void);
      virtual void getParameters(void) override;

      virtual void setProjectBField(
         FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & perBGrid,
         FsGrid< std::array<Real, fsgrids::bgbfield::N_BGB>, FS_STENCIL_WIDTH> & BgBGrid,
         FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid
         ) override;

      virtual Realf fillPhaseSpace(spatial_cell::SpatialCell *cell,
                                   const uint popID,
                                   const uint nRequested) const override;
      virtual Realf probePhaseSpace(spatial_cell::SpatialCell *cell,
                                    const uint popID,
                                    Real vx_in, Real vy_in, Real vz_in) const override;
      virtual std::vector<std::array<Real, 3>> getV0(creal x, creal y, creal z, const uint popID) const override;
      //virtual void calcCellParameters(Real* cellParams,creal& t);
      virtual void calcCellParameters(spatial_cell::SpatialCell* cell, creal& t) override;
      bool refineSpatialCells( dccrg::Dccrg<spatial_cell::SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid ) const override;
      // Interpolate between up- and downstream quantities
      // based on position
      Real interpolate(Real u, Real d, Real x) const;

      // Upstream bulk values
      Real B0u[3];
      Real B0utangential;
      // Downstream bulk values
      Real B0d[3];
      Real B0dtangential;

      // Flow direction definitions
      Real Bucosphi;
      Real Bdcosphi;
      int Byusign;
      int Bydsign;
      int Bzusign;
      int Bzdsign;

      Real Shockwidth;
      Real AMR_L1width;
      Real AMR_L2width;
      Real AMR_L3width;
      Real AMR_L4width;

      std::vector<IPShockSpeciesParameters> speciesParams;

   } ; //class IPShock
} // namespace projects

#endif
