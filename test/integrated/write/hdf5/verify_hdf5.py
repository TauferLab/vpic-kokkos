#!/usr/bin/env python3
"""
HDF5 I/O Verification with Automatic Cleanup
Verifies both M2M (many-to-many) and M2O (many-to-one) HDF5 output modes
"""

import sys
import os
import argparse
import shutil
import h5py
import numpy as np

# Add common utilities to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from verify_utils import *

# =============================================================================
# CONFIGURATION
# =============================================================================
NX, NY, NZ = 4, 4, 4

# =============================================================================
# CLEANUP
# =============================================================================
def cleanup_test_output():
    """Remove test output directories"""
    dirs_to_remove = ["field", "ehydro", "particle", "rundata"]
    for dirname in dirs_to_remove:
        if os.path.exists(dirname):
            try:
                shutil.rmtree(dirname)
                print(f"  [CLEANUP] Removed {dirname}/")
            except Exception as e:
                print(f"  [WARN] Could not remove {dirname}: {e}")

# =============================================================================
# HDF5-SPECIFIC READERS
# =============================================================================
def read_hdf5_particles(fname):
    """Read HDF5 particle data (both M2M and M2O formats)"""
    if not os.path.exists(fname):
        return None
    
    try:
        with h5py.File(fname, 'r') as f:
            # New format: compound dataset named 'particles'
            if 'particles' in f:
                dset = f['particles']
                return {
                    'ux': dset['ux'][:],
                    'uy': dset['uy'][:],
                    'uz': dset['uz'][:]
                }
            # Fallback: individual datasets (legacy format)
            elif 'ux' in f:
                return {
                    'ux': f['ux'][:],
                    'uy': f['uy'][:],
                    'uz': f['uz'][:]
                }
            else:
                raise Exception("Unknown HDF5 particle format")
    except Exception as e:
        print(f"    [ERR] Failed to read {fname}: {e}")
        return None

def read_hdf5_field_component(fname, component):
    """Read single field component from HDF5 file"""
    if not os.path.exists(fname):
        return None
    
    try:
        with h5py.File(fname, 'r') as f:
            if component not in f:
                return None
            return f[component][:]
    except Exception as e:
        print(f"    [ERR] Failed to read {fname}/{component}: {e}")
        return None

def read_hdf5_hydro_component(fname, component):
    """Read single hydro component from HDF5 file"""
    return read_hdf5_field_component(fname, component)  # Same format

# =============================================================================
# SINGLE-RANK TESTS
# =============================================================================
def verify_single():
    """Verify single-rank HDF5 output (M2M and M2O modes)"""
    suite = TestSuite("HDF5 Single-Rank")
    
    # Test 1: Particle M2M vs M2O Consistency
    def test_particle_consistency():
        m2m = read_hdf5_particles("particle/test_particles_m2m.electron.0.0.h5")
        m2o = read_hdf5_particles("particle/test_particles_m2o.electron.0.h5")
        
        if m2m is None or m2o is None:
            raise Exception("Failed to read particle files")
        
        # Check counts match
        if len(m2m['ux']) != len(m2o['ux']):
            raise Exception(f"Particle count mismatch: M2M={len(m2m['ux'])}, M2O={len(m2o['ux'])}")
        
        # Check data matches (after sorting, since ordering may differ)
        for comp in ['ux', 'uy', 'uz']:
            m2m_sorted = np.sort(m2m[comp])
            m2o_sorted = np.sort(m2o[comp])
            diff = np.max(np.abs(m2m_sorted - m2o_sorted))
            if diff > 1e-6:
                raise Exception(f"Particle {comp} mismatch between M2M and M2O: {diff}")
    
    suite.run_test("Particle M2M vs M2O Consistency", test_particle_consistency)
    
    # Test 2: Particle Anisotropic Velocity Pattern
    def test_particle_physics():
        m2m = read_hdf5_particles("particle/test_particles_m2m.electron.0.0.h5")
        if m2m is None:
            raise Exception("Failed to read particle file")
        
        # Generate expected pattern (anisotropic: ux=1x, uy=2x, uz=3x)
        expected_base = []
        for iz in range(1, NZ+1):
            for iy in range(1, NY+1):
                for ix in range(1, NX+1):
                    gid = iz * NY * NX + iy * NX + ix
                    expected_base.append(gid * 1.0e-9)
        expected_base = np.sort(np.array(expected_base))
        
        # Verify anisotropy
        ux_sorted = np.sort(m2m['ux'])
        uy_sorted = np.sort(m2m['uy'])
        uz_sorted = np.sort(m2m['uz'])
        
        diff_x = np.max(np.abs(ux_sorted - expected_base))
        diff_y = np.max(np.abs(uy_sorted - expected_base * 2.0))
        diff_z = np.max(np.abs(uz_sorted - expected_base * 3.0))
        
        if max(diff_x, diff_y, diff_z) > 1e-6:
            raise Exception(f"Particle anisotropy mismatch: X={diff_x:.1e}, Y={diff_y:.1e}, Z={diff_z:.1e}")
    
    suite.run_test("Particle Anisotropic Pattern", test_particle_physics)
    
    # Test 3: Field M2M vs M2O Consistency
    def test_field_consistency():
        components = ['ex', 'ey', 'ez', 'cbx', 'cby', 'cbz']
        
        for comp in components:
            m2m = read_hdf5_field_component("field/T.0/test_fields_m2m.0.0.h5", comp)
            m2o = read_hdf5_field_component("field/T.0/test_fields_m2o.0.h5", comp)
            
            if m2m is None or m2o is None:
                raise Exception(f"Failed to read field component {comp}")
            
            diff = np.max(np.abs(m2m - m2o))
            if diff > 1e-4:
                raise Exception(f"Field {comp} mismatch between M2M and M2O: {diff}")
    
    suite.run_test("Field M2M vs M2O Consistency", test_field_consistency)
    
    # Test 4: Field Data Pattern
    def test_field_pattern():
        # Generate expected pattern (1000 + cell_idx * 1e-8 + component_offset)
        expected_base = []
        for iz in range(1, NZ+1):
            for iy in range(1, NY+1):
                for ix in range(1, NX+1):
                    cell_idx = iz * (NY+2) * (NX+2) + iy * (NX+2) + ix
                    expected_base.append(1000.0 + cell_idx * 1.0e-8)
        expected_base = np.array(expected_base).reshape(NZ, NY, NX)
        
        components = [
            ('ex', 0.1), ('ey', 0.2), ('ez', 0.3),
            ('cbx', 0.4), ('cby', 0.5), ('cbz', 0.6)
        ]
        
        for name, offset in components:
            data = read_hdf5_field_component("field/T.0/test_fields_m2m.0.0.h5", name)
            if data is None:
                raise Exception(f"Failed to read field component {name}")
            
            expected = expected_base + offset
            diff = np.max(np.abs(data - expected))
            
            if diff > 1e-4:
                raise Exception(f"Field {name} pattern mismatch: {diff}")
    
    suite.run_test("Field Data Pattern", test_field_pattern)
    
    # Test 5: Hydro M2M vs M2O Consistency
    def test_hydro_consistency():
        components = ['jx', 'jy', 'jz', 'rho']
        
        for comp in components:
            m2m = read_hdf5_hydro_component("ehydro/T.0/test_hydro_m2m.0.0.h5", comp)
            m2o = read_hdf5_hydro_component("ehydro/T.0/test_hydro_m2o.0.h5", comp)
            
            if m2m is None or m2o is None:
                raise Exception(f"Failed to read hydro component {comp}")
            
            diff = np.max(np.abs(m2m - m2o))
            if diff > 1e-5:
                raise Exception(f"Hydro {comp} mismatch between M2M and M2O: {diff}")
    
    suite.run_test("Hydro M2M vs M2O Consistency", test_hydro_consistency)
    
    # Test 6: Hydro Physics Recovery
    def test_hydro_physics():
        jx = read_hdf5_hydro_component("ehydro/T.0/test_hydro_m2m.0.0.h5", 'jx')
        jy = read_hdf5_hydro_component("ehydro/T.0/test_hydro_m2m.0.0.h5", 'jy')
        jz = read_hdf5_hydro_component("ehydro/T.0/test_hydro_m2m.0.0.h5", 'jz')
        rho = read_hdf5_hydro_component("ehydro/T.0/test_hydro_m2m.0.0.h5", 'rho')
        
        if any(x is None for x in [jx, jy, jz, rho]):
            raise Exception("Failed to read hydro data")
        
        # Recover velocities from current density and charge density
        valid = np.abs(rho) > 1e-15
        if not np.any(valid):
            raise Exception("No valid charge density found")
        
        ux = np.zeros_like(jx); ux[valid] = jx[valid] / rho[valid]
        uy = np.zeros_like(jy); uy[valid] = jy[valid] / rho[valid]
        uz = np.zeros_like(jz); uz[valid] = jz[valid] / rho[valid]
        
        # Generate expected velocity pattern
        expected_u = np.zeros((NZ, NY, NX))
        for iz in range(1, NZ+1):
            for iy in range(1, NY+1):
                for ix in range(1, NX+1):
                    gid = iz * NY * NX + iy * NX + ix
                    expected_u[iz-1, iy-1, ix-1] = gid * 1.0e-9
        
        # Verify anisotropy (ux=1x, uy=2x, uz=3x)
        diff_x = np.max(np.abs(ux[valid] - expected_u[valid]))
        diff_y = np.max(np.abs(uy[valid] - expected_u[valid] * 2.0))
        diff_z = np.max(np.abs(uz[valid] - expected_u[valid] * 3.0))
        
        if max(diff_x, diff_y, diff_z) > 1e-6:
            raise Exception(f"Hydro physics mismatch: X={diff_x:.1e}, Y={diff_y:.1e}, Z={diff_z:.1e}")
    
    suite.run_test("Hydro Physics Recovery", test_hydro_physics)
    
    return suite.summary()

# =============================================================================
# 4-RANK TESTS
# =============================================================================
def verify_4rank():
    """Verify 4-rank HDF5 output (M2M and M2O modes)"""
    suite = TestSuite("HDF5 4-Rank")
    
    NX_L, NY_L, NZ_L = 4, 4, 4
    NX_G, NY_G, NZ_G = 8, 8, 4
    NUM_RANKS = 4
    
    # Test 1: All Files Exist
    def test_files_exist():
        # M2M files (one per rank)
        for rank in range(NUM_RANKS):
            files = [
                f"field/T.0/test_fields_m2m.0.{rank}.h5",
                f"ehydro/T.0/test_hydro_m2m.0.{rank}.h5",
                f"particle/test_particles_m2m.electron.{rank}.0.h5"
            ]
            for fname in files:
                if not os.path.exists(fname):
                    raise Exception(f"Missing M2M file: {fname}")
        
        # M2O files (one shared file)
        files = [
            "field/T.0/test_fields_m2o.0.h5",
            "ehydro/T.0/test_hydro_m2o.0.h5",
            "particle/test_particles_m2o.electron.0.h5"
        ]
        for fname in files:
            if not os.path.exists(fname):
                raise Exception(f"Missing M2O file: {fname}")
    
    suite.run_test("All HDF5 Files Present", test_files_exist)
    
    # Test 2: Field M2O Global Pattern
    def test_field_m2o():
        # Generate global expected pattern
        expected_global = np.zeros((NZ_G, NY_G, NX_G), dtype=np.float32)
        for iz in range(1, NZ_G+1):
            for iy in range(1, NY_G+1):
                for ix in range(1, NX_G+1):
                    gid = iz * NY_G * NX_G + iy * NX_G + ix
                    expected_global[iz-1, iy-1, ix-1] = float(gid)
        
        components = [
            ('ex', 0.1), ('ey', 0.2), ('ez', 0.3),
            ('cbx', 0.4), ('cby', 0.5), ('cbz', 0.6)
        ]
        
        with h5py.File("field/T.0/test_fields_m2o.0.h5", 'r') as f:
            for name, offset in components:
                if name not in f:
                    raise Exception(f"Missing field component {name} in M2O file")
                
                data = f[name][:]
                expected = expected_global + offset
                diff = np.max(np.abs(data - expected))
                
                if diff > 1e-5:
                    raise Exception(f"Field {name} M2O mismatch: {diff}")
    
    suite.run_test("Field M2O Global Pattern", test_field_m2o)
    
    # Test 3: Field M2M Stitching Matches M2O
    def test_field_stitching():
        components = ['ex', 'ey', 'ez', 'cbx', 'cby', 'cbz']
        
        # Read M2O data
        with h5py.File("field/T.0/test_fields_m2o.0.h5", 'r') as f:
            m2o_data = {comp: f[comp][:] for comp in components}
        
        # Stitch M2M data
        for comp in components:
            stitched = stitch_m2m_to_global_hdf5(
                "field/T.0/test_fields_m2m.0.{rank}.h5",
                NUM_RANKS,
                (1, 2, 2),
                (NZ_L, NY_L, NX_L),
                comp
            )
            
            if stitched is None:
                raise Exception(f"Failed to stitch {comp}")
            
            diff = np.max(np.abs(stitched - m2o_data[comp]))
            if diff > 1e-5:
                raise Exception(f"Field {comp} M2M stitching != M2O: {diff}")
    
    suite.run_test("Field M2M Stitching Matches M2O", test_field_stitching)
    
    # Test 4: Particle M2O Global Distribution
    def test_particle_m2o():
        with h5py.File("particle/test_particles_m2o.electron.0.h5", 'r') as f:
            if 'particles' not in f:
                raise Exception("Missing particles dataset in M2O file")
            
            dset = f['particles']
            total_particles = len(dset)
            expected_total = NX_G * NY_G * NZ_G
            
            if total_particles != expected_total:
                raise Exception(f"Particle count mismatch: {total_particles} != {expected_total}")
            
            # Verify anisotropic pattern
            ux = np.sort(dset['ux'][:])
            uy = np.sort(dset['uy'][:])
            uz = np.sort(dset['uz'][:])
            
            expected_base = []
            for iz in range(1, NZ_G+1):
                for iy in range(1, NY_G+1):
                    for ix in range(1, NX_G+1):
                        gid = iz * NY_G * NX_G + iy * NX_G + ix
                        expected_base.append(gid * 1.0e-9)
            expected_base = np.sort(np.array(expected_base))
            
            diff_x = np.max(np.abs(ux - expected_base))
            diff_y = np.max(np.abs(uy - expected_base * 2.0))
            diff_z = np.max(np.abs(uz - expected_base * 3.0))
            
            if max(diff_x, diff_y, diff_z) > 1e-6:
                raise Exception(f"Particle anisotropy mismatch: X={diff_x:.1e}, Y={diff_y:.1e}, Z={diff_z:.1e}")
    
    suite.run_test("Particle M2O Global Distribution", test_particle_m2o)
    
    # Test 5: Particle M2M Aggregation Matches M2O
    def test_particle_aggregation():
        # Read M2O data
        with h5py.File("particle/test_particles_m2o.electron.0.h5", 'r') as f:
            m2o_ux = np.sort(f['particles']['ux'][:])
            m2o_uy = np.sort(f['particles']['uy'][:])
            m2o_uz = np.sort(f['particles']['uz'][:])
        
        # Aggregate M2M data
        all_ux, all_uy, all_uz = [], [], []
        for rank in range(NUM_RANKS):
            fname = f"particle/test_particles_m2m.electron.{rank}.0.h5"
            with h5py.File(fname, 'r') as f:
                dset = f['particles']
                all_ux.extend(dset['ux'][:])
                all_uy.extend(dset['uy'][:])
                all_uz.extend(dset['uz'][:])
        
        m2m_ux = np.sort(np.array(all_ux))
        m2m_uy = np.sort(np.array(all_uy))
        m2m_uz = np.sort(np.array(all_uz))
        
        diff_x = np.max(np.abs(m2m_ux - m2o_ux))
        diff_y = np.max(np.abs(m2m_uy - m2o_uy))
        diff_z = np.max(np.abs(m2m_uz - m2o_uz))
        
        if max(diff_x, diff_y, diff_z) > 1e-6:
            raise Exception(f"Particle M2M aggregation != M2O: X={diff_x:.1e}, Y={diff_y:.1e}, Z={diff_z:.1e}")
    
    suite.run_test("Particle M2M Aggregation Matches M2O", test_particle_aggregation)
    
    # Test 6: Hydro M2M Stitching Matches M2O
    def test_hydro_stitching():
        components = ['jx', 'jy', 'jz', 'rho']
        
        # Read M2O data
        with h5py.File("ehydro/T.0/test_hydro_m2o.0.h5", 'r') as f:
            m2o_data = {comp: f[comp][:] for comp in components}
        
        # Stitch M2M data
        for comp in components:
            stitched = stitch_m2m_to_global_hdf5(
                "ehydro/T.0/test_hydro_m2m.0.{rank}.h5",
                NUM_RANKS,
                (1, 2, 2),
                (NZ_L, NY_L, NX_L),
                comp
            )
            
            if stitched is None:
                raise Exception(f"Failed to stitch {comp}")
            
            diff = np.max(np.abs(stitched - m2o_data[comp]))
            if diff > 1e-5:
                raise Exception(f"Hydro {comp} M2M stitching != M2O: {diff}")
    
    suite.run_test("Hydro M2M Stitching Matches M2O", test_hydro_stitching)
    
    # Test 7: Hydro Global Physics Recovery
    def test_hydro_physics():
        # Read M2O data
        with h5py.File("ehydro/T.0/test_hydro_m2o.0.h5", 'r') as f:
            jx = f['jx'][:]
            jy = f['jy'][:]
            jz = f['jz'][:]
            rho = f['rho'][:]
        
        # Recover velocities
        valid = np.abs(rho) > 1e-15
        if not np.any(valid):
            raise Exception("No valid charge density found")
        
        ux = np.zeros_like(jx); ux[valid] = jx[valid] / rho[valid]
        uy = np.zeros_like(jy); uy[valid] = jy[valid] / rho[valid]
        uz = np.zeros_like(jz); uz[valid] = jz[valid] / rho[valid]
        
        # Generate expected global pattern
        expected_u = np.zeros((NZ_G, NY_G, NX_G))
        for iz in range(1, NZ_G+1):
            for iy in range(1, NY_G+1):
                for ix in range(1, NX_G+1):
                    gid = iz * NY_G * NX_G + iy * NX_G + ix
                    expected_u[iz-1, iy-1, ix-1] = gid * 1.0e-9
        
        # Verify anisotropy (ux=1x, uy=2x, uz=3x)
        diff_x = np.max(np.abs(ux[valid] - expected_u[valid]))
        diff_y = np.max(np.abs(uy[valid] - expected_u[valid] * 2.0))
        diff_z = np.max(np.abs(uz[valid] - expected_u[valid] * 3.0))
        
        if max(diff_x, diff_y, diff_z) > 1e-6:
            raise Exception(f"Hydro physics mismatch: X={diff_x:.1e}, Y={diff_y:.1e}, Z={diff_z:.1e}")
    
    suite.run_test("Hydro Global Physics Recovery", test_hydro_physics)
    
    return suite.summary()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def stitch_m2m_to_global_hdf5(file_pattern, num_ranks, topology, local_dims, component):
    """
    Stitch M2M HDF5 files into a global array
    
    Args:
        file_pattern: String with {rank} placeholder (e.g., "field/T.0/test_fields_m2m.0.{rank}.h5")
        num_ranks: Total number of ranks
        topology: Tuple (pz, py, px) of process grid
        local_dims: Tuple (nz, ny, nx) of local grid per rank
        component: Component name to read (e.g., 'ex', 'jx')
    
    Returns:
        Global stitched array or None on failure
    """
    pz, py, px = topology
    nz, ny, nx = local_dims
    
    global_dims = (nz * pz, ny * py, nx * px)
    global_data = np.zeros(global_dims, dtype=np.float32)
    
    for rank in range(num_ranks):
        # Compute rank position in topology
        rx = rank % px
        ry = (rank // px) % py
        rz = rank // (px * py)
        
        # Compute offset in global array
        oz = rz * nz
        oy = ry * ny
        ox = rx * nx
        
        # Read rank's data
        fname = file_pattern.format(rank=rank)
        if not os.path.exists(fname):
            return None
        
        try:
            with h5py.File(fname, 'r') as f:
                if component not in f:
                    return None
                local_data = f[component][:]
                global_data[oz:oz+nz, oy:oy+ny, ox:ox+nx] = local_data
        except Exception as e:
            print(f"    [ERR] Failed to read {fname}: {e}")
            return None
    
    return global_data

class TestSuite:
    """Simple test suite for organized verification"""
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.failures = []
    
    def run_test(self, test_name, test_func):
        """Run a single test and record results"""
        try:
            test_func()
            self.passed += 1
            print(f"  [\033[92mPASS\033[0m] {test_name}")
        except Exception as e:
            self.failed += 1
            self.failures.append((test_name, str(e)))
            print(f"  [\033[91mFAIL\033[0m] {test_name}: {e}")
    
    def summary(self):
        """Print summary and return success status"""
        total = self.passed + self.failed
        print(f"\n{self.name} Summary: {self.passed}/{total} tests passed")
        
        if self.failed > 0:
            print(f"\n\033[91mFailed Tests:\033[0m")
            for name, error in self.failures:
                print(f"  - {name}: {error}")
            return False
        return True

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Verify HDF5 I/O correctness')
    parser.add_argument('--mode', choices=['single', '4rank'], required=True,
                       help='Test mode: single-rank or 4-rank')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Skip cleanup of test output')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"HDF5 I/O Verification Suite")
    print(f"Mode: {args.mode}")
    print(f"{'='*70}\n")
    
    # Run appropriate verification
    if args.mode == 'single':
        success = verify_single()
    else:  # 4rank
        success = verify_4rank()
    
    # Cleanup
    if not args.no_cleanup:
        print("\n" + "="*70)
        print("Cleaning up test output...")
        print("="*70)
        cleanup_test_output()
    
    # Final result
    print("\n" + "="*70)
    if success:
        print("\033[92m✓ ALL TESTS PASSED\033[0m")
        print("="*70)
        return 0
    else:
        print("\033[91m✗ TESTS FAILED\033[0m")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())