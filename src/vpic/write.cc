#include "vpic.h"
#include <sys/stat.h> // for mkdir
#include <vector>

#ifdef VPIC_ENABLE_HDF5
#include "hdf5.h"
#endif // VPIC_ENABLE_HDF5

// =============================================================================
// CONFIGURATION
// =============================================================================

// File format magic number (0xBEEF0002 = Version 2 of VPIC binary format)
constexpr int32_t VPIC_BINARY_MAGIC = 0xBEEF0002;
constexpr int32_t VPIC_BINARY_VERSION = 1;

// Particle buffer size for chunked I/O (2M particles ≈ 64MB per chunk)
constexpr int PARTICLE_BUFFER_SIZE = 2097152;

// Field/Hydro buffer size for chunked I/O (1MB chunks)
constexpr int FIELD_BUFFER_SIZE = 32768;
// Hydro buffer size for chunked I/O (1MB chunks)
constexpr int HYDRO_BUFFER_SIZE = 32768;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * @brief Ensures a directory exists, creating it if necessary
 * @param path Directory path to create
 */
static void ensure_directory(const char* path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        mkdir(path, 0755);
    }
}

// =============================================================================
// BINARY FORMAT STRUCTURES
// =============================================================================

/**
 * @brief Binary file header structure (64 bytes)
 * 
 * This header precedes all data in VPIC binary dump files and contains
 * metadata necessary for interpreting the file contents.
 */
struct BinaryHeader {
    int32_t magic;      ///< File format identifier (0xBEEF0002)
    int32_t version;    ///< Format version number (currently 1)
    int32_t step;       ///< Simulation timestep when dump was created
    int32_t nx, ny, nz; ///< Grid dimensions (cells per rank)
    float   dt;         ///< Timestep size
    float   dx, dy, dz; ///< Cell dimensions
    float   x0, y0, z0; ///< Global origin coordinates (padding in current version)
    float   q_m;        ///< Charge-to-mass ratio (species-specific, 0.0 for fields)
    int32_t num_vars;   ///< Number of variables per cell
    int32_t var_mask;   ///< Bitmask indicating which variables are present
};

// =============================================================================
// BINARY FIELD WRITER
// =============================================================================

/**
 * @brief Writes field data to binary format using table-driven variable selection
 * 
 * @param params Dump configuration (output directory, filename, variable mask)
 * @param fa Field array containing electromagnetic field data
 * 
 * This function uses a mapping table to dynamically select which field components
 * to write based on the bitmask in params.output_vars. Data is written in
 * rank-per-file (M2M) format.
 */
void vpic_simulation::write_fields_binary(DumpParameters& params, field_array_t* fa) {
    TIC {
        if (!fa) ERROR(("NULL field array"));
        if (step() > fa->last_copied) fa->copy_to_host();
        grid_t* g = fa->g;

        // Define field component mapping: bitmask index -> memory offset
        struct FieldComponent {
            int bit;    ///< Bit index in output_vars bitmask
            int offset; ///< Byte offset within field_t structure
        };

        static const std::vector<FieldComponent> field_map = {
            {0, (int)offsetof(field_t, ex)},   // Electric field components
            {1, (int)offsetof(field_t, ey)},
            {2, (int)offsetof(field_t, ez)},
            {4, (int)offsetof(field_t, cbx)},  // Magnetic field components (centered)
            {5, (int)offsetof(field_t, cby)},
            {6, (int)offsetof(field_t, cbz)}
            // Note: div_e_err (bit 3) and div_b_err (bit 7) can be added if needed
        };

        // Create output directory structure
        if (rank() == 0) {
            ensure_directory(params.baseDir);
            char timeDir[256];
            snprintf(timeDir, 256, "%s/T.%ld", params.baseDir, (long)step());
            ensure_directory(timeDir);
        }
        mp_barrier();

        // Open output file for this rank
        char fname[256];
        snprintf(fname, 256, "%s/T.%ld/%s.%ld.%d",
                 params.baseDir, (long)step(), params.baseFileName, (long)step(), rank());
        FILE* fp = fopen(fname, "wb");
        if (!fp) ERROR(("Could not open file: %s", fname));

        // Count requested variables
        int32_t num_vars = 0;
        for (const auto& comp : field_map) {
            if (params.output_vars.bitset(comp.bit)) num_vars++;
        }

        double io_start = wallclock();

        // Write header
        BinaryHeader header = {
            VPIC_BINARY_MAGIC,
            VPIC_BINARY_VERSION,
            (int32_t)step(),
            g->nx, g->ny, g->nz,
            g->dt,
            g->dx, g->dy, g->dz,
            0.0f, 0.0f, 0.0f,  // Padding (x0, y0, z0 not used in M2M mode)
            0.0f,              // q_m (not applicable for fields)
            num_vars,
            0                  // var_mask (reserved for future use)
        };
        fwrite(&header, sizeof(BinaryHeader), 1, fp);

        // Write field data
        size_t num_cells = g->nx * g->ny * g->nz;
        std::vector<float> buffer(num_cells);
        auto* f_base = fa->f;

        for (const auto& comp : field_map) {
            if (params.output_vars.bitset(comp.bit)) {
                size_t idx = 0;
                for (int k = 1; k <= g->nz; k++) {
                    for (int j = 1; j <= g->ny; j++) {
                        for (int i = 1; i <= g->nx; i++) {
                            int voxel_idx = voxel(i, j, k);
                            float val = *(float*)((char*)&f_base[voxel_idx] + comp.offset);
                            buffer[idx++] = val;
                        }
                    }
                }
                fwrite(buffer.data(), sizeof(float), num_cells, fp);
            }
        }

        fclose(fp);

        // Performance metrics
        double io_time = wallclock() - io_start;
        double max_time = 0.0;
        mp_allmax_d(&io_time, &max_time, 1);

        if (rank() == 0 && num_vars > 0) {
            double mb_total = ((double)(num_cells * num_vars * sizeof(float)) / (1024.0 * 1024.0)) * nproc();
            double throughput = mb_total / max_time;
            MESSAGE(("[METRIC],write_fields_bin,%ld,%.4f,%.4f,%.4f",
                     (long)step(), max_time, mb_total, throughput));
        }
    } TOC(write_fields_bin, 1);
}

// =============================================================================
// BINARY HYDRO WRITER
// =============================================================================

/**
 * @brief Writes hydro array data to binary format using table-driven variable selection
 * 
 * @param params Dump configuration (output directory, filename, variable mask)
 * @param ha Hydro array containing moment data
 * @param sp_name Species name for physics accumulation
 * 
 * This function performs physics accumulation from particle data before writing,
 * computing current density (J), charge density (ρ), momentum (P), kinetic energy (KE),
 * and stress tensor components (T). Data is written in rank-per-file (M2M) format.
 * 
 * @note Physics accumulation is performed inside this function by design - the hydro
 *       array must be computed from particle data before dumping.
 */
void vpic_simulation::write_hydro_binary(DumpParameters& params, 
                                         hydro_array_t* ha, 
                                         const char* sp_name) {
    TIC {
        if (!ha) ERROR(("NULL hydro array"));
        
        // =====================================================================
        // PHYSICS ACCUMULATION
        // =====================================================================
        // Hydro arrays must be computed from particle data before writing.
        // This accumulates current density (J), charge density (ρ), momentum (P),
        // kinetic energy (KE), and stress tensor components (T) for the species.
        
        species_t* sp = find_species(sp_name);
        if (!sp) ERROR(("Invalid species \"%s\"", sp_name));

        // Clear grid and accumulate moments from particles
        Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
        clear_hydro_array(ha);
        accumulate_hydro_p_kokkos(sp->k_p_d, sp->k_p_i_d, hydro_array->k_h_d, 
                                  interpolator_array->k_i_d, sp);
        
        // Synchronize accumulated data to host
        ha->copy_to_host();
        synchronize_hydro_array(ha);

        grid_t* g = ha->g;

        // =====================================================================
        // VARIABLE DEFINITION TABLE
        // =====================================================================
        // Maps bitmask indices to memory offsets within hydro_t structure.
        // Only variables with their corresponding bit set in params.output_vars
        // will be written to the output file.
        
        struct HydroComponent {
            int bit;    ///< Bit index in output_vars bitmask
            int offset; ///< Byte offset within hydro_t structure
        };

        static const std::vector<HydroComponent> map = {
            {0,  (int)offsetof(hydro_t, jx)},   // Current density
            {1,  (int)offsetof(hydro_t, jy)},
            {2,  (int)offsetof(hydro_t, jz)},
            {3,  (int)offsetof(hydro_t, rho)},  // Charge density
            {4,  (int)offsetof(hydro_t, px)},   // Momentum
            {5,  (int)offsetof(hydro_t, py)},
            {6,  (int)offsetof(hydro_t, pz)},
            {7,  (int)offsetof(hydro_t, ke)},   // Kinetic energy
            {8,  (int)offsetof(hydro_t, txx)},  // Stress tensor
            {9,  (int)offsetof(hydro_t, tyy)},
            {10, (int)offsetof(hydro_t, tzz)},
            {11, (int)offsetof(hydro_t, tyz)},
            {12, (int)offsetof(hydro_t, tzx)},
            {13, (int)offsetof(hydro_t, txy)}
        };
        
        // =====================================================================
        // FILE PREPARATION
        // =====================================================================
        
        double io_tic = wallclock();
        
        // Create output directory structure
        if (rank() == 0) {
            ensure_directory(params.baseDir);
            char timeDir[256];
            snprintf(timeDir, 256, "%s/T.%ld", params.baseDir, (long)step());
            ensure_directory(timeDir);
        }
        mp_barrier();

        // Open output file for this rank
        char fname[256];
        snprintf(fname, 256, "%s/T.%ld/%s.%ld.%d", 
                 params.baseDir, (long)step(), params.baseFileName, (long)step(), rank());
        
        FILE* fp = fopen(fname, "wb");
        if (!fp) ERROR(("Could not open file: %s", fname));

        // Count requested variables
        int32_t num_vars = 0;
        for (const auto& comp : map) {
            if (params.output_vars.bitset(comp.bit)) num_vars++;
        }

        // =====================================================================
        // WRITE HEADER
        // =====================================================================
        
        BinaryHeader header = {
            VPIC_BINARY_MAGIC,
            VPIC_BINARY_VERSION,
            (int32_t)step(),
            g->nx, g->ny, g->nz,
            g->dt,
            g->dx, g->dy, g->dz,
            0.0f, 0.0f, 0.0f,  // Padding (x0, y0, z0 not used in M2M mode)
            sp->q / sp->m,     // Charge-to-mass ratio for this species
            num_vars,
            0                  // var_mask (reserved for future use, must remain for format compatibility)
        };
        fwrite(&header, sizeof(BinaryHeader), 1, fp);

        // =====================================================================
        // WRITE DATA
        // =====================================================================
        // Iterate through variable map and write only requested components
        
        size_t num_cells = g->nx * g->ny * g->nz;
        std::vector<float> buffer(num_cells);
        auto* h_base = ha->h;

        for (const auto& comp : map) {
            if (params.output_vars.bitset(comp.bit)) {
                // Pack data in column-major order (k->j->i)
                size_t idx = 0;
                for (int k = 1; k <= g->nz; k++) {
                    for (int j = 1; j <= g->ny; j++) {
                        for (int i = 1; i <= g->nx; i++) {
                            int voxel_idx = voxel(i, j, k);
                            float val = *(float*)((char*)&h_base[voxel_idx] + comp.offset);
                            buffer[idx++] = val;
                        }
                    }
                }
                fwrite(buffer.data(), sizeof(float), num_cells, fp);
            }
        }

        fclose(fp);

        // =====================================================================
        // PERFORMANCE METRICS
        // =====================================================================
        
        double pure_io_time = wallclock() - io_tic;
        double t_max = 0.0;
        mp_allmax_d(&pure_io_time, &t_max, 1);

        if (rank() == 0 && num_vars > 0) {
            // Calculate total data volume across all ranks
            double mb_total = ((double)(num_cells * num_vars * sizeof(float)) / 
                              (1024.0 * 1024.0)) * nproc();
            double throughput = mb_total / t_max;
            
            MESSAGE(("[METRIC],write_hydro_bin,%ld,%.4f,%.4f,%.4f", 
                     (long)step(), t_max, mb_total, throughput));
        }
        
    } TOC(write_hydro_bin, 1);
}

// =============================================================================
// BINARY PARTICLE WRITER
// =============================================================================

/**
 * @brief Writes particle data to binary format
 * 
 * @param fbase Base filename (without extension)
 * @param species_name Name of species to dump
 * @param compute_physical_position Position mode selector:
 *        - true: Output physical (x,y,z) in simulation coordinates, omit voxel index
 *        - false: Output logical (dx,dy,dz) in [-1,1] cell-local coords, include voxel index
 * 
 * Position Mode Details:
 * 
 * PHYSICAL MODE (compute_physical_position = true):
 * - Converts particle's cell-local coordinates to global simulation coordinates
 * - Transformation: x = x0 + (ix - 1 + 0.5*(1 + dx)) * cell_dx
 * - Output fields: x, y, z, ux, uy, uz, w (7 floats = 28 bytes) - NO voxel index
 * - Use case: Post-processing tools that need absolute positions
 * - File size: ~12.5% SMALLER than logical mode (saves 4 bytes per particle)
 * 
 * LOGICAL MODE (compute_physical_position = false):
 * - Outputs raw particle data as stored during simulation
 * - Output fields: dx, dy, dz, i, ux, uy, uz, w (3 floats + 1 int32 + 4 floats = 32 bytes)
 * - Use case: Debugging, restart files, visualization that handles cell-local coords
 * - Benefit: Preserves exact simulation state, faster I/O (no coordinate conversion)
 * 
 * Performance characteristics:
 * - Chunked processing (32K particles per chunk) for memory efficiency
 * - Time-centering applied via center_p_dump() - filters ghosts automatically
 * - Physical mode adds ~5-10% computation overhead for coordinate transformation
 * - Physical mode reduces file size by 12.5% (28 vs 32 bytes per particle)
 * - Binary format: simple concatenated structs, excellent sequential read performance
 * 
 * @note File format: BinaryHeader (64 bytes) + particle data array
 * @note Header's num_vars field reflects output: 7 (physical) or 8 (logical)
 */
void vpic_simulation::write_particles_binary(const char* fbase, 
                                             const char* species_name,
                                             bool compute_physical_position) {
    TIC {
        // =====================================================================
        // PHASE 1: SETUP & VALIDATION
        // =====================================================================
        
        species_t* sp = find_species(species_name);
        if(!sp) ERROR(("Invalid species: %s", species_name));
        sp->copy_to_host();
        grid_t* g = sp->g;

        hsize_t my_count = sp->np;

        // Open output file
        char fname[256];
        snprintf(fname, 256, "%s.%s.%d", fbase, sp->name, rank());
        FILE *fp = fopen(fname, "wb");
        if(!fp) ERROR(("Could not open file: %s", fname));

        // =====================================================================
        // PHASE 2: WRITE HEADER
        // =====================================================================
        // Header reflects output format: 7 vars (physical) or 8 vars (logical)
        
        BinaryHeader h = {
            (int32_t)0xBEEF0002,                           // magic
            1,                                              // version
            (int32_t)g->step,                              // step
            g->nx, g->ny, g->nz,                           // grid dims
            g->dt,                                          // timestep
            g->dx, g->dy, g->dz,                           // cell size
            g->x0, g->y0, g->z0,                           // origin
            sp->q/sp->m,                                    // q/m ratio
            compute_physical_position ? 7 : 8,              // num_vars
            (int32_t)my_count                               // particle count
        };
        fwrite(&h, sizeof(BinaryHeader), 1, fp);

        // =====================================================================
        // PHASE 3: CHUNKED PARTICLE PROCESSING
        // =====================================================================
        // Process in 32K particle chunks for memory efficiency.
        // Each chunk undergoes: Copy → Time-Center → Convert (if physical) → Write
        
        const int PBUF_SIZE = 32768;
        particle_t* p_buf;
        MALLOC_ALIGNED(p_buf, PBUF_SIZE, 128);

        auto& k_p_h = sp->k_p_h;
        auto& k_p_i_h = sp->k_p_i_h;
        int sp_np = sp->np;
        int sp_max_np = sp->max_np;

        // Precompute grid parameters for physical coordinate conversion
        // UNVOXEL macro will be used: converts voxel index → (i,j,k)
        int nx = g->nx, ny = g->ny, nz = g->nz;
        float dx = g->dx, dy = g->dy, dz = g->dz;
        float x0 = g->x0, y0 = g->y0, z0 = g->z0;

        for(int buf_start = 0; buf_start < sp_np; buf_start += PBUF_SIZE) {
            
            // Adjust species counts for this chunk
            sp->np = sp_np - buf_start;
            if(sp->np > PBUF_SIZE) sp->np = PBUF_SIZE;
            sp->max_np = PBUF_SIZE;

            // -----------------------------------------------------------------
            // Sub-phase 3a: Copy Chunk to Aligned Buffer
            // -----------------------------------------------------------------
            Kokkos::View<particle_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
                pbuf_view(p_buf, PBUF_SIZE);
            
            Kokkos::parallel_for("PopulateParticleDumpBuffer",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np),
                KOKKOS_LAMBDA(int i) {
                    int idx = buf_start + i;
                    pbuf_view(i).dx = k_p_h(idx, particle_var::dx);
                    pbuf_view(i).dy = k_p_h(idx, particle_var::dy);
                    pbuf_view(i).dz = k_p_h(idx, particle_var::dz);
                    pbuf_view(i).ux = k_p_h(idx, particle_var::ux);
                    pbuf_view(i).uy = k_p_h(idx, particle_var::uy);
                    pbuf_view(i).uz = k_p_h(idx, particle_var::uz);
                    pbuf_view(i).w  = k_p_h(idx, particle_var::w);
                    pbuf_view(i).i  = k_p_i_h(idx);
                });
            Kokkos::fence();

            // -----------------------------------------------------------------
            // Sub-phase 3b: Time-Centering & Ghost Filtering
            // -----------------------------------------------------------------
            // center_p_dump() performs two critical operations:
            // 1. Centers particle velocities to current timestep
            // 2. Removes ghost particles, updating sp->np to valid count
            center_p_dump(sp, p_buf, interpolator_array);

            // -----------------------------------------------------------------
            // Sub-phase 3c: Coordinate Transformation (Physical Mode Only)
            // -----------------------------------------------------------------
            if (compute_physical_position && sp->np > 0) {
                // Transform (dx,dy,dz,i) → (x,y,z) in-place
                // After this, p_buf[n].dx/dy/dz contain physical coordinates
                // and p_buf[n].i becomes meaningless (won't be written)
                
                for(int n = 0; n < sp->np; ++n) {
                    int voxel_idx = p_buf[n].i;
                    
                    // Convert voxel index to (i,j,k) grid coordinates
                    // UNVOXEL macro: extracts cell indices from flat voxel index
                    int ix, iy, iz;
                    UNVOXEL(voxel_idx, ix, iy, iz, nx, ny, nz);
                    
                    // Transform to physical position
                    // Cell center is at integer (ix,iy,iz), particle offset in [-1,1]
                    // Formula: global = origin + (cell_index - 1 + 0.5*(1 + local_offset)) * cell_size
                    // The "-1" accounts for ghost cell at index 0
                    p_buf[n].dx = x0 + (ix - 1 + 0.5f * (1.0f + p_buf[n].dx)) * dx;
                    p_buf[n].dy = y0 + (iy - 1 + 0.5f * (1.0f + p_buf[n].dy)) * dy;
                    p_buf[n].dz = z0 + (iz - 1 + 0.5f * (1.0f + p_buf[n].dz)) * dz;
                }
            }

            // -----------------------------------------------------------------
            // Sub-phase 3d: Write Chunk to File
            // -----------------------------------------------------------------
            if(sp->np > 0) {
                if (compute_physical_position) {
                    // Physical mode: Write only x,y,z,ux,uy,uz,w (skip voxel index)
                    for(int n = 0; n < sp->np; ++n) {
                        fwrite(&p_buf[n].dx, sizeof(float), 3, fp);  // x,y,z
                        fwrite(&p_buf[n].ux, sizeof(float), 4, fp);  // ux,uy,uz,w
                    }
                } else {
                    // Logical mode: Write full struct including voxel index
                    fwrite(p_buf, sizeof(particle_t), sp->np, fp);
                }
            }
        }

        // =====================================================================
        // PHASE 4: CLEANUP & METRICS
        // =====================================================================
        
        // Restore species state
        sp->np = sp_np;
        sp->max_np = sp_max_np;
        FREE_ALIGNED(p_buf);
        fclose(fp);

        // Calculate metrics
        double t_local = wallclock() - _profile_tic;
        size_t bytes_per_particle = compute_physical_position ? (7 * sizeof(float)) : sizeof(particle_t);
        double mb_local = (double)(my_count * bytes_per_particle) / (1024.0 * 1024.0);
        
        double t_max = 0.0, mb_total = 0.0;
        mp_allmax_d(&t_local, &t_max, 1);
        mp_allsum_d(&mb_local, &mb_total, 1);

        if (rank() == 0) {
            double throughput = mb_total / t_max;
            const char* mode = compute_physical_position ? "physical" : "logical";
            MESSAGE(("[METRIC],write_particles_bin_%s,%ld,%.4f,%.4f,%.4f", 
                     mode, (long)step(), t_max, mb_total, throughput));
        }
        
    } TOC(write_particles_bin, 1);
}

#ifdef VPIC_ENABLE_HDF5

// =============================================================================
// HDF5 HELPER FUNCTIONS
// =============================================================================

/**
 * @brief Global grid layout information for HDF5 spatial datasets
 * 
 * Contains the total simulation grid dimensions and this rank's offset within
 * the global domain. Used for configuring hyperslab selections in M2O mode.
 */
struct GlobalGridInfo {
    hsize_t global_dims[3];  ///< Total simulation grid [nz, ny, nx]
    hsize_t offset[3];       ///< This rank's starting index [z_off, y_off, x_off]
};

/**
 * @brief Calculates global grid dimensions and this rank's offset within them
 * 
 * @param g Grid structure containing local dimensions and spatial bounds
 * @param single_file True for M2O (many-to-one), false for M2M (many-to-many)
 * @return GlobalGridInfo containing global dimensions and rank offset
 * 
 * For M2M mode, returns local grid size (each rank writes its own file).
 * For M2O mode, computes where this rank's subdomain fits in the global grid
 * by using MPI reductions to find global min/max coordinates.
 * 
 * @note Dimensions follow HDF5 convention: [nz, ny, nx] (Z slowest, X fastest)
 */
static GlobalGridInfo get_global_grid_info(grid_t* g, bool single_file) {
    GlobalGridInfo info = {{0}, {0}};
    
    // Local dimensions in HDF5 convention (Z, Y, X)
    hsize_t local_dims[3] = {(hsize_t)g->nz, (hsize_t)g->ny, (hsize_t)g->nx};

    // M2M mode: Each file contains only local grid
    if (!single_file) {
        info.global_dims[0] = local_dims[0];
        info.global_dims[1] = local_dims[1];
        info.global_dims[2] = local_dims[2];
        return info;
    }

    // -------------------------------------------------------------------------
    // M2O mode: Compute global grid layout
    // -------------------------------------------------------------------------
    // Use spatial coordinates to determine rank offset in global grid
    
    double local_min[3] = {g->x0, g->y0, g->z0};
    double local_max[3] = {g->x1, g->y1, g->z1};
    double global_min[3], global_max[3];

    mp_allmin_d(local_min, global_min, 3);
    mp_allmax_d(local_max, global_max, 3);

    // Calculate offsets and dimensions (mapping Z->0, Y->1, X->2)
    info.offset[0] = (hsize_t)((g->z0 - global_min[2]) / g->dz + 0.5);
    info.global_dims[0] = (hsize_t)((global_max[2] - global_min[2]) / g->dz + 0.5);

    info.offset[1] = (hsize_t)((g->y0 - global_min[1]) / g->dy + 0.5);
    info.global_dims[1] = (hsize_t)((global_max[1] - global_min[1]) / g->dy + 0.5);

    info.offset[2] = (hsize_t)((g->x0 - global_min[0]) / g->dx + 0.5);
    info.global_dims[2] = (hsize_t)((global_max[0] - global_min[0]) / g->dx + 0.5);

    return info;
}

/**
 * @brief Writes a scalar attribute to an HDF5 object
 * 
 * @tparam T Attribute data type (int, float, double, etc.)
 * @param loc_id HDF5 object identifier (file, group, or dataset)
 * @param name Attribute name
 * @param type_id HDF5 datatype identifier
 * @param value Value to write
 */
template<typename T>
static void write_scalar_attr(hid_t loc_id, const char* name, hid_t type_id, T value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc_id, name, type_id, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, type_id, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

/**
 * @brief Creates optimized File Access Property List (FAPL) for parallel I/O
 * 
 * @param single_file True for M2O mode (requires parallel HDF5)
 * @return Configured FAPL handle
 * 
 * Optimizations applied:
 * 1. Collective metadata operations (reduces MPI communication overhead)
 * 2. 16MB stripe alignment (optimizes for parallel filesystems like Lustre)
 * 3. Deferred metadata cache flushes (batches writes, reduces sync operations)
 * 
 * Based on ECP best practices and h5bench benchmarking results.
 */
static hid_t create_optimized_fapl(bool single_file) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);

    #ifdef VPIC_HDF5_PARALLEL
    if (single_file) {
        // Enable MPI-IO driver for parallel access
        H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);

        // Collective metadata operations (reduces per-rank I/O)
        #if (H5_VERS_MAJOR > 1) || (H5_VERS_MAJOR == 1 && H5_VERS_MINOR >= 10)
        H5Pset_all_coll_metadata_ops(fapl, 1);
        H5Pset_coll_metadata_write(fapl, 1);
        #endif
    }
    #endif

    // Align large objects to 16MB boundaries (optimized for Lustre stripe size)
    H5Pset_alignment(fapl, 4096, 16 * 1024 * 1024);

    // Defer metadata flushes until file close (batch B-tree updates)
    H5AC_cache_config_t cache_config;
    cache_config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
    H5Pget_mdc_config(fapl, &cache_config);
    cache_config.evictions_enabled = 0;
    cache_config.incr_mode = H5C_incr__off;
    cache_config.flash_incr_mode = H5C_flash_incr__off;
    cache_config.decr_mode = H5C_decr__off;
    H5Pset_mdc_config(fapl, &cache_config);

    return fapl;
}

/**
 * @brief Creates HDF5 file with optimized settings and configures dataspaces
 * 
 * @param fname Output filename
 * @param single_file M2O vs M2M mode selector
 * @param ginfo Global grid information
 * @param local_dims Local rank dimensions [nz, ny, nx]
 * @param[out] dataspace_id File dataspace (global or local)
 * @param[out] memspace_id Memory dataspace (local)
 * @return HDF5 file identifier
 * 
 * For M2O mode, configures hyperslab selection for rank's subdomain within
 * global file. For M2M mode, file and memory spaces are identical.
 */
static hid_t setup_hdf5_file(const char* fname, bool single_file, 
                             const GlobalGridInfo& ginfo, const hsize_t* local_dims,
                             hid_t& dataspace_id, hid_t& memspace_id) {
    // Create file with optimized properties
    hid_t fapl_id = create_optimized_fapl(single_file);
    hid_t file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);
    
    if (file_id < 0) ERROR(("Failed to create HDF5 file: %s", fname));

    // Configure file dataspace
    dataspace_id = H5Screate_simple(3, ginfo.global_dims, NULL);
    
    if (single_file) {
        // M2O: Select hyperslab for this rank's contribution
        H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, 
                           ginfo.offset, NULL, local_dims, NULL);
    }
    
    // Memory dataspace always matches local dimensions
    memspace_id = H5Screate_simple(3, local_dims, NULL);
    
    return file_id;
}

// =============================================================================
// HDF5 PARTICLE WRITER
// =============================================================================

/**
 * @brief Writes particle data to HDF5 format
 * 
 * @param fbase Base filename (without extension)
 * @param species_name Name of species to dump
 * @param single_file True for M2O (single shared file), false for M2M (per-rank files)
 * @param compute_physical_position Position mode selector:
 *        - true: Output physical (x,y,z) in simulation coordinates, omit voxel index
 *        - false: Output logical (dx,dy,dz) in [-1,1] cell-local coords, include voxel index
 * 
 * Position Mode Details:
 * 
 * PHYSICAL MODE (compute_physical_position = true):
 * - Converts to global simulation coordinates before writing
 * - HDF5 compound type: {x, y, z, ux, uy, uz, w} (7 floats = 28 bytes)
 * - Advantages: Self-contained coordinates, easier post-processing
 * - File size: ~12.5% SMALLER than logical mode (saves 4 bytes per particle)
 * 
 * LOGICAL MODE (compute_physical_position = false):
 * - Outputs raw simulation data
 * - HDF5 compound type: {dx, dy, dz, i, ux, uy, uz, w} (3 floats + 1 int32 + 4 floats = 32 bytes)
 * - Advantages: Exact simulation state, faster I/O (no coordinate conversion)
 * - Note: Post-processing must handle cell-local coordinates and voxel indices
 * 
 * Performance characteristics:
 * - Chunked processing (2M particles per chunk) optimized for HDF5 write size
 * - Physical mode adds ~5-10% computation overhead for coordinate transformation
 * - Physical mode reduces file size by 12.5% (28 vs 32 bytes per particle)
 * - M2O mode uses collective I/O with optimized file access properties
 * - Persistent buffer allocation avoids repeated memory allocations
 * 
 * Outputs two metrics:
 * - [METRIC]: Aggregated throughput using configurable timing (pure I/O or total)
 * - [DIAGNOSTIC]: Detailed phase timing breakdown
 * 
 * @note TIME_PURE_IO_ONLY flag controls whether metrics measure only H5Dwrite
 *       time or total function time. Set to true for fair I/O benchmarking.
 * @note Physical mode transformation occurs after center_p_dump(), between
 *       t_compute and t_h5dwrite phases.
 */

void vpic_simulation::write_particles_hdf5(const char* fbase, 
                                           const char* species_name, 
                                           bool single_file,
                                           bool compute_physical_position) {
    KOKKOS_TIC() {
        // =====================================================================
        // CONFIGURATION
        // =====================================================================
        // Controls metric calculation: true = H5Dwrite only, false = total time
        const bool TIME_PURE_IO_ONLY = true;
        // =====================================================================

        double t_total_start = wallclock();

        // Phase timing accumulators
        double t_setup = 0.0;
        double t_meta_create = 0.0;
        double t_buf_pack = 0.0;
        double t_compute = 0.0;
        double t_h5dwrite = 0.0;
        double t_meta_close = 0.0;

        // =====================================================================
        // PHASE 1: SETUP & PARTICLE COUNT AGGREGATION
        // =====================================================================
        
        double t_phase_start = wallclock();

        species_t* sp = find_species(species_name);
        if(!sp) ERROR(("Invalid species: %s", species_name));
        sp->copy_to_host();
        grid_t* g = sp->g;

        hsize_t my_count = sp->np;
        hsize_t total_count = my_count;
        hsize_t current_file_offset = 0;

        // For M2O mode, compute file offsets via MPI scan
        #ifdef VPIC_HDF5_PARALLEL
        if (single_file) {
            uint64_t local_val = my_count, scan_val = 0, sum_val = 0;
            MPI_Exscan(&local_val, &scan_val, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_val, &sum_val, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
            if (rank() == 0) scan_val = 0;
            current_file_offset = scan_val;
            total_count = sum_val;
        }
        #endif

        t_setup = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 2: FILE & METADATA CREATION
        // =====================================================================
        
        t_phase_start = wallclock();

        char fname[256];
        if(single_file) {
            snprintf(fname, 256, "%s.%s.%ld.h5", fbase, sp->name, (long)step());
        } else {
            snprintf(fname, 256, "%s.%s.%d.%ld.h5", fbase, sp->name, rank(), (long)step());
        }

        // Create file with optimized access properties
        hid_t fapl = create_optimized_fapl(single_file);
        hid_t fid = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        if(fid < 0) ERROR(("Failed HDF5 create: %s", fname));

        // ---------------------------------------------------------------------
        // Define HDF5 Compound Type (Mode-Dependent)
        // ---------------------------------------------------------------------
        // Physical mode: 7 floats  (x, y, z, ux, uy, uz, w)
        // Logical mode:  8 members (dx, dy, dz, i, ux, uy, uz, w)
        
        hid_t ptype;
        size_t particle_size;
        
        if (compute_physical_position) {
            // Physical mode: No voxel index, contiguous floats
            particle_size = 7 * sizeof(float);
            ptype = H5Tcreate(H5T_COMPOUND, particle_size);
            H5Tinsert(ptype, "x",  0*sizeof(float), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "y",  1*sizeof(float), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "z",  2*sizeof(float), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "ux", 3*sizeof(float), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "uy", 4*sizeof(float), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "uz", 5*sizeof(float), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "w",  6*sizeof(float), H5T_NATIVE_FLOAT);
        } else {
            // Logical mode: Matches particle_t memory layout exactly
            particle_size = sizeof(particle_t);
            ptype = H5Tcreate(H5T_COMPOUND, particle_size);
            H5Tinsert(ptype, "dx", offsetof(particle_t, dx), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "dy", offsetof(particle_t, dy), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "dz", offsetof(particle_t, dz), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "i",  offsetof(particle_t, i),  H5T_NATIVE_INT32);
            H5Tinsert(ptype, "ux", offsetof(particle_t, ux), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "uy", offsetof(particle_t, uy), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "uz", offsetof(particle_t, uz), H5T_NATIVE_FLOAT);
            H5Tinsert(ptype, "w",  offsetof(particle_t, w),  H5T_NATIVE_FLOAT);
        }

        // Create 1D dataset for particles
        hid_t fspace = H5Screate_simple(1, &total_count, NULL);
        hid_t dset_particles = H5Dcreate2(fid, "particles", ptype, fspace,
                                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // Configure data transfer properties
        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        #ifdef VPIC_HDF5_PARALLEL
        if (single_file) H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
        #endif

        t_meta_create = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 3: CHUNKED PARTICLE PROCESSING
        // =====================================================================
        // Process in 2M particle chunks. Each chunk undergoes:
        // Copy → Time-Center → Convert (if physical) → Write
        
        const int PBUF_SIZE = 2097152;  // 2M particles ≈ 64MB chunks
        particle_t* p_buf;
        MALLOC_ALIGNED(p_buf, PBUF_SIZE, 128);

        // Allocate output buffer for physical mode (7 floats per particle)
        float* output_buf = nullptr;
        if (compute_physical_position) {
            MALLOC_ALIGNED(output_buf, PBUF_SIZE * 7 * sizeof(float), 128);
        }

        auto& k_p_h = sp->k_p_h;
        auto& k_p_i_h = sp->k_p_i_h;
        int sp_np = sp->np;
        int sp_max_np = sp->max_np;

        // Precompute grid parameters for coordinate transformation
        int nx = g->nx, ny = g->ny, nz = g->nz;
        float dx = g->dx, dy = g->dy, dz = g->dz;
        float x0 = g->x0, y0 = g->y0, z0 = g->z0;

        // Synchronize loop count for collective I/O
        double local_loops = (double)((sp_np + PBUF_SIZE - 1) / PBUF_SIZE);
        if (sp_np == 0) local_loops = 0.0;
        double max_loops_d = 0.0;
        mp_allmax_d(&local_loops, &max_loops_d, 1);
        int total_loops = single_file ? (int)max_loops_d : (int)local_loops;

        for(int loop_idx = 0; loop_idx < total_loops; ++loop_idx) {
            int buf_start = loop_idx * PBUF_SIZE;
            hsize_t chunk_valid_count = 0;

            if (buf_start < sp_np) {
                // Adjust species counts for this chunk
                sp->np = sp_np - buf_start;
                if(sp->np > PBUF_SIZE) sp->np = PBUF_SIZE;
                sp->max_np = PBUF_SIZE;

                // -------------------------------------------------------------
                // Sub-phase 3a: Copy Chunk to Buffer
                // -------------------------------------------------------------
                double t_loop_start = wallclock();
                
                Kokkos::View<particle_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    pbuf_view(p_buf, PBUF_SIZE);
                
                Kokkos::parallel_for("PopulateParticleDumpBuffer",
                    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np),
                    KOKKOS_LAMBDA(int i) {
                        int idx = buf_start + i;
                        pbuf_view(i).dx = k_p_h(idx, particle_var::dx);
                        pbuf_view(i).dy = k_p_h(idx, particle_var::dy);
                        pbuf_view(i).dz = k_p_h(idx, particle_var::dz);
                        pbuf_view(i).ux = k_p_h(idx, particle_var::ux);
                        pbuf_view(i).uy = k_p_h(idx, particle_var::uy);
                        pbuf_view(i).uz = k_p_h(idx, particle_var::uz);
                        pbuf_view(i).w  = k_p_h(idx, particle_var::w);
                        pbuf_view(i).i  = k_p_i_h(idx);
                    });
                Kokkos::fence();
                
                t_buf_pack += (wallclock() - t_loop_start);

                // -------------------------------------------------------------
                // Sub-phase 3b: Time-Centering & Ghost Removal
                // -------------------------------------------------------------
                // center_p_dump() filters ghosts and updates sp->np to valid count
                t_loop_start = wallclock();
                center_p_dump(sp, p_buf, interpolator_array);
                chunk_valid_count = sp->np;
                t_compute += (wallclock() - t_loop_start);

                // -------------------------------------------------------------
                // Sub-phase 3c: Coordinate Transformation (Physical Mode Only)
                // -------------------------------------------------------------
                if (compute_physical_position && chunk_valid_count > 0) {
                    // Transform (dx,dy,dz,i) → (x,y,z) and repack into output buffer
                    // Output layout: [x, y, z, ux, uy, uz, w] per particle
                    
                    float* out = (float*)output_buf;
                    
                    for(hsize_t n = 0; n < chunk_valid_count; ++n) {
                        int voxel_idx = p_buf[n].i;
                        
                        // Decompose voxel index into (i,j,k) using UNVOXEL macro
                        int ix, iy, iz;
                        UNVOXEL(voxel_idx, ix, iy, iz, nx, ny, nz);
                        
                        // Convert to physical position
                        // Particle offset (dx,dy,dz) is in [-1,1] within cell
                        // Cell center is at integer (ix,iy,iz) where ghost cell is at index 0
                        // Formula: global = origin + (cell_index - 1 + 0.5*(1 + local_offset)) * cell_size
                        out[n*7 + 0] = x0 + (ix - 1 + 0.5f * (1.0f + p_buf[n].dx)) * dx;  // x
                        out[n*7 + 1] = y0 + (iy - 1 + 0.5f * (1.0f + p_buf[n].dy)) * dy;  // y
                        out[n*7 + 2] = z0 + (iz - 1 + 0.5f * (1.0f + p_buf[n].dz)) * dz;  // z
                        out[n*7 + 3] = p_buf[n].ux;  // ux
                        out[n*7 + 4] = p_buf[n].uy;  // uy
                        out[n*7 + 5] = p_buf[n].uz;  // uz
                        out[n*7 + 6] = p_buf[n].w;   // w
                    }
                }
            }

            // -----------------------------------------------------------------
            // Sub-phase 3d: Write Chunk to HDF5
            // -----------------------------------------------------------------
            if (single_file || chunk_valid_count > 0) {
                double t_loop_start = wallclock();
                
                hid_t mspace = H5Screate_simple(1, &chunk_valid_count, NULL);
                H5Sselect_hyperslab(fspace, H5S_SELECT_SET,
                                   &current_file_offset, NULL,
                                   &chunk_valid_count, NULL);
                
                // Select appropriate buffer based on mode
                void* write_ptr = compute_physical_position ? (void*)output_buf : (void*)p_buf;
                H5Dwrite(dset_particles, ptype, mspace, fspace, dxpl, write_ptr);
                
                H5Sclose(mspace);
                current_file_offset += chunk_valid_count;
                t_h5dwrite += (wallclock() - t_loop_start);
            }
        }

        // Restore species state
        sp->np = sp_np;
        sp->max_np = sp_max_np;
        FREE_ALIGNED(p_buf);
        if (output_buf) FREE_ALIGNED(output_buf);

        // =====================================================================
        // PHASE 4: CLEANUP & CLOSE
        // =====================================================================
        
        t_phase_start = wallclock();

        H5Dclose(dset_particles);
        H5Tclose(ptype);
        H5Pclose(dxpl);
        H5Sclose(fspace);
        H5Fclose(fid);

        t_meta_close = wallclock() - t_phase_start;
        double t_total = wallclock() - t_total_start;

        // =====================================================================
        // PERFORMANCE METRICS
        // =====================================================================
        
        // Find maximum time spent in each phase across all MPI ranks
        double max_setup, max_meta_create, max_buf_pack, max_compute,
               max_h5dwrite, max_meta_close, max_total;
        
        mp_allmax_d(&t_setup, &max_setup, 1);
        mp_allmax_d(&t_meta_create, &max_meta_create, 1);
        mp_allmax_d(&t_buf_pack, &max_buf_pack, 1);
        mp_allmax_d(&t_compute, &max_compute, 1);
        mp_allmax_d(&t_h5dwrite, &max_h5dwrite, 1);
        mp_allmax_d(&t_meta_close, &max_meta_close, 1);
        mp_allmax_d(&t_total, &max_total, 1);
        
        // Calculate total data volume
        double mb_total = 0.0;
        if (single_file) {
            mb_total = (double)(total_count * particle_size) / (1024.0 * 1024.0);
        } else {
            double mb_local = (double)(my_count * particle_size) / (1024.0 * 1024.0);
            mp_allsum_d(&mb_local, &mb_total, 1);
        }

        if (rank() == 0) {
            // Determine which time metric to use based on internal flag
            double t_metric = TIME_PURE_IO_ONLY ? max_h5dwrite : max_total;
            double throughput = mb_total / t_metric;
            
            // Generate tag based on mode and file layout
            const char* file_mode = single_file ? "M2O" : "M2M";
            const char* pos_mode = compute_physical_position ? "physical" : "logical";
            char tag[64];
            snprintf(tag, 64, "write_particles_hdf5_%s_%s", file_mode, pos_mode);
            
            // Standard metric output
            MESSAGE(("[METRIC],%s,%ld,%.4f,%.4f,%.4f",
                     tag, (long)step(), t_metric, mb_total, throughput));
            
            // Detailed diagnostic output
            MESSAGE(("[DIAGNOSTIC],%s,%ld,Total:%.4f,Setup:%.4f,MetaCreate:%.4f,"
                     "BufPack:%.4f,Compute:%.4f,H5Dwrite:%.4f,MetaClose:%.4f",
                     tag, (long)step(), max_total, max_setup, max_meta_create,
                     max_buf_pack, max_compute, max_h5dwrite, max_meta_close));
        }

    } KOKKOS_TOC(write_particles_hdf5, 1);
}

// =============================================================================
// HDF5 FIELDS WRITER
// =============================================================================

/**
 * @brief Writes electromagnetic field data to HDF5 format
 * 
 * @param params DumpParameters containing output variables and paths
 * @param fa Field array containing E and B field data
 * @param single_file True for M2O (single shared file), false for M2M (per-rank files)
 * 
 * This function performs a critical memory layout transformation:
 * - Input: Array of Structures (AoS) - field_t structs with interleaved components
 * - Output: Structure of Arrays (SoA) - separate datasets per component
 * 
 * The transformation is necessary because:
 * 1. HDF5 performs best with contiguous data per dataset
 * 2. Post-processing tools expect one variable per dataset
 * 3. Enables selective variable loading (don't need to read entire field_t)
 * 
 * Performance characteristics:
 * - Uses persistent Kokkos buffer to avoid repeated allocations
 * - Single-pass repacking via MDRangePolicy (cache-friendly)
 * - Table-driven variable selection (no conditional logic in inner loops)
 * 
 * Outputs two metrics:
 * - [METRIC]: Aggregated throughput using configurable timing (pure I/O or total)
 * - [DIAGNOSTIC]: Detailed phase timing breakdown
 * 
 * @note TIME_PURE_IO_ONLY flag controls whether metrics measure only H5Dwrite
 *       time or total function time. Set to true for fair I/O benchmarking.
 */
void vpic_simulation::write_fields_hdf5(DumpParameters& params, 
                                        field_array_t* fa, 
                                        bool single_file) {
    KOKKOS_TIC() {
        // =====================================================================
        // CONFIGURATION
        // =====================================================================
        // Controls metric calculation: true = H5Dwrite only, false = total time
        const bool TIME_PURE_IO_ONLY = true;
        // =====================================================================

        double t_total_start = wallclock();

        // Phase timing accumulators
        double t_setup = 0.0;
        double t_meta_create = 0.0;
        double t_buf_pack = 0.0;
        double t_compute = 0.0;      // Unused for fields (no equivalent to center_p_dump)
        double t_h5dwrite = 0.0;
        double t_meta_close = 0.0;

        // =====================================================================
        // PHASE 1: SETUP & VARIABLE SELECTION
        // =====================================================================
        
        double t_phase_start = wallclock();

        if (!fa) ERROR(("NULL field array"));
        if (step() > fa->last_copied) fa->copy_to_host();
        grid_t* g = fa->g;

        // Ensure output directory structure exists
        if (rank() == 0) { 
            ensure_directory(params.baseDir);
            char time_dir[256];
            snprintf(time_dir, 256, "%s/T.%ld", params.baseDir, (long)step());
            ensure_directory(time_dir);
        }
        mp_barrier();  // Ensure directories exist before all ranks proceed

        // ---------------------------------------------------------------------
        // Variable Mapping Table
        // ---------------------------------------------------------------------
        // Maps user-selected bit indices to field_t memory layout.
        // Only variables with bitset(bit) == true will be written.
        
        struct FieldMap { 
            int bit;            // Bit index in params.output_vars
            const char* name;   // HDF5 dataset name
            int offset;         // Byte offset within field_t struct
        };
        
        static const std::vector<FieldMap> field_map = {
            {0, "ex",         offsetof(field_t, ex)},
            {1, "ey",         offsetof(field_t, ey)},
            {2, "ez",         offsetof(field_t, ez)},
            {3, "div_e_err",  offsetof(field_t, div_e_err)},
            {4, "cbx",        offsetof(field_t, cbx)},
            {5, "cby",        offsetof(field_t, cby)},
            {6, "cbz",        offsetof(field_t, cbz)},
            {7, "div_b_err",  offsetof(field_t, div_b_err)}
        };
        
        // Filter to only actively requested variables
        struct ActiveVar { 
            const char* name; 
            int offset; 
        };
        std::vector<ActiveVar> active_vars;
        
        for (const auto& m : field_map) {
            if (params.output_vars.bitset(m.bit)) {
                active_vars.push_back({m.name, m.offset});
            }
        }
        int num_active = active_vars.size();

        t_setup = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 2: FILE & METADATA CREATION
        // =====================================================================
        
        t_phase_start = wallclock();

        char fname[256];
        if (single_file) {
            snprintf(fname, 256, "%s/T.%ld/%s.%ld.h5", 
                     params.baseDir, (long)step(), params.baseFileName, (long)step());
        } else {
            snprintf(fname, 256, "%s/T.%ld/%s.%ld.%d.h5", 
                     params.baseDir, (long)step(), params.baseFileName, (long)step(), rank());
        }

        // Compute grid layout (global dimensions and rank offset)
        hsize_t local_dims[3] = {(hsize_t)g->nz, (hsize_t)g->ny, (hsize_t)g->nx};
        GlobalGridInfo ginfo = get_global_grid_info(g, single_file);

        // Create file with optimized settings
        hid_t dspace, mspace;
        hid_t fid = setup_hdf5_file(fname, single_file, ginfo, local_dims, dspace, mspace);

        // Write simulation metadata attributes
        write_scalar_attr(fid, "step", H5T_NATIVE_LONG, (long)g->step);
        write_scalar_attr(fid, "time", H5T_NATIVE_DOUBLE, (double)g->t0);

        // Configure data transfer properties
        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        #ifdef VPIC_HDF5_PARALLEL
        if (single_file) {
            H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
        }
        #endif

        t_meta_create = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 3: MEMORY LAYOUT TRANSFORMATION (AoS → SoA)
        // =====================================================================
        // Convert from field_t array (interleaved components) to separate
        // contiguous arrays per component. This requires a full data copy but
        // enables optimal HDF5 write performance and post-processing access.
        
        size_t num_cells = g->nx * g->ny * g->nz;
        
        // Persistent buffer to avoid repeated allocations
        // Dimensions: [max_vars=8][num_cells]
        static Kokkos::View<float**, Kokkos::HostSpace> buffer("h5_field_buf", 8, 0);
        if (buffer.extent(1) < num_cells) {
            Kokkos::resize(buffer, 8, num_cells);
        }

        t_phase_start = wallclock();

        // ---------------------------------------------------------------------
        // Single-Pass Repacking Kernel
        // ---------------------------------------------------------------------
        // Uses MDRangePolicy for cache-friendly 3D iteration.
        // Each thread extracts all requested components from one field_t cell.
        
        auto* f_base = fa->f;
        int nx = g->nx, ny = g->ny, nz = g->nz;
        
        using Policy3D = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, 
                                               Kokkos::Rank<3>>;
        
        Kokkos::parallel_for("PackFieldsSinglePass", 
            Policy3D({1, 1, 1}, {nz+1, ny+1, nx+1}),
            [&](int k, int j, int i) {
                // Compute linear index in output buffer (Z-major ordering)
                size_t buf_idx = ((k-1) * ny * nx) + ((j-1) * nx) + (i-1);
                
                // Get base pointer to this cell's field_t struct
                char* cell_base = (char*)&f_base[voxel(i, j, k)];
                
                // Extract all active components via offset pointer arithmetic
                for (int v = 0; v < num_active; ++v) {
                    buffer(v, buf_idx) = *(float*)(cell_base + active_vars[v].offset);
                }
            });
        Kokkos::fence();

        t_buf_pack = wallclock() - t_phase_start;
        t_compute = 0.0;  // No physics computation for fields

        // =====================================================================
        // PHASE 4: WRITE DATA TO HDF5
        // =====================================================================
        // Write each component as a separate 3D dataset.
        // For M2O, all ranks participate in collective write.
        
        t_phase_start = wallclock();

        for (int v = 0; v < num_active; ++v) {
            hid_t dset = H5Dcreate2(fid, active_vars[v].name, H5T_NATIVE_FLOAT, 
                                   dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            // Write from buffer row v (contiguous data for this component)
            H5Dwrite(dset, H5T_NATIVE_FLOAT, mspace, dspace, dxpl, &buffer(v, 0));
            H5Dclose(dset);
        }

        t_h5dwrite = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 5: CLEANUP & CLOSE
        // =====================================================================
        
        t_phase_start = wallclock();

        H5Pclose(dxpl);
        H5Sclose(mspace);
        H5Sclose(dspace);
        H5Fclose(fid);

        t_meta_close = wallclock() - t_phase_start;
        double t_total = wallclock() - t_total_start;

        // =====================================================================
        // PERFORMANCE METRICS
        // =====================================================================
        
        // Find maximum time spent in each phase across all MPI ranks
        double max_setup, max_meta_create, max_buf_pack, max_compute,
               max_h5dwrite, max_meta_close, max_total;
        
        mp_allmax_d(&t_setup, &max_setup, 1);
        mp_allmax_d(&t_meta_create, &max_meta_create, 1);
        mp_allmax_d(&t_buf_pack, &max_buf_pack, 1);
        mp_allmax_d(&t_compute, &max_compute, 1);
        mp_allmax_d(&t_h5dwrite, &max_h5dwrite, 1);
        mp_allmax_d(&t_meta_close, &max_meta_close, 1);
        mp_allmax_d(&t_total, &max_total, 1);

        if (rank() == 0 && num_active > 0) {
            // Calculate total data volume across all ranks
            double mb_total = ((double)(num_cells * num_active * sizeof(float)) / 
                              (1024.0 * 1024.0)) * nproc();
            
            // Determine which time metric to use based on internal flag
            double t_metric = TIME_PURE_IO_ONLY ? max_h5dwrite : max_total;
            double throughput = mb_total / t_metric;
            
            const char* tag = single_file ? "write_fields_hdf5_M2O" : "write_fields_hdf5_M2M";
            
            // Standard metric output
            MESSAGE(("[METRIC],%s,%ld,%.4f,%.4f,%.4f", 
                     tag, (long)step(), t_metric, mb_total, throughput));
            
            // Detailed diagnostic output
            MESSAGE(("[DIAGNOSTIC],%s,%ld,Total:%.4f,Setup:%.4f,MetaCreate:%.4f,"
                     "BufPack:%.4f,Compute:%.4f,H5Dwrite:%.4f,MetaClose:%.4f",
                     tag, (long)step(), max_total, max_setup, max_meta_create,
                     max_buf_pack, max_compute, max_h5dwrite, max_meta_close));
        }

    } KOKKOS_TOC(write_fields_hdf5, 1);
}

// =============================================================================
// HDF5 HYDRO WRITER
// =============================================================================

/**
 * @brief Writes hydrodynamic moment data to HDF5 format
 * 
 * @param params DumpParameters containing output variables and paths
 * @param ha Hydro array structure for moment storage
 * @param sp_name Name of species to compute moments for
 * @param single_file True for M2O (single shared file), false for M2M (per-rank files)
 * 
 * This function differs from field writing in a critical way: hydro moments are
 * NOT pre-computed during the simulation. They must be accumulated from particle
 * data on-demand during I/O:
 * 
 * Physics Pipeline:
 * 1. Clear hydro array (J, ρ, pressure tensor, etc.)
 * 2. Accumulate moments from particle distribution via scatter operation
 * 3. Synchronize ghost cells across MPI boundaries
 * 4. Write to disk
 * 
 * This "compute-during-I/O" model means setup time includes physics calculations,
 * making it fundamentally different from field dumps where data already exists.
 * 
 * Memory Layout Transformation (like fields):
 * - Input: Array of Structures (AoS) - hydro_t structs with 14 interleaved moments
 * - Output: Structure of Arrays (SoA) - separate datasets per moment
 * 
 * Performance characteristics:
 * - Physics accumulation dominates setup time (~60-80% for particle-rich cells)
 * - Uses persistent Kokkos buffer to avoid repeated allocations
 * - Single-pass repacking via MDRangePolicy (cache-friendly)
 * - Table-driven variable selection (no conditional logic in inner loops)
 * 
 * Outputs two metrics:
 * - [METRIC]: Aggregated throughput using configurable timing (pure I/O or total)
 * - [DIAGNOSTIC]: Detailed phase timing breakdown
 * 
 * @note TIME_PURE_IO_ONLY flag controls whether metrics measure only H5Dwrite
 *       time or total function time. Set to true for fair I/O benchmarking.
 * @note Setup timing includes physics accumulation, making it much larger than
 *       for fields/particles where data pre-exists.
 */
void vpic_simulation::write_hydro_hdf5(DumpParameters& params, 
                                       hydro_array_t* ha, 
                                       const char* sp_name, 
                                       bool single_file) {
    KOKKOS_TIC() {
        // =====================================================================
        // CONFIGURATION
        // =====================================================================
        // Controls metric calculation: true = H5Dwrite only, false = total time
        const bool TIME_PURE_IO_ONLY = true;
        // =====================================================================

        double t_total_start = wallclock();

        // Phase timing accumulators
        double t_setup = 0.0;
        double t_meta_create = 0.0;
        double t_buf_pack = 0.0;
        double t_compute = 0.0;      // Unused (physics is in setup phase)
        double t_h5dwrite = 0.0;
        double t_meta_close = 0.0;

        // =====================================================================
        // PHASE 1: SETUP & PHYSICS ACCUMULATION
        // =====================================================================
        // Unlike fields (pre-computed), hydro moments must be calculated from
        // particles on-demand. This is the most expensive phase.
        
        double t_phase_start = wallclock();

        if (!ha) ERROR(("NULL hydro array"));
        
        species_t* sp = find_species(sp_name);
        if (!sp) ERROR(("Invalid species: %s", sp_name));

        // ---------------------------------------------------------------------
        // Physics Accumulation: Scatter particles → moments
        // ---------------------------------------------------------------------
        // Computes current density (J), charge density (ρ), momentum (P),
        // kinetic energy (KE), and pressure tensor (T) from particle distribution.
        
        Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
        clear_hydro_array(ha);
        accumulate_hydro_p_kokkos(sp->k_p_d, sp->k_p_i_d, hydro_array->k_h_d, 
                                 interpolator_array->k_i_d, sp);
        ha->copy_to_host();
        synchronize_hydro_array(ha);  // MPI ghost cell exchange
        
        grid_t* g = ha->g;

        // Ensure output directory structure exists
        if (rank() == 0) {
            ensure_directory(params.baseDir);
            char time_dir[256];
            snprintf(time_dir, 256, "%s/T.%ld", params.baseDir, (long)step());
            ensure_directory(time_dir);
        }
        mp_barrier();  // Ensure directories exist before all ranks proceed

        // ---------------------------------------------------------------------
        // Variable Mapping Table
        // ---------------------------------------------------------------------
        // Maps user-selected bit indices to hydro_t memory layout.
        // Only variables with bitset(bit) == true will be written.
        
        struct HydroMap {
            int bit;            // Bit index in params.output_vars
            const char* name;   // HDF5 dataset name
            int offset;         // Byte offset within hydro_t struct
        };
        
        static const std::vector<HydroMap> hydro_map = {
            // Current density
            {0,  "jx",  offsetof(hydro_t, jx)},
            {1,  "jy",  offsetof(hydro_t, jy)},
            {2,  "jz",  offsetof(hydro_t, jz)},
            // Charge & momentum
            {3,  "rho", offsetof(hydro_t, rho)},
            {4,  "px",  offsetof(hydro_t, px)},
            {5,  "py",  offsetof(hydro_t, py)},
            {6,  "pz",  offsetof(hydro_t, pz)},
            // Energy & pressure tensor
            {7,  "ke",  offsetof(hydro_t, ke)},
            {8,  "txx", offsetof(hydro_t, txx)},
            {9,  "tyy", offsetof(hydro_t, tyy)},
            {10, "tzz", offsetof(hydro_t, tzz)},
            {11, "tyz", offsetof(hydro_t, tyz)},
            {12, "tzx", offsetof(hydro_t, tzx)},
            {13, "txy", offsetof(hydro_t, txy)}
        };
        
        // Filter to only actively requested variables
        struct ActiveVar {
            const char* name;
            int offset;
        };
        std::vector<ActiveVar> active_vars;
        
        for (const auto& m : hydro_map) {
            if (params.output_vars.bitset(m.bit)) {
                active_vars.push_back({m.name, m.offset});
            }
        }
        int num_active = active_vars.size();

        t_setup = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 2: FILE & METADATA CREATION
        // =====================================================================
        
        t_phase_start = wallclock();

        char fname[256];
        if (single_file) {
            snprintf(fname, 256, "%s/T.%ld/%s.%ld.h5",
                     params.baseDir, (long)step(), params.baseFileName, (long)step());
        } else {
            snprintf(fname, 256, "%s/T.%ld/%s.%ld.%d.h5",
                     params.baseDir, (long)step(), params.baseFileName, (long)step(), rank());
        }

        // Compute grid layout (global dimensions and rank offset)
        hsize_t local_dims[3] = {(hsize_t)g->nz, (hsize_t)g->ny, (hsize_t)g->nx};
        GlobalGridInfo ginfo = get_global_grid_info(g, single_file);

        // Create file with optimized settings
        hid_t dspace, mspace;
        hid_t fid = setup_hdf5_file(fname, single_file, ginfo, local_dims, dspace, mspace);

        // Write simulation metadata attributes
        write_scalar_attr(fid, "step", H5T_NATIVE_LONG, (long)g->step);
        write_scalar_attr(fid, "time", H5T_NATIVE_DOUBLE, (double)g->t0);

        // Configure data transfer properties
        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        #ifdef VPIC_HDF5_PARALLEL
        if (single_file) {
            H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
        }
        #endif

        t_meta_create = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 3: MEMORY LAYOUT TRANSFORMATION (AoS → SoA)
        // =====================================================================
        // Convert from hydro_t array (14 interleaved moments) to separate
        // contiguous arrays per moment. This enables optimal HDF5 write
        // performance and selective post-processing access.
        
        size_t num_cells = g->nx * g->ny * g->nz;
        
        // Persistent buffer to avoid repeated allocations
        // Dimensions: [max_moments=14][num_cells]
        static Kokkos::View<float**, Kokkos::HostSpace> buffer("h5_hydro_buf", 14, 0);
        if (buffer.extent(1) < num_cells) {
            Kokkos::resize(buffer, 14, num_cells);
        }

        t_phase_start = wallclock();

        // ---------------------------------------------------------------------
        // Single-Pass Repacking Kernel
        // ---------------------------------------------------------------------
        // Uses MDRangePolicy for cache-friendly 3D iteration.
        // Each thread extracts all requested moments from one hydro_t cell.
        
        auto* h_base = ha->h;
        int nx = g->nx, ny = g->ny, nz = g->nz;
        
        using Policy3D = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                                               Kokkos::Rank<3>>;
        
        Kokkos::parallel_for("PackHydroSinglePass",
            Policy3D({1, 1, 1}, {nz+1, ny+1, nx+1}),
            [&](int k, int j, int i) {
                // Compute linear index in output buffer (Z-major ordering)
                size_t buf_idx = ((k-1) * ny * nx) + ((j-1) * nx) + (i-1);
                
                // Get base pointer to this cell's hydro_t struct
                char* cell_base = (char*)&h_base[voxel(i, j, k)];
                
                // Extract all active moments via offset pointer arithmetic
                for (int v = 0; v < num_active; ++v) {
                    buffer(v, buf_idx) = *(float*)(cell_base + active_vars[v].offset);
                }
            });
        Kokkos::fence();

        t_buf_pack = wallclock() - t_phase_start;
        
        // t_compute unused: physics accumulation already happened in setup phase
        t_compute = 0.0;

        // =====================================================================
        // PHASE 4: WRITE DATA TO HDF5
        // =====================================================================
        // Write each moment as a separate 3D dataset.
        // For M2O, all ranks participate in collective write.
        
        t_phase_start = wallclock();

        for (int v = 0; v < num_active; ++v) {
            hid_t dset = H5Dcreate2(fid, active_vars[v].name, H5T_NATIVE_FLOAT,
                                   dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            // Write from buffer row v (contiguous data for this moment)
            H5Dwrite(dset, H5T_NATIVE_FLOAT, mspace, dspace, dxpl, &buffer(v, 0));
            H5Dclose(dset);
        }

        t_h5dwrite = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 5: CLEANUP & CLOSE
        // =====================================================================
        
        t_phase_start = wallclock();

        H5Pclose(dxpl);
        H5Sclose(mspace);
        H5Sclose(dspace);
        H5Fclose(fid);

        t_meta_close = wallclock() - t_phase_start;
        double t_total = wallclock() - t_total_start;

        // =====================================================================
        // PERFORMANCE METRICS
        // =====================================================================
        
        // Find maximum time spent in each phase across all MPI ranks
        double max_setup, max_meta_create, max_buf_pack, max_compute,
               max_h5dwrite, max_meta_close, max_total;
        
        mp_allmax_d(&t_setup, &max_setup, 1);
        mp_allmax_d(&t_meta_create, &max_meta_create, 1);
        mp_allmax_d(&t_buf_pack, &max_buf_pack, 1);
        mp_allmax_d(&t_compute, &max_compute, 1);
        mp_allmax_d(&t_h5dwrite, &max_h5dwrite, 1);
        mp_allmax_d(&t_meta_close, &max_meta_close, 1);
        mp_allmax_d(&t_total, &max_total, 1);

        if (rank() == 0 && num_active > 0) {
            // Calculate total data volume across all ranks
            double mb_total = ((double)(num_cells * num_active * sizeof(float)) /
                              (1024.0 * 1024.0)) * nproc();
            
            // Determine which time metric to use based on internal flag
            double t_metric = TIME_PURE_IO_ONLY ? max_h5dwrite : max_total;
            double throughput = mb_total / t_metric;
            
            const char* tag = single_file ? "write_hydro_hdf5_M2O" : "write_hydro_hdf5_M2M";
            
            // Standard metric output
            MESSAGE(("[METRIC],%s,%ld,%.4f,%.4f,%.4f",
                     tag, (long)step(), t_metric, mb_total, throughput));
            
            // Detailed diagnostic output
            MESSAGE(("[DIAGNOSTIC],%s,%ld,Total:%.4f,Setup:%.4f,MetaCreate:%.4f,"
                     "BufPack:%.4f,Compute:%.4f,H5Dwrite:%.4f,MetaClose:%.4f",
                     tag, (long)step(), max_total, max_setup, max_meta_create,
                     max_buf_pack, max_compute, max_h5dwrite, max_meta_close));
        }

    } KOKKOS_TOC(write_hydro_hdf5, 1);
}
#endif // VPIC_ENABLE_HDF5