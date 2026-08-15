#include "vpic.h"
#include <sys/stat.h> // for mkdir
#include <vector>

#ifdef VPIC_ENABLE_HDF5
#include "hdf5.h"
#endif // VPIC_ENABLE_HDF5

// -----------------------------------------------------------------------------
// HELPER: DIRECTORY CREATION
// -----------------------------------------------------------------------------
static void ensure_directory(const char* path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        mkdir(path, 0755);
    }
}

// -----------------------------------------------------------------------------
// BINARY HEADER
// -----------------------------------------------------------------------------
struct BinaryHeader {
    int32_t magic;      // 0xBEEF0002
    int32_t version;    // 1
    int32_t step;       // Current step
    int32_t nx, ny, nz; // Physical Grid Dimensions
    float dt;           // Timestep
    float dx, dy, dz;   // Cell Size
    float x0, y0, z0;   // Global Origin
    float q_m;          // Species Charge/Mass (or 0.0)
    int32_t num_vars;   // Number of floats per cell
    int32_t var_mask;   // [NEW] Bitmask of variables (replaces padding[0])
};

// -----------------------------------------------------------------------------
// BINARY FIELD WRITER (Table-Driven Design)
// -----------------------------------------------------------------------------
void vpic_simulation::write_fields_binary(DumpParameters& params, field_array_t* fa) {
    TIC {
        if(!fa) ERROR(("NULL field array"));
        if(step() > fa->last_copied) fa->copy_to_host();
        grid_t* g = fa->g;

        // --- 1. DEFINITION TABLE ---
        // Maps Bitmask Index -> Memory Offset
        struct FieldComponent {
            int bit;      // The bit index in vpic.h (e.g., 0 for Ex, 4 for Bx)
            int offset;   // The byte offset in field_t
        };

        static const std::vector<FieldComponent> map = {
            {0, (int)offsetof(field_t, ex)},  // Electric
            {1, (int)offsetof(field_t, ey)},
            {2, (int)offsetof(field_t, ez)},
            // Bit 3 (div_e_err) skipped for standard output, add here if desired
            {4, (int)offsetof(field_t, cbx)}, // Magnetic
            {5, (int)offsetof(field_t, cby)},
            {6, (int)offsetof(field_t, cbz)}
        };

        // --- 2. PREPARE ---
        if(rank() == 0) {
            ensure_directory(params.baseDir);
            char timeDir[256]; snprintf(timeDir, 256, "%s/T.%ld", params.baseDir, (long)step());
            ensure_directory(timeDir);
        }
        mp_barrier();

        char fname[256];
        snprintf(fname, 256, "%s/T.%ld/%s.%ld.%d", 
                 params.baseDir, (long)step(), params.baseFileName, (long)step(), rank());

        FILE *fp = fopen(fname, "wb");
        if(!fp) ERROR(("Could not open file: %s", fname));

        // --- 3. DYNAMIC HEADER ---
        // Count only the variables requested by the deck AND present in our map
        int32_t num_vars = 0;
        for(const auto& comp : map) {
            if(params.output_vars.bitset(comp.bit)) num_vars++;
        }
        double io_tic = wallclock();
        // Standard Header
        int32_t magic = (int32_t)0xBEEF0002;
        int32_t version = 0;
        int32_t step_i = (int32_t)step();
        int32_t nx = g->nx, ny = g->ny, nz = g->nz;
        float dt = g->dt, dx = g->dx, dy = g->dy, dz = g->dz;
        float q_m = 1.0; 
        
        fwrite(&magic, 4, 1, fp);
        fwrite(&version, 4, 1, fp);
        fwrite(&step_i, 4, 1, fp);
        fwrite(&nx, 4, 1, fp);
        fwrite(&ny, 4, 1, fp);
        fwrite(&nz, 4, 1, fp);
        fwrite(&dt, 4, 1, fp);
        fwrite(&dx, 4, 1, fp);
        fwrite(&dy, 4, 1, fp);
        fwrite(&dz, 4, 1, fp);
        float padding[3] = {0,0,0}; fwrite(padding, 4, 3, fp);
        fwrite(&q_m, 4, 1, fp);
        fwrite(&num_vars, 4, 1, fp); // Dynamic Count
        int32_t p_count = 0; fwrite(&p_count, 4, 1, fp);

        // --- 4. DATA WRITE ---
        size_t num_cells = g->nx * g->ny * g->nz;
        std::vector<float> buffer(num_cells);
        auto* f_base = fa->f;

        // Iterate through the MAP, not an arbitrary loop
        for(const auto& comp : map) {
            // Only write if the user requested this specific bit
            if(params.output_vars.bitset(comp.bit)) {
                
                size_t idx = 0;
                for(int k=1; k<=g->nz; k++) {
                    for(int j=1; j<=g->ny; j++) {
                        for(int i=1; i<=g->nx; i++) {
                            int voxel_idx = voxel(i, j, k);
                            // Safe offset access
                            float val = *(float*)((char*)&f_base[voxel_idx] + comp.offset);
                            buffer[idx++] = val;
                        }
                    }
                }
                fwrite(buffer.data(), sizeof(float), num_cells, fp);
            }
        }

        fclose(fp);
        double pure_io_time = wallclock() - io_tic;
        double t_local = wallclock() - _profile_tic; 
        double t_max = 0.0;
        mp_allmax_d(&pure_io_time, &t_max, 1);

        // Make sure to use 'vars_written' or 'num_vars' depending on the function!
        if (rank() == 0 && num_vars > 0) { 
            // Local volume * number of ranks
            double mb_total = ((double)(num_cells * num_vars * sizeof(float)) / (1024.0 * 1024.0)) * nproc();
            double throughput = mb_total / t_max;
            MESSAGE(("[METRIC],write_fields_bin,%ld,%.4f,%.4f,%.4f", 
                     (long)step(), t_max, mb_total, throughput));
        }
    } TOC(write_fields_bin, 1);
}

// -----------------------------------------------------------------------------
// BINARY HYDRO WRITER (Table-Driven Design)
// -----------------------------------------------------------------------------
void vpic_simulation::write_hydro_binary(DumpParameters& params, hydro_array_t* ha, const char* sp_name) {
    TIC {
        if(!ha) ERROR(("NULL hydro array"));
        
        // --- FIX BEGIN: Perform Physics Accumulation ---
        // 1. Find the species
        species_t *sp = find_species(sp_name);
        if( !sp ) ERROR(( "Invalid species \"%s\"", sp_name ));

        // 2. Clear Grid & Accumulate from Particles (Standard VPIC-Kokkos)
        Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
        clear_hydro_array(ha);
        accumulate_hydro_p_kokkos(sp->k_p_d, sp->k_p_i_d, hydro_array->k_h_d, interpolator_array->k_i_d, sp);
        
        // 3. Sync to Host
        ha->copy_to_host();
        synchronize_hydro_array(ha);
        // --- FIX END ---

        grid_t* g = ha->g;

        struct HydroComponent { int bit; int offset; };
        static const std::vector<HydroComponent> map = {
            {0, (int)offsetof(hydro_t, jx)}, {1, (int)offsetof(hydro_t, jy)}, {2, (int)offsetof(hydro_t, jz)},
            {3, (int)offsetof(hydro_t, rho)}, {4, (int)offsetof(hydro_t, px)}, {5, (int)offsetof(hydro_t, py)},
            {6, (int)offsetof(hydro_t, pz)}, {7, (int)offsetof(hydro_t, ke)}, {8, (int)offsetof(hydro_t, txx)},
            {9, (int)offsetof(hydro_t, tyy)}, {10, (int)offsetof(hydro_t, tzz)}, {11, (int)offsetof(hydro_t, tyz)},
            {12, (int)offsetof(hydro_t, tzx)}, {13, (int)offsetof(hydro_t, txy)}
        };
        
        double io_tic = wallclock();
        
        if(rank() == 0) {
            ensure_directory(params.baseDir);
            char timeDir[256]; snprintf(timeDir, 256, "%s/T.%ld", params.baseDir, (long)step());
            ensure_directory(timeDir);
        }
        mp_barrier();

        char fname[256];
        snprintf(fname, 256, "%s/T.%ld/%s.%ld.%d", params.baseDir, (long)step(), params.baseFileName, (long)step(), rank());
        FILE *fp = fopen(fname, "wb");
        if(!fp) ERROR(("Could not open file: %s", fname));

        int32_t num_vars=0;
        for(const auto& comp : map) if(params.output_vars.bitset(comp.bit)) num_vars++;

        int32_t magic=(int32_t)0xBEEF0002, version=0, step_i=(int32_t)step();
        int32_t nx=g->nx, ny=g->ny, nz=g->nz;
        float dt=g->dt, dx=g->dx, dy=g->dy, dz=g->dz, q_m=1.0, padding[3]={0};
        
        fwrite(&magic,4,1,fp); fwrite(&version,4,1,fp); fwrite(&step_i,4,1,fp);
        fwrite(&nx,4,1,fp); fwrite(&ny,4,1,fp); fwrite(&nz,4,1,fp);
        fwrite(&dt,4,1,fp); fwrite(&dx,4,1,fp); fwrite(&dy,4,1,fp); fwrite(&dz,4,1,fp);
        fwrite(padding,4,3,fp); fwrite(&q_m,4,1,fp); fwrite(&num_vars,4,1,fp);
        int32_t p_count=0; fwrite(&p_count,4,1,fp);

        size_t num_cells = g->nx * g->ny * g->nz;
        std::vector<float> buffer(num_cells);
        auto* h_base = ha->h;

        for(const auto& comp : map) {
            if(params.output_vars.bitset(comp.bit)) {
                size_t idx = 0;
                for(int k=1; k<=g->nz; k++) {
                    for(int j=1; j<=g->ny; j++) {
                        for(int i=1; i<=g->nx; i++) {
                            float val = *(float*)((char*)&h_base[voxel(i,j,k)] + comp.offset);
                            buffer[idx++] = val;
                        }
                    }
                }
                fwrite(buffer.data(), sizeof(float), num_cells, fp);
            }
        }
        fclose(fp);
        double pure_io_time = wallclock() - io_tic;
        double t_local = wallclock() - _profile_tic; 
        double t_max = 0.0;
        mp_allmax_d(&pure_io_time, &t_max, 1);

        // Make sure to use 'vars_written' or 'num_vars' depending on the function!
        if (rank() == 0 && num_vars > 0) { 
            // Local volume * number of ranks
            double mb_total = ((double)(num_cells * num_vars * sizeof(float)) / (1024.0 * 1024.0)) * nproc();
            double throughput = mb_total / t_max;
            MESSAGE(("[METRIC],write_hydro_bin,%ld,%.4f,%.4f,%.4f", 
                     (long)step(), t_max, mb_total, throughput));
        }
    } TOC(write_hydro_bin, 1);
}

// -----------------------------------------------------------------------------
// PARTICLE WRITER
// -----------------------------------------------------------------------------
void vpic_simulation::write_particles_binary(const char* fbase, const char* species_name) {
    TIC {
        species_t* sp = find_species(species_name);
        if(!sp) ERROR(("Invalid species: %s", species_name));
        sp->copy_to_host();
        grid_t* g = sp->g;

        // =========================================================================
        // 1. All particles in sp->np are valid (ghosts are cleared pre-I/O)
        // =========================================================================
        hsize_t my_count = sp->np;

        char fname[256];
        snprintf(fname, 256, "%s.%s.%d", fbase, sp->name, rank());
        FILE *fp = fopen(fname, "wb");
        if(!fp) ERROR(("Could not open file: %s", fname));

        // 2. Write 64-Byte Header
        //struct BinaryHeader { int32_t m, v, s, nx, ny, nz; float dt, dx, dy, dz, x0, y0, z0, qm; int32_t n, vm; };
        BinaryHeader h = {(int32_t)0xBEEF0002, 1, (int32_t)g->step, g->nx, g->ny, g->nz, g->dt, g->dx, g->dy, g->dz, g->x0, g->y0, g->z0, sp->q/sp->m, 8, my_count};
        fwrite(&h, sizeof(BinaryHeader), 1, fp);

        // 3. Time-Centering & Chunked Bulk Writing
        const int PBUF_SIZE = 32768; // 1MB chunks
        particle_t* p_buf;
        MALLOC_ALIGNED(p_buf, PBUF_SIZE, 128);

        auto& k_p_h = sp->k_p_h;
        auto& k_p_i_h = sp->k_p_i_h;
        int sp_np = sp->np;
        int sp_max_np = sp->max_np;

        for(int buf_start = 0; buf_start < sp_np; buf_start += PBUF_SIZE) {
            
            // Spoof species counts for center_p_dump
            sp->np = sp_np - buf_start;
            if(sp->np > PBUF_SIZE) sp->np = PBUF_SIZE;
            sp->max_np = PBUF_SIZE;

            // Copy chunk to alignment buffer
            Kokkos::View<particle_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> pbuf_view(p_buf, PBUF_SIZE);
            Kokkos::parallel_for("Populate particle dump buffer",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np),
                KOKKOS_LAMBDA(int i) {
                    pbuf_view(i).dx = k_p_h(buf_start + i, particle_var::dx);
                    pbuf_view(i).dy = k_p_h(buf_start + i, particle_var::dy);
                    pbuf_view(i).dz = k_p_h(buf_start + i, particle_var::dz);
                    pbuf_view(i).ux = k_p_h(buf_start + i, particle_var::ux);
                    pbuf_view(i).uy = k_p_h(buf_start + i, particle_var::uy);
                    pbuf_view(i).uz = k_p_h(buf_start + i, particle_var::uz);
                    pbuf_view(i).w  = k_p_h(buf_start + i, particle_var::w);
                    pbuf_view(i).i  = k_p_i_h(buf_start + i);
                });
            Kokkos::fence();

            // Apply Time-Centering Physics
            center_p_dump(sp, p_buf, interpolator_array);

            // Filter Ghosts & Bulk Write
            if(sp->np > 0) {
                fwrite(p_buf, sizeof(particle_t), sp->np, fp);
            }
        }

        // Restore species state & Cleanup
        sp->np = sp_np;
        sp->max_np = sp_max_np;
        FREE_ALIGNED(p_buf);
        fclose(fp);
        
        
        
        // 1. Get exact local metrics
        double t_local = wallclock() - _profile_tic;
        double mb_local = (double)(my_count * sizeof(particle_t)) / (1024.0 * 1024.0);

        // 2. Global reductions via VPIC wrappers
        double t_max = 0.0;
        double mb_total = 0.0;
        
        mp_allmax_d(&t_local, &t_max, 1);
        mp_allsum_d(&mb_local, &mb_total, 1);

        if (rank() == 0) {
            double throughput = mb_total / t_max;
            // Format: [METRIC], Function, Step, Max_Time(s), Total_MB, Throughput(MB/s)
            MESSAGE(("[METRIC],write_particles_bin,%ld,%.4f,%.4f,%.4f", 
                     (long)step(), t_max, mb_total, throughput));
        }
    } TOC( write_particles_bin, 1 ); 
}

#ifdef VPIC_ENABLE_HDF5

// -----------------------------------------------------------------------------
// HELPER: Global Grid Info
// Calculates where this rank's grid patch lives in the global simulation box
// -----------------------------------------------------------------------------
struct GlobalGridInfo {
    hsize_t global_dims[3]; // Total [nx, ny, nz]
    hsize_t offset[3];      // This rank's [i, j, k] offset
};

static GlobalGridInfo get_global_grid_info(grid_t* g, bool single_file) {
    GlobalGridInfo info = {{0}, {0}};
    
    // FIX 1: Local Dims must be Z, Y, X
    // (Z is slowest/dims[0], X is fastest/dims[2])
    hsize_t local_dims[3] = {(hsize_t)g->nz, (hsize_t)g->ny, (hsize_t)g->nx};

    if (!single_file) {
        // Multi-file (M2M): The file is the size of the local grid (Z, Y, X)
        info.global_dims[0] = local_dims[0];
        info.global_dims[1] = local_dims[1];
        info.global_dims[2] = local_dims[2];
        return info;
    }

    // --- Parallel Offset Calculation ---
    double local_min[3] = {g->x0, g->y0, g->z0};
    double global_min[3];
    
    double local_max[3] = {g->x1, g->y1, g->z1};
    double global_max[3];

    mp_allmin_d(local_min, global_min, 3);
    mp_allmax_d(local_max, global_max, 3);

    // FIX 2: Global Offsets must map Z->0, Y->1, X->2
    
    // Index 0 (Slowest) -> Z
    info.offset[0] = (hsize_t)((g->z0 - global_min[2]) / g->dz + 0.5);
    info.global_dims[0] = (hsize_t)((global_max[2] - global_min[2]) / g->dz + 0.5);

    // Index 1 -> Y
    info.offset[1] = (hsize_t)((g->y0 - global_min[1]) / g->dy + 0.5);
    info.global_dims[1] = (hsize_t)((global_max[1] - global_min[1]) / g->dy + 0.5);

    // Index 2 (Fastest) -> X
    info.offset[2] = (hsize_t)((g->x0 - global_min[0]) / g->dx + 0.5);
    info.global_dims[2] = (hsize_t)((global_max[0] - global_min[0]) / g->dx + 0.5);

    return info;
}

// -----------------------------------------------------------------------------
// HELPER: Parallel Filter (Scan)
// Returns a View of indices for all valid (non-ghost) particles
// -----------------------------------------------------------------------------
static Kokkos::View<int*, Kokkos::HostSpace> 
filter_valid_particles(species_t* sp) {
    // CHANGE: Get the Integer View
    auto particles_i = sp->k_p_i_h; 
    int np = sp->np;
    
    grid_t* g = sp->g;
    int nx_stride = g->nx + 2;
    int ny_stride = g->ny + 2;

    Kokkos::View<int*, Kokkos::HostSpace> mask("valid_mask", np);
    
    Kokkos::parallel_for("MarkValid", 
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, np), 
        KOKKOS_LAMBDA(const int n) {
            // CHANGE: Access 'i' from the integer view
            int i_idx = particles_i(n); 
            int ix = i_idx % nx_stride;
            int iy = (i_idx / nx_stride) % ny_stride;
            int iz = i_idx / (nx_stride * ny_stride);
            
            bool is_valid = (ix >= 1 && ix <= g->nx && 
                             iy >= 1 && iy <= g->ny && 
                             iz >= 1 && iz <= g->nz);
            mask(n) = is_valid ? 1 : 0;
    });

    // ... (Rest of the Scan/Compact logic remains the same) ...
    Kokkos::View<int*, Kokkos::HostSpace> offsets("offsets", np);
    int total_valid = 0;
    Kokkos::parallel_scan("ScanOffsets", 
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, np),
        KOKKOS_LAMBDA(const int n, int& update, const bool final) {
            int is_valid = mask(n);
            if (final) offsets(n) = update;
            update += is_valid;
        }, total_valid);

    Kokkos::View<int*, Kokkos::HostSpace> valid_indices("valid_indices", total_valid);
    Kokkos::parallel_for("CompactIndices", 
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, np), 
        KOKKOS_LAMBDA(const int n) {
            if (mask(n)) valid_indices(offsets(n)) = n;
    });

    return valid_indices;
}

// -----------------------------------------------------------------------------
// HELPER: Write Attribute (Templated)
// -----------------------------------------------------------------------------
template<typename T>
static void write_scalar_attr(hid_t loc_id, const char* name, hid_t type_id, T value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc_id, name, type_id, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, type_id, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

// -----------------------------------------------------------------------------
// HELPER: Optimized HDF5 File Access Property List (FAPL)
// Implements ECP & h5bench best practices for parallel filesystems
// -----------------------------------------------------------------------------
static hid_t create_optimized_fapl(bool single_file) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);

    #ifdef VPIC_HDF5_PARALLEL
    if (single_file) {
        H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);

        // 1. Collective Metadata Operations (Prevents Read/Write Storms)
        #if (H5_VERS_MAJOR > 1) || (H5_VERS_MAJOR == 1 && H5_VERS_MINOR >= 10)
        H5Pset_all_coll_metadata_ops(fapl, 1);
        H5Pset_coll_metadata_write(fapl, 1);
        #endif
    }
    #endif

    // 2. Stripe Alignment (Align objects >4KB to 16MB boundaries)
    H5Pset_alignment(fapl, 4096, 16 * 1024 * 1024);

    // 3. Defer Metadata Cache Flushes (Hold B-Tree updates in RAM until close)
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

// Update your existing setup_hdf5_file to use the new helper:
static hid_t setup_hdf5_file(const char* fname, bool single_file, 
                             const GlobalGridInfo& ginfo, const hsize_t* local_dims,
                             hid_t& dataspace_id, hid_t& memspace_id) {
    hid_t fapl_id = create_optimized_fapl(single_file);
    hid_t file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);
    if (file_id < 0) ERROR(("Failed to create HDF5 file: %s", fname));

    dataspace_id = H5Screate_simple(3, ginfo.global_dims, NULL);
    if (single_file) {
        H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, ginfo.offset, NULL, local_dims, NULL);
    }
    memspace_id = H5Screate_simple(3, local_dims, NULL);
    return file_id;
}

void vpic_simulation::write_particles_hdf5(const char* fbase, const char* species_name, bool single_file) {
    KOKKOS_TIC() {
        // =====================================================================
        // INTERNAL CONFIGURATION
        // =====================================================================
        // If true, [METRIC] throughput uses ONLY the time spent inside H5Dwrite.
        // If false, [METRIC] throughput uses the total time of this entire function.
        const bool TIME_PURE_IO_ONLY = true; 
        // =====================================================================

        double t_total_start = wallclock();

        // Diagnostic Timers
        double t_setup = 0.0;
        double t_meta_create = 0.0;
        double t_buf_pack = 0.0;
        double t_compute = 0.0;
        double t_h5dwrite = 0.0;
        double t_meta_close = 0.0;
        
        double t_phase_start = wallclock();

        // ---------------------------------------------------------------------
        // Phase 1: Setup & Staging
        // ---------------------------------------------------------------------
        species_t* sp = find_species(species_name);
        if(!sp) ERROR(("Invalid species: %s", species_name));
        sp->copy_to_host();

        hsize_t my_count = sp->np;
        hsize_t total_count = my_count;
        hsize_t current_file_offset = 0;

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
        t_phase_start = wallclock();

        // ---------------------------------------------------------------------
        // Phase 2: Metadata & File Creation
        // ---------------------------------------------------------------------
        char fname[256];
        if(single_file) snprintf(fname, 256, "%s.%s.%ld.h5", fbase, sp->name,(long)step());
        else snprintf(fname, 256, "%s.%s.%d.%ld.h5", fbase, sp->name, rank(),(long)step());
        
        // Use our new Optimized FAPL
        hid_t fapl = create_optimized_fapl(single_file);
        hid_t fid = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        if(fid < 0) ERROR(("Failed HDF5 create: %s", fname));

        // Define the HDF5 Compound Datatype to match the memory struct exactly
        hid_t ptype = H5Tcreate(H5T_COMPOUND, sizeof(particle_t));
        H5Tinsert(ptype, "dx", offsetof(particle_t, dx), H5T_NATIVE_FLOAT);
        H5Tinsert(ptype, "dy", offsetof(particle_t, dy), H5T_NATIVE_FLOAT);
        H5Tinsert(ptype, "dz", offsetof(particle_t, dz), H5T_NATIVE_FLOAT);
        H5Tinsert(ptype, "i",  offsetof(particle_t, i),  H5T_NATIVE_INT32);
        H5Tinsert(ptype, "ux", offsetof(particle_t, ux), H5T_NATIVE_FLOAT);
        H5Tinsert(ptype, "uy", offsetof(particle_t, uy), H5T_NATIVE_FLOAT);
        H5Tinsert(ptype, "uz", offsetof(particle_t, uz), H5T_NATIVE_FLOAT);
        H5Tinsert(ptype, "w",  offsetof(particle_t, w),  H5T_NATIVE_FLOAT);

        hid_t fspace = H5Screate_simple(1, &total_count, NULL);
        hid_t dset_particles = H5Dcreate2(fid, "particles", ptype, fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        #ifdef VPIC_HDF5_PARALLEL
        if (single_file) H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
        #endif

        t_meta_create = wallclock() - t_phase_start;

        // =========================================================================
        // Phase 3: Monolithic Pipeline (Ablation Test)
        // =========================================================================
        hsize_t chunk_valid_count = sp->np;
        particle_t* p_buf;
        
        // Massive monolithic allocation (e.g., 16M particles = ~512 MB per rank)
        MALLOC_ALIGNED(p_buf, chunk_valid_count, 128); 

        auto& k_p_h = sp->k_p_h;
        auto& k_p_i_h = sp->k_p_i_h;

        // 3a. Untimed Buffer Packing (Just to get data into contiguous AoS memory)
        Kokkos::View<particle_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> pbuf_view(p_buf, chunk_valid_count);
        Kokkos::parallel_for("Populate monolithic dump buffer",
            Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, chunk_valid_count),
            KOKKOS_LAMBDA(int i) {
                pbuf_view(i).dx = k_p_h(i, particle_var::dx);
                pbuf_view(i).dy = k_p_h(i, particle_var::dy);
                pbuf_view(i).dz = k_p_h(i, particle_var::dz);
                pbuf_view(i).ux = k_p_h(i, particle_var::ux);
                pbuf_view(i).uy = k_p_h(i, particle_var::uy);
                pbuf_view(i).uz = k_p_h(i, particle_var::uz);
                pbuf_view(i).w  = k_p_h(i, particle_var::w);
                pbuf_view(i).i  = k_p_i_h(i);
            });
        Kokkos::fence();

        // 3b. Skip Compute (Ablation)
        t_buf_pack = 0.0; // Ignored for pure I/O test
        t_compute = 0.0;  // Skipped entirely

        // 3c. Time Pure I/O (H5Dwrite) - THE MONOLITHIC FIREHOSE
        if (single_file || chunk_valid_count > 0) {
            double t_loop_start = wallclock();
            hid_t mspace = H5Screate_simple(1, &chunk_valid_count, NULL);
            H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &current_file_offset, NULL, &chunk_valid_count, NULL);
            
            // Blast the entire 512MB array over the network in a single call
            H5Dwrite(dset_particles, ptype, mspace, fspace, dxpl, p_buf);
            
            H5Sclose(mspace);
            current_file_offset += chunk_valid_count;
            t_h5dwrite += (wallclock() - t_loop_start);
        }

        FREE_ALIGNED(p_buf);

        t_phase_start = wallclock();

        // ---------------------------------------------------------------------
        // Phase 4: Cleanup & Close
        // ---------------------------------------------------------------------
        H5Dclose(dset_particles);
        H5Tclose(ptype);
        H5Pclose(dxpl);
        H5Sclose(fspace);
        H5Fclose(fid);

        t_meta_close = wallclock() - t_phase_start;
        double t_total = wallclock() - t_total_start;

        // =====================================================================
        // Aggregation & Output
        // =====================================================================
        
        // Find the maximum time spent in each phase across all MPI ranks
        double max_setup, max_meta_create, max_buf_pack, max_compute, max_h5dwrite, max_meta_close, max_total;
        mp_allmax_d(&t_setup, &max_setup, 1);
        mp_allmax_d(&t_meta_create, &max_meta_create, 1);
        mp_allmax_d(&t_buf_pack, &max_buf_pack, 1);
        mp_allmax_d(&t_compute, &max_compute, 1);
        mp_allmax_d(&t_h5dwrite, &max_h5dwrite, 1);
        mp_allmax_d(&t_meta_close, &max_meta_close, 1);
        mp_allmax_d(&t_total, &max_total, 1);
        
        double mb_total = 0.0;
        if (single_file) mb_total = (double)(total_count * sizeof(particle_t)) / (1024.0 * 1024.0);
        else {
            double mb_local = (double)(my_count * sizeof(particle_t)) / (1024.0 * 1024.0);
            mp_allsum_d(&mb_local, &mb_total, 1);
        }

        if (rank() == 0) {
            // Determine which time metric to use based on the internal flag
            double t_metric = TIME_PURE_IO_ONLY ? max_h5dwrite : max_total;
            double throughput = mb_total / t_metric;
            
            const char* tag = single_file ? "write_particles_hdf5_M2O" : "write_particles_hdf5_M2M";
            
            // Standard Metric Output
            MESSAGE(("[METRIC],%s,%ld,%.4f,%.4f,%.4f", tag, (long)step(), t_metric, mb_total, throughput));
            
            // New Granular Diagnostic Output
            MESSAGE(("[DIAGNOSTIC],%s,%ld,Total:%.4f,Setup:%.4f,MetaCreate:%.4f,BufPack:%.4f,Compute:%.4f,H5Dwrite:%.4f,MetaClose:%.4f", 
                     tag, (long)step(), max_total, max_setup, max_meta_create, max_buf_pack, max_compute, max_h5dwrite, max_meta_close));
        }
    } KOKKOS_TOC( write_particles_hdf5, 1 );
}

// -----------------------------------------------------------------------------
// FIELDS WRITER (High-Performance + Table-Driven Fix)
// -----------------------------------------------------------------------------
void vpic_simulation::write_fields_hdf5(DumpParameters& params, field_array_t* fa, bool single_file) {
    KOKKOS_TIC() {
        if(!fa) ERROR(("NULL field array"));
        if(step() > fa->last_copied) fa->copy_to_host();
        grid_t* g = fa->g;

        if(rank()==0) { 
            ensure_directory(params.baseDir); 
            char td[256]; 
            snprintf(td,256,"%s/T.%ld",params.baseDir,(long)step()); 
            ensure_directory(td); 
        }
        mp_barrier();

        char fname[256];
        if(single_file) snprintf(fname,256,"%s/T.%ld/%s.%ld.h5",params.baseDir,(long)step(),params.baseFileName,(long)step());
        else snprintf(fname,256,"%s/T.%ld/%s.%ld.%d.h5",params.baseDir,(long)step(),params.baseFileName,(long)step(),rank());

        hsize_t local_dims[3] = {(hsize_t)g->nz, (hsize_t)g->ny, (hsize_t)g->nx};
        GlobalGridInfo ginfo = get_global_grid_info(g, single_file);

        hid_t dspace, mspace;
        hid_t fid = setup_hdf5_file(fname, single_file, ginfo, local_dims, dspace, mspace);

        write_scalar_attr(fid, "step", H5T_NATIVE_LONG, (long)g->step);
        write_scalar_attr(fid, "time", H5T_NATIVE_DOUBLE, (double)g->t0);

        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        #ifdef VPIC_HDF5_PARALLEL
        if(single_file) H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
        #endif

        // ==========================================================
        // 1. Define Variable Map & Identify Active Variables
        // ==========================================================
        struct FieldMap { int bit; const char* name; int offset; };
        static const std::vector<FieldMap> map = {
            {0,"ex",offsetof(field_t,ex)}, {1,"ey",offsetof(field_t,ey)}, {2,"ez",offsetof(field_t,ez)},
            {3,"div_e_err",offsetof(field_t,div_e_err)}, {4,"cbx",offsetof(field_t,cbx)},
            {5,"cby",offsetof(field_t,cby)}, {6,"cbz",offsetof(field_t,cbz)}, {7,"div_b_err",offsetof(field_t,div_b_err)}
        };
        
        struct ActiveVar { const char* name; int offset; };
        std::vector<ActiveVar> active_vars;
        for(const auto& m : map) {
            if(params.output_vars.bitset(m.bit)) {
                active_vars.push_back({m.name, m.offset});
            }
        }
        int num_active = active_vars.size();

        // ==========================================================
        // 2. High-Water Mark 2D Buffer (Max 8 Field Variables)
        // ==========================================================
        size_t num_cells = g->nx * g->ny * g->nz;
        static Kokkos::View<float**, Kokkos::HostSpace> buffer("h5_field_buf", 8, 0);
        if (buffer.extent(1) < num_cells) {
            Kokkos::resize(buffer, 8, num_cells);
        }

        // ==========================================================
        // 3. SINGLE-PASS AoS to SoA REPACK (Bypasses L3 Cache Cliff)
        // -> NOT TIMED AS I/O
        // ==========================================================
        auto* f_base = fa->f; 
        int nx=g->nx, ny=g->ny, nz=g->nz;
        using P = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>;
        
        Kokkos::parallel_for("PackFieldsSinglePass", P({1,1,1}, {nz+1,ny+1,nx+1}), KOKKOS_LAMBDA(int k, int j, int i) {
            size_t bidx = ((k-1)*ny*nx) + ((j-1)*nx) + (i-1);
            char* cell_base = (char*)&f_base[voxel(i,j,k)];
            // Extract all active variables from this cache-line simultaneously
            for(int v=0; v<num_active; ++v) {
                buffer(v, bidx) = *(float*)(cell_base + active_vars[v].offset);
            }
        });
        Kokkos::fence();

        // ==========================================================
        // 4. PURE I/O WRITE BLOCK
        // ==========================================================
        double io_tic = wallclock(); // START PURE I/O TIMER

        for(int v=0; v<num_active; ++v) {
            hid_t dset = H5Dcreate2(fid, active_vars[v].name, H5T_NATIVE_FLOAT, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            // &buffer(v, 0) gives the contiguous pointer for this specific variable
            H5Dwrite(dset, H5T_NATIVE_FLOAT, mspace, dspace, dxpl, &buffer(v, 0));
            H5Dclose(dset);
        }

        H5Pclose(dxpl); 
        H5Sclose(mspace); 
        H5Sclose(dspace); 
        H5Fclose(fid);
        
        double pure_io_time = wallclock() - io_tic; // STOP PURE I/O TIMER

        double t_local = wallclock() - _profile_tic; 
        double t_max = 0.0;
        mp_allmax_d(&pure_io_time, &t_max, 1);

        // Make sure to use 'vars_written' or 'num_vars' depending on the function!
        if (rank() == 0 && num_active > 0) { 
            // Local volume * number of ranks
            double mb_total = ((double)(num_cells * num_active * sizeof(float)) / (1024.0 * 1024.0)) * nproc();
            double throughput = mb_total / t_max;
            if(single_file){
            // Adjust tag as appropriate (e.g. write_fields_bin, write_hydro_h5_M2M)
                MESSAGE(("[METRIC],write_fields_hdf5_M2O,%ld,%.4f,%.4f,%.4f", 
                        (long)step(), t_max, mb_total, throughput));
            } else {
                MESSAGE(("[METRIC],write_fields_hdf5_M2M,%ld,%.4f,%.4f,%.4f", 
                        (long)step(), t_max, mb_total, throughput));
            }
        }

    } KOKKOS_TOC( write_fields_hdf5, 1 );
}

// -----------------------------------------------------------------------------
// HYDRO WRITER (High-Performance + Table-Driven Fix)
// -----------------------------------------------------------------------------
void vpic_simulation::write_hydro_hdf5(DumpParameters& params, hydro_array_t* ha, const char* sp_name, bool single_file) {
    KOKKOS_TIC() {
        if(!ha) ERROR(("NULL hydro"));
        species_t* sp=find_species(sp_name); if(!sp) ERROR(("Invalid sp"));

        // Accumulate Physics Moments (J/Rho)
        Kokkos::deep_copy(hydro_array->k_h_d, 0.0f); 
        clear_hydro_array(ha);
        accumulate_hydro_p_kokkos(sp->k_p_d, sp->k_p_i_d, hydro_array->k_h_d, interpolator_array->k_i_d, sp);
        ha->copy_to_host(); 
        synchronize_hydro_array(ha); 
        grid_t* g = ha->g;

        if(rank()==0) { 
            ensure_directory(params.baseDir); 
            char td[256]; 
            snprintf(td,256,"%s/T.%ld",params.baseDir,(long)step()); 
            ensure_directory(td); 
        }
        mp_barrier();

        char fname[256];
        if(single_file) snprintf(fname,256,"%s/T.%ld/%s.%ld.h5",params.baseDir,(long)step(),params.baseFileName,(long)step());
        else snprintf(fname,256,"%s/T.%ld/%s.%ld.%d.h5",params.baseDir,(long)step(),params.baseFileName,(long)step(),rank());

        hsize_t local_dims[3] = {(hsize_t)g->nz, (hsize_t)g->ny, (hsize_t)g->nx};
        GlobalGridInfo ginfo = get_global_grid_info(g, single_file);

        hid_t dspace, mspace;
        hid_t fid = setup_hdf5_file(fname, single_file, ginfo, local_dims, dspace, mspace);
        
        write_scalar_attr(fid, "step", H5T_NATIVE_LONG, (long)g->step);
        write_scalar_attr(fid, "time", H5T_NATIVE_DOUBLE, (double)g->t0);

        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        #ifdef VPIC_HDF5_PARALLEL
        if(single_file) H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
        #endif

        // ==========================================================
        // 1. Define Variable Map & Identify Active Variables
        // ==========================================================
        struct HydroMap { int bit; const char* name; int offset; };
        static const std::vector<HydroMap> map = {
            {0,"jx",offsetof(hydro_t,jx)}, {1,"jy",offsetof(hydro_t,jy)}, {2,"jz",offsetof(hydro_t,jz)},
            {3,"rho",offsetof(hydro_t,rho)}, {4,"px",offsetof(hydro_t,px)}, {5,"py",offsetof(hydro_t,py)},
            {6,"pz",offsetof(hydro_t,pz)}, {7,"ke",offsetof(hydro_t,ke)}, {8,"txx",offsetof(hydro_t,txx)},
            {9,"tyy",offsetof(hydro_t,tyy)}, {10,"tzz",offsetof(hydro_t,tzz)}, {11,"tyz",offsetof(hydro_t,tyz)},
            {12,"tzx",offsetof(hydro_t,tzx)}, {13,"txy",offsetof(hydro_t,txy)}
        };
        
        struct ActiveVar { const char* name; int offset; };
        std::vector<ActiveVar> active_vars;
        for(const auto& m : map) {
            if(params.output_vars.bitset(m.bit)) {
                active_vars.push_back({m.name, m.offset});
            }
        }
        int num_active = active_vars.size();

        // ==========================================================
        // 2. High-Water Mark 2D Buffer (Max 14 Hydro Variables)
        // ==========================================================
        size_t num_cells = g->nx * g->ny * g->nz;
        static Kokkos::View<float**, Kokkos::HostSpace> buffer("h5_hydro_buf", 14, 0);
        if (buffer.extent(1) < num_cells) {
            Kokkos::resize(buffer, 14, num_cells);
        }

        // ==========================================================
        // 3. SINGLE-PASS AoS to SoA REPACK
        // -> NOT TIMED AS I/O
        // ==========================================================
        auto* h_base = ha->h; 
        int nx=g->nx, ny=g->ny, nz=g->nz;
        using P = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<3>>;
        
        Kokkos::parallel_for("PackHydroSinglePass", P({1,1,1}, {nz+1,ny+1,nx+1}), KOKKOS_LAMBDA(int k, int j, int i) {
            size_t bidx = ((k-1)*ny*nx) + ((j-1)*nx) + (i-1);
            char* cell_base = (char*)&h_base[voxel(i,j,k)];
            for(int v=0; v<num_active; ++v) {
                buffer(v, bidx) = *(float*)(cell_base + active_vars[v].offset);
            }
        });
        Kokkos::fence();

        // ==========================================================
        // 4. PURE I/O WRITE BLOCK
        // ==========================================================
        double io_tic = wallclock(); // START PURE I/O TIMER

        for(int v=0; v<num_active; ++v) {
            hid_t dset = H5Dcreate2(fid, active_vars[v].name, H5T_NATIVE_FLOAT, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dset, H5T_NATIVE_FLOAT, mspace, dspace, dxpl, &buffer(v, 0));
            H5Dclose(dset);
        }

        H5Pclose(dxpl); 
        H5Sclose(mspace); 
        H5Sclose(dspace); 
        H5Fclose(fid);
        
        double pure_io_time = wallclock() - io_tic; // STOP PURE I/O TIMER

        double t_local = wallclock() - _profile_tic; 
        double t_max = 0.0;
        mp_allmax_d(&pure_io_time, &t_max, 1);

        // Make sure to use 'vars_written' or 'num_vars' depending on the function!
        if (rank() == 0 && num_active > 0) { 
            // Local volume * number of ranks
            double mb_total = ((double)(num_cells * num_active * sizeof(float)) / (1024.0 * 1024.0)) * nproc();
            double throughput = mb_total / t_max;
            if(single_file){
            // Adjust tag as appropriate (e.g. write_fields_bin, write_hydro_h5_M2M)
                MESSAGE(("[METRIC],write_hydro_hdf5_M2O,%ld,%.4f,%.4f,%.4f", 
                        (long)step(), t_max, mb_total, throughput));
            } else {
                MESSAGE(("[METRIC],write_hydro_hdf5_M2M,%ld,%.4f,%.4f,%.4f", 
                        (long)step(), t_max, mb_total, throughput));
            }
        }

    } KOKKOS_TOC( write_hydro_hdf5, 1 );
}
#endif // VPIC_ENABLE_HDF5