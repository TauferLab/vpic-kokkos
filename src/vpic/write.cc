#include "vpic.h"

// -----------------------------------------------------------------------------
// BINARY HEADER LAYOUT (Strict 68 bytes)
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
    int32_t num_vars;   // Components per element
    int32_t padding[1]; // Align to 68 bytes (Holds Particle Count)
};

// -----------------------------------------------------------------------------
// GHOST-FREE FIELD WRITER
// -----------------------------------------------------------------------------
void vpic_simulation::write_fields_binary(const char* fbase, field_array_t* fa) {
    if(!fa) ERROR(("NULL field array"));
    
    // FIX: Respect the last_copied flag to allow manual host modifications (Nano.cxx)
    if (step() > fa->last_copied) {
        fa->copy_to_host();
    }
    
    grid_t* g = fa->g;
    char fname[256];
    snprintf(fname, 256, "%s.%d", fbase, rank()); 
    
    FileIO fio;
    if(fio.open(fname, io_write) == fail) ERROR(("Could not open file %s", fname));

    // Write Header
    BinaryHeader h = {0};
    h.magic = 0xBEEF0002; h.version = 1; h.step = g->step;
    h.nx = g->nx; h.ny = g->ny; h.nz = g->nz;
    h.dt = g->dt; h.dx = g->dx; h.dy = g->dy; h.dz = g->dz;
    h.x0 = g->x0; h.y0 = g->y0; h.z0 = g->z0;
    h.num_vars = 6;
    fio.write(&h, 1);

    // Loop PHYSICAL cells only
    for(int k=1; k <= g->nz; ++k) {
        for(int j=1; j <= g->ny; ++j) {
            for(int i=1; i <= g->nx; ++i) {
                // Use class helper voxel()
                field_t* f = &fa->f[ voxel(i, j, k) ]; 

                fio.write(&f->ex, 1);
                fio.write(&f->ey, 1);
                fio.write(&f->ez, 1);
                fio.write(&f->cbx, 1);
                fio.write(&f->cby, 1);
                fio.write(&f->cbz, 1);
            }
        }
    }
    fio.close();
    MESSAGE((" [Write] Wrote fields (no ghosts) to %s", fname));
}

// -----------------------------------------------------------------------------
// GHOST-FREE HYDRO WRITER
// -----------------------------------------------------------------------------
void vpic_simulation::write_hydro_binary(const char* fbase, hydro_array_t* ha, const char* species_name) {
    if(!ha) ERROR(("NULL hydro array"));
    
    // 1. SETUP
    species_t *sp = find_species(species_name);
    if( !sp ) ERROR(( "Invalid species \"%s\"", species_name ));

    // 2. SCORCHED EARTH CLEAR (Host + Device)
    // We must ensure NO residual data exists from previous dumps.
    
    // Clear Device (Kokkos View)
    Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
    
    // Clear Host (Legacy Array)
    // This is critical because 'synchronize' accumulates. If 'copy_to_host' 
    // misses anything or if sync runs on dirty data, we get artifacts.
    // Using the internal helper from hydro_array.cc
    clear_hydro_array(ha); 

    // 3. RE-ACCUMULATE
    accumulate_hydro_p_kokkos(
        sp->k_p_d,
        sp->k_p_i_d,
        hydro_array->k_h_d,
        interpolator_array->k_i_d,
        sp
    );

    // 4. PREPARE HOST
    // Overwrite Host with fresh Device data
    ha->copy_to_host();
    
    // Reduce Ghosts (Validates boundaries)
    synchronize_hydro_array(ha);

    // 5. WRITE
    grid_t* g = ha->g;
    char fname[256];
    snprintf(fname, 256, "%s.%s.%d", fbase, species_name, rank());
    
    FileIO fio;
    if(fio.open(fname, io_write) == fail) ERROR(("Could not open file %s", fname));

    BinaryHeader h = {0};
    h.magic = 0xBEEF0002; h.version = 1; h.step = g->step;
    h.nx = g->nx; h.ny = g->ny; h.nz = g->nz;
    h.dt = g->dt; h.dx = g->dx; h.dy = g->dy; h.dz = g->dz;
    h.x0 = g->x0; h.y0 = g->y0; h.z0 = g->z0;
    h.num_vars = 4;
    fio.write(&h, 1);

    for(int k=1; k <= g->nz; ++k) {
        for(int j=1; j <= g->ny; ++j) {
            for(int i=1; i <= g->nx; ++i) {
                hydro_t* hy = &ha->h[ voxel(i, j, k) ];
                fio.write(&hy->jx, 1);
                fio.write(&hy->jy, 1);
                fio.write(&hy->jz, 1);
                fio.write(&hy->rho, 1);
            }
        }
    }
    
    fio.close();
    MESSAGE((" [Write] Wrote hydro (fresh accumulation) to %s", fname));
}

// -----------------------------------------------------------------------------
// GHOST-FREE PARTICLE WRITER
// -----------------------------------------------------------------------------
void vpic_simulation::write_particles_binary(const char* fbase, const char* species_name) {
    species_t* sp = find_species(species_name);
    if(!sp) ERROR(("Invalid species name: %s", species_name));
    
    sp->copy_to_host();
    
    grid_t* g = sp->g;
    char fname[256];
    snprintf(fname, 256, "%s.%s.%d", fbase, sp->name, rank());
    
    // 1. COUNT PASS (Memory only - fast)
    long valid_count = 0;
    int nx_stride = g->nx + 2;
    int ny_stride = g->ny + 2;
    
    for(int n=0; n<sp->np; ++n) {
        particle_t* p = &sp->p[n];
        int i_idx = p->i;
        int ix = i_idx % nx_stride;
        int iy = (i_idx / nx_stride) % ny_stride;
        int iz = i_idx / (nx_stride * ny_stride);
        
        bool is_ghost = (ix < 1 || ix > g->nx ||
                         iy < 1 || iy > g->ny ||
                         iz < 1 || iz > g->nz);
        if(!is_ghost) valid_count++;
    }

    // 2. WRITE PASS
    FileIO fio;
    if(fio.open(fname, io_write) == fail) ERROR(("Could not open file %s", fname));

    float q_m = (sp->m != 0) ? sp->q / sp->m : 0;

    BinaryHeader h = {0};
    h.magic = 0xBEEF0002; h.version = 1; h.step = g->step;
    h.nx = g->nx; h.ny = g->ny; h.nz = g->nz; 
    h.dt = g->dt; h.dx = g->dx; h.dy = g->dy; h.dz = g->dz;
    h.x0 = g->x0; h.y0 = g->y0; h.z0 = g->z0;
    h.q_m = q_m;
    h.num_vars = 8; 
    h.padding[0] = (int32_t)valid_count; 
    
    fio.write(&h, 1);

    for(int n=0; n<sp->np; ++n) {
        particle_t* p = &sp->p[n];
        int i_idx = p->i;
        int ix = i_idx % nx_stride;
        int iy = (i_idx / nx_stride) % ny_stride;
        int iz = i_idx / (nx_stride * ny_stride);
        
        bool is_ghost = (ix < 1 || ix > g->nx ||
                         iy < 1 || iy > g->ny ||
                         iz < 1 || iz > g->nz);
                         
        if (!is_ghost) {
            fio.write(&p->dx, 1);
            fio.write(&p->dy, 1);
            fio.write(&p->dz, 1);
            fio.write(&p->i, 1);
            fio.write(&p->ux, 1);
            fio.write(&p->uy, 1);
            fio.write(&p->uz, 1);
            fio.write(&p->w, 1);
        }
    }
    
    fio.close();
    MESSAGE((" [Write] Wrote %ld particles to %s", valid_count, fname));
}