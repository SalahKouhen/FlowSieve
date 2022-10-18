#include <math.h>
#include <algorithm>
#include "../constants.hpp"
#include "../functions.hpp"
#include "../netcdf_io.hpp"
#include <cassert>


// Class constructor
dataset::dataset() {
};

void dataset::load_time( const std::string dim_name, const std::string filename ) {
    if ( ( dim_name == "DNE" ) or ( dim_name == "DOES_NOT_EXIST" ) ) {
        time.resize(1);
        time[0] = 0.;
        #if DEBUG >= 1
        int wRank=-1;
        MPI_Comm_rank( MPI_COMM_WORLD, &wRank );
        if (wRank == 0) { fprintf(stdout, "Time dimension DNE, so setting as singleton.\n"); }
        #endif
    } else {
        read_var_from_file(time, dim_name, filename);
    }
    full_Ntime = time.size();
};

void dataset::load_depth( const std::string dim_name, const std::string filename ) {
    if ( ( dim_name == "DNE" ) or ( dim_name == "DOES_NOT_EXIST" ) ) {
        depth.resize(1);
        depth[0] = 0.;
        #if DEBUG >= 1
        int wRank=-1;
        MPI_Comm_rank( MPI_COMM_WORLD, &wRank );
        if (wRank == 0) { 
            fprintf(stdout, "Depth dimension DNE, so setting as singleton.\n\n");
            fflush( stdout );
        }
        #endif
    } else {
        read_var_from_file(depth, dim_name, filename);
    }
    full_Ndepth = depth.size();
};

void dataset::load_latitude( const std::string dim_name, const std::string filename ) {
    read_var_from_file(latitude, dim_name, filename);
    Nlat = latitude.size();
};

void dataset::load_longitude( const std::string dim_name, const std::string filename ) {
    read_var_from_file(longitude, dim_name, filename);
    Nlon = longitude.size();
};

void dataset::compute_cell_areas() {

    assert( (Nlat > 0) and (Nlon > 0) );    

    areas.resize( Nlat * Nlon );
    compute_areas( areas, longitude, latitude );
}

void dataset::load_variable( 
        const std::string var_name, 
        const std::string var_name_in_file, 
        const std::string filename,
        const bool read_mask,
        const bool load_counts,
        const bool do_splits
        ) {

    // Add a new entry to the variables dictionary with an empty array
    variables.insert( std::pair< std::string, std::vector<double> >( var_name, std::vector<double>() ) );

    // Now read in from the file and store in the variables dictionary
    read_var_from_file( variables.at( var_name ), var_name_in_file, filename, 
                        read_mask ? &mask : NULL, 
                        load_counts ? &myCounts : NULL, 
                        load_counts ? &myStarts : NULL, 
                        Nprocs_in_time, Nprocs_in_depth,
                        do_splits );
};

void dataset::check_processor_divisions( const int Nprocs_in_time_input, const int Nprocs_in_depth_input, const MPI_Comm ) {

    assert( (full_Ntime > 0) and (full_Ndepth > 0) and (Nlon > 0) and (Nlat > 0) ); // Must read in dimensions before checking processor divisions.

    int wRank=-1, wSize=-1;
    MPI_Comm_rank( MPI_COMM_WORLD, &wRank );
    MPI_Comm_size( MPI_COMM_WORLD, &wSize );

    // Apply some cleaning to the processor allotments if necessary. 
    Nprocs_in_time  = ( full_Ntime  == 1 ) ? 1 : 
                      ( full_Ndepth == 1 ) ? wSize : 
                                             Nprocs_in_time_input;
    Nprocs_in_depth = ( full_Ndepth == 1 ) ? 1 : 
                      ( full_Ntime  == 1 ) ? wSize : 
                                             Nprocs_in_depth_input;

    #if DEBUG >= 0
    if (Nprocs_in_time != Nprocs_in_time_input) { 
        if (wRank == 0) { fprintf(stdout, " WARNING!! Changing number of processors in time to %'d from %'d\n", Nprocs_in_time, Nprocs_in_time_input); }
    }
    if (Nprocs_in_depth != Nprocs_in_depth_input) { 
        if (wRank == 0) { fprintf(stdout, " WARNING!! Changing number of processors in depth to %'d from %'d\n", Nprocs_in_depth, Nprocs_in_depth_input); }
    }
    if (wRank == 0) { fprintf(stdout, " Nproc(time, depth) = (%'d, %'d)\n\n", Nprocs_in_time, Nprocs_in_depth); }
    #endif

    assert( Nprocs_in_time * Nprocs_in_depth == wSize );

    // Now that processor divisions have been tested, also create the sub-communicator items
    int color, key;

    // communicator for ranks with the same times
    color = wRank / Nprocs_in_depth;
    key   = wRank % Nprocs_in_depth;
    MPI_Comm_split( MPI_Comm, color, key, MPI_subcomm_sametimes); 

    // communicator for ranks with the same depths
    color = wRank % Nprocs_in_depth;
    key   = wRank / Nprocs_in_depth;
    MPI_Comm_split( MPI_Comm, color, key, MPI_subcomm_samedepths); 
}


void dataset::compute_region_areas() {

    assert( mask.size() > 0 ); // must read in mask before computing region areas
    assert( areas.size() > 0 ); // must compute cell areas before computing region areas
    const size_t num_regions = region_names.size();
    region_areas.resize( num_regions * Ntime * Ndepth );
    if (constants::FILTER_OVER_LAND) { region_areas_water_only.resize( region_areas.size() ); }

    #if DEBUG >= 2
    int wRank=-1;
    MPI_Comm_rank( MPI_COMM_WORLD, &wRank );

    if (wRank == 0) { fprintf(stdout, "  Computing geographic region areas\n"); }
    #endif

    double local_area, local_area_water_only;
    size_t Ilat, Ilon, reg_index, index, area_index;

    const int chunk_size = get_omp_chunksize(Nlat, Nlon);

    for (size_t Iregion = 0; Iregion < num_regions; ++Iregion) {
        for (size_t Itime = 0; Itime < Ntime; ++Itime) {
            for (size_t Idepth = 0; Idepth < Ndepth; ++Idepth) {

                local_area = 0.;
                local_area_water_only = 0.;

                #pragma omp parallel default(none)\
                private( Ilat, Ilon, index, area_index )\
                shared( mask, areas, Iregion, Itime, Idepth ) \
                reduction(+ : local_area, local_area_water_only)
                { 
                    #pragma omp for collapse(2) schedule(dynamic, chunk_size)
                    for (Ilat = 0; Ilat < Nlat; ++Ilat) {
                        for (Ilon = 0; Ilon < Nlon; ++Ilon) {

                            index = Index(Itime, Idepth, Ilat, Ilon, Ntime, Ndepth, Nlat, Nlon);

                            if ( mask.at(index) ) { // Skip land areas

                                area_index = Index(0, 0, Ilat, Ilon, 1, 1, Nlat, Nlon);

                                if ( regions.at( region_names.at( Iregion ) ).at(area_index) ) {
                                    local_area += areas.at(area_index); 
                                }
                            }

                            if (constants::FILTER_OVER_LAND) {
                                if ( reference_mask.at(index) ) { // Skip land areas

                                    area_index = Index(0, 0, Ilat, Ilon, 1, 1, Nlat, Nlon);

                                    if ( regions.at( region_names.at( Iregion ) ).at(area_index) ) {
                                        local_area_water_only += areas.at(area_index); 
                                    }
                                }
                            }
                        }
                    }
                }
                reg_index = Index(0, Itime, Idepth, Iregion, 1, Ntime, Ndepth, num_regions);

                region_areas.at(reg_index) = local_area;
                if (constants::FILTER_OVER_LAND) { region_areas_water_only.at(reg_index) = local_area_water_only; }
                
            }
        }
    }
}
