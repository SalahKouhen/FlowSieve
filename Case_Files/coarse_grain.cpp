#include <fenv.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cassert>

#include "../netcdf_io.hpp"
#include "../functions.hpp"
#include "../constants.hpp"

/*
 * \brief Case file to coarse-grain raw velocity fields (NOT FOR HELMHOLTZ DECOMPOSED FIELDS)
 *
 * @param   --input_file            Filename for the primary input. (default is input.nc)
 * @param   --time                  Name of the time dimension (default is time)
 * @param   --depth                 Name of the depth dimension (default is depth)
 * @param   --latitude
 * @param   --longitude
 * @param   --is_degrees
 * @param   --Nprocs_in_time
 * @param   --Nprocs_in_depth
 * @param   --zonal_vel
 * @param   --merid_vel
 * @param   --density
 * @param   --pressure
 * @param   --region_definitions_file
 * @param   --region_definitions_dim
 * @param   --region_definitions_var
 *
 */
int main(int argc, char *argv[]) {

    // PERIODIC_Y implies UNIFORM_LAT_GRID
    static_assert( (constants::UNIFORM_LAT_GRID) or (not(constants::PERIODIC_Y)),
            "PERIODIC_Y requires UNIFORM_LAT_GRID.\n"
            "Please update constants.hpp accordingly.\n");

    // NO_FULL_OUTPUTS implies APPLY_POSTPROCESS
    static_assert( (constants::APPLY_POSTPROCESS) or (not(constants::NO_FULL_OUTPUTS)),
            "If NO_FULL_OUTPUTS is true, then APPLY_POSTPROCESS must also be true, "
            "otherwise no outputs will be produced.\n"
            "Please update constants.hpp accordingly.");

    // NO_FULL_OUTPUTS implies MINIMAL_OUTPUT
    static_assert( (constants::MINIMAL_OUTPUT) or (not(constants::NO_FULL_OUTPUTS)),
            "NO_FULL_OUTPUTS implies MINIMAL_OUTPUT. "
            "You must either change NO_FULL_OUTPUTS to false, "
            "or MINIMAL_OUTPUT to true.\n" 
            "Please update constants.hpp accordingly.");
    
    // Cannot extend to poles AND be Cartesian
    static_assert( not( (constants::EXTEND_DOMAIN_TO_POLES) and (constants::CARTESIAN) ),
            "Cartesian implies that there are no poles, so cannot extend to poles."
            "Please update constants.hpp accordingly.");

    // Enable all floating point exceptions but FE_INEXACT and FE_UNDERFLOW
    //      for reasons that I do not understand, FE_ALL_EXCEPT is __NOT__ equal
    //      to the bit-wise or of the five exceptions. So instead of say "all except these"
    //      we'll just list the ones that we want
    //feenableexcept( FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW );
    //fprintf( stdout, " %d : %d \n", FE_ALL_EXCEPT, FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_INEXACT | FE_UNDERFLOW );
    feenableexcept( FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

    // Specify the number of OpenMP threads
    //   and initialize the MPI world
    int thread_safety_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_safety_provided);
    //MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI::ERRORS_THROW_EXCEPTIONS);
    const double start_time = MPI_Wtime();

    int wRank=-1, wSize=-1;
    MPI_Comm_rank( MPI_COMM_WORLD, &wRank );
    MPI_Comm_size( MPI_COMM_WORLD, &wSize );

    //
    //// Parse command-line arguments
    //
    InputParser input(argc, argv);
    if(input.cmdOptionExists("--version")){
        if (wRank == 0) { print_compile_info(NULL); } 
        return 0;
    }

    // first argument is the flag, second argument is default value (for when flag is not present)
    const std::string &input_fname       = input.getCmdOption("--input_file",  "input.nc");

    const std::string   &time_dim_name      = input.getCmdOption("--time",        "time"),
                        &depth_dim_name     = input.getCmdOption("--depth",       "depth"),
                        &latitude_dim_name  = input.getCmdOption("--latitude",    "latitude"),
                        &longitude_dim_name = input.getCmdOption("--longitude",   "longitude");

    const std::string &latlon_in_degrees  = input.getCmdOption("--is_degrees",   "true");

    const std::string   &Nprocs_in_time_string  = input.getCmdOption("--Nprocs_in_time",  "1"),
                        &Nprocs_in_depth_string = input.getCmdOption("--Nprocs_in_depth", "1");
    const int   Nprocs_in_time_input  = stoi(Nprocs_in_time_string),
                Nprocs_in_depth_input = stoi(Nprocs_in_depth_string);

    const std::string   &zonal_vel_name    = input.getCmdOption("--zonal_vel",   "uo"),
                        &merid_vel_name    = input.getCmdOption("--merid_vel",   "vo"),
                        &density_var_name  = input.getCmdOption("--density",     "rho"),
                        &pressure_var_name = input.getCmdOption("--pressure",    "p");

    const std::string   &region_defs_fname    = input.getCmdOption("--region_definitions_file",    "region_definitions.nc"),
                        &region_defs_dim_name = input.getCmdOption("--region_definitions_dim",     "region"),
                        &region_defs_var_name = input.getCmdOption("--region_definitions_var",     "region_definition");

    // Also read in the filter scales from the commandline
    //   e.g. --filter_scales "10.e3 150.76e3 1000e3" (units are in metres)
    std::vector<double> filter_scales;
    input.getFilterScales( filter_scales, "--filter_scales" );

    // Set OpenMP thread number
    const int max_threads = omp_get_max_threads();
    omp_set_num_threads( max_threads );

    // Print some header info, depending on debug level
    print_header_info();

    // Initialize dataset class instance
    dataset source_data;

    // Read in source data / get size information
    #if DEBUG >= 1
    if (wRank == 0) { fprintf(stdout, "Reading in source data.\n\n"); }
    #endif

    // Read in the grid coordinates
    source_data.load_time(      time_dim_name,      input_fname );
    source_data.load_depth(     depth_dim_name,     input_fname );
    source_data.load_latitude(  latitude_dim_name,  input_fname );
    source_data.load_longitude( longitude_dim_name, input_fname );

    // Apply some cleaning to the processor allotments if necessary. 
    source_data.check_processor_divisions( Nprocs_in_time_input, Nprocs_in_depth_input );
     
    // Convert to radians, if appropriate
    if ( (latlon_in_degrees == "true") and (not(constants::CARTESIAN)) ) {
        convert_coordinates( source_data.longitude, source_data.latitude );
    }

    // Compute the area of each 'cell' which will be necessary for integration
    source_data.compute_cell_areas();

    // Read in the velocity fields
    source_data.load_variable( "u_lon", zonal_vel_name, input_fname, true, true );
    source_data.load_variable( "u_lat", merid_vel_name, input_fname, true, true );

    // Get the MPI-local dimension sizes
    source_data.Ntime  = source_data.myCounts[0];
    source_data.Ndepth = source_data.myCounts[1];

    // No u_r in inputs, so initialize as zero
    source_data.variables.insert( std::pair< std::string, std::vector<double> >
                                           ( "u_r",       std::vector<double>(source_data.variables.at("u_lon").size(), 0.) ) 
                                );

    if (constants::COMP_BC_TRANSFERS) {
        // If desired, read in rho and p
        source_data.load_variable( "rho", density_var_name,  input_fname, false, false );
        source_data.load_variable( "p",   pressure_var_name, input_fname, false, false );
    }



    if ( not(constants::EXTEND_DOMAIN_TO_POLES) ) {
        // Mask out the pole, if necessary (i.e. set lat = 90 to land)
        mask_out_pole( source_data.latitude, source_data.mask, source_data.Ntime, source_data.Ndepth, source_data.Nlat, source_data.Nlon );
    }

    // If we're using FILTER_OVER_LAND, then the mask has been wiped out. Load in a mask that still includes land references
    //      so that we have both. Will be used to get 'water-only' region areas.
    if (constants::FILTER_OVER_LAND) { 
        read_mask_from_file( source_data.reference_mask, zonal_vel_name, input_fname,
               source_data.Nprocs_in_time, source_data.Nprocs_in_depth );
    }

    // Read in the region definitions and compute region areas
    if ( check_file_existence( region_defs_fname ) ) {
        // If the file exists, then read in from that
        source_data.load_region_definitions( region_defs_fname, region_defs_dim_name, region_defs_var_name );
    } else {
        // Otherwise, just make a single region which is the entire domain
        source_data.region_names.push_back("full_domain");
        source_data.regions.insert( std::pair< std::string, std::vector<bool> >( 
                                    "full_domain", std::vector<bool>( source_data.Nlat * source_data.Nlon, true) ) 
                );
        source_data.compute_region_areas();
    }


    //
    //// If necessary, extend the domain to reach the poles
    //
    
    if ( constants::EXTEND_DOMAIN_TO_POLES ) {
        #if DEBUG >= 0
        if (wRank == 0) { fprintf( stdout, "Extending the domain to the poles\n" ); }
        #endif

        // Extend the latitude grid to reach the poles and update source_data with the new info.
        std::vector<double> extended_latitude;
        int orig_lat_start_in_extend;
        #if DEBUG >= 2
        if (wRank == 0) { fprintf( stdout, "    Extending latitude to poles\n" ); }
        #endif
        extend_latitude_to_poles( source_data.latitude, extended_latitude, orig_lat_start_in_extend );

        // Extend out the mask
        #if DEBUG >= 2
        if (wRank == 0) { fprintf( stdout, "    Extending mask to poles\n" ); }
        #endif
        extend_mask_to_poles( source_data.mask,           source_data, extended_latitude, orig_lat_start_in_extend );
        if (constants::FILTER_OVER_LAND) { 
            extend_mask_to_poles( source_data.reference_mask, source_data, extended_latitude, orig_lat_start_in_extend, false );
        }

        // Extend out all of the variable fields
        for(const auto& var_data : source_data.variables) {
            #if DEBUG >= 2
            if (wRank == 0) { fprintf( stdout, "    Extending variable %s to poles\n", var_data.first.c_str() ); }
            #endif
            extend_field_to_poles( source_data.variables[var_data.first], source_data, extended_latitude, orig_lat_start_in_extend );
        }

        // Extend out all of the region definitions
        for(const auto& reg_data : source_data.regions) {
            #if DEBUG >= 2
            if (wRank == 0) { fprintf( stdout, "    Extending region %s to poles\n", reg_data.first.c_str() ); }
            #endif
            extend_mask_to_poles( source_data.regions[reg_data.first], source_data, extended_latitude, orig_lat_start_in_extend, false );
        }

        // Update source_data to use the extended latitude
        source_data.latitude = extended_latitude;
        source_data.Nlat = source_data.latitude.size();
        source_data.myCounts[2] = source_data.Nlat;

        // Mask out the pole, if necessary (i.e. set lat = 90 to land)
        mask_out_pole( source_data.latitude, source_data.mask, source_data.Ntime, source_data.Ndepth, source_data.Nlat, source_data.Nlon );

        // Re-compute cell areas and region areas
        source_data.compute_cell_areas();
        source_data.compute_region_areas();
    }


    //
    //// Now pass the arrays along to the filtering routines
    //
    const double pre_filter_time = MPI_Wtime();
    filtering( source_data, filter_scales );
    const double post_filter_time = MPI_Wtime();

    // Done!
    #if DEBUG >= 0
    const double delta_clock = MPI_Wtick();
    if (wRank == 0) {
        fprintf(stdout, "\n\n");
        fprintf(stdout, "Process completed.\n");
        fprintf(stdout, "\n");
        fprintf(stdout, "Start-up time  = %.13g\n", pre_filter_time - start_time);
        fprintf(stdout, "Filtering time = %.13g\n", post_filter_time - pre_filter_time);
        fprintf(stdout, "   (clock resolution = %.13g)\n", delta_clock);
    }
    #endif

    #if DEBUG >= 1
    fprintf(stdout, "Processor %d / %d waiting to finalize.\n", wRank + 1, wSize);
    #endif
    MPI_Finalize();
    return 0;
}
