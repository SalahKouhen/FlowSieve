#include <math.h>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include "../../functions.hpp"
#include "../../netcdf_io.hpp"
#include "../../constants.hpp"
#include "../../postprocess.hpp"
#include "../../preprocess.hpp"

/*!
 * \brief Main filtering driver for Helmholtz decomposed error data
 *
 * This function is the main filtering driver. It sets up the appropriate
 * loop sequences, calls the other funcations (velocity conversions), and
 * calls the IO functionality.
 *
 * @param[in]   source_data     dataset class instance containing data (Psi, Phi, etc)
 * @param[in]   scales          scales at which to filter the data
 * @param[in]   comm            MPI communicator (default MPI_COMM_WORLD)
 *
 */
void filtering_helmholtz_error(
        const dataset & source_data,
        const std::vector<double> & scales,
        const MPI_Comm comm
        ) {

    // Get dimension sizes
    const int   Nscales = scales.size(),
                Ntime   = source_data.Ntime,    // this is the MPI-local Ntime, not the full Ntime
                Ndepth  = source_data.Ndepth,   // this is the MPI-local Ndepth, not the full Ndepth
                Nlat    = source_data.Nlat,
                Nlon    = source_data.Nlon;
    const unsigned int num_pts = Ntime * Ndepth * Nlat * Nlon;

    const std::vector<double> zero_vector( num_pts, 0. );

    // Create some tidy names for variables
    const std::vector<double>   &latitude   = source_data.latitude,
                                &longitude  = source_data.longitude;

    const std::vector<double>   &F_potential    = source_data.variables.at("F_potential"),
                                &F_toroidal     = source_data.variables.at("F_toroidal"),
                                &u_r            = source_data.variables.at("u_r");

    const std::vector<double>   &uiuj_F_r   = ( constants::COMP_PI_HELMHOLTZ ) ? source_data.variables.at("uiuj_F_r")   : zero_vector,
                                &uiuj_F_Phi = ( constants::COMP_PI_HELMHOLTZ ) ? source_data.variables.at("uiuj_F_Phi") : zero_vector,
                                &uiuj_F_Psi = ( constants::COMP_PI_HELMHOLTZ ) ? source_data.variables.at("uiuj_F_Psi") : zero_vector;

    const std::vector<double>   &wind_tau_Psi = ( constants::COMP_WIND_FORCE ) ? source_data.variables.at("wind_tau_Psi") : zero_vector,
                                &wind_tau_Phi = ( constants::COMP_WIND_FORCE ) ? source_data.variables.at("wind_tau_Phi") : zero_vector;

    const std::vector<bool> &mask = source_data.mask;

    const std::vector<int>  &myStarts = source_data.myStarts;

    // Create some tidy names for the second dataset
    const std::vector<double>   &F_potential_2    = source_data.variables.at("F_potential_2"),
                                &F_toroidal_2     = source_data.variables.at("F_toroidal_2"),
                                &u_r_2            = source_data.variables.at("u_r_2");

    const std::vector<double>   &uiuj_F_r_2   = ( constants::COMP_PI_HELMHOLTZ ) ? source_data.variables.at("uiuj_F_r_2")   : zero_vector,
                                &uiuj_F_Phi_2 = ( constants::COMP_PI_HELMHOLTZ ) ? source_data.variables.at("uiuj_F_Phi_2") : zero_vector,
                                &uiuj_F_Psi_2 = ( constants::COMP_PI_HELMHOLTZ ) ? source_data.variables.at("uiuj_F_Psi_2") : zero_vector;

    const std::vector<double>   &wind_tau_Psi_2 = ( constants::COMP_WIND_FORCE ) ? source_data.variables.at("wind_tau_Psi_2") : zero_vector,
                                &wind_tau_Phi_2 = ( constants::COMP_WIND_FORCE ) ? source_data.variables.at("wind_tau_Phi_2") : zero_vector;



    // Get some MPI info
    int wRank, wSize;
    MPI_Comm_rank( comm, &wRank );
    MPI_Comm_size( comm, &wSize );

    #if DEBUG >= 2
    if (wRank == 0) { fprintf(stdout, "\nEntered filtering_helmholtz_error\n\n"); }
    #endif

    // If we've passed the DO_TIMING flag, then create some timing vars
    Timing_Records timing_records;
    double clock_on;

    #if DEBUG >= 1
    if (wRank == 0) { fprintf( stdout, "\nPreparing to apply %d filters to data with (MPI-local) sizes (%'d - %'d - %'d - %'d) \n", Nscales, Ntime, Ndepth, Nlat, Nlon ); }
    #endif

    char fname [50];
    
    const int ndims = 4;
    size_t starts[ndims] = { size_t(myStarts.at(0)), size_t(myStarts.at(1)), size_t(myStarts.at(2)), size_t(myStarts.at(3)) };
    size_t counts[ndims] = { size_t(Ntime),          size_t(Ndepth),         size_t(Nlat),           size_t(Nlon)           };
    size_t index;
    std::vector<std::string> vars_to_write;

    int LAT_lb, LAT_ub;

    std::vector<double> local_kernel( Nlat * Nlon, 0.), 
                        local_dl_kernel( Nlat*Nlon, 0. ),
                        local_dll_kernel( Nlat*Nlon, 0. );

    std::vector<double> null_vector(0);

    #if DEBUG >= 2
    if (wRank == 0) { fprintf(stdout, "\nInitializing storage arrays.\n"); }
    #endif
    std::vector<double> 

        //
        //// Arrays to store filtered Helmholtz scalars and their ell-derivatives
        //

        // Psi
        coarse_F_tor(   num_pts, 0. ),
        dl_coarse_Psi(  num_pts, 0. ),
        dll_coarse_Psi( num_pts, 0. ),

        // Phi
        coarse_F_pot(   num_pts, 0. ),
        dl_coarse_Phi(  num_pts, 0. ),
        dll_coarse_Phi( num_pts, 0. ),

        // u_r
        u_r_coarse(     num_pts, 0. ),
        dl_coarse_u_r(  num_pts, 0. ),
        dll_coarse_u_r( num_pts, 0. ),

        // Helmholtz for uiuj
        coarse_uiuj_F_r(   num_pts, 0. ),
        coarse_uiuj_F_Phi( num_pts, 0. ),
        coarse_uiuj_F_Psi( num_pts, 0. ),

        // Original KE
        KE_tor_orig(    num_pts, 0. ),
        KE_pot_orig(    num_pts, 0. ),
        KE_tot_orig(    num_pts, 0. ),

        // Coarse KE (computed from velocities)
        KE_tor_coarse(  num_pts, 0. ),
        KE_pot_coarse(  num_pts, 0. ),
        KE_tot_coarse(  num_pts, 0. ),

        // Fine KE ( tau(uu) = bar(uu) - bar(u)bar(u) )
        KE_tor_fine(    num_pts, 0. ),
        KE_pot_fine(    num_pts, 0. ),
        KE_tot_fine(    num_pts, 0. ),

        // Fine KE modified ( uu - bar(u)bar(u) )
        KE_tor_fine_mod(    num_pts, 0. ),
        KE_pot_fine_mod(    num_pts, 0. ),
        KE_tot_fine_mod(    num_pts, 0. ),

        // Filtered KE (used to compute fine KE)
        KE_tor_filt(    num_pts, 0. ),
        KE_pot_filt(    num_pts, 0. ),
        KE_tot_filt(    num_pts, 0. ),

        //
        //// KE Spectra and Spectral Slopes
        //
        u_spectrum_tot( num_pts, 0. ),
        v_spectrum_tot( num_pts, 0. ),

        u_spectrum_tor( num_pts, 0. ),
        v_spectrum_tor( num_pts, 0. ),

        u_spectrum_pot( num_pts, 0. ),
        v_spectrum_pot( num_pts, 0. ),

        spec_slope_tot( num_pts, 0. ),
        spec_slope_tor( num_pts, 0. ),
        spec_slope_pot( num_pts, 0. ),


        // Energy transport
        div_J_tor( num_pts, 0. ),
        div_J_pot( num_pts, 0. ),
        div_J_tot( num_pts, 0. ),

        // Enstrophy
        Enst_tor( num_pts, 0. ),
        Enst_pot( num_pts, 0. ),
        Enst_tot( num_pts, 0. ),

        // Velocity divergences
        div_tor(        num_pts, 0. ),
        div_pot(        num_pts, 0. ),
        div_tot(        num_pts, 0. ),

        // Cartensian velocities
        u_x_tor( num_pts, 0. ),
        u_y_tor( num_pts, 0. ),
        u_z_tor( num_pts, 0. ),

        u_x_pot( num_pts, 0. ),
        u_y_pot( num_pts, 0. ),
        u_z_pot( num_pts, 0. ),

        u_x_tot( num_pts, 0. ),
        u_y_tot( num_pts, 0. ),
        u_z_tot( num_pts, 0. ),

        u_x_tor_coarse( num_pts, 0. ),
        u_y_tor_coarse( num_pts, 0. ),
        u_z_tor_coarse( num_pts, 0. ),

        u_x_pot_coarse( num_pts, 0. ),
        u_y_pot_coarse( num_pts, 0. ),
        u_z_pot_coarse( num_pts, 0. ),

        u_x_tot_coarse( num_pts, 0. ),
        u_y_tot_coarse( num_pts, 0. ),
        u_z_tot_coarse( num_pts, 0. ),

        //
        //// Diadic (Cartesian) velocity components
        //

        // tor
        ux_ux_tor( num_pts, 0. ),
        ux_uy_tor( num_pts, 0. ),
        ux_uz_tor( num_pts, 0. ),
        uy_uy_tor( num_pts, 0. ),
        uy_uz_tor( num_pts, 0. ),
        uz_uz_tor( num_pts, 0. ),

        vort_ux_tor( num_pts, 0. ),
        vort_uy_tor( num_pts, 0. ),
        vort_uz_tor( num_pts, 0. ),

        // pot
        ux_ux_pot( num_pts, 0. ),
        ux_uy_pot( num_pts, 0. ),
        ux_uz_pot( num_pts, 0. ),
        uy_uy_pot( num_pts, 0. ),
        uy_uz_pot( num_pts, 0. ),
        uz_uz_pot( num_pts, 0. ),

        vort_ux_pot( num_pts, 0. ),
        vort_uy_pot( num_pts, 0. ),
        vort_uz_pot( num_pts, 0. ),

        // tot
        ux_ux_tot( num_pts, 0. ),
        ux_uy_tot( num_pts, 0. ),
        ux_uz_tot( num_pts, 0. ),
        uy_uy_tot( num_pts, 0. ),
        uy_uz_tot( num_pts, 0. ),
        uz_uz_tot( num_pts, 0. ),

        vort_ux_tot( num_pts, 0. ),
        vort_uy_tot( num_pts, 0. ),
        vort_uz_tot( num_pts, 0. ),

        //
        //// Spherical velocity components
        //
        zero_array( num_pts, 0.),

        // Spherical - radial velocity
        //  by definition this is potential-only since toroidal is incompressible
        //u_r( num_pts, 0.),

        // Spherical - zonal velocities
        u_lon_tor( num_pts, 0. ),
        u_lon_pot( num_pts, 0. ),
        u_lon_tot( num_pts, 0. ),

        // Spherical - meridional velocities
        u_lat_tor( num_pts, 0. ),
        u_lat_pot( num_pts, 0. ),
        u_lat_tot( num_pts, 0. ),

        // Spherical - dyadix products
        ulon_ulon( num_pts, 0. ),
        ulon_ulat( num_pts, 0. ),
        ulat_ulat( num_pts, 0. ),

        // Vorticity (only r component)
        vort_tor_r( num_pts, 0. ),
        vort_pot_r( num_pts, 0. ),
        vort_tot_r( num_pts, 0. ),

        // Full vorticity components
        full_vort_tor_r( num_pts, 0. ),
        full_vort_pot_r( num_pts, 0. ),
        full_vort_tot_r( num_pts, 0. ),

        // Okubo-Weiss values
        OkuboWeiss_tor( num_pts, 0. ),
        OkuboWeiss_pot( num_pts, 0. ),
        OkuboWeiss_tot( num_pts, 0. ),

        // Pi
        Pi_tor(  num_pts, 0. ),
        Pi_pot(  num_pts, 0. ),
        Pi_tot(  num_pts, 0. ),
        Pi_DVV(  num_pts, 0. ),
        Pi_DTT(  num_pts, 0. ),
        Pi_VDD(  num_pts, 0. ),
        Pi_VTT(  num_pts, 0. ),
        Pi_Helm( num_pts, 0. ),

        // Z ( enstrophy cascade )
        Z_tor( num_pts, 0. ),
        Z_pot( num_pts, 0. ),
        Z_tot( num_pts, 0. );

    std::vector<double> 
        //
        //// Arrays to store filtered Helmholtz scalars and their ell-derivatives for field 2
        //

        // Psi
        coarse_F_tor_2(   num_pts, 0. ),
        dl_coarse_Psi_2(  num_pts, 0. ),
        dll_coarse_Psi_2( num_pts, 0. ),

        // Phi
        coarse_F_pot_2(   num_pts, 0. ),
        dl_coarse_Phi_2(  num_pts, 0. ),
        dll_coarse_Phi_2( num_pts, 0. ),

        // u_r
        u_r_coarse_2(     num_pts, 0. ),
        dl_coarse_u_r_2(  num_pts, 0. ),
        dll_coarse_u_r_2( num_pts, 0. ),

        // Helmholtz for uiuj
        coarse_uiuj_F_r_2(   num_pts, 0. ),
        coarse_uiuj_F_Phi_2( num_pts, 0. ),
        coarse_uiuj_F_Psi_2( num_pts, 0. ),

        // Original KE
        KE_tor_orig_2(    num_pts, 0. ),
        KE_pot_orig_2(    num_pts, 0. ),
        KE_tot_orig_2(    num_pts, 0. ),

        // Coarse KE (computed from velocities)
        KE_tor_coarse_2(  num_pts, 0. ),
        KE_pot_coarse_2(  num_pts, 0. ),
        KE_tot_coarse_2(  num_pts, 0. ),

        // Fine KE ( tau(uu) = bar(uu) - bar(u)bar(u) )
        KE_tor_fine_2(    num_pts, 0. ),
        KE_pot_fine_2(    num_pts, 0. ),
        KE_tot_fine_2(    num_pts, 0. ),

        // Fine KE modified ( uu - bar(u)bar(u) )
        KE_tor_fine_mod_2(    num_pts, 0. ),
        KE_pot_fine_mod_2(    num_pts, 0. ),
        KE_tot_fine_mod_2(    num_pts, 0. ),

        // Filtered KE (used to compute fine KE)
        KE_tor_filt_2(    num_pts, 0. ),
        KE_pot_filt_2(    num_pts, 0. ),
        KE_tot_filt_2(    num_pts, 0. ),

        //
        //// KE Spectra and Spectral Slopes
        //
        u_spectrum_tot_2( num_pts, 0. ),
        v_spectrum_tot_2( num_pts, 0. ),

        u_spectrum_tor_2( num_pts, 0. ),
        v_spectrum_tor_2( num_pts, 0. ),

        u_spectrum_pot_2( num_pts, 0. ),
        v_spectrum_pot_2( num_pts, 0. ),

        spec_slope_tot_2( num_pts, 0. ),
        spec_slope_tor_2( num_pts, 0. ),
        spec_slope_pot_2( num_pts, 0. ),


        // Energy transport
        div_J_tor_2( num_pts, 0. ),
        div_J_pot_2( num_pts, 0. ),
        div_J_tot_2( num_pts, 0. ),

        // Enstrophy
        Enst_tor_2( num_pts, 0. ),
        Enst_pot_2( num_pts, 0. ),
        Enst_tot_2( num_pts, 0. ),

        // Velocity divergences
        div_tor_2(        num_pts, 0. ),
        div_pot_2(        num_pts, 0. ),
        div_tot_2(        num_pts, 0. ),

        // Cartensian velocities
        u_x_tor_2( num_pts, 0. ),
        u_y_tor_2( num_pts, 0. ),
        u_z_tor_2( num_pts, 0. ),

        u_x_pot_2( num_pts, 0. ),
        u_y_pot_2( num_pts, 0. ),
        u_z_pot_2( num_pts, 0. ),

        u_x_tot_2( num_pts, 0. ),
        u_y_tot_2( num_pts, 0. ),
        u_z_tot_2( num_pts, 0. ),

        u_x_tor_coarse_2( num_pts, 0. ),
        u_y_tor_coarse_2( num_pts, 0. ),
        u_z_tor_coarse_2( num_pts, 0. ),

        u_x_pot_coarse_2( num_pts, 0. ),
        u_y_pot_coarse_2( num_pts, 0. ),
        u_z_pot_coarse_2( num_pts, 0. ),

        u_x_tot_coarse_2( num_pts, 0. ),
        u_y_tot_coarse_2( num_pts, 0. ),
        u_z_tot_coarse_2( num_pts, 0. ),

        //
        //// Diadic (Cartesian) velocity components
        //

        // tor
        ux_ux_tor_2( num_pts, 0. ),
        ux_uy_tor_2( num_pts, 0. ),
        ux_uz_tor_2( num_pts, 0. ),
        uy_uy_tor_2( num_pts, 0. ),
        uy_uz_tor_2( num_pts, 0. ),
        uz_uz_tor_2( num_pts, 0. ),

        vort_ux_tor_2( num_pts, 0. ),
        vort_uy_tor_2( num_pts, 0. ),
        vort_uz_tor_2( num_pts, 0. ),

        // pot
        ux_ux_pot_2( num_pts, 0. ),
        ux_uy_pot_2( num_pts, 0. ),
        ux_uz_pot_2( num_pts, 0. ),
        uy_uy_pot_2( num_pts, 0. ),
        uy_uz_pot_2( num_pts, 0. ),
        uz_uz_pot_2( num_pts, 0. ),

        vort_ux_pot_2( num_pts, 0. ),
        vort_uy_pot_2( num_pts, 0. ),
        vort_uz_pot_2( num_pts, 0. ),

        // tot
        ux_ux_tot_2( num_pts, 0. ),
        ux_uy_tot_2( num_pts, 0. ),
        ux_uz_tot_2( num_pts, 0. ),
        uy_uy_tot_2( num_pts, 0. ),
        uy_uz_tot_2( num_pts, 0. ),
        uz_uz_tot_2( num_pts, 0. ),

        vort_ux_tot_2( num_pts, 0. ),
        vort_uy_tot_2( num_pts, 0. ),
        vort_uz_tot_2( num_pts, 0. ),

        //
        //// Spherical velocity components
        //
        zero_array_2( num_pts, 0.),

        // Spherical - radial velocity
        //  by definition this is potential-only since toroidal is incompressible
        //u_r( num_pts, 0.),

        // Spherical - zonal velocities
        u_lon_tor_2( num_pts, 0. ),
        u_lon_pot_2( num_pts, 0. ),
        u_lon_tot_2( num_pts, 0. ),

        // Spherical - meridional velocities
        u_lat_tor_2( num_pts, 0. ),
        u_lat_pot_2( num_pts, 0. ),
        u_lat_tot_2( num_pts, 0. ),

        // Spherical - dyadix products
        ulon_ulon_2( num_pts, 0. ),
        ulon_ulat_2( num_pts, 0. ),
        ulat_ulat_2( num_pts, 0. ),

        // Vorticity (only r component)
        vort_tor_r_2( num_pts, 0. ),
        vort_pot_r_2( num_pts, 0. ),
        vort_tot_r_2( num_pts, 0. ),

        // Full vorticity components
        full_vort_tor_r_2( num_pts, 0. ),
        full_vort_pot_r_2( num_pts, 0. ),
        full_vort_tot_r_2( num_pts, 0. ),

        // Okubo-Weiss values
        OkuboWeiss_tor_2( num_pts, 0. ),
        OkuboWeiss_pot_2( num_pts, 0. ),
        OkuboWeiss_tot_2( num_pts, 0. ),

        // Pi
        Pi_tor_2(  num_pts, 0. ),
        Pi_pot_2(  num_pts, 0. ),
        Pi_tot_2(  num_pts, 0. ),
        Pi_DVV_2(  num_pts, 0. ),
        Pi_DTT_2(  num_pts, 0. ),
        Pi_VDD_2(  num_pts, 0. ),
        Pi_VTT_2(  num_pts, 0. ),
        Pi_Helm_2( num_pts, 0. ),

        // Z ( enstrophy cascade )
        Z_tor_2( num_pts, 0. ),
        Z_pot_2( num_pts, 0. ),
        Z_tot_2( num_pts, 0. );

    std::vector<double>
        //
        //// Arrays to store filtered Helmholtz scalars and their ell-derivatives for error field
        //

        // Error F fields (dataset 1 - dataset 2)
        F_potential_error(  num_pts, 0. ),
        F_toroidal_error(   num_pts, 0. ),
        u_r_error(          num_pts, 0. ),

        // Psi
        coarse_F_tor_error(   num_pts, 0. ),
        dl_coarse_Psi_error(  num_pts, 0. ),
        dll_coarse_Psi_error( num_pts, 0. ),

        // Phi
        coarse_F_pot_error(   num_pts, 0. ),
        dl_coarse_Phi_error(  num_pts, 0. ),
        dll_coarse_Phi_error( num_pts, 0. ),

        // u_r
        u_r_coarse_error(     num_pts, 0. ),
        dl_coarse_u_r_error(  num_pts, 0. ),
        dll_coarse_u_r_error( num_pts, 0. ),

        // Helmholtz for uiuj
        coarse_uiuj_F_r_error(   num_pts, 0. ),
        coarse_uiuj_F_Phi_error( num_pts, 0. ),
        coarse_uiuj_F_Psi_error( num_pts, 0. ),

        // Original KE
        KE_tor_orig_error(    num_pts, 0. ),
        KE_pot_orig_error(    num_pts, 0. ),
        KE_tot_orig_error(    num_pts, 0. ),

        // Coarse KE (computed from velocities)
        KE_tor_coarse_error(  num_pts, 0. ),
        KE_pot_coarse_error(  num_pts, 0. ),
        KE_tot_coarse_error(  num_pts, 0. ),

        // Fine KE ( tau(uu) = bar(uu) - bar(u)bar(u) )
        KE_tor_fine_error(    num_pts, 0. ),
        KE_pot_fine_error(    num_pts, 0. ),
        KE_tot_fine_error(    num_pts, 0. ),

        // Fine KE modified ( uu - bar(u)bar(u) )
        KE_tor_fine_mod_error(    num_pts, 0. ),
        KE_pot_fine_mod_error(    num_pts, 0. ),
        KE_tot_fine_mod_error(    num_pts, 0. ),

        // Filtered KE (used to compute fine KE)
        KE_tor_filt_error(    num_pts, 0. ),
        KE_pot_filt_error(    num_pts, 0. ),
        KE_tot_filt_error(    num_pts, 0. ),

        //
        //// KE Spectra and Spectral Slopes
        //
        u_spectrum_tot_error( num_pts, 0. ),
        v_spectrum_tot_error( num_pts, 0. ),

        u_spectrum_tor_error( num_pts, 0. ),
        v_spectrum_tor_error( num_pts, 0. ),

        u_spectrum_pot_error( num_pts, 0. ),
        v_spectrum_pot_error( num_pts, 0. ),

        spec_slope_tot_error( num_pts, 0. ),
        spec_slope_tor_error( num_pts, 0. ),
        spec_slope_pot_error( num_pts, 0. ),


        // Energy transport
        div_J_tor_error( num_pts, 0. ),
        div_J_pot_error( num_pts, 0. ),
        div_J_tot_error( num_pts, 0. ),

        // Enstrophy
        Enst_tor_error( num_pts, 0. ),
        Enst_pot_error( num_pts, 0. ),
        Enst_tot_error( num_pts, 0. ),

        // Velocity divergences
        div_tor_error(        num_pts, 0. ),
        div_pot_error(        num_pts, 0. ),
        div_tot_error(        num_pts, 0. ),

        // Cartensian velocities
        u_x_tor_error( num_pts, 0. ),
        u_y_tor_error( num_pts, 0. ),
        u_z_tor_error( num_pts, 0. ),

        u_x_pot_error( num_pts, 0. ),
        u_y_pot_error( num_pts, 0. ),
        u_z_pot_error( num_pts, 0. ),

        u_x_tot_error( num_pts, 0. ),
        u_y_tot_error( num_pts, 0. ),
        u_z_tot_error( num_pts, 0. ),

        u_x_tor_coarse_error( num_pts, 0. ),
        u_y_tor_coarse_error( num_pts, 0. ),
        u_z_tor_coarse_error( num_pts, 0. ),

        u_x_pot_coarse_error( num_pts, 0. ),
        u_y_pot_coarse_error( num_pts, 0. ),
        u_z_pot_coarse_error( num_pts, 0. ),

        u_x_tot_coarse_error( num_pts, 0. ),
        u_y_tot_coarse_error( num_pts, 0. ),
        u_z_tot_coarse_error( num_pts, 0. ),

        //
        //// Diadic (Cartesian) velocity components
        //

        // tor
        ux_ux_tor_error( num_pts, 0. ),
        ux_uy_tor_error( num_pts, 0. ),
        ux_uz_tor_error( num_pts, 0. ),
        uy_uy_tor_error( num_pts, 0. ),
        uy_uz_tor_error( num_pts, 0. ),
        uz_uz_tor_error( num_pts, 0. ),

        vort_ux_tor_error( num_pts, 0. ),
        vort_uy_tor_error( num_pts, 0. ),
        vort_uz_tor_error( num_pts, 0. ),

        // pot
        ux_ux_pot_error( num_pts, 0. ),
        ux_uy_pot_error( num_pts, 0. ),
        ux_uz_pot_error( num_pts, 0. ),
        uy_uy_pot_error( num_pts, 0. ),
        uy_uz_pot_error( num_pts, 0. ),
        uz_uz_pot_error( num_pts, 0. ),

        vort_ux_pot_error( num_pts, 0. ),
        vort_uy_pot_error( num_pts, 0. ),
        vort_uz_pot_error( num_pts, 0. ),

        // tot
        ux_ux_tot_error( num_pts, 0. ),
        ux_uy_tot_error( num_pts, 0. ),
        ux_uz_tot_error( num_pts, 0. ),
        uy_uy_tot_error( num_pts, 0. ),
        uy_uz_tot_error( num_pts, 0. ),
        uz_uz_tot_error( num_pts, 0. ),

        vort_ux_tot_error( num_pts, 0. ),
        vort_uy_tot_error( num_pts, 0. ),
        vort_uz_tot_error( num_pts, 0. ),

        //
        //// Spherical velocity components
        //
        zero_array_error( num_pts, 0.),

        // Spherical - radial velocity
        //  by definition this is potential-only since toroidal is incompressible
        //u_r( num_pts, 0.),

        // Spherical - zonal velocities
        u_lon_tor_error( num_pts, 0. ),
        u_lon_pot_error( num_pts, 0. ),
        u_lon_tot_error( num_pts, 0. ),

        // Spherical - meridional velocities
        u_lat_tor_error( num_pts, 0. ),
        u_lat_pot_error( num_pts, 0. ),
        u_lat_tot_error( num_pts, 0. ),

        // Spherical - dyadix products
        ulon_ulon_error( num_pts, 0. ),
        ulon_ulat_error( num_pts, 0. ),
        ulat_ulat_error( num_pts, 0. ),

        // Vorticity (only r component)
        vort_tor_r_error( num_pts, 0. ),
        vort_pot_r_error( num_pts, 0. ),
        vort_tot_r_error( num_pts, 0. ),

        // Full vorticity components
        full_vort_tor_r_error( num_pts, 0. ),
        full_vort_pot_r_error( num_pts, 0. ),
        full_vort_tot_r_error( num_pts, 0. ),

        // Okubo-Weiss values
        OkuboWeiss_tor_error( num_pts, 0. ),
        OkuboWeiss_pot_error( num_pts, 0. ),
        OkuboWeiss_tot_error( num_pts, 0. ),

        // Pi
        Pi_tor_error(  num_pts, 0. ),
        Pi_pot_error(  num_pts, 0. ),
        Pi_tot_error(  num_pts, 0. ),
        Pi_DVV_error(  num_pts, 0. ),
        Pi_DTT_error(  num_pts, 0. ),
        Pi_VDD_error(  num_pts, 0. ),
        Pi_VTT_error(  num_pts, 0. ),
        Pi_Helm_error( num_pts, 0. ),

        // Z ( enstrophy cascade )
        Z_tor_error( num_pts, 0. ),
        Z_pot_error( num_pts, 0. ),
        Z_tot_error( num_pts, 0. );


    // Wind stress analysis not supported for error analysis
    std::vector<double> tau_wind_x_tor( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),
                        tau_wind_y_tor( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),
                        tau_wind_x_pot( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),
                        tau_wind_y_pot( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),
                        tau_wind_dot_u_tor( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),
                        tau_wind_dot_u_pot( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),
                        local_wind_forcing_tor( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),
                        local_wind_forcing_pot( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),
                        local_wind_forcing_tot( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),

                        coarse_tau_wind_dot_u_tor( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),
                        coarse_tau_wind_dot_u_pot( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),
                        coarse_tau_wind_dot_u_tot( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),

                        coarse_wind_tau_Psi( constants::COMP_WIND_FORCE ? num_pts : 1, 0. ),
                        coarse_wind_tau_Phi( constants::COMP_WIND_FORCE ? num_pts : 1, 0. );

    //
    //// Compute error F fields (dataset 1 - dataset 2)
    //
    #if DEBUG >= 2
    if (wRank == 0) { fprintf(stdout, "\nComputing error F fields\n"); }
    #endif
    #pragma omp parallel for collapse(1) schedule(guided)
    for (int idx = 0; idx < num_pts; ++idx) {
        F_potential_error.at(idx) = F_potential.at(idx) - F_potential_2.at(idx);
        F_toroidal_error.at(idx)  = F_toroidal.at(idx)  - F_toroidal_2.at(idx);
        u_r_error.at(idx)         = u_r.at(idx)         - u_r_2.at(idx);
    }

    //
    //// Compute original (unfiltered) KE
    //
     
    #if DEBUG >= 2
    if (wRank == 0) { fprintf(stdout, "\nExtracting velocities from Phi and Psi\n"); }
    #endif
    // Get pot and tor velocities
    if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
    
    // Dataset 1 velocities
    toroidal_vel_from_F(  u_lon_tor, u_lat_tor, F_toroidal,  longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask );
    potential_vel_from_F( u_lon_pot, u_lat_pot, F_potential, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask );
    
    // Dataset 2 velocities
    toroidal_vel_from_F(  u_lon_tor_2, u_lat_tor_2, F_toroidal_2,  longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask );
    potential_vel_from_F( u_lon_pot_2, u_lat_pot_2, F_potential_2, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask );
    
    // Error velocities (leveraging linearity of derivatives: u_error = u(F1) - u(F2))
    #pragma omp parallel for collapse(1) schedule(guided)
    for (int idx = 0; idx < u_lon_tor.size(); ++idx) {
        u_lon_tor_error.at(idx) = u_lon_tor.at(idx) - u_lon_tor_2.at(idx);
        u_lat_tor_error.at(idx) = u_lat_tor.at(idx) - u_lat_tor_2.at(idx);
        u_lon_pot_error.at(idx) = u_lon_pot.at(idx) - u_lon_pot_2.at(idx);
        u_lat_pot_error.at(idx) = u_lat_pot.at(idx) - u_lat_pot_2.at(idx);
    }
    
    if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute velocities from F"); }

    #if DEBUG >= 2
    if (wRank == 0) { fprintf(stdout, "\nComputing KE of unfiltered velocities\n"); }
    #endif
    if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
    
    // Dataset 1 KE
    #pragma omp parallel \
    default( none ) \
    shared( KE_tor_orig, KE_pot_orig, KE_tot_orig, mask, \
            u_r, u_lon_tor, u_lat_tor, u_lon_pot, u_lat_pot, u_lon_tot, u_lat_tot) \
    private( index )
    {
        #pragma omp for collapse(1) schedule(guided)
        for (index = 0; index < u_lon_tor.size(); ++index) {
            u_lon_tot.at(index) = u_lon_tor.at(index) + u_lon_pot.at(index);
            u_lat_tot.at(index) = u_lat_tor.at(index) + u_lat_pot.at(index);
            if ( mask.at(index) ) { 
                KE_tor_orig.at(index) = 0.5 * constants::rho0 * (   pow(u_lon_tor.at(index), 2.) 
                                                                  + pow(u_lat_tor.at(index), 2.) );
                KE_pot_orig.at(index) = 0.5 * constants::rho0 * (   pow(u_lon_pot.at(index), 2.) 
                                                                  + pow(u_lat_pot.at(index), 2.)
                                                                  + pow(u_r.at(      index), 2.) );
                KE_tot_orig.at(index) = 0.5 * constants::rho0 * (   pow(u_lon_tot.at(index), 2.) 
                                                                  + pow(u_lat_tot.at(index), 2.)
                                                                  + pow(u_r.at(      index), 2.) );
            } else {
                KE_tor_orig.at(index) = 0.;
                KE_pot_orig.at(index) = 0.;
                KE_tot_orig.at(index) = 0.;
            }
        }
    }

    // Dataset 2 KE
    #pragma omp parallel \
    default( none ) \
    shared( KE_tor_orig_2, KE_pot_orig_2, KE_tot_orig_2, mask, \
            u_r_2, u_lon_tor_2, u_lat_tor_2, u_lon_pot_2, u_lat_pot_2, u_lon_tot_2, u_lat_tot_2) \
    private( index )
    {
        #pragma omp for collapse(1) schedule(guided)
        for (index = 0; index < u_lon_tor_2.size(); ++index) {
            u_lon_tot_2.at(index) = u_lon_tor_2.at(index) + u_lon_pot_2.at(index);
            u_lat_tot_2.at(index) = u_lat_tor_2.at(index) + u_lat_pot_2.at(index);
            if ( mask.at(index) ) { 
                KE_tor_orig_2.at(index) = 0.5 * constants::rho0 * (   pow(u_lon_tor_2.at(index), 2.) 
                                                                  + pow(u_lat_tor_2.at(index), 2.) );
                KE_pot_orig_2.at(index) = 0.5 * constants::rho0 * (   pow(u_lon_pot_2.at(index), 2.) 
                                                                  + pow(u_lat_pot_2.at(index), 2.)
                                                                  + pow(u_r_2.at(   index), 2.) );
                KE_tot_orig_2.at(index) = 0.5 * constants::rho0 * (   pow(u_lon_tot_2.at(index), 2.) 
                                                                  + pow(u_lat_tot_2.at(index), 2.)
                                                                  + pow(u_r_2.at(   index), 2.) );
            } else {
                KE_tor_orig_2.at(index) = 0.;
                KE_pot_orig_2.at(index) = 0.;
                KE_tot_orig_2.at(index) = 0.;
            }
        }
    }

    // Error KE (differences)
    #pragma omp parallel \
    default( none ) \
    shared( KE_tor_orig_error, KE_pot_orig_error, KE_tot_orig_error, mask, \
            u_r_error, u_lon_tor_error, u_lat_tor_error, u_lon_pot_error, u_lat_pot_error, u_lon_tot_error, u_lat_tot_error) \
    private( index )
    {
        #pragma omp for collapse(1) schedule(guided)
        for (index = 0; index < u_lon_tor_error.size(); ++index) {
            u_lon_tot_error.at(index) = u_lon_tor_error.at(index) + u_lon_pot_error.at(index);
            u_lat_tot_error.at(index) = u_lat_tor_error.at(index) + u_lat_pot_error.at(index);
            if ( mask.at(index) ) { 
                KE_tor_orig_error.at(index) = 0.5 * constants::rho0 * (   pow(u_lon_tor_error.at(index), 2.) 
                                                                  + pow(u_lat_tor_error.at(index), 2.) );
                KE_pot_orig_error.at(index) = 0.5 * constants::rho0 * (   pow(u_lon_pot_error.at(index), 2.) 
                                                                  + pow(u_lat_pot_error.at(index), 2.)
                                                                  + pow(u_r_error.at( index), 2.) );
                KE_tot_orig_error.at(index) = 0.5 * constants::rho0 * (   pow(u_lon_tot_error.at(index), 2.) 
                                                                  + pow(u_lat_tot_error.at(index), 2.)
                                                                  + pow(u_r_error.at( index), 2.) );
            } else {
                KE_tor_orig_error.at(index) = 0.;
                KE_pot_orig_error.at(index) = 0.;
                KE_tot_orig_error.at(index) = 0.;
            }
        }
    }
    if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute KE and Enstrophy"); }

    // Also get some wind-based terms, if applicable (Error not supported)
    if ( constants::COMP_WIND_FORCE ){ 
        toroidal_vel_from_F(  tau_wind_x_tor, tau_wind_y_tor, wind_tau_Psi, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask );
        potential_vel_from_F( tau_wind_x_pot, tau_wind_y_pot, wind_tau_Phi, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask );

        #pragma omp parallel \
        default( none ) \
        shared( tau_wind_dot_u_tor, tau_wind_dot_u_pot, mask, \
                tau_wind_x_tor, tau_wind_x_pot, tau_wind_y_tor, tau_wind_y_pot, \
                u_lon_tor, u_lat_tor, u_lon_pot, u_lat_pot ) \
        private( index )
        {
            #pragma omp for collapse(1) schedule(guided)
            for (index = 0; index < u_lon_tor.size(); ++index) {

                if ( mask.at(index) ) { 
                    tau_wind_dot_u_tor.at( index ) =    u_lon_tor.at(index) * ( tau_wind_x_tor.at( index ) + tau_wind_x_pot.at( index ) )
                                                      + u_lat_tor.at(index) * ( tau_wind_y_tor.at( index ) + tau_wind_y_pot.at( index ) );
                    tau_wind_dot_u_pot.at( index ) =    u_lon_pot.at(index) * ( tau_wind_x_tor.at( index ) + tau_wind_x_pot.at( index ) )
                                                      + u_lat_pot.at(index) * ( tau_wind_y_tor.at( index ) + tau_wind_y_pot.at( index ) );
                } else {
                    tau_wind_dot_u_tor.at( index ) = 0.;
                    tau_wind_dot_u_pot.at( index ) = 0.;
                }
            }
        }
    }

    // Get vorticities
    if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
    
    // Dataset 1 vorticities
    compute_vorticity( full_vort_tor_r, null_vector, null_vector, null_vector, null_vector,
                null_vector, null_vector, null_vector, null_vector,
                source_data, zero_array, u_lon_tor, u_lat_tor);
    compute_vorticity( full_vort_pot_r, null_vector, null_vector, null_vector, null_vector,
                null_vector, null_vector, null_vector, null_vector,
                source_data, u_r, u_lon_pot, u_lat_pot);
    compute_vorticity( full_vort_tot_r, null_vector, null_vector, null_vector, null_vector,
                null_vector, null_vector, null_vector, null_vector,
                source_data, u_r, u_lon_tot, u_lat_tot);
    
    // Dataset 2 vorticities
    compute_vorticity( full_vort_tor_r_2, null_vector, null_vector, null_vector, null_vector,
                null_vector, null_vector, null_vector, null_vector,
                source_data, zero_array, u_lon_tor_2, u_lat_tor_2);
    compute_vorticity( full_vort_pot_r_2, null_vector, null_vector, null_vector, null_vector,
                null_vector, null_vector, null_vector, null_vector,
                source_data, u_r_2, u_lon_pot_2, u_lat_pot_2);
    compute_vorticity( full_vort_tot_r_2, null_vector, null_vector, null_vector, null_vector,
                null_vector, null_vector, null_vector, null_vector,
                source_data, u_r_2, u_lon_tot_2, u_lat_tot_2);
    
    // Error vorticities (dataset 1 - dataset 2, leveraging linearity of derivatives)
    #pragma omp parallel for collapse(1) schedule(guided)
    for (int idx = 0; idx < full_vort_tor_r.size(); ++idx) {
        full_vort_tor_r_error.at(idx) = full_vort_tor_r.at(idx) - full_vort_tor_r_2.at(idx);
        full_vort_pot_r_error.at(idx) = full_vort_pot_r.at(idx) - full_vort_pot_r_2.at(idx);
        full_vort_tot_r_error.at(idx) = full_vort_tot_r.at(idx) - full_vort_tot_r_2.at(idx);
    }
    
    if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute vorticity"); }


    #if DEBUG >= 2
    if (wRank == 0) { fprintf(stdout, "\nGetting Cartesian velocity components\n"); }
    #endif
    // Get Cartesian velocities, will need them for Pi
    if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
    
    // Dataset 1 Cartesian velocities
    vel_Spher_to_Cart( u_x_tor, u_y_tor, u_z_tor, zero_array, u_lon_tor, u_lat_tor, source_data );
    vel_Spher_to_Cart( u_x_pot, u_y_pot, u_z_pot, u_r,        u_lon_pot, u_lat_pot, source_data );
    vel_Spher_to_Cart( u_x_tot, u_y_tot, u_z_tot, u_r,        u_lon_tot, u_lat_tot, source_data );
    
    // Dataset 2 Cartesian velocities
    vel_Spher_to_Cart( u_x_tor_2, u_y_tor_2, u_z_tor_2, zero_array, u_lon_tor_2, u_lat_tor_2, source_data );
    vel_Spher_to_Cart( u_x_pot_2, u_y_pot_2, u_z_pot_2, u_r_2,        u_lon_pot_2, u_lat_pot_2, source_data );
    vel_Spher_to_Cart( u_x_tot_2, u_y_tot_2, u_z_tot_2, u_r_2,        u_lon_tot_2, u_lat_tot_2, source_data );
    
    // Error Cartesian velocities (leveraging linearity of coordinate transformations)
    #pragma omp parallel for collapse(1) schedule(guided)
    for (int idx = 0; idx < u_x_tor.size(); ++idx) {
        u_x_tor_error.at(idx) = u_x_tor.at(idx) - u_x_tor_2.at(idx);
        u_y_tor_error.at(idx) = u_y_tor.at(idx) - u_y_tor_2.at(idx);
        u_z_tor_error.at(idx) = u_z_tor.at(idx) - u_z_tor_2.at(idx);
        
        u_x_pot_error.at(idx) = u_x_pot.at(idx) - u_x_pot_2.at(idx);
        u_y_pot_error.at(idx) = u_y_pot.at(idx) - u_y_pot_2.at(idx);
        u_z_pot_error.at(idx) = u_z_pot.at(idx) - u_z_pot_2.at(idx);
        
        u_x_tot_error.at(idx) = u_x_tot.at(idx) - u_x_tot_2.at(idx);
        u_y_tot_error.at(idx) = u_y_tot.at(idx) - u_y_tot_2.at(idx);
        u_z_tot_error.at(idx) = u_z_tot.at(idx) - u_z_tot_2.at(idx);
    }
    
    if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "Sphere to Cart Conversion"); }

    #if DEBUG >= 2
    if (wRank == 0) { fprintf(stdout, "\nFlagging variables for output\n"); }
    #endif
    if (not(constants::NO_FULL_OUTPUTS)) {
        //
        // These variables are output unless full outputs are turned off
        // 

        vars_to_write.push_back("coarse_F_tor");
        vars_to_write.push_back("coarse_F_pot");

        if ( source_data.compute_radial_vel ) {
            vars_to_write.push_back("u_r");
        }

        if ( constants::COMP_PI_HELMHOLTZ ) {
            vars_to_write.push_back("coarse_uiuj_F_r");
            vars_to_write.push_back("coarse_uiuj_F_Phi");
            vars_to_write.push_back("coarse_uiuj_F_Psi");
        }

        vars_to_write.push_back("u_lon_tor");
        vars_to_write.push_back("u_lat_tor");

        vars_to_write.push_back("u_lon_pot");
        vars_to_write.push_back("u_lat_pot");

        vars_to_write.push_back("KE_tor_fine");
        vars_to_write.push_back("KE_pot_fine");
        vars_to_write.push_back("KE_tot_fine");

        // KE spectra and spectral slopes
        vars_to_write.push_back("u_spectrum_tot");
        vars_to_write.push_back("v_spectrum_tot");

        vars_to_write.push_back("u_spectrum_tor");
        vars_to_write.push_back("v_spectrum_tor");

        vars_to_write.push_back("u_spectrum_pot");
        vars_to_write.push_back("v_spectrum_pot");

        vars_to_write.push_back("KE_spectral_slope_tot");
        vars_to_write.push_back("KE_spectral_slope_tor");
        vars_to_write.push_back("KE_spectral_slope_pot");

        // KE scale-transfer
        vars_to_write.push_back("Pi_tor");
        vars_to_write.push_back("Pi_pot");
        vars_to_write.push_back("Pi_tot");
        if ( constants::COMP_PI_HELMHOLTZ ) {
            vars_to_write.push_back("Pi_Helm");
        }

        vars_to_write.push_back("Z_tor");
        vars_to_write.push_back("Z_pot");
        vars_to_write.push_back("Z_tot");

        if ( constants::COMP_WIND_FORCE ){
            vars_to_write.push_back("local_wind_forcing_tor");
            vars_to_write.push_back("local_wind_forcing_pot");

            //vars_to_write.push_back("tau_wind_x_tor");
            //vars_to_write.push_back("tau_wind_y_tor");

            //vars_to_write.push_back("tau_wind_x_pot");
            //vars_to_write.push_back("tau_wind_y_pot");

            vars_to_write.push_back("tau_wind_dot_u_tor");
            vars_to_write.push_back("tau_wind_dot_u_pot");
        }
        
        // Second dataset outputs

        vars_to_write.push_back("coarse_F_tor_2");
        vars_to_write.push_back("coarse_F_pot_2");

        if ( source_data.compute_radial_vel ) {
            vars_to_write.push_back("u_r_2");
        }

        if ( constants::COMP_PI_HELMHOLTZ ) {
            vars_to_write.push_back("coarse_uiuj_F_r_2");
            vars_to_write.push_back("coarse_uiuj_F_Phi_2");
            vars_to_write.push_back("coarse_uiuj_F_Psi_2");
        }

        vars_to_write.push_back("u_lon_tor_2");
        vars_to_write.push_back("u_lat_tor_2");

        vars_to_write.push_back("u_lon_pot_2");
        vars_to_write.push_back("u_lat_pot_2");

        vars_to_write.push_back("KE_tor_fine_2");
        vars_to_write.push_back("KE_pot_fine_2");
        vars_to_write.push_back("KE_tot_fine_2");

        // KE spectra and spectral slopes
        vars_to_write.push_back("u_spectrum_tot_2");
        vars_to_write.push_back("v_spectrum_tot_2");

        vars_to_write.push_back("u_spectrum_tor_2");
        vars_to_write.push_back("v_spectrum_tor_2");

        vars_to_write.push_back("u_spectrum_pot_2");
        vars_to_write.push_back("v_spectrum_pot_2");

        vars_to_write.push_back("KE_spectral_slope_tot_2");
        vars_to_write.push_back("KE_spectral_slope_tor_2");
        vars_to_write.push_back("KE_spectral_slope_pot_2");

        // KE scale-transfer
        vars_to_write.push_back("Pi_tor_2");
        vars_to_write.push_back("Pi_pot_2");
        vars_to_write.push_back("Pi_tot_2");
        if ( constants::COMP_PI_HELMHOLTZ ) {
            vars_to_write.push_back("Pi_Helm_2");
        }

        vars_to_write.push_back("Z_tor_2");
        vars_to_write.push_back("Z_pot_2");
        vars_to_write.push_back("Z_tot_2");

        // Error outputs

        vars_to_write.push_back("coarse_F_tor_error");
        vars_to_write.push_back("coarse_F_pot_error");

        if ( source_data.compute_radial_vel ) {
            vars_to_write.push_back("u_r_error");
        }

        vars_to_write.push_back("u_lon_tor_error");
        vars_to_write.push_back("u_lat_tor_error");

        vars_to_write.push_back("u_lon_pot_error");
        vars_to_write.push_back("u_lat_pot_error");

        vars_to_write.push_back("KE_tor_fine_error");
        vars_to_write.push_back("KE_pot_fine_error");
        vars_to_write.push_back("KE_tot_fine_error");

        // KE spectra and spectral slopes
        vars_to_write.push_back("u_spectrum_tot_error");
        vars_to_write.push_back("v_spectrum_tot_error");

        vars_to_write.push_back("u_spectrum_tor_error");
        vars_to_write.push_back("v_spectrum_tor_error");

        vars_to_write.push_back("u_spectrum_pot_error");
        vars_to_write.push_back("v_spectrum_pot_error");

        vars_to_write.push_back("KE_spectral_slope_tot_error");
        vars_to_write.push_back("KE_spectral_slope_tor_error");
        vars_to_write.push_back("KE_spectral_slope_pot_error");

        // KE scale-transfer
        vars_to_write.push_back("Pi_tor_error");
        vars_to_write.push_back("Pi_pot_error");
        vars_to_write.push_back("Pi_tot_error");
        if ( constants::COMP_PI_HELMHOLTZ ) {
            vars_to_write.push_back("Pi_Helm_error");
        }

        vars_to_write.push_back("Z_tor_error");
        vars_to_write.push_back("Z_pot_error");
        vars_to_write.push_back("Z_tot_error");
    }

    if (not(constants::MINIMAL_OUTPUT)) {
        //
        // These outputs are only included if not set to minimal outputs
        //

        vars_to_write.push_back("KE_tor_fine_mod");
        vars_to_write.push_back("KE_pot_fine_mod");
        vars_to_write.push_back("KE_tot_fine_mod");

        vars_to_write.push_back("div_tor");
        vars_to_write.push_back("div_pot");
        vars_to_write.push_back("div_tot");

        if (constants::DO_OKUBOWEISS_ANALYSIS) {
            vars_to_write.push_back("OkuboWeiss_tor");
            vars_to_write.push_back("OkuboWeiss_pot");
            vars_to_write.push_back("OkuboWeiss_tot");
        }

        vars_to_write.push_back("KE_tor_filt");
        vars_to_write.push_back("KE_pot_filt");
        vars_to_write.push_back("KE_tot_filt");

        vars_to_write.push_back("Enstrophy_tor");
        vars_to_write.push_back("Enstrophy_pot");
        vars_to_write.push_back("Enstrophy_tot");

        vars_to_write.push_back("vort_r_tor");
        vars_to_write.push_back("vort_r_pot");
        vars_to_write.push_back("vort_r_tot");

        if ( constants::COMP_PI_HELMHOLTZ ) {
            vars_to_write.push_back("coarse_uu");
            vars_to_write.push_back("coarse_uv");
            vars_to_write.push_back("coarse_vv");
        }
    }

    // Compute the kernal alpha value (for baroclinic transfers)
    const double kern_alpha = kernel_alpha();

    // Now prepare to filter
    double scale;
    int Itime, Idepth, Ilat, Ilon, Ilatlon, thread_id, num_threads, prev_Ilat = -1;
    const bool can_roll_in_longitude = ( (constants::PERIODIC_X) and (constants::UNIFORM_LON_GRID) and (constants::FULL_LON_SPAN) );

    int perc_base = 5;
    int perc, perc_count=0;

    //
    //// Set up filtering vectors
    //
    std::vector<double*> filtered_vals, dl_filter_vals, dll_filter_vals, 
        dl_kernel_vals, dll_kernel_vals;
    std::vector<double*> filtered_vals_2, dl_filter_vals_2, dll_filter_vals_2, 
        dl_kernel_vals_2, dll_kernel_vals_2;
    std::vector<double*> filtered_vals_error, dl_filter_vals_error, dll_filter_vals_error, 
        dl_kernel_vals_error, dll_kernel_vals_error;
    std::vector<bool> filt_use_mask;
    std::vector<const std::vector<double>*> filter_fields;
    std::vector<const std::vector<double>*> filter_fields_2;
    std::vector<const std::vector<double>*> filter_fields_error;

    double F_pot_tmp;
    filter_fields.push_back(&F_potential);
    filt_use_mask.push_back(false);

    double F_tor_tmp;
    filter_fields.push_back(&F_toroidal);
    filt_use_mask.push_back(false);

    double u_r_tmp;
    if ( source_data.compute_radial_vel ) {
        filter_fields.push_back(&u_r);
        filt_use_mask.push_back(false);
    }

    // For spectra and spectral slopes
    double dl_Psi_tmp, dll_Psi_tmp, dl_Phi_tmp, dll_Phi_tmp, dl_ur_tmp, dll_ur_tmp,
           dl_kernel_val, dll_kernel_val;


    double uxux_tmp, uxuy_tmp, uxuz_tmp, uyuy_tmp, uyuz_tmp, uzuz_tmp, vort_ux_tmp, vort_uy_tmp, vort_uz_tmp;


    // Field 2

    double F_pot_tmp_2;
    filter_fields_2.push_back(&F_potential_2);

    double F_tor_tmp_2;
    filter_fields_2.push_back(&F_toroidal_2);

    double u_r_tmp_2;
    if ( source_data.compute_radial_vel ) {
        filter_fields_2.push_back(&u_r_2);
    }

    // For spectra and spectral slopes
    double dl_Psi_tmp_2, dll_Psi_tmp_2, dl_Phi_tmp_2, dll_Phi_tmp_2, dl_ur_tmp_2, dll_ur_tmp_2,
           dl_kernel_val_2, dll_kernel_val_2;


    double uxux_tmp_2, uxuy_tmp_2, uxuz_tmp_2, uyuy_tmp_2, uyuz_tmp_2, uzuz_tmp_2, vort_ux_tmp_2, vort_uy_tmp_2, vort_uz_tmp_2;

    // Error

    double F_pot_tmp_error;
    filter_fields_error.push_back(&F_potential_error);

    double F_tor_tmp_error;
    filter_fields_error.push_back(&F_toroidal_error);

    double u_r_tmp_error;
    if ( source_data.compute_radial_vel ) {
        filter_fields_error.push_back(&u_r_error);
    }

    // For spectra and spectral slopes
    double dl_Psi_tmp_error, dll_Psi_tmp_error, dl_Phi_tmp_error, dll_Phi_tmp_error, dl_ur_tmp_error, dll_ur_tmp_error,
           dl_kernel_val_error, dll_kernel_val_error;


    double uxux_tmp_error, uxuy_tmp_error, uxuz_tmp_error, uyuy_tmp_error, uyuz_tmp_error, uzuz_tmp_error, vort_ux_tmp_error, vort_uy_tmp_error, vort_uz_tmp_error;

    //
    //// Set up post-processing variables
    //
    #if DEBUG >= 2
    if (wRank == 0) { fprintf(stdout, "\nFlagging variables for post-processing\n"); }
    #endif
    std::vector<const std::vector<double>*> postprocess_fields_tor, postprocess_fields_pot, postprocess_fields_tot;
    std::vector<std::string> postprocess_names;

    postprocess_names.push_back( "F" );
    postprocess_fields_tor.push_back( &coarse_F_tor );
    postprocess_fields_pot.push_back( &coarse_F_pot );
    postprocess_fields_tot.push_back( &u_r_coarse   );

    postprocess_names.push_back( "coarse_KE" );
    postprocess_fields_tor.push_back( &KE_tor_coarse );
    postprocess_fields_pot.push_back( &KE_pot_coarse );
    postprocess_fields_tot.push_back( &KE_tot_coarse );

    postprocess_names.push_back( "fine_KE" );
    postprocess_fields_tor.push_back( &KE_tor_fine );
    postprocess_fields_pot.push_back( &KE_pot_fine );
    postprocess_fields_tot.push_back( &KE_tot_fine );

    /*
    postprocess_names.push_back( "Fine_KE_mod" );
    postprocess_fields_tor.push_back( &KE_tor_fine_mod );
    postprocess_fields_pot.push_back( &KE_pot_fine_mod );
    postprocess_fields_tot.push_back( &KE_tot_fine_mod );
    */

    // KE spectra
    postprocess_names.push_back( "u_lon_spectrum" );
    postprocess_fields_tor.push_back( &u_spectrum_tor );
    postprocess_fields_pot.push_back( &u_spectrum_pot );
    postprocess_fields_tot.push_back( &u_spectrum_tot );

    postprocess_names.push_back( "u_lat_spectrum" );
    postprocess_fields_tor.push_back( &v_spectrum_tor );
    postprocess_fields_pot.push_back( &v_spectrum_pot );
    postprocess_fields_tot.push_back( &v_spectrum_tot );

    // KE spectral slopes
    postprocess_names.push_back( "KE_spectral_slope" );
    postprocess_fields_tor.push_back( &spec_slope_tor );
    postprocess_fields_pot.push_back( &spec_slope_pot );
    postprocess_fields_tot.push_back( &spec_slope_tot );

    /*
    postprocess_names.push_back( "div_J_transport" );
    postprocess_fields_tor.push_back( &div_J_tor );
    postprocess_fields_pot.push_back( &div_J_pot );
    postprocess_fields_tot.push_back( &div_J_tot );
    */

    postprocess_names.push_back( "enstrophy" );
    postprocess_fields_tor.push_back( &Enst_tor );
    postprocess_fields_pot.push_back( &Enst_pot );
    postprocess_fields_tot.push_back( &Enst_tot );

    postprocess_names.push_back( "u_lon" );
    postprocess_fields_tor.push_back( &u_lon_tor );
    postprocess_fields_pot.push_back( &u_lon_pot );
    postprocess_fields_tot.push_back( &u_lon_tot );

    postprocess_names.push_back( "u_lat" );
    postprocess_fields_tor.push_back( &u_lat_tor );
    postprocess_fields_pot.push_back( &u_lat_pot );
    postprocess_fields_tot.push_back( &u_lat_tot );

    if (constants::DO_OKUBOWEISS_ANALYSIS) {
        postprocess_names.push_back( "OkuboWeiss" );
        postprocess_fields_tor.push_back( &OkuboWeiss_tor );
        postprocess_fields_pot.push_back( &OkuboWeiss_pot );
        postprocess_fields_tot.push_back( &OkuboWeiss_tot );
    }

    postprocess_names.push_back( "Pi" );
    postprocess_fields_tor.push_back( &Pi_tor );
    postprocess_fields_pot.push_back( &Pi_pot );
    postprocess_fields_tot.push_back( &Pi_tot );

    postprocess_names.push_back( "Pi_Dversus" );
    postprocess_fields_tor.push_back( &Pi_DVV );
    postprocess_fields_pot.push_back( &Pi_pot );
    postprocess_fields_tot.push_back( &Pi_DTT );

    postprocess_names.push_back( "Pi_Vversus" );
    postprocess_fields_tor.push_back( &Pi_tor );
    postprocess_fields_pot.push_back( &Pi_VDD );
    postprocess_fields_tot.push_back( &Pi_VTT );

    if ( constants::COMP_PI_HELMHOLTZ ) {
        postprocess_names.push_back( "Pi_Helm" );
        postprocess_fields_tor.push_back( &Pi_Helm );
        postprocess_fields_pot.push_back( &Pi_Helm );
        postprocess_fields_tot.push_back( &Pi_Helm );
    }

    postprocess_names.push_back( "Z" );
    postprocess_fields_tor.push_back( &Z_tor );
    postprocess_fields_pot.push_back( &Z_pot );
    postprocess_fields_tot.push_back( &Z_tot );

    postprocess_names.push_back( "velocity_divergence" );
    postprocess_fields_tor.push_back( &div_tor );
    postprocess_fields_pot.push_back( &div_pot );
    postprocess_fields_tot.push_back( &div_tot );

    if ( constants::COMP_WIND_FORCE ) {
        postprocess_names.push_back( "wind_forcing_to_small_scales" );
        postprocess_fields_tor.push_back( &local_wind_forcing_tor );
        postprocess_fields_pot.push_back( &local_wind_forcing_pot );
        postprocess_fields_tot.push_back( &local_wind_forcing_tot );

        postprocess_names.push_back( "coarse_wind_forcing" );
        postprocess_fields_tor.push_back( &coarse_tau_wind_dot_u_tor );
        postprocess_fields_pot.push_back( &coarse_tau_wind_dot_u_pot );
        postprocess_fields_tot.push_back( &coarse_tau_wind_dot_u_tot );
    }

    // Field 2

    postprocess_names.push_back( "F_2" );
    postprocess_fields_tor.push_back( &coarse_F_tor_2 );
    postprocess_fields_pot.push_back( &coarse_F_pot_2 );
    postprocess_fields_tot.push_back( &u_r_coarse_2   );

    postprocess_names.push_back( "coarse_KE_2" );
    postprocess_fields_tor.push_back( &KE_tor_coarse_2);
    postprocess_fields_pot.push_back( &KE_pot_coarse_2 );
    postprocess_fields_tot.push_back( &KE_tot_coarse_2 );

    postprocess_names.push_back( "fine_KE_2" );
    postprocess_fields_tor.push_back( &KE_tor_fine_2 );
    postprocess_fields_pot.push_back( &KE_pot_fine_2 );
    postprocess_fields_tot.push_back( &KE_tot_fine_2 );

    /*
    postprocess_names.push_back( "Fine_KE_mod" );
    postprocess_fields_tor.push_back( &KE_tor_fine_mod );
    postprocess_fields_pot.push_back( &KE_pot_fine_mod );
    postprocess_fields_tot.push_back( &KE_tot_fine_mod );
    */

    // KE spectra
    postprocess_names.push_back( "u_lon_spectrum_2" );
    postprocess_fields_tor.push_back( &u_spectrum_tor_2 );
    postprocess_fields_pot.push_back( &u_spectrum_pot_2 );
    postprocess_fields_tot.push_back( &u_spectrum_tot_2 );

    postprocess_names.push_back( "u_lat_spectrum_2" );
    postprocess_fields_tor.push_back( &v_spectrum_tor_2 );
    postprocess_fields_pot.push_back( &v_spectrum_pot_2 );
    postprocess_fields_tot.push_back( &v_spectrum_tot_2 );

    // KE spectral slopes
    postprocess_names.push_back( "KE_spectral_slope_2" );
    postprocess_fields_tor.push_back( &spec_slope_tor_2 );
    postprocess_fields_pot.push_back( &spec_slope_pot_2 );
    postprocess_fields_tot.push_back( &spec_slope_tot_2 );

    /*
    postprocess_names.push_back( "div_J_transport" );
    postprocess_fields_tor.push_back( &div_J_tor );
    postprocess_fields_pot.push_back( &div_J_pot );
    postprocess_fields_tot.push_back( &div_J_tot );
    */

    postprocess_names.push_back( "enstrophy_2" );
    postprocess_fields_tor.push_back( &Enst_tor_2 );
    postprocess_fields_pot.push_back( &Enst_pot_2 );
    postprocess_fields_tot.push_back( &Enst_tot_2 );

    postprocess_names.push_back( "u_lon_2" );
    postprocess_fields_tor.push_back( &u_lon_tor_2 );
    postprocess_fields_pot.push_back( &u_lon_pot_2 );
    postprocess_fields_tot.push_back( &u_lon_tot_2 );

    postprocess_names.push_back( "u_lat_2" );
    postprocess_fields_tor.push_back( &u_lat_tor_2 );
    postprocess_fields_pot.push_back( &u_lat_pot_2 );
    postprocess_fields_tot.push_back( &u_lat_tot_2 );

    if (constants::DO_OKUBOWEISS_ANALYSIS) {
        postprocess_names.push_back( "OkuboWeiss_2" );
        postprocess_fields_tor.push_back( &OkuboWeiss_tor_2 );
        postprocess_fields_pot.push_back( &OkuboWeiss_pot_2 );
        postprocess_fields_tot.push_back( &OkuboWeiss_tot_2 );
    }

    postprocess_names.push_back( "Pi_2" );
    postprocess_fields_tor.push_back( &Pi_tor_2 );
    postprocess_fields_pot.push_back( &Pi_pot_2 );
    postprocess_fields_tot.push_back( &Pi_tot_2 );

    postprocess_names.push_back( "Pi_Dversus_2" );
    postprocess_fields_tor.push_back( &Pi_DVV_2 );
    postprocess_fields_pot.push_back( &Pi_pot_2 );
    postprocess_fields_tot.push_back( &Pi_DTT_2 );

    postprocess_names.push_back( "Pi_Vversus_2" );
    postprocess_fields_tor.push_back( &Pi_tor_2 );
    postprocess_fields_pot.push_back( &Pi_VDD_2 );
    postprocess_fields_tot.push_back( &Pi_VTT_2 );

    if ( constants::COMP_PI_HELMHOLTZ ) {
        postprocess_names.push_back( "Pi_Helm_2" );
        postprocess_fields_tor.push_back( &Pi_Helm_2 );
        postprocess_fields_pot.push_back( &Pi_Helm_2 );
        postprocess_fields_tot.push_back( &Pi_Helm_2 );
    }

    postprocess_names.push_back( "Z_2" );
    postprocess_fields_tor.push_back( &Z_tor_2 );
    postprocess_fields_pot.push_back( &Z_pot_2 );
    postprocess_fields_tot.push_back( &Z_tot_2 );

    postprocess_names.push_back( "velocity_divergence_2" );
    postprocess_fields_tor.push_back( &div_tor_2 );
    postprocess_fields_pot.push_back( &div_pot_2 );
    postprocess_fields_tot.push_back( &div_tot_2 );

    // Error

    postprocess_names.push_back( "F_error" );
    postprocess_fields_tor.push_back( &coarse_F_tor_error );
    postprocess_fields_pot.push_back( &coarse_F_pot_error );
    postprocess_fields_tot.push_back( &u_r_coarse_error   );

    postprocess_names.push_back( "coarse_KE_error" );
    postprocess_fields_tor.push_back( &KE_tor_coarse_error);
    postprocess_fields_pot.push_back( &KE_pot_coarse_error );
    postprocess_fields_tot.push_back( &KE_tot_coarse_error );

    postprocess_names.push_back( "fine_KE_error" );
    postprocess_fields_tor.push_back( &KE_tor_fine_error );
    postprocess_fields_pot.push_back( &KE_pot_fine_error );
    postprocess_fields_tot.push_back( &KE_tot_fine_error );

    /*
    postprocess_names.push_back( "Fine_KE_mod" );
    postprocess_fields_tor.push_back( &KE_tor_fine_mod );
    postprocess_fields_pot.push_back( &KE_pot_fine_mod );
    postprocess_fields_tot.push_back( &KE_tot_fine_mod );
    */

    // KE spectra
    postprocess_names.push_back( "u_lon_spectrum_error" );
    postprocess_fields_tor.push_back( &u_spectrum_tor_error );
    postprocess_fields_pot.push_back( &u_spectrum_pot_error );
    postprocess_fields_tot.push_back( &u_spectrum_tot_error );

    postprocess_names.push_back( "u_lat_spectrum_error" );
    postprocess_fields_tor.push_back( &v_spectrum_tor_error );
    postprocess_fields_pot.push_back( &v_spectrum_pot_error );
    postprocess_fields_tot.push_back( &v_spectrum_tot_error );

    // KE spectral slopes
    postprocess_names.push_back( "KE_spectral_slope_error" );
    postprocess_fields_tor.push_back( &spec_slope_tor_error );
    postprocess_fields_pot.push_back( &spec_slope_pot_error );
    postprocess_fields_tot.push_back( &spec_slope_tot_error );

    /*
    postprocess_names.push_back( "div_J_transport" );
    postprocess_fields_tor.push_back( &div_J_tor );
    postprocess_fields_pot.push_back( &div_J_pot );
    postprocess_fields_tot.push_back( &div_J_tot );
    */

    postprocess_names.push_back( "enstrophy_error" );
    postprocess_fields_tor.push_back( &Enst_tor_error );
    postprocess_fields_pot.push_back( &Enst_pot_error );
    postprocess_fields_tot.push_back( &Enst_tot_error );

    postprocess_names.push_back( "u_lon_error" );
    postprocess_fields_tor.push_back( &u_lon_tor_error );
    postprocess_fields_pot.push_back( &u_lon_pot_error );
    postprocess_fields_tot.push_back( &u_lon_tot_error );

    postprocess_names.push_back( "u_lat_error" );
    postprocess_fields_tor.push_back( &u_lat_tor_error );
    postprocess_fields_pot.push_back( &u_lat_pot_error );
    postprocess_fields_tot.push_back( &u_lat_tot_error );

    if (constants::DO_OKUBOWEISS_ANALYSIS) {
        postprocess_names.push_back( "OkuboWeiss_error" );
        postprocess_fields_tor.push_back( &OkuboWeiss_tor_error );
        postprocess_fields_pot.push_back( &OkuboWeiss_pot_error );
        postprocess_fields_tot.push_back( &OkuboWeiss_tot_error );
    }

    postprocess_names.push_back( "Pi_error" );
    postprocess_fields_tor.push_back( &Pi_tor_error );
    postprocess_fields_pot.push_back( &Pi_pot_error );
    postprocess_fields_tot.push_back( &Pi_tot_error );

    postprocess_names.push_back( "Pi_Dversus_error" );
    postprocess_fields_tor.push_back( &Pi_DVV_error );
    postprocess_fields_pot.push_back( &Pi_pot_error );
    postprocess_fields_tot.push_back( &Pi_DTT_error );

    postprocess_names.push_back( "Pi_Vversus_error" );
    postprocess_fields_tor.push_back( &Pi_tor_error );
    postprocess_fields_pot.push_back( &Pi_VDD_error );
    postprocess_fields_tot.push_back( &Pi_VTT_error );

    if ( constants::COMP_PI_HELMHOLTZ ) {
        postprocess_names.push_back( "Pi_Helm_error" );
        postprocess_fields_tor.push_back( &Pi_Helm_error );
        postprocess_fields_pot.push_back( &Pi_Helm_error );
        postprocess_fields_tot.push_back( &Pi_Helm_error );
    }

    postprocess_names.push_back( "Z_error" );
    postprocess_fields_tor.push_back( &Z_tor_error );
    postprocess_fields_pot.push_back( &Z_pot_error );
    postprocess_fields_tot.push_back( &Z_tot_error );

    postprocess_names.push_back( "velocity_divergence_error" );
    postprocess_fields_tor.push_back( &div_tor_error );
    postprocess_fields_pot.push_back( &div_pot_error );
    postprocess_fields_tot.push_back( &div_tot_error );

    //
    //// Begin the main filtering loop
    //
    #if DEBUG>=1
    if (wRank == 0) { fprintf(stdout, "\nBeginning main filtering loop.\n\n"); }
    #endif
    for (int Iscale = 0; Iscale < Nscales; Iscale++) {


        // Reset our timing records
        timing_records.reset();

        // Create the output file
        snprintf(fname, 50, "filter_%.6gkm.nc", scales.at(Iscale)/1e3);
        if (not(constants::NO_FULL_OUTPUTS)) {
            initialize_output_file( source_data, vars_to_write, fname, scales.at(Iscale));

            // Add some attributes to the file
            add_attr_to_file("kernel_alpha", kern_alpha, fname);
        }

        #if DEBUG >= 0
        if (wRank == 0) { 
            fprintf(stdout, "\nScale %d of %d (%.5g km)\n", 
                Iscale+1, Nscales, scales.at(Iscale)/1e3); 
        }
        #endif

        scale = scales.at(Iscale);
        perc  = perc_base;

        #if DEBUG >= 1
        if (wRank == 0) { fprintf(stdout, "  filtering: "); }
        fflush(stdout);
        #endif

        // Debug: Check if we get here before OMP parallel
        fprintf(stderr, "[DEBUG] About to enter OMP parallel for scale %.5g km\n", scales.at(Iscale)/1e3);
        fflush(stderr);

        #pragma omp parallel \
        default(none) \
        shared( source_data, mask, stdout, stderr, perc_base, \
                filter_fields, filter_fields_2, filter_fields_error, filt_use_mask, \
                timing_records, clock_on, \
                longitude, latitude, scale, \
                F_potential, F_toroidal, coarse_F_tor, coarse_F_pot, u_r, u_r_coarse, \
                dl_coarse_Phi, dll_coarse_Phi, dl_coarse_Psi, dll_coarse_Psi, \
                dl_coarse_u_r, dll_coarse_u_r, \
                full_vort_tor_r, full_vort_pot_r, full_vort_tot_r, \
                u_x_tor, u_y_tor, u_z_tor, u_x_pot, u_y_pot, u_z_pot, u_x_tot, u_y_tot, u_z_tot, \
                ux_ux_tor, ux_uy_tor, ux_uz_tor, uy_uy_tor, uy_uz_tor, uz_uz_tor,\
                ux_ux_pot, ux_uy_pot, ux_uz_pot, uy_uy_pot, uy_uz_pot, uz_uz_pot,\
                ux_ux_tot, ux_uy_tot, ux_uz_tot, uy_uy_tot, uy_uz_tot, uz_uz_tot,\
                vort_ux_tor, vort_uy_tor, vort_uz_tor, \
                vort_ux_pot, vort_uy_pot, vort_uz_pot, \
                vort_ux_tot, vort_uy_tot, vort_uz_tot, \
                KE_tor_filt, KE_pot_filt, KE_tot_filt, \
                uiuj_F_r, uiuj_F_Phi, uiuj_F_Psi, coarse_uiuj_F_r, coarse_uiuj_F_Phi, coarse_uiuj_F_Psi, \
                wind_tau_Psi, wind_tau_Phi, tau_wind_dot_u_tor, tau_wind_dot_u_pot, \
                coarse_wind_tau_Psi, coarse_wind_tau_Phi, \
                coarse_tau_wind_dot_u_tor, coarse_tau_wind_dot_u_pot, coarse_tau_wind_dot_u_tot, \
                F_potential_2, F_toroidal_2, coarse_F_tor_2, coarse_F_pot_2, u_r_2, u_r_coarse_2, \
                dl_coarse_Phi_2, dll_coarse_Phi_2, dl_coarse_Psi_2, dll_coarse_Psi_2, \
                dl_coarse_u_r_2, dll_coarse_u_r_2, \
                full_vort_tor_r_2, full_vort_pot_r_2, full_vort_tot_r_2, \
                u_x_tor_2, u_y_tor_2, u_z_tor_2, u_x_pot_2, u_y_pot_2, u_z_pot_2, u_x_tot_2, u_y_tot_2, u_z_tot_2, \
                ux_ux_tor_2, ux_uy_tor_2, ux_uz_tor_2, uy_uy_tor_2, uy_uz_tor_2, uz_uz_tor_2,\
                ux_ux_pot_2, ux_uy_pot_2, ux_uz_pot_2, uy_uy_pot_2, uy_uz_pot_2, uz_uz_pot_2,\
                ux_ux_tot_2, ux_uy_tot_2, ux_uz_tot_2, uy_uy_tot_2, uy_uz_tot_2, uz_uz_tot_2,\
                vort_ux_tor_2, vort_uy_tor_2, vort_uz_tor_2, \
                vort_ux_pot_2, vort_uy_pot_2, vort_uz_pot_2, \
                vort_ux_tot_2, vort_uy_tot_2, vort_uz_tot_2, \
                KE_tor_filt_2, KE_pot_filt_2, KE_tot_filt_2, \
                uiuj_F_r_2, uiuj_F_Phi_2, uiuj_F_Psi_2, coarse_uiuj_F_r_2, coarse_uiuj_F_Phi_2, coarse_uiuj_F_Psi_2, \
                F_potential_error, F_toroidal_error, coarse_F_tor_error, coarse_F_pot_error, u_r_error, u_r_coarse_error, \
                dl_coarse_Phi_error, dll_coarse_Phi_error, dl_coarse_Psi_error, dll_coarse_Psi_error, \
                dl_coarse_u_r_error, dll_coarse_u_r_error, \
                full_vort_tor_r_error, full_vort_pot_r_error, full_vort_tot_r_error, \
                u_x_tor_error, u_y_tor_error, u_z_tor_error, u_x_pot_error, u_y_pot_error, u_z_pot_error, u_x_tot_error, u_y_tot_error, u_z_tot_error, \
                ux_ux_tor_error, ux_uy_tor_error, ux_uz_tor_error, uy_uy_tor_error, uy_uz_tor_error, uz_uz_tor_error,\
                ux_ux_pot_error, ux_uy_pot_error, ux_uz_pot_error, uy_uy_pot_error, uy_uz_pot_error, uz_uz_pot_error,\
                ux_ux_tot_error, ux_uy_tot_error, ux_uz_tot_error, uy_uy_tot_error, uy_uz_tot_error, uz_uz_tot_error,\
                vort_ux_tor_error, vort_uy_tor_error, vort_uz_tor_error, \
                vort_ux_pot_error, vort_uy_pot_error, vort_uz_pot_error, \
                vort_ux_tot_error, vort_uy_tot_error, vort_uz_tot_error, \
                KE_tor_filt_error, KE_pot_filt_error, KE_tot_filt_error, \
                local_kernel, local_dl_kernel, local_dll_kernel \
                ) \
        private(Itime, Idepth, Ilat, Ilon, index, prev_Ilat, Ilatlon, \
                F_tor_tmp, F_pot_tmp, u_r_tmp, uxux_tmp, uxuy_tmp, uxuz_tmp, uyuy_tmp, uyuz_tmp, uzuz_tmp, \
                vort_ux_tmp, vort_uy_tmp, vort_uz_tmp, LAT_lb, LAT_ub, thread_id, num_threads, \
                dl_kernel_val, dll_kernel_val, \
                dl_Psi_tmp, dll_Psi_tmp, dl_Phi_tmp, dll_Phi_tmp, dl_ur_tmp, dll_ur_tmp, \
                F_tor_tmp_2, F_pot_tmp_2, u_r_tmp_2, uxux_tmp_2, uxuy_tmp_2, uxuz_tmp_2, uyuy_tmp_2, uyuz_tmp_2, uzuz_tmp_2, \
                vort_ux_tmp_2, vort_uy_tmp_2, vort_uz_tmp_2, \
                dl_kernel_val_2, dll_kernel_val_2, \
                dl_Psi_tmp_2, dll_Psi_tmp_2, dl_Phi_tmp_2, dll_Phi_tmp_2, dl_ur_tmp_2, dll_ur_tmp_2, \
                F_tor_tmp_error, F_pot_tmp_error, u_r_tmp_error, uxux_tmp_error, uxuy_tmp_error, uxuz_tmp_error, uyuy_tmp_error, uyuz_tmp_error, uzuz_tmp_error, \
                vort_ux_tmp_error, vort_uy_tmp_error, vort_uz_tmp_error, \
                dl_kernel_val_error, dll_kernel_val_error, \
                dl_Psi_tmp_error, dll_Psi_tmp_error, dl_Phi_tmp_error, dll_Phi_tmp_error, dl_ur_tmp_error, dll_ur_tmp_error, \
                filtered_vals, dl_filter_vals, dll_filter_vals, \
                filtered_vals_2, dl_filter_vals_2, dll_filter_vals_2, \
                filtered_vals_error, dl_filter_vals_error, dll_filter_vals_error \
                ) \
        firstprivate(perc, wRank, \
                perc_count, Nlon, Nlat, Ndepth, Ntime )
        {
            // Debug: start of parallel region
            fprintf(stderr, "[DEBUG Thread %d] Entered parallel region\n", omp_get_thread_num());
            fflush(stderr);

            filtered_vals.clear();

            // Phi
            filtered_vals.push_back(&F_pot_tmp);
            dl_filter_vals.push_back( &dl_Phi_tmp );
            dll_filter_vals.push_back( &dll_Phi_tmp );

            // Psi
            filtered_vals.push_back(&F_tor_tmp);
            dl_filter_vals.push_back( &dl_Psi_tmp );
            dll_filter_vals.push_back( &dll_Psi_tmp );

            // u_r
            if ( source_data.compute_radial_vel ) {
                filtered_vals.push_back(&u_r_tmp);
                dl_filter_vals.push_back( &dl_ur_tmp );
                dll_filter_vals.push_back( &dll_ur_tmp );
            }

            filtered_vals_2.clear();

            // Phi
            filtered_vals_2.push_back(&F_pot_tmp_2);
            dl_filter_vals_2.push_back( &dl_Phi_tmp_2 );
            dll_filter_vals_2.push_back( &dll_Phi_tmp_2 );

            // Psi
            filtered_vals_2.push_back(&F_tor_tmp_2);
            dl_filter_vals_2.push_back( &dl_Psi_tmp_2 );
            dll_filter_vals_2.push_back( &dll_Psi_tmp_2 );

            // u_r
            if ( source_data.compute_radial_vel ) {
                filtered_vals_2.push_back(&u_r_tmp_2);
                dl_filter_vals_2.push_back( &dl_ur_tmp_2 );
                dll_filter_vals_2.push_back( &dll_ur_tmp_2 );
            }

            filtered_vals_error.clear();

            // Phi
            filtered_vals_error.push_back(&F_pot_tmp_error);
            dl_filter_vals_error.push_back( &dl_Phi_tmp_error );
            dll_filter_vals_error.push_back( &dll_Phi_tmp_error );

            // Psi
            filtered_vals_error.push_back(&F_tor_tmp_error);
            dl_filter_vals_error.push_back( &dl_Psi_tmp_error );
            dll_filter_vals_error.push_back( &dll_Psi_tmp_error );

            // u_r
            if ( source_data.compute_radial_vel ) {
                filtered_vals_error.push_back(&u_r_tmp_error);
                dl_filter_vals_error.push_back( &dl_ur_tmp_error );
                dll_filter_vals_error.push_back( &dll_ur_tmp_error );
            }

            thread_id = omp_get_thread_num();  // thread ID
            num_threads = omp_get_num_threads(); // number of threads

            prev_Ilat = -1; // forces kernel computation on first iteration

            // Melt latitude and longitude into one loop
            //   This is done manually (instead of simply using a 'collapse(2)'
            //   for optimization purposes. If threads jump between latitudes, 
            //   they need to keep recomputing the kernel, which is costly.
            //   Here we force them to iterate through one latitude at a time, 
            //   splitting the longitudes evenly between the group
            // If cost per point varies heavily, this could have load balancing issues,
            //   but in the case of filtering over land (the typical Helmholtz case)
            //   there is no land/water distinction, and so filtering cost should be
            //   similar at all longitudes.
            for (Ilatlon = thread_id; Ilatlon < Nlat * Nlon; Ilatlon = Ilatlon + num_threads) {
                Ilon = Ilatlon % Nlon;
                Ilat = Ilatlon / Nlon;

                if (thread_id == 0 && Ilatlon == 0) {
                    fprintf(stderr, "[DEBUG] Entered main loop, Ilat=0, Ilon=0\n");
                    fflush(stderr);
                }

                get_lat_bounds(LAT_lb, LAT_ub, latitude,  Ilat, scale); 

                // If our longitude grid is uniform, and spans the full periodic domain,
                // AND we're at the same latitude as the last iteration of the loop,
                // then we can just re-use the kernel.
                // Otherwise, we need to compute the kernel.
                if ( can_roll_in_longitude and (Ilat == prev_Ilat) ) {
                    // just re-use the kernel from last time
                } else if ( can_roll_in_longitude ) {
                    // At a new latitude, so compute kernel at reference longitude (index 0)
                    if ( (constants::DO_TIMING) and (thread_id == 0) ) { clock_on = MPI_Wtime(); }
                    compute_local_kernel( local_kernel, local_dl_kernel, local_dll_kernel,
                            scale, source_data, Ilat, 0, LAT_lb, LAT_ub );
                    if ( (constants::DO_TIMING) and (thread_id == 0) ) { timing_records.add_to_record(MPI_Wtime() - clock_on, "kernel_computation"); }
                } else {
                    // Otherwise, we need to compute the whole kernel every time. Boo.
                    if ( (constants::DO_TIMING) and (thread_id == 0) ) { clock_on = MPI_Wtime(); }
                    compute_local_kernel( local_kernel, local_dl_kernel, local_dll_kernel,
                            scale, source_data, Ilat, Ilon, LAT_lb, LAT_ub );
                    if ( (constants::DO_TIMING) and (thread_id == 0) ) { timing_records.add_to_record(MPI_Wtime() - clock_on, "kernel_computation_all"); }
                }
                // And set prev_Ilat before we forget
                prev_Ilat = Ilat;

                //fprintf( stdout, "Finished computing local kernel (tid %d of %d).\n", thread_id, num_threads );

                #if DEBUG >= 0
                if ( (thread_id == 0) and (wRank == 0) ) {
                    // Every perc_base percent, print a dot, but only the first thread
                    while ( ((double)(Ilat*Nlon + Ilon + 1) / (Nlon*Nlat)) * 100 >= perc ) {
                        perc_count++;
                        if (perc_count % 5 == 0) { fprintf(stdout, "|"); }
                        else                     { fprintf(stdout, "."); }
                        fflush(stdout);
                        perc += perc_base;
                    }
                }
                #endif

                for (Itime = 0; Itime < Ntime; Itime++) {
                    for (Idepth = 0; Idepth < Ndepth; Idepth++) {

                        // Convert our four-index to a one-index
                        index = Index(Itime, Idepth, Ilat, Ilon, Ntime, Ndepth, Nlat, Nlon);

                        // The F_tor and F_pot fields exist over land from the projection
                        //     procedure, so do those filtering operations on land as well.
                        // The other stuff (KE, etc), will only be done on water cells

                        // Apply the filter at the point
                        if ( (constants::DO_TIMING) and (thread_id == 0) ) { clock_on = MPI_Wtime(); }
                        
                        if (thread_id == 0 && Ilatlon == 0) {
                            fprintf(stderr, "[DEBUG] About to call apply_filter_at_point #1\n");
                            fprintf(stderr, "  filtered_vals.size()=%zu, dl_filter_vals.size()=%zu\n", filtered_vals.size(), dl_filter_vals.size());
                            fprintf(stderr, "  filter_fields.size()=%zu\n", filter_fields.size());
                            fprintf(stderr, "  local_kernel.size()=%zu, local_dl_kernel.size()=%zu, local_dll_kernel.size()=%zu\n", 
                                    local_kernel.size(), local_dl_kernel.size(), local_dll_kernel.size());
                            fprintf(stderr, "  Nlat=%d, Nlon=%d, Ntime=%d, Ndepth=%d\n", Nlat, Nlon, Ntime, Ndepth);
                            fprintf(stderr, "  latitude.size()=%zu, longitude.size()=%zu\n", 
                                    source_data.latitude.size(), source_data.longitude.size());
                            fflush(stderr);
                        }
                        
                        apply_filter_at_point(  
                                filtered_vals, dl_filter_vals, dll_filter_vals,
                                dl_kernel_val, dll_kernel_val,
                                filter_fields, source_data, Itime, Idepth, Ilat, Ilon, 
                                LAT_lb, LAT_ub, scale, filt_use_mask, 
                                local_kernel, local_dl_kernel, local_dll_kernel );
                        apply_filter_at_point(  
                                filtered_vals_2, dl_filter_vals_2, dll_filter_vals_2,
                                dl_kernel_val_2, dll_kernel_val_2,
                                filter_fields_2, source_data, Itime, Idepth, Ilat, Ilon, 
                                LAT_lb, LAT_ub, scale, filt_use_mask, 
                                local_kernel, local_dl_kernel, local_dll_kernel );
                        apply_filter_at_point(  
                                filtered_vals_error, dl_filter_vals_error, dll_filter_vals_error,
                                dl_kernel_val_error, dll_kernel_val_error,
                                filter_fields_error, source_data, Itime, Idepth, Ilat, Ilon, 
                                LAT_lb, LAT_ub, scale, filt_use_mask, 
                                local_kernel, local_dl_kernel, local_dll_kernel );
                        if ( (constants::DO_TIMING) and (thread_id == 0) ) { timing_records.add_to_record(MPI_Wtime() - clock_on, "filter_at_point"); }

                        // Store the filtered values in the appropriate arrays

                        // Phi
                        coarse_F_pot.at(index) = F_pot_tmp;
                        double dPhi = (dl_Phi_tmp - F_pot_tmp) * dl_kernel_val;
                        dl_coarse_Phi.at(index) = dPhi;
                        double ddPhi = ( dll_Phi_tmp - F_pot_tmp ) * dll_kernel_val - 2 * dPhi * dl_kernel_val;
                        dll_coarse_Phi.at(index) = ddPhi;

                        // Psi
                        coarse_F_tor.at(index) = F_tor_tmp;
                        double dPsi = (dl_Psi_tmp - F_tor_tmp) * dl_kernel_val;
                        dl_coarse_Psi.at(index) = dPsi;
                        double ddPsi = ( dll_Psi_tmp - F_tor_tmp ) * dll_kernel_val - 2 * dPsi * dl_kernel_val;
                        dll_coarse_Psi.at(index) = ddPsi;



                        // Phi
                        coarse_F_pot_2.at(index) = F_pot_tmp_2;
                        double dPhi_2 = (dl_Phi_tmp_2 - F_pot_tmp_2) * dl_kernel_val;
                        dl_coarse_Phi_2.at(index) = dPhi_2;
                        double ddPhi_2 = ( dll_Phi_tmp_2 - F_pot_tmp_2 ) * dll_kernel_val - 2 * dPhi_2 * dl_kernel_val;
                        dll_coarse_Phi_2.at(index) = ddPhi_2;

                        // Psi
                        coarse_F_tor_2.at(index) = F_tor_tmp_2;
                        double dPsi_2 = (dl_Psi_tmp_2 - F_tor_tmp_2) * dl_kernel_val;
                        dl_coarse_Psi_2.at(index) = dPsi_2;
                        double ddPsi_2 = ( dll_Psi_tmp_2 - F_tor_tmp_2 ) * dll_kernel_val - 2 * dPsi_2 * dl_kernel_val;
                        dll_coarse_Psi_2.at(index) = ddPsi_2;



                        // Phi
                        coarse_F_pot_error.at(index) = F_pot_tmp - F_pot_tmp_2;
                        dl_coarse_Phi_error.at(index) = dPhi - dPhi_2;
                        dll_coarse_Phi_error.at(index) = ddPhi - ddPhi_2;

                        // Psi
                        coarse_F_tor_error.at(index) = F_tor_tmp - F_tor_tmp_2;
                        dl_coarse_Psi_error.at(index) = dPsi - dPsi_2;
                        dll_coarse_Psi_error.at(index) = ddPsi - ddPsi_2;

                        // u_r
                        if ( source_data.compute_radial_vel ) {
 
                            u_r_coarse.at(index) = u_r_tmp;
                            double du_r = (dl_ur_tmp - u_r_tmp) * dl_kernel_val;
                            dl_coarse_u_r.at(index) = du_r;
                            double ddu_r = ( dll_ur_tmp - u_r_tmp ) * dll_kernel_val - 2 * du_r * dl_kernel_val;
                            dll_coarse_u_r.at(index) = ddu_r;

                            u_r_coarse_2.at(index) = u_r_tmp_2;
                            double du_r_2 = (dl_ur_tmp_2 - u_r_tmp_2) * dl_kernel_val;
                            dl_coarse_u_r_2.at(index) = du_r_2;
                            double ddu_r_2 = ( dll_ur_tmp_2 - u_r_tmp_2 ) * dll_kernel_val - 2 * du_r_2 * dl_kernel_val;
                            dll_coarse_u_r_2.at(index) = ddu_r_2;

                            u_r_coarse_error.at(index) = u_r_tmp - u_r_tmp_2;
                            dl_coarse_u_r_error.at(index) = du_r - du_r_2;
                            dll_coarse_u_r_error.at(index) = ddu_r - ddu_r_2;
                        }

                        if ( mask.at(index) ) {
                            if ( (constants::DO_TIMING) and (thread_id == 0) ) { clock_on = MPI_Wtime(); }

                            //
                            //// Also get (uiuj)_bar from Cartesian velocities
                            //

                            // tor
                            apply_filter_at_point_for_quadratics(
                                    uxux_tmp, uxuy_tmp, uxuz_tmp, uyuy_tmp, uyuz_tmp, uzuz_tmp, vort_ux_tmp, vort_uy_tmp, vort_uz_tmp,
                                    u_x_tor,  u_y_tor,  u_z_tor, full_vort_tor_r, source_data, Itime, Idepth, Ilat, Ilon,
                                    LAT_lb, LAT_ub, scale, local_kernel);

                            ux_ux_tor.at(index) = uxux_tmp;
                            ux_uy_tor.at(index) = uxuy_tmp;
                            ux_uz_tor.at(index) = uxuz_tmp;
                            uy_uy_tor.at(index) = uyuy_tmp;
                            uy_uz_tor.at(index) = uyuz_tmp;
                            uz_uz_tor.at(index) = uzuz_tmp;

                            vort_ux_tor.at(index) = vort_ux_tmp;
                            vort_uy_tor.at(index) = vort_uy_tmp;
                            vort_uz_tor.at(index) = vort_uz_tmp;

                            KE_tor_filt.at(index) = 0.5 * constants::rho0 * (uxux_tmp + uyuy_tmp + uzuz_tmp);

                            // pot
                            apply_filter_at_point_for_quadratics(
                                    uxux_tmp, uxuy_tmp, uxuz_tmp, uyuy_tmp, uyuz_tmp, uzuz_tmp, vort_ux_tmp, vort_uy_tmp, vort_uz_tmp,
                                    u_x_pot,  u_y_pot,  u_z_pot, full_vort_pot_r, source_data, Itime, Idepth, Ilat, Ilon,
                                    LAT_lb, LAT_ub, scale, local_kernel);

                            ux_ux_pot.at(index) = uxux_tmp;
                            ux_uy_pot.at(index) = uxuy_tmp;
                            ux_uz_pot.at(index) = uxuz_tmp;
                            uy_uy_pot.at(index) = uyuy_tmp;
                            uy_uz_pot.at(index) = uyuz_tmp;
                            uz_uz_pot.at(index) = uzuz_tmp;

                            vort_ux_pot.at(index) = vort_ux_tmp;
                            vort_uy_pot.at(index) = vort_uy_tmp;
                            vort_uz_pot.at(index) = vort_uz_tmp;

                            KE_pot_filt.at(index) = 0.5 * constants::rho0 * (uxux_tmp + uyuy_tmp + uzuz_tmp);

                            // tot
                            apply_filter_at_point_for_quadratics(
                                    uxux_tmp, uxuy_tmp, uxuz_tmp, uyuy_tmp, uyuz_tmp, uzuz_tmp, vort_ux_tmp, vort_uy_tmp, vort_uz_tmp,
                                    u_x_tot,  u_y_tot,  u_z_tot, full_vort_tot_r, source_data, Itime, Idepth, Ilat, Ilon,
                                    LAT_lb, LAT_ub, scale, local_kernel);

                            ux_ux_tot.at(index) = uxux_tmp;
                            ux_uy_tot.at(index) = uxuy_tmp;
                            ux_uz_tot.at(index) = uxuz_tmp;
                            uy_uy_tot.at(index) = uyuy_tmp;
                            uy_uz_tot.at(index) = uyuz_tmp;
                            uz_uz_tot.at(index) = uzuz_tmp;

                            vort_ux_tot.at(index) = vort_ux_tmp;
                            vort_uy_tot.at(index) = vort_uy_tmp;
                            vort_uz_tot.at(index) = vort_uz_tmp;

                            KE_tot_filt.at(index) = 0.5 * constants::rho0 * (uxux_tmp + uyuy_tmp + uzuz_tmp);

                            //
                            //// Also get (uiuj)_bar_2 from Cartesian velocities
                            //

                            // tor
                            apply_filter_at_point_for_quadratics(
                                    uxux_tmp_2, uxuy_tmp_2, uxuz_tmp_2, uyuy_tmp_2, uyuz_tmp_2, uzuz_tmp_2, vort_ux_tmp_2, vort_uy_tmp_2, vort_uz_tmp_2,
                                    u_x_tor_2,  u_y_tor_2,  u_z_tor_2, full_vort_tor_r_2, source_data, Itime, Idepth, Ilat, Ilon,
                                    LAT_lb, LAT_ub, scale, local_kernel);

                            ux_ux_tor_2.at(index) = uxux_tmp_2;
                            ux_uy_tor_2.at(index) = uxuy_tmp_2;
                            ux_uz_tor_2.at(index) = uxuz_tmp_2;
                            uy_uy_tor_2.at(index) = uyuy_tmp_2;
                            uy_uz_tor_2.at(index) = uyuz_tmp_2;
                            uz_uz_tor_2.at(index) = uzuz_tmp_2;

                            vort_ux_tor_2.at(index) = vort_ux_tmp_2;
                            vort_uy_tor_2.at(index) = vort_uy_tmp_2;
                            vort_uz_tor_2.at(index) = vort_uz_tmp_2;

                            KE_tor_filt_2.at(index) = 0.5 * constants::rho0 * (uxux_tmp_2 + uyuy_tmp_2 + uzuz_tmp_2);

                            // pot
                            apply_filter_at_point_for_quadratics(
                                    uxux_tmp_2, uxuy_tmp_2, uxuz_tmp_2, uyuy_tmp_2, uyuz_tmp_2, uzuz_tmp_2, vort_ux_tmp_2, vort_uy_tmp_2, vort_uz_tmp_2,
                                    u_x_pot_2,  u_y_pot_2,  u_z_pot_2, full_vort_pot_r_2, source_data, Itime, Idepth, Ilat, Ilon,
                                    LAT_lb, LAT_ub, scale, local_kernel);

                            ux_ux_pot_2.at(index) = uxux_tmp_2;
                            ux_uy_pot_2.at(index) = uxuy_tmp_2;
                            ux_uz_pot_2.at(index) = uxuz_tmp_2;
                            uy_uy_pot_2.at(index) = uyuy_tmp_2;
                            uy_uz_pot_2.at(index) = uyuz_tmp_2;
                            uz_uz_pot_2.at(index) = uzuz_tmp_2;

                            vort_ux_pot_2.at(index) = vort_ux_tmp_2;
                            vort_uy_pot_2.at(index) = vort_uy_tmp_2;
                            vort_uz_pot_2.at(index) = vort_uz_tmp_2;

                            KE_pot_filt_2.at(index) = 0.5 * constants::rho0 * (uxux_tmp_2 + uyuy_tmp_2 + uzuz_tmp_2);

                            // tot
                            apply_filter_at_point_for_quadratics(
                                    uxux_tmp_2, uxuy_tmp_2, uxuz_tmp_2, uyuy_tmp_2, uyuz_tmp_2, uzuz_tmp_2, vort_ux_tmp_2, vort_uy_tmp_2, vort_uz_tmp_2,
                                    u_x_tot_2,  u_y_tot_2,  u_z_tot_2, full_vort_tot_r_2, source_data, Itime, Idepth, Ilat, Ilon,
                                    LAT_lb, LAT_ub, scale, local_kernel);

                            ux_ux_tot_2.at(index) = uxux_tmp_2;
                            ux_uy_tot_2.at(index) = uxuy_tmp_2;
                            ux_uz_tot_2.at(index) = uxuz_tmp_2;
                            uy_uy_tot_2.at(index) = uyuy_tmp_2;
                            uy_uz_tot_2.at(index) = uyuz_tmp_2;
                            uz_uz_tot_2.at(index) = uzuz_tmp_2;

                            vort_ux_tot_2.at(index) = vort_ux_tmp_2;
                            vort_uy_tot_2.at(index) = vort_uy_tmp_2;
                            vort_uz_tot_2.at(index) = vort_uz_tmp_2;

                            KE_tot_filt_2.at(index) = 0.5 * constants::rho0 * (uxux_tmp_2 + uyuy_tmp_2 + uzuz_tmp_2);

                            if ( (constants::DO_TIMING) and (thread_id == 0) ) { 
                                timing_records.add_to_record(MPI_Wtime() - clock_on, "filter_at_point_for_quadratics"); 
                            }

                            //
                            //// Also get (uiuj)_bar_error from Cartesian velocities, note (uiuj)_bar_error = (eiej)_bar I'm doing it this way to try and minimise typos
                            //

                            // tor
                            apply_filter_at_point_for_quadratics(
                                    uxux_tmp_error, uxuy_tmp_error, uxuz_tmp_error, uyuy_tmp_error, uyuz_tmp_error, uzuz_tmp_error, vort_ux_tmp_error, vort_uy_tmp_error, vort_uz_tmp_error,
                                    u_x_tor_error,  u_y_tor_error,  u_z_tor_error, full_vort_tor_r_error, source_data, Itime, Idepth, Ilat, Ilon,
                                    LAT_lb, LAT_ub, scale, local_kernel);

                            ux_ux_tor_error.at(index) = uxux_tmp_error;
                            ux_uy_tor_error.at(index) = uxuy_tmp_error;
                            ux_uz_tor_error.at(index) = uxuz_tmp_error;
                            uy_uy_tor_error.at(index) = uyuy_tmp_error;
                            uy_uz_tor_error.at(index) = uyuz_tmp_error;
                            uz_uz_tor_error.at(index) = uzuz_tmp_error;

                            vort_ux_tor_error.at(index) = vort_ux_tmp_error;
                            vort_uy_tor_error.at(index) = vort_uy_tmp_error;
                            vort_uz_tor_error.at(index) = vort_uz_tmp_error;

                            KE_tor_filt_error.at(index) = 0.5 * constants::rho0 * (uxux_tmp_error + uyuy_tmp_error + uzuz_tmp_error);

                            // pot
                            apply_filter_at_point_for_quadratics(
                                    uxux_tmp_error, uxuy_tmp_error, uxuz_tmp_error, uyuy_tmp_error, uyuz_tmp_error, uzuz_tmp_error, vort_ux_tmp_error, vort_uy_tmp_error, vort_uz_tmp_error,
                                    u_x_pot_error,  u_y_pot_error,  u_z_pot_error, full_vort_pot_r_error, source_data, Itime, Idepth, Ilat, Ilon,
                                    LAT_lb, LAT_ub, scale, local_kernel);

                            ux_ux_pot_error.at(index) = uxux_tmp_error;
                            ux_uy_pot_error.at(index) = uxuy_tmp_error;
                            ux_uz_pot_error.at(index) = uxuz_tmp_error;
                            uy_uy_pot_error.at(index) = uyuy_tmp_error;
                            uy_uz_pot_error.at(index) = uyuz_tmp_error;
                            uz_uz_pot_error.at(index) = uzuz_tmp_error;

                            vort_ux_pot_error.at(index) = vort_ux_tmp_error;
                            vort_uy_pot_error.at(index) = vort_uy_tmp_error;
                            vort_uz_pot_error.at(index) = vort_uz_tmp_error;

                            KE_pot_filt_error.at(index) = 0.5 * constants::rho0 * (uxux_tmp_error + uyuy_tmp_error + uzuz_tmp_error);

                            // tot
                            apply_filter_at_point_for_quadratics(
                                    uxux_tmp_error, uxuy_tmp_error, uxuz_tmp_error, uyuy_tmp_error, uyuz_tmp_error, uzuz_tmp_error, vort_ux_tmp_error, vort_uy_tmp_error, vort_uz_tmp_error,
                                    u_x_tot_error,  u_y_tot_error,  u_z_tot_error, full_vort_tot_r_error, source_data, Itime, Idepth, Ilat, Ilon,
                                    LAT_lb, LAT_ub, scale, local_kernel);

                            ux_ux_tot_error.at(index) = uxux_tmp_error;
                            ux_uy_tot_error.at(index) = uxuy_tmp_error;
                            ux_uz_tot_error.at(index) = uxuz_tmp_error;
                            uy_uy_tot_error.at(index) = uyuy_tmp_error;
                            uy_uz_tot_error.at(index) = uyuz_tmp_error;
                            uz_uz_tot_error.at(index) = uzuz_tmp_error;

                            vort_ux_tot_error.at(index) = vort_ux_tmp_error;
                            vort_uy_tot_error.at(index) = vort_uy_tmp_error;
                            vort_uz_tot_error.at(index) = vort_uz_tmp_error;

                            KE_tot_filt_error.at(index) = 0.5 * constants::rho0 * (uxux_tmp_error + uyuy_tmp_error + uzuz_tmp_error);

                        }  // end if(masked) block
                    }  // end for(depth) block
                }  // end for(time) block
            }  // end for(latitude-longitude) block
        }  // end pragma parallel block




        #if DEBUG >= 0
        if (wRank == 0) { fprintf(stdout, "\n"); }
        #endif

        #if DEBUG >= 2
        fprintf(stdout, "  = Rank %d finished filtering loop =\n", wRank);
        fflush(stdout);
        #endif

        // Write to file
        if (not(constants::NO_FULL_OUTPUTS)) {
            if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            // Don't mask these fields, since they are filled over land from the projection
            write_field_to_output(coarse_F_tor, "coarse_F_tor", starts, counts, fname, NULL);
            write_field_to_output(coarse_F_pot, "coarse_F_pot", starts, counts, fname, NULL);

            write_field_to_output(coarse_F_tor_2, "coarse_F_tor_2", starts, counts, fname, NULL);
            write_field_to_output(coarse_F_pot_2, "coarse_F_pot_2", starts, counts, fname, NULL);

            write_field_to_output(coarse_F_tor_error, "coarse_F_tor_error", starts, counts, fname, NULL);
            write_field_to_output(coarse_F_pot_error, "coarse_F_pot_error", starts, counts, fname, NULL);

            if ( constants::COMP_PI_HELMHOLTZ ) {
                write_field_to_output(coarse_uiuj_F_r,   "coarse_uiuj_F_r",   starts, counts, fname, NULL);
                write_field_to_output(coarse_uiuj_F_Phi, "coarse_uiuj_F_Phi", starts, counts, fname, NULL);
                write_field_to_output(coarse_uiuj_F_Psi, "coarse_uiuj_F_Psi", starts, counts, fname, NULL);

                write_field_to_output(coarse_uiuj_F_r_2,   "coarse_uiuj_F_r_2",   starts, counts, fname, NULL);
                write_field_to_output(coarse_uiuj_F_Phi_2, "coarse_uiuj_F_Phi_2", starts, counts, fname, NULL);
                write_field_to_output(coarse_uiuj_F_Psi_2, "coarse_uiuj_F_Psi_2", starts, counts, fname, NULL);
            }
            if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "writing");  }
        }

        // Get pot and tor velocities
        if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
        toroidal_vel_from_F( u_lon_tor, u_lat_tor, coarse_F_tor, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask);
        potential_vel_from_F(u_lon_pot, u_lat_pot, coarse_F_pot, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask);

        toroidal_vel_from_F( u_lon_tor_2, u_lat_tor_2, coarse_F_tor_2, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask);
        potential_vel_from_F(u_lon_pot_2, u_lat_pot_2, coarse_F_pot_2, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask);

        toroidal_vel_from_F( u_lon_tor_error, u_lat_tor_error, coarse_F_tor_error, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask);
        potential_vel_from_F(u_lon_pot_error, u_lat_pot_error, coarse_F_pot_error, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask);

        #pragma omp parallel \
        default( none ) \
        shared( mask, u_lon_tor, u_lat_tor, u_lon_pot, u_lat_pot, u_lon_tot, u_lat_tot, \
                u_lon_tor_2, u_lat_tor_2, u_lon_pot_2, u_lat_pot_2, u_lon_tot_2, u_lat_tot_2, \
                u_lon_tor_error, u_lat_tor_error, u_lon_pot_error, u_lat_pot_error, u_lon_tot_error, u_lat_tot_error ) \
        private( index )
        {
            #pragma omp for collapse(1) schedule(guided)
            for (index = 0; index < u_lon_tor.size(); ++index) {
                if ( mask.at(index) ) {
                    u_lon_tot.at(index) = u_lon_tor.at(index) + u_lon_pot.at(index);
                    u_lat_tot.at(index) = u_lat_tor.at(index) + u_lat_pot.at(index);

                    u_lon_tot_2.at(index) = u_lon_tor_2.at(index) + u_lon_pot_2.at(index);
                    u_lat_tot_2.at(index) = u_lat_tor_2.at(index) + u_lat_pot_2.at(index);

                    u_lon_tot_error.at(index) = u_lon_tor_error.at(index) + u_lon_pot_error.at(index);
                    u_lat_tot_error.at(index) = u_lat_tor_error.at(index) + u_lat_pot_error.at(index);
                }
            }
        }
        if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute velocities from F"); }

        // Get the Cartesian velocities
        if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
        vel_Spher_to_Cart( u_x_tor_coarse, u_y_tor_coarse, u_z_tor_coarse, zero_array, u_lon_tor, u_lat_tor, source_data );
        vel_Spher_to_Cart( u_x_pot_coarse, u_y_pot_coarse, u_z_pot_coarse, u_r_coarse, u_lon_pot, u_lat_pot, source_data );
        vel_Spher_to_Cart( u_x_tot_coarse, u_y_tot_coarse, u_z_tot_coarse, u_r_coarse, u_lon_tot, u_lat_tot, source_data );

        vel_Spher_to_Cart( u_x_tor_coarse_2, u_y_tor_coarse_2, u_z_tor_coarse_2, zero_array, u_lon_tor_2, u_lat_tor_2, source_data );
        vel_Spher_to_Cart( u_x_pot_coarse_2, u_y_pot_coarse_2, u_z_pot_coarse_2, u_r_coarse_2, u_lon_pot_2, u_lat_pot_2, source_data );
        vel_Spher_to_Cart( u_x_tot_coarse_2, u_y_tot_coarse_2, u_z_tot_coarse_2, u_r_coarse_2, u_lon_tot_2, u_lat_tot_2, source_data );

        vel_Spher_to_Cart( u_x_tor_coarse_error, u_y_tor_coarse_error, u_z_tor_coarse_error, zero_array, u_lon_tor_error, u_lat_tor_error, source_data );
        vel_Spher_to_Cart( u_x_pot_coarse_error, u_y_pot_coarse_error, u_z_pot_coarse_error, u_r_coarse_error, u_lon_pot_error, u_lat_pot_error, source_data );
        vel_Spher_to_Cart( u_x_tot_coarse_error, u_y_tot_coarse_error, u_z_tot_coarse_error, u_r_coarse_error, u_lon_tot_error, u_lat_tot_error, source_data );

        if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "Sphere to Cart Conversion"); }

        if (not(constants::NO_FULL_OUTPUTS)) {
            if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            write_field_to_output(u_lon_tor, "u_lon_tor", starts, counts, fname, &mask);
            write_field_to_output(u_lat_tor, "u_lat_tor", starts, counts, fname, &mask);

            write_field_to_output(u_lon_pot, "u_lon_pot", starts, counts, fname, &mask);
            write_field_to_output(u_lat_pot, "u_lat_pot", starts, counts, fname, &mask);

            if ( source_data.compute_radial_vel ) {
                write_field_to_output(u_r_coarse, "u_r", starts, counts, fname, NULL);
            }

            write_field_to_output(u_lon_tor_2, "u_lon_tor_2", starts, counts, fname, &mask);
            write_field_to_output(u_lat_tor_2, "u_lat_tor_2", starts, counts, fname, &mask);

            write_field_to_output(u_lon_pot_2, "u_lon_pot_2", starts, counts, fname, &mask);
            write_field_to_output(u_lat_pot_2, "u_lat_pot_2", starts, counts, fname, &mask);

            if ( source_data.compute_radial_vel ) {
                write_field_to_output(u_r_coarse_2, "u_r_2", starts, counts, fname, NULL);
            }

            write_field_to_output(u_lon_tor_error, "u_lon_tor_error", starts, counts, fname, &mask);
            write_field_to_output(u_lat_tor_error, "u_lat_tor_error", starts, counts, fname, &mask);

            write_field_to_output(u_lon_pot_error, "u_lon_pot_error", starts, counts, fname, &mask);
            write_field_to_output(u_lat_pot_error, "u_lat_pot_error", starts, counts, fname, &mask);

            if ( source_data.compute_radial_vel ) {
                write_field_to_output(u_r_coarse_error, "u_r_error", starts, counts, fname, NULL);
            }

            if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "writing");  }
        }

        // KE spectra and spectral slopes
        compute_KE_spectra_and_slopes( 
                u_spectrum_tot, u_spectrum_tor, u_spectrum_pot,
                v_spectrum_tot, v_spectrum_tor, v_spectrum_pot,
                spec_slope_tot, spec_slope_tor, spec_slope_pot,
                u_lon_tot, u_lon_tor, u_lon_pot,
                u_lat_tot, u_lat_tor, u_lat_pot,
                dl_coarse_Phi, dl_coarse_Psi,
                dll_coarse_Phi, dll_coarse_Psi,
                source_data, scale
                );

        compute_KE_spectra_and_slopes( 
                u_spectrum_tot_2, u_spectrum_tor_2, u_spectrum_pot_2,
                v_spectrum_tot_2, v_spectrum_tor_2, v_spectrum_pot_2,
                spec_slope_tot_2, spec_slope_tor_2, spec_slope_pot_2,
                u_lon_tot_2, u_lon_tor_2, u_lon_pot_2,
                u_lat_tot_2, u_lat_tor_2, u_lat_pot_2,
                dl_coarse_Phi_2, dl_coarse_Psi_2,
                dll_coarse_Phi_2, dll_coarse_Psi_2,
                source_data, scale
                );

        if (wRank == 0) { 
            fprintf(stdout, "DEBUG: Before compute_KE_spectra_and_slopes\n");
            fprintf(stdout, "  u_lon_tor_error[0] = %.15e, u_lon_tor_error[1] = %.15e\n", 
                    u_lon_tor_error[0], u_lon_tor_error[1]);
            fprintf(stdout, "  coarse_F_tor_error[0] = %.15e\n", coarse_F_tor_error[0]);
            fflush(stdout);
            }

        compute_KE_spectra_and_slopes( 
                u_spectrum_tot_error, u_spectrum_tor_error, u_spectrum_pot_error,
                v_spectrum_tot_error, v_spectrum_tor_error, v_spectrum_pot_error,
                spec_slope_tot_error, spec_slope_tor_error, spec_slope_pot_error,
                u_lon_tot_error, u_lon_tor_error, u_lon_pot_error,
                u_lat_tot_error, u_lat_tor_error, u_lat_pot_error,
                dl_coarse_Phi_error, dl_coarse_Psi_error,
                dll_coarse_Phi_error, dll_coarse_Psi_error,
                source_data, scale
                );

        if (not(constants::NO_FULL_OUTPUTS)) {
            if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }

            write_field_to_output( u_spectrum_tot, "u_spectrum_tot", starts, counts, fname, &mask);
            write_field_to_output( v_spectrum_tot, "v_spectrum_tot", starts, counts, fname, &mask);

            write_field_to_output( u_spectrum_tor, "u_spectrum_tor", starts, counts, fname, &mask);
            write_field_to_output( v_spectrum_tor, "v_spectrum_tor", starts, counts, fname, &mask);

            write_field_to_output( u_spectrum_pot, "u_spectrum_pot", starts, counts, fname, &mask);
            write_field_to_output( v_spectrum_pot, "v_spectrum_pot", starts, counts, fname, &mask);

            write_field_to_output( spec_slope_tot, "KE_spectral_slope_tot", starts, counts, fname, &mask);
            write_field_to_output( spec_slope_tor, "KE_spectral_slope_tor", starts, counts, fname, &mask);
            write_field_to_output( spec_slope_pot, "KE_spectral_slope_pot", starts, counts, fname, &mask);

            // Field 2

            write_field_to_output( u_spectrum_tot_2, "u_spectrum_tot_2", starts, counts, fname, &mask);
            write_field_to_output( v_spectrum_tot_2, "v_spectrum_tot_2", starts, counts, fname, &mask);

            write_field_to_output( u_spectrum_tor_2, "u_spectrum_tor_2", starts, counts, fname, &mask);
            write_field_to_output( v_spectrum_tor_2, "v_spectrum_tor_2", starts, counts, fname, &mask);

            write_field_to_output( u_spectrum_pot_2, "u_spectrum_pot_2", starts, counts, fname, &mask);
            write_field_to_output( v_spectrum_pot_2, "v_spectrum_pot_2", starts, counts, fname, &mask);

            write_field_to_output( spec_slope_tot_2, "KE_spectral_slope_tot_2", starts, counts, fname, &mask);
            write_field_to_output( spec_slope_tor_2, "KE_spectral_slope_tor_2", starts, counts, fname, &mask);
            write_field_to_output( spec_slope_pot_2, "KE_spectral_slope_pot_2", starts, counts, fname, &mask);

            // Error

            write_field_to_output( u_spectrum_tot_error, "u_spectrum_tot_error", starts, counts, fname, &mask);
            write_field_to_output( v_spectrum_tot_error, "v_spectrum_tot_error", starts, counts, fname, &mask);

            write_field_to_output( u_spectrum_tor_error, "u_spectrum_tor_error", starts, counts, fname, &mask);
            write_field_to_output( v_spectrum_tor_error, "v_spectrum_tor_error", starts, counts, fname, &mask);

            write_field_to_output( u_spectrum_pot_error, "u_spectrum_pot_error", starts, counts, fname, &mask);
            write_field_to_output( v_spectrum_pot_error, "v_spectrum_pot_error", starts, counts, fname, &mask);

            write_field_to_output( spec_slope_tot_error, "KE_spectral_slope_tot_error", starts, counts, fname, &mask);
            write_field_to_output( spec_slope_tor_error, "KE_spectral_slope_tor_error", starts, counts, fname, &mask);
            write_field_to_output( spec_slope_pot_error, "KE_spectral_slope_pot_error", starts, counts, fname, &mask);


            if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "writing");  }
        }

        // Get uiuj from corresponding Helmholtz
        if ( constants::COMP_PI_HELMHOLTZ ) {
            uiuj_from_Helmholtz( ulon_ulon, ulon_ulat, ulat_ulat, coarse_uiuj_F_r, coarse_uiuj_F_Phi, coarse_uiuj_F_Psi, source_data );
            uiuj_from_Helmholtz( ulon_ulon_2, ulon_ulat_2, ulat_ulat_2, coarse_uiuj_F_r_2, coarse_uiuj_F_Phi_2, coarse_uiuj_F_Psi_2, source_data );

            if (not(constants::MINIMAL_OUTPUT)) {
                if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
                write_field_to_output( ulon_ulon, "coarse_uu", starts, counts, fname, &mask );
                write_field_to_output( ulon_ulat, "coarse_uv", starts, counts, fname, &mask );
                write_field_to_output( ulat_ulat, "coarse_vv", starts, counts, fname, &mask );
                
                // Field 2

                write_field_to_output( ulon_ulon_2, "coarse_uu_2", starts, counts, fname, &mask );
                write_field_to_output( ulon_ulat_2, "coarse_uv_2", starts, counts, fname, &mask );
                write_field_to_output( ulat_ulat_2, "coarse_vv_2", starts, counts, fname, &mask );
                if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "writing");  }
            }
        }

        // compute_vorticity gives vorticity, divergence, and OkuboWeiss
        if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
        compute_vorticity(
                vort_tor_r, null_vector, null_vector, div_tor, OkuboWeiss_tor, 
                null_vector, null_vector, null_vector, null_vector,
                source_data, zero_array, u_lon_tor, u_lat_tor);

        compute_vorticity(
                vort_pot_r, null_vector, null_vector, div_pot, OkuboWeiss_pot,
                null_vector, null_vector, null_vector, null_vector,
                source_data, u_r_coarse, u_lon_pot, u_lat_pot);

        compute_vorticity(
                vort_tot_r, null_vector, null_vector, div_tot, OkuboWeiss_tot,
                null_vector, null_vector, null_vector, null_vector,
                source_data, u_r_coarse, u_lon_tot, u_lat_tot);

        // Field 2

        compute_vorticity(
                vort_tor_r_2, null_vector, null_vector, div_tor_2, OkuboWeiss_tor_2, 
                null_vector, null_vector, null_vector, null_vector,
                source_data, zero_array, u_lon_tor_2, u_lat_tor_2);

        compute_vorticity(
                vort_pot_r_2, null_vector, null_vector, div_pot_2, OkuboWeiss_pot_2,
                null_vector, null_vector, null_vector, null_vector,
                source_data, u_r_coarse_2, u_lon_pot_2, u_lat_pot_2);

        compute_vorticity(
                vort_tot_r_2, null_vector, null_vector, div_tot_2, OkuboWeiss_tot_2,
                null_vector, null_vector, null_vector, null_vector,
                source_data, u_r_coarse_2, u_lon_tot_2, u_lat_tot_2);

        // Error

        compute_vorticity(
                vort_tor_r_error, null_vector, null_vector, div_tor_error, OkuboWeiss_tor_error, 
                null_vector, null_vector, null_vector, null_vector,
                source_data, zero_array, u_lon_tor_error, u_lat_tor_error);

        compute_vorticity(
                vort_pot_r_error, null_vector, null_vector, div_pot_error, OkuboWeiss_pot_error,
                null_vector, null_vector, null_vector, null_vector,
                source_data, u_r_coarse_error, u_lon_pot_error, u_lat_pot_error);

        compute_vorticity(
                vort_tot_r_error, null_vector, null_vector, div_tot_error, OkuboWeiss_tot_error,
                null_vector, null_vector, null_vector, null_vector,
                source_data, u_r_coarse_error, u_lon_tot_error, u_lat_tot_error);

        if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute vorticity"); }

        if (not(constants::MINIMAL_OUTPUT)) {
            if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            write_field_to_output(div_tor, "div_tor", starts, counts, fname, &mask);
            write_field_to_output(div_pot, "div_pot", starts, counts, fname, &mask);
            write_field_to_output(div_tot, "div_tot", starts, counts, fname, &mask);

            // Field 2

            write_field_to_output(div_tor_2, "div_tor_2", starts, counts, fname, &mask);
            write_field_to_output(div_pot_2, "div_pot_2", starts, counts, fname, &mask);
            write_field_to_output(div_tot_2, "div_tot_2", starts, counts, fname, &mask);

            // Error

            write_field_to_output(div_tor_error, "div_tor_error", starts, counts, fname, &mask);
            write_field_to_output(div_pot_error, "div_pot_error", starts, counts, fname, &mask);
            write_field_to_output(div_tot_error, "div_tot_error", starts, counts, fname, &mask);


            if (constants::DO_OKUBOWEISS_ANALYSIS) {
                write_field_to_output(OkuboWeiss_tor, "OkuboWeiss_tor", starts, counts, fname, &mask);
                write_field_to_output(OkuboWeiss_pot, "OkuboWeiss_pot", starts, counts, fname, &mask);
                write_field_to_output(OkuboWeiss_tot, "OkuboWeiss_tot", starts, counts, fname, &mask);

                // Field 2

                write_field_to_output(OkuboWeiss_tor_2, "OkuboWeiss_tor_2", starts, counts, fname, &mask);
                write_field_to_output(OkuboWeiss_pot_2, "OkuboWeiss_pot_2", starts, counts, fname, &mask);
                write_field_to_output(OkuboWeiss_tot_2, "OkuboWeiss_tot_2", starts, counts, fname, &mask);  

                // Error    

                write_field_to_output(OkuboWeiss_tor_error, "OkuboWeiss_tor_error", starts, counts, fname, &mask);
                write_field_to_output(OkuboWeiss_pot_error, "OkuboWeiss_pot_error", starts, counts, fname, &mask);
                write_field_to_output(OkuboWeiss_tot_error, "OkuboWeiss_tot_error", starts, counts, fname, &mask);
            }
            if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "writing");  }
        }


        //
        //// Toroidal diagnostics
        //

        // If we need depth derivatives, we'll need to communicated across MPI
        //    ranks in order to rebuild the depth profile. We'll do that here.
        std::vector<double> u_x_coarse_DEPTH, u_y_coarse_DEPTH, u_z_coarse_DEPTH,
                            ux_ux_DEPTH, ux_uy_DEPTH, ux_uz_DEPTH, uy_uy_DEPTH, uy_uz_DEPTH, uz_uz_DEPTH,
                            vort_r_DEPTH, vort_ux_DEPTH, vort_uy_DEPTH, vort_uz_DEPTH;

        std::vector<double> u_x_coarse_DEPTH_2, u_y_coarse_DEPTH_2, u_z_coarse_DEPTH_2,
                            ux_ux_DEPTH_2, ux_uy_DEPTH_2, ux_uz_DEPTH_2, uy_uy_DEPTH_2, uy_uz_DEPTH_2, uz_uz_DEPTH_2,
                            vort_r_DEPTH_2, vort_ux_DEPTH_2, vort_uy_DEPTH_2, vort_uz_DEPTH_2;

        if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            // Energy cascade (Pi)
            compute_Pi( Pi_tor, source_data, u_x_tor_coarse, u_y_tor_coarse, u_z_tor_coarse, 
                        ux_ux_tor, ux_uy_tor, ux_uz_tor, uy_uy_tor, uy_uz_tor, uz_uz_tor );

            // Enstrophy cascade (Z)
            compute_Z(  Z_tor, source_data, u_x_tor_coarse, u_y_tor_coarse, u_z_tor_coarse, 
                        vort_tor_r, vort_ux_tor, vort_uy_tor, vort_uz_tor );

            // Field 2
            compute_Pi( Pi_tor_2, source_data, u_x_tor_coarse_2, u_y_tor_coarse_2, u_z_tor_coarse_2, 
                        ux_ux_tor_2, ux_uy_tor_2, ux_uz_tor_2, uy_uy_tor_2, uy_uz_tor_2, uz_uz_tor_2 );

            compute_Z(  Z_tor_2, source_data, u_x_tor_coarse_2, u_y_tor_coarse_2, u_z_tor_coarse_2, 
                        vort_tor_r_2, vort_ux_tor_2, vort_uy_tor_2, vort_uz_tor_2 );

            // Error

            compute_Pi_error( Pi_tor_error, source_data, u_x_tor_coarse, u_y_tor_coarse, u_z_tor_coarse, 
                        ux_ux_tor, ux_uy_tor, ux_uz_tor, uy_uy_tor, uy_uz_tor, uz_uz_tor, 
                        u_x_tor_coarse_2, u_y_tor_coarse_2, u_z_tor_coarse_2, 
                        ux_ux_tor_2, ux_uy_tor_2, ux_uz_tor_2, uy_uy_tor_2, uy_uz_tor_2, uz_uz_tor_2, 
                        nullptr, nullptr, nullptr,   // ux_in_tau etc
                        nullptr, nullptr, nullptr);   // ux_in_tau_2 etc


        if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute_Pi_and_Z"); }

        // Energy transport
        if (source_data.use_depth_derivatives and ( source_data.Nprocs_in_depth > 1 )) {
            if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            MPI_Barrier(source_data.MPI_subcomm_sametimes);
            source_data.gather_variable_across_depth( u_x_tor_coarse, u_x_coarse_DEPTH );
            source_data.gather_variable_across_depth( u_y_tor_coarse, u_y_coarse_DEPTH );
            source_data.gather_variable_across_depth( u_z_tor_coarse, u_z_coarse_DEPTH );

            source_data.gather_variable_across_depth( ux_ux_tor, ux_ux_DEPTH );
            source_data.gather_variable_across_depth( ux_uy_tor, ux_uy_DEPTH );
            source_data.gather_variable_across_depth( ux_uz_tor, ux_uz_DEPTH );
            source_data.gather_variable_across_depth( uy_uy_tor, uy_uy_DEPTH );
            source_data.gather_variable_across_depth( uy_uz_tor, uy_uz_DEPTH );
            source_data.gather_variable_across_depth( uz_uz_tor, uz_uz_DEPTH );

            if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "MPI_COMM_depth_merging"); }
            if (wRank == 0) { fprintf( stdout, "Merged variables across depth.\n" ); fflush(stdout); }
        }
        if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            if (source_data.use_depth_derivatives and ( source_data.Nprocs_in_depth > 1 )) {
                compute_div_transport( div_J_tor, source_data, u_x_coarse_DEPTH, u_y_coarse_DEPTH, u_z_coarse_DEPTH, 
                                       ux_ux_DEPTH, ux_uy_DEPTH, ux_uz_DEPTH, uy_uy_DEPTH, uy_uz_DEPTH, uz_uz_DEPTH, 
                                       zero_array);
                compute_div_transport( div_J_tor_2, source_data, u_x_coarse_DEPTH_2, u_y_coarse_DEPTH_2, u_z_coarse_DEPTH_2, 
                                       ux_ux_DEPTH_2, ux_uy_DEPTH_2, ux_uz_DEPTH_2, uy_uy_DEPTH_2, uy_uz_DEPTH_2, uz_uz_DEPTH_2, 
                                       zero_array);                                      
            } else {
                compute_div_transport( div_J_tor, source_data, u_x_tor_coarse, u_y_tor_coarse, u_z_tor_coarse, 
                                       ux_ux_tor, ux_uy_tor, ux_uz_tor, uy_uy_tor, uy_uz_tor, uz_uz_tor, 
                                       zero_array);
                compute_div_transport( div_J_tor_2, source_data, u_x_tor_coarse_2, u_y_tor_coarse_2, u_z_tor_coarse_2, 
                                       ux_ux_tor_2, ux_uy_tor_2, ux_uz_tor_2, uy_uy_tor_2, uy_uz_tor_2, uz_uz_tor_2, 
                                       zero_array);
            }
        if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute_transport"); }

        //
        //// Potential diagnostics
        //

        if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            // Energy cascade (Pi)
            compute_Pi( Pi_pot, source_data, u_x_pot_coarse, u_y_pot_coarse, u_z_pot_coarse, 
                        ux_ux_pot, ux_uy_pot, ux_uz_pot, uy_uy_pot, uy_uz_pot, uz_uz_pot );

            // Enstrophy cascade (Z)
            compute_Z(  Z_pot, source_data, u_x_pot_coarse, u_y_pot_coarse, u_z_pot_coarse, 
                        vort_pot_r, vort_ux_pot, vort_uy_pot, vort_uz_pot );

            // Field 2

            // Energy cascade (Pi)
            compute_Pi( Pi_pot_2, source_data, u_x_pot_coarse_2, u_y_pot_coarse_2, u_z_pot_coarse_2, 
                        ux_ux_pot_2, ux_uy_pot_2, ux_uz_pot_2, uy_uy_pot_2, uy_uz_pot_2, uz_uz_pot_2 );

            // Enstrophy cascade (Z)
            compute_Z(  Z_pot_2, source_data, u_x_pot_coarse_2, u_y_pot_coarse_2, u_z_pot_coarse_2, 
                        vort_pot_r_2, vort_ux_pot_2, vort_uy_pot_2, vort_uz_pot_2 );

            // Error

            compute_Pi_error( Pi_pot_error, source_data, u_x_pot_coarse, u_y_pot_coarse, u_z_pot_coarse, 
            ux_ux_pot, ux_uy_pot, ux_uz_pot, uy_uy_pot, uy_uz_pot, uz_uz_pot, 
            u_x_pot_coarse_2, u_y_pot_coarse_2, u_z_pot_coarse_2, 
            ux_ux_pot_2, ux_uy_pot_2, ux_uz_pot_2, uy_uy_pot_2, uy_uz_pot_2, uz_uz_pot_2, 
            nullptr, nullptr, nullptr,   // ux_in_tau etc
            nullptr, nullptr, nullptr);   // ux_in_tau_2 etc

        if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute_Pi_and_Z"); }

        // Energy transport
        if (source_data.use_depth_derivatives and ( source_data.Nprocs_in_depth > 1 )) {
            if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            MPI_Barrier(source_data.MPI_subcomm_sametimes);
            source_data.gather_variable_across_depth( u_x_pot_coarse, u_x_coarse_DEPTH );
            source_data.gather_variable_across_depth( u_y_pot_coarse, u_y_coarse_DEPTH );
            source_data.gather_variable_across_depth( u_z_pot_coarse, u_z_coarse_DEPTH );

            source_data.gather_variable_across_depth( ux_ux_pot, ux_ux_DEPTH );
            source_data.gather_variable_across_depth( ux_uy_pot, ux_uy_DEPTH );
            source_data.gather_variable_across_depth( ux_uz_pot, ux_uz_DEPTH );
            source_data.gather_variable_across_depth( uy_uy_pot, uy_uy_DEPTH );
            source_data.gather_variable_across_depth( uy_uz_pot, uy_uz_DEPTH );
            source_data.gather_variable_across_depth( uz_uz_pot, uz_uz_DEPTH );
            
            if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "MPI_COMM_depth_merging"); }
        }
        if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            if (source_data.use_depth_derivatives and ( source_data.Nprocs_in_depth > 1 )) {
                compute_div_transport( div_J_pot, source_data, u_x_coarse_DEPTH, u_y_coarse_DEPTH, u_z_coarse_DEPTH, 
                                       ux_ux_DEPTH, ux_uy_DEPTH, ux_uz_DEPTH, uy_uy_DEPTH, uy_uz_DEPTH, uz_uz_DEPTH, 
                                       zero_array);
                compute_div_transport( div_J_pot_2, source_data, u_x_coarse_DEPTH_2, u_y_coarse_DEPTH_2, u_z_coarse_DEPTH_2, 
                                       ux_ux_DEPTH_2, ux_uy_DEPTH_2, ux_uz_DEPTH_2, uy_uy_DEPTH_2, uy_uz_DEPTH_2, uz_uz_DEPTH_2, 
                                       zero_array);
            } else {
                compute_div_transport( div_J_pot, source_data, u_x_pot_coarse, u_y_pot_coarse, u_z_pot_coarse, 
                                       ux_ux_pot, ux_uy_pot, ux_uz_pot, uy_uy_pot, uy_uz_pot, uz_uz_pot, 
                                       zero_array);
                compute_div_transport( div_J_pot_2, source_data, u_x_pot_coarse_2, u_y_pot_coarse_2, u_z_pot_coarse_2, 
                                       ux_ux_pot_2, ux_uy_pot_2, ux_uz_pot_2, uy_uy_pot_2, uy_uz_pot_2, uz_uz_pot_2, 
                                       zero_array);
            }
        if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute_transport"); }

        //
        //// Total velocity diagnostics
        //

        if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            // Energy cascade (Pi)
            compute_Pi( Pi_tot, source_data, u_x_tot_coarse, u_y_tot_coarse, u_z_tot_coarse, 
                        ux_ux_tot, ux_uy_tot, ux_uz_tot, uy_uy_tot, uy_uz_tot, uz_uz_tot );

            // Enstrophy cascade (Z)
            compute_Z(  Z_tot, source_data, u_x_tot_coarse, u_y_tot_coarse, u_z_tot_coarse, 
                        vort_tot_r, vort_ux_tot, vort_uy_tot, vort_uz_tot );

            // Field 2
                    
            // Energy cascade (Pi)
            compute_Pi( Pi_tot_2, source_data, u_x_tot_coarse_2, u_y_tot_coarse_2, u_z_tot_coarse_2, 
                        ux_ux_tot_2, ux_uy_tot_2, ux_uz_tot_2, uy_uy_tot_2, uy_uz_tot_2, uz_uz_tot_2 );

            // Enstrophy cascade (Z)
            compute_Z(  Z_tot_2, source_data, u_x_tot_coarse_2, u_y_tot_coarse_2, u_z_tot_coarse_2, 
                        vort_tot_r_2, vort_ux_tot_2, vort_uy_tot_2, vort_uz_tot_2 );

            // Error

            compute_Pi_error( Pi_tot_error, source_data, u_x_tot_coarse, u_y_tot_coarse, u_z_tot_coarse, 
            ux_ux_tot, ux_uy_tot, ux_uz_tot, uy_uy_tot, uy_uz_tot, uz_uz_tot, 
            u_x_tot_coarse_2, u_y_tot_coarse_2, u_z_tot_coarse_2, 
            ux_ux_tot_2, ux_uy_tot_2, ux_uz_tot_2, uy_uy_tot_2, uy_uz_tot_2, uz_uz_tot_2, 
            nullptr, nullptr, nullptr,   // ux_in_tau etc
            nullptr, nullptr, nullptr);   // ux_in_tau_2 etc


        if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute_Pi_and_Z"); }

        // Energy transport
        if (source_data.use_depth_derivatives and ( source_data.Nprocs_in_depth > 1 )) {
            if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            MPI_Barrier(source_data.MPI_subcomm_sametimes);
            source_data.gather_variable_across_depth( u_x_tot_coarse, u_x_coarse_DEPTH );
            source_data.gather_variable_across_depth( u_y_tot_coarse, u_y_coarse_DEPTH );
            source_data.gather_variable_across_depth( u_z_tot_coarse, u_z_coarse_DEPTH );

            source_data.gather_variable_across_depth( ux_ux_tot, ux_ux_DEPTH );
            source_data.gather_variable_across_depth( ux_uy_tot, ux_uy_DEPTH );
            source_data.gather_variable_across_depth( ux_uz_tot, ux_uz_DEPTH );
            source_data.gather_variable_across_depth( uy_uy_tot, uy_uy_DEPTH );
            source_data.gather_variable_across_depth( uy_uz_tot, uy_uz_DEPTH );
            source_data.gather_variable_across_depth( uz_uz_tot, uz_uz_DEPTH );

            if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "MPI_COMM_depth_merging"); }
        }
        if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            if (source_data.use_depth_derivatives and ( source_data.Nprocs_in_depth > 1 )) {
                compute_div_transport( div_J_tot, source_data, u_x_coarse_DEPTH, u_y_coarse_DEPTH, u_z_coarse_DEPTH, 
                                       ux_ux_DEPTH, ux_uy_DEPTH, ux_uz_DEPTH, uy_uy_DEPTH, uy_uz_DEPTH, uz_uz_DEPTH, 
                                       zero_array);
                compute_div_transport( div_J_tot_2, source_data, u_x_coarse_DEPTH_2, u_y_coarse_DEPTH_2, u_z_coarse_DEPTH_2, 
                                       ux_ux_DEPTH_2, ux_uy_DEPTH_2, ux_uz_DEPTH_2, uy_uy_DEPTH_2, uy_uz_DEPTH_2, uz_uz_DEPTH_2, 
                                       zero_array);
            } else {
                compute_div_transport( div_J_tot, source_data, u_x_tot_coarse, u_y_tot_coarse, u_z_tot_coarse, 
                                       ux_ux_tot, ux_uy_tot, ux_uz_tot, uy_uy_tot, uy_uz_tot, uz_uz_tot, 
                                       zero_array);
                compute_div_transport( div_J_tot_2, source_data, u_x_tot_coarse_2, u_y_tot_coarse_2, u_z_tot_coarse_2, 
                                       ux_ux_tot_2, ux_uy_tot_2, ux_uz_tot_2, uy_uy_tot_2, uy_uz_tot_2, uz_uz_tot_2, 
                                       zero_array);
            }
        if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute_transport"); }

        //
        //// Cross-term Pi [these should now give all combinations, when mixed appropriated]
        //

        if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
        compute_Pi( Pi_DVV, source_data, u_x_pot_coarse, u_y_pot_coarse, u_z_pot_coarse, 
                ux_ux_tor, ux_uy_tor, ux_uz_tor, uy_uy_tor, uy_uz_tor, uz_uz_tor,
                &u_x_tor_coarse, &u_y_tor_coarse, &u_z_tor_coarse );
        compute_Pi( Pi_DTT, source_data, u_x_pot_coarse, u_y_pot_coarse, u_z_pot_coarse, 
                ux_ux_tot, ux_uy_tot, ux_uz_tot, uy_uy_tot, uy_uz_tot, uz_uz_tot,
                &u_x_tot_coarse, &u_y_tot_coarse, &u_z_tot_coarse );

        compute_Pi( Pi_VDD, source_data, u_x_tor_coarse, u_y_tor_coarse, u_z_tor_coarse, 
                ux_ux_pot, ux_uy_pot, ux_uz_pot, uy_uy_pot, uy_uz_pot, uz_uz_pot,
                &u_x_pot_coarse, &u_y_pot_coarse, &u_z_pot_coarse );
        compute_Pi( Pi_VTT, source_data, u_x_tor_coarse, u_y_tor_coarse, u_z_tor_coarse, 
                ux_ux_tot, ux_uy_tot, ux_uz_tot, uy_uy_tot, uy_uz_tot, uz_uz_tot,
                &u_x_tot_coarse, &u_y_tot_coarse, &u_z_tot_coarse );

        // Field 2
        compute_Pi( Pi_DVV_2, source_data, u_x_pot_coarse_2, u_y_pot_coarse_2, u_z_pot_coarse_2, 
                ux_ux_tor_2, ux_uy_tor_2, ux_uz_tor_2, uy_uy_tor_2, uy_uz_tor_2, uz_uz_tor_2,
                &u_x_tor_coarse_2, &u_y_tor_coarse_2, &u_z_tor_coarse_2 );
        compute_Pi( Pi_DTT_2, source_data, u_x_pot_coarse_2, u_y_pot_coarse_2, u_z_pot_coarse_2, 
                ux_ux_tot_2, ux_uy_tot_2, ux_uz_tot_2, uy_uy_tot_2, uy_uz_tot_2, uz_uz_tot_2,
                &u_x_tot_coarse_2, &u_y_tot_coarse_2, &u_z_tot_coarse_2 );

        compute_Pi( Pi_VDD_2, source_data, u_x_tor_coarse_2, u_y_tor_coarse_2, u_z_tor_coarse_2, 
                ux_ux_pot_2, ux_uy_pot_2, ux_uz_pot_2, uy_uy_pot_2, uy_uz_pot_2, uz_uz_pot_2,
                &u_x_pot_coarse_2, &u_y_pot_coarse_2, &u_z_pot_coarse_2 );
        compute_Pi( Pi_VTT_2, source_data, u_x_tor_coarse_2, u_y_tor_coarse_2, u_z_tor_coarse_2, 
                ux_ux_tot_2, ux_uy_tot_2, ux_uz_tot_2, uy_uy_tot_2, uy_uz_tot_2, uz_uz_tot_2,
                &u_x_tot_coarse_2, &u_y_tot_coarse_2, &u_z_tot_coarse_2 );
        if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute_Pi_and_Z"); }


        //
        //  Helmholtz Pi
        //
        if ( constants::COMP_PI_HELMHOLTZ ) {
            compute_Pi_Helmholtz( Pi_Helm, source_data, u_lon_tot, u_lat_tot, ulon_ulon, ulon_ulat, ulat_ulat );
            compute_Pi_Helmholtz( Pi_Helm_2, source_data, u_lon_tot_2, u_lat_tot_2, ulon_ulon_2, ulon_ulat_2, ulat_ulat_2 );
            compute_Pi_Helmholtz_error( Pi_Helm_error, source_data, u_lon_tot, u_lat_tot, ulon_ulon, ulon_ulat, ulat_ulat,
                                        u_lon_tot_2, u_lat_tot_2, ulon_ulon_2, ulon_ulat_2, ulat_ulat_2 );
        }

        // Writing
        if (not(constants::NO_FULL_OUTPUTS)) {
            if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            write_field_to_output( Pi_tor, "Pi_tor", starts, counts, fname, &mask);
            write_field_to_output( Pi_pot, "Pi_pot", starts, counts, fname, &mask);
            write_field_to_output( Pi_tot, "Pi_tot", starts, counts, fname, &mask);
            
            write_field_to_output( Pi_tor_2, "Pi_tor_2", starts, counts, fname, &mask);
            write_field_to_output( Pi_pot_2, "Pi_pot_2", starts, counts, fname, &mask);
            write_field_to_output( Pi_tot_2, "Pi_tot_2", starts, counts, fname, &mask);
            
            write_field_to_output( Pi_tor_error, "Pi_tor_error", starts, counts, fname, &mask);
            write_field_to_output( Pi_pot_error, "Pi_pot_error", starts, counts, fname, &mask);
            write_field_to_output( Pi_tot_error, "Pi_tot_error", starts, counts, fname, &mask);

            if ( constants::COMP_PI_HELMHOLTZ ) {
                write_field_to_output( Pi_Helm, "Pi_Helm", starts, counts, fname, &mask);
                write_field_to_output( Pi_Helm_2, "Pi_Helm_2", starts, counts, fname, &mask);
                write_field_to_output( Pi_Helm_error, "Pi_Helm_error", starts, counts, fname, &mask);
            }

            write_field_to_output( Z_tor, "Z_tor", starts, counts, fname, &mask);
            write_field_to_output( Z_pot, "Z_pot", starts, counts, fname, &mask);
            write_field_to_output( Z_tot, "Z_tot", starts, counts, fname, &mask);

            write_field_to_output( Z_tor_2, "Z_tor_2", starts, counts, fname, &mask);
            write_field_to_output( Z_pot_2, "Z_pot_2", starts, counts, fname, &mask);
            write_field_to_output( Z_tot_2, "Z_tot_2", starts, counts, fname, &mask);
            if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "writing");  }
        }

        if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
        #pragma omp parallel \
        default( none ) \
        shared( KE_tor_coarse, KE_tor_fine, KE_tor_filt, KE_tor_fine_mod, KE_tor_orig, \
                KE_pot_coarse, KE_pot_fine, KE_pot_filt, KE_pot_fine_mod, KE_pot_orig, \
                KE_tot_coarse, KE_tot_fine, KE_tot_filt, KE_tot_fine_mod, KE_tot_orig, \
                Enst_tor, Enst_pot, Enst_tot, mask, \
                u_lon_tor, u_lat_tor, u_lon_pot, u_lat_pot, u_lon_tot, u_lat_tot, \
                vort_tor_r, vort_pot_r, vort_tot_r, \
                KE_tor_coarse_2, KE_tor_fine_2, KE_tor_filt_2, KE_tor_fine_mod_2, KE_tor_orig_2, \
                KE_pot_coarse_2, KE_pot_fine_2, KE_pot_filt_2, KE_pot_fine_mod_2, KE_pot_orig_2, \
                KE_tot_coarse_2, KE_tot_fine_2, KE_tot_filt_2, KE_tot_fine_mod_2, KE_tot_orig_2, \
                Enst_tor_2, Enst_pot_2, Enst_tot_2,  \
                u_lon_tor_2, u_lat_tor_2, u_lon_pot_2, u_lat_pot_2, u_lon_tot_2, u_lat_tot_2, \
                vort_tor_r_2, vort_pot_r_2, vort_tot_r_2, \
                KE_tor_coarse_error, KE_tor_fine_error, KE_tor_filt_error, KE_tor_fine_mod_error, KE_tor_orig_error, \
                KE_pot_coarse_error, KE_pot_fine_error, KE_pot_filt_error, KE_pot_fine_mod_error, KE_pot_orig_error, \
                KE_tot_coarse_error, KE_tot_fine_error, KE_tot_filt_error, KE_tot_fine_mod_error, KE_tot_orig_error, \
                Enst_tor_error, Enst_pot_error, Enst_tot_error,  \
                u_lon_tor_error, u_lat_tor_error, u_lon_pot_error, u_lat_pot_error, u_lon_tot_error, u_lat_tot_error, \
                vort_tor_r_error, vort_pot_r_error, vort_tot_r_error \
            ) \
        private( index )
        {
            #pragma omp for collapse(1) schedule(guided)
            for (index = 0; index < u_lon_tor.size(); ++index) {
                if ( mask.at(index) ) { 
                    KE_tor_coarse.at(index) = 0.5 * constants::rho0 * ( pow(u_lon_tor.at(index), 2.) + pow(u_lat_tor.at(index), 2.) );
                    KE_pot_coarse.at(index) = 0.5 * constants::rho0 * ( pow(u_lon_pot.at(index), 2.) + pow(u_lat_pot.at(index), 2.) );
                    KE_tot_coarse.at(index) = 0.5 * constants::rho0 * ( pow(u_lon_tot.at(index), 2.) + pow(u_lat_tot.at(index), 2.) );

                    KE_tor_coarse_2.at(index) = 0.5 * constants::rho0 * ( pow(u_lon_tor_2.at(index), 2.) + pow(u_lat_tor_2.at(index), 2.) );
                    KE_pot_coarse_2.at(index) = 0.5 * constants::rho0 * ( pow(u_lon_pot_2.at(index), 2.) + pow(u_lat_pot_2.at(index), 2.) );
                    KE_tot_coarse_2.at(index) = 0.5 * constants::rho0 * ( pow(u_lon_tot_2.at(index), 2.) + pow(u_lat_tot_2.at(index), 2.) );

                    KE_tor_coarse_error.at(index) = 0.5 * constants::rho0 * ( pow(u_lon_tor_error.at(index), 2.) + pow(u_lat_tor_error.at(index), 2.) );
                    KE_pot_coarse_error.at(index) = 0.5 * constants::rho0 * ( pow(u_lon_pot_error.at(index), 2.) + pow(u_lat_pot_error.at(index), 2.) );
                    KE_tot_coarse_error.at(index) = 0.5 * constants::rho0 * ( pow(u_lon_tot_error.at(index), 2.) + pow(u_lat_tot_error.at(index), 2.) );

                    KE_tor_fine.at(index) = KE_tor_filt.at(index) - KE_tor_coarse.at(index);
                    KE_pot_fine.at(index) = KE_pot_filt.at(index) - KE_pot_coarse.at(index);
                    KE_tot_fine.at(index) = KE_tot_filt.at(index) - KE_tot_coarse.at(index);

                    KE_tor_fine_2.at(index) = KE_tor_filt_2.at(index) - KE_tor_coarse_2.at(index);
                    KE_pot_fine_2.at(index) = KE_pot_filt_2.at(index) - KE_pot_coarse_2.at(index);
                    KE_tot_fine_2.at(index) = KE_tot_filt_2.at(index) - KE_tot_coarse_2.at(index);

                    KE_tor_fine_error.at(index) = KE_tor_filt_error.at(index) - KE_tor_coarse_error.at(index);
                    KE_pot_fine_error.at(index) = KE_pot_filt_error.at(index) - KE_pot_coarse_error.at(index);
                    KE_tot_fine_error.at(index) = KE_tot_filt_error.at(index) - KE_tot_coarse_error.at(index);

                    KE_tor_fine_mod.at(index) = KE_tor_orig.at(index) - KE_tor_coarse.at(index);
                    KE_pot_fine_mod.at(index) = KE_pot_orig.at(index) - KE_pot_coarse.at(index);
                    KE_tot_fine_mod.at(index) = KE_tot_orig.at(index) - KE_tot_coarse.at(index);

                    KE_tor_fine_mod_2.at(index) = KE_tor_orig_2.at(index) - KE_tor_coarse_2.at(index);
                    KE_pot_fine_mod_2.at(index) = KE_pot_orig_2.at(index) - KE_pot_coarse_2.at(index);
                    KE_tot_fine_mod_2.at(index) = KE_tot_orig_2.at(index) - KE_tot_coarse_2.at(index);

                    KE_tor_fine_mod_error.at(index) = KE_tor_orig_error.at(index) - KE_tor_coarse_error.at(index);
                    KE_pot_fine_mod_error.at(index) = KE_pot_orig_error.at(index) - KE_pot_coarse_error.at(index);
                    KE_tot_fine_mod_error.at(index) = KE_tot_orig_error.at(index) - KE_tot_coarse_error.at(index);

                    Enst_tor.at(index) = 0.5 * constants::rho0 * ( pow(vort_tor_r.at(index), 2.) );
                    Enst_pot.at(index) = 0.5 * constants::rho0 * ( pow(vort_pot_r.at(index), 2.) );
                    Enst_tot.at(index) = 0.5 * constants::rho0 * ( pow(vort_tot_r.at(index), 2.) );

                    Enst_tor_2.at(index) = 0.5 * constants::rho0 * ( pow(vort_tor_r_2.at(index), 2.) );
                    Enst_pot_2.at(index) = 0.5 * constants::rho0 * ( pow(vort_pot_r_2.at(index), 2.) );
                    Enst_tot_2.at(index) = 0.5 * constants::rho0 * ( pow(vort_tot_r_2.at(index), 2.) );

                    Enst_tor_error.at(index) = 0.5 * constants::rho0 * ( pow(vort_tor_r_error.at(index), 2.) );
                    Enst_pot_error.at(index) = 0.5 * constants::rho0 * ( pow(vort_pot_r_error.at(index), 2.) );
                    Enst_tot_error.at(index) = 0.5 * constants::rho0 * ( pow(vort_tot_r_error.at(index), 2.) );
                }
            }
        }
        if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "compute KE and Enstrophy"); }

        if (not(constants::NO_FULL_OUTPUTS)) {
            if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            write_field_to_output( KE_tor_filt, "KE_tor_filt", starts, counts, fname, &mask);
            write_field_to_output( KE_pot_filt, "KE_pot_filt", starts, counts, fname, &mask);
            write_field_to_output( KE_tot_filt, "KE_tot_filt", starts, counts, fname, &mask);

            write_field_to_output( KE_tor_filt_2, "KE_tor_filt_2", starts, counts, fname, &mask);
            write_field_to_output( KE_pot_filt_2, "KE_pot_filt_2", starts, counts, fname, &mask);
            write_field_to_output( KE_tot_filt_2, "KE_tot_filt_2", starts, counts, fname, &mask);

            write_field_to_output( KE_tor_filt_error, "KE_tor_filt_error", starts, counts, fname, &mask);
            write_field_to_output( KE_pot_filt_error, "KE_pot_filt_error", starts, counts, fname, &mask);
            write_field_to_output( KE_tot_filt_error, "KE_tot_filt_error", starts, counts, fname, &mask);

            write_field_to_output( KE_tor_fine, "KE_tor_fine", starts, counts, fname, &mask);
            write_field_to_output( KE_pot_fine, "KE_pot_fine", starts, counts, fname, &mask);
            write_field_to_output( KE_tot_fine, "KE_tot_fine", starts, counts, fname, &mask);

            write_field_to_output( KE_tor_fine_2, "KE_tor_fine_2", starts, counts, fname, &mask);
            write_field_to_output( KE_pot_fine_2, "KE_pot_fine_2", starts, counts, fname, &mask);
            write_field_to_output( KE_tot_fine_2, "KE_tot_fine_2", starts, counts, fname, &mask);

            write_field_to_output( KE_tor_fine_error, "KE_tor_fine_error", starts, counts, fname, &mask);
            write_field_to_output( KE_pot_fine_error, "KE_pot_fine_error", starts, counts, fname, &mask);
            write_field_to_output( KE_tot_fine_error, "KE_tot_fine_error", starts, counts, fname, &mask);
            if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "writing");  }
        }

        if (not(constants::MINIMAL_OUTPUT)) {
            if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
            write_field_to_output( KE_tor_fine_mod, "KE_tor_fine_mod", starts, counts, fname, &mask);
            write_field_to_output( KE_pot_fine_mod, "KE_pot_fine_mod", starts, counts, fname, &mask);
            write_field_to_output( KE_tot_fine_mod, "KE_tot_fine_mod", starts, counts, fname, &mask);

            write_field_to_output( KE_tor_fine_mod_2, "KE_tor_fine_mod_2", starts, counts, fname, &mask);
            write_field_to_output( KE_pot_fine_mod_2, "KE_pot_fine_mod_2", starts, counts, fname, &mask);
            write_field_to_output( KE_tot_fine_mod_2, "KE_tot_fine_mod_2", starts, counts, fname, &mask);

            write_field_to_output( KE_tor_fine_mod_error, "KE_tor_fine_mod_error", starts, counts, fname, &mask);
            write_field_to_output( KE_pot_fine_mod_error, "KE_pot_fine_mod_error", starts, counts, fname, &mask);
            write_field_to_output( KE_tot_fine_mod_error, "KE_tot_fine_mod_error", starts, counts, fname, &mask);

            write_field_to_output( Enst_tor, "Enstrophy_tor", starts, counts, fname, &mask);
            write_field_to_output( Enst_pot, "Enstrophy_pot", starts, counts, fname, &mask);
            write_field_to_output( Enst_tot, "Enstrophy_tot", starts, counts, fname, &mask);

            write_field_to_output( Enst_tor_2, "Enstrophy_tor_2", starts, counts, fname, &mask);
            write_field_to_output( Enst_pot_2, "Enstrophy_pot_2", starts, counts, fname, &mask);
            write_field_to_output( Enst_tot_2, "Enstrophy_tot_2", starts, counts, fname, &mask);

            write_field_to_output( Enst_tor_error, "Enstrophy_tor_error", starts, counts, fname, &mask);
            write_field_to_output( Enst_pot_error, "Enstrophy_pot_error", starts, counts, fname, &mask);
            write_field_to_output( Enst_tot_error, "Enstrophy_tot_error", starts, counts, fname, &mask);

            write_field_to_output( vort_tor_r, "vort_r_tor", starts, counts, fname, &mask);
            write_field_to_output( vort_pot_r, "vort_r_pot", starts, counts, fname, &mask);
            write_field_to_output( vort_tot_r, "vort_r_tot", starts, counts, fname, &mask);

            write_field_to_output( vort_tor_r_2, "vort_r_tor_2", starts, counts, fname, &mask);
            write_field_to_output( vort_pot_r_2, "vort_r_pot_2", starts, counts, fname, &mask);
            write_field_to_output( vort_tot_r_2, "vort_r_tot_2", starts, counts, fname, &mask);

            write_field_to_output( vort_tor_r_error, "vort_r_tor_error", starts, counts, fname, &mask);
            write_field_to_output( vort_pot_r_error, "vort_r_pot_error", starts, counts, fname, &mask);
            write_field_to_output( vort_tot_r_error, "vort_r_tot_error", starts, counts, fname, &mask);
            if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "writing");  }
        }
        
        
        // Compute some wind-based terms, if applicable
        if ( constants::COMP_WIND_FORCE ){ 
            toroidal_vel_from_F(  tau_wind_x_tor, tau_wind_y_tor, coarse_wind_tau_Psi, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask );
            potential_vel_from_F( tau_wind_x_pot, tau_wind_y_pot, coarse_wind_tau_Phi, longitude, latitude, Ntime, Ndepth, Nlat, Nlon, mask );

            #pragma omp parallel \
            default( none ) \
            shared( coarse_tau_wind_dot_u_tor, coarse_tau_wind_dot_u_pot, mask, \
                    tau_wind_x_tor, tau_wind_y_tor, tau_wind_x_pot, tau_wind_y_pot, \
                    local_wind_forcing_tor, local_wind_forcing_pot, local_wind_forcing_tot, \
                    u_lon_tor, u_lat_tor, u_lon_pot, u_lat_pot ) \
            private( index )
            {
                #pragma omp for collapse(1) schedule(guided)
                for (index = 0; index < u_lon_tor.size(); ++index) {

                    if ( mask.at(index) ) { 
                        local_wind_forcing_tor.at(index) = 
                                                  coarse_tau_wind_dot_u_tor.at( index ) - 
                                                  (   u_lon_tor.at(index) * ( tau_wind_x_tor.at( index ) + tau_wind_x_pot.at( index ) )
                                                    + u_lat_tor.at(index) * ( tau_wind_y_tor.at( index ) + tau_wind_y_pot.at( index ) )
                                                  );
                        local_wind_forcing_pot.at(index) = 
                                                  coarse_tau_wind_dot_u_pot.at( index ) - 
                                                  (   u_lon_pot.at(index) * ( tau_wind_x_tor.at( index ) + tau_wind_x_pot.at( index ) )
                                                    + u_lat_pot.at(index) * ( tau_wind_y_tor.at( index ) + tau_wind_y_pot.at( index ) )
                                                  );

                        local_wind_forcing_tot.at( index ) = local_wind_forcing_tor.at( index ) + local_wind_forcing_pot.at( index );
                    }
                }
            }

            if (not(constants::NO_FULL_OUTPUTS)) {
                if (constants::DO_TIMING) { clock_on = MPI_Wtime(); }
                write_field_to_output( local_wind_forcing_tor, "local_wind_forcing_tor", starts, counts, fname, &mask);
                write_field_to_output( local_wind_forcing_pot, "local_wind_forcing_pot", starts, counts, fname, &mask);

                write_field_to_output( coarse_tau_wind_dot_u_tor, "tau_wind_dot_u_tor", starts, counts, fname, &mask );
                write_field_to_output( coarse_tau_wind_dot_u_pot, "tau_wind_dot_u_pot", starts, counts, fname, &mask );
                if (constants::DO_TIMING) { timing_records.add_to_record(MPI_Wtime() - clock_on, "writing");  }
            }
        }

        //
        //// on-line postprocessing, if desired
        //

        if (constants::APPLY_POSTPROCESS) {
            MPI_Barrier(MPI_COMM_WORLD);

            // If we're doing post-processing, then spectral slopes need to be scaled by
            //   the spectrum. We scale it in here. The Postprocessing routine then uses the
            //   spectrum-weighted normalization factors
            #pragma omp parallel default(none) \
            private(index) \
            shared( spec_slope_tot, u_spectrum_tot, v_spectrum_tot, \
                    spec_slope_pot, u_spectrum_pot, v_spectrum_pot, \
                    spec_slope_tor, u_spectrum_tor, v_spectrum_tor, \
                    spec_slope_tot_2, u_spectrum_tot_2, v_spectrum_tot_2, \
                    spec_slope_pot_2, u_spectrum_pot_2, v_spectrum_pot_2, \
                    spec_slope_tor_2, u_spectrum_tor_2, v_spectrum_tor_2, \
                    spec_slope_tot_error, u_spectrum_tot_error, v_spectrum_tot_error, \
                    spec_slope_pot_error, u_spectrum_pot_error, v_spectrum_pot_error, \
                    spec_slope_tor_error, u_spectrum_tor_error, v_spectrum_tor_error, \
                    mask )
            {
                #pragma omp for schedule(static)
                for ( index = 0; index < spec_slope_tot.size(); index++ ) {
                    if ( mask.at(index) ) {
                        spec_slope_tot.at(index) *= u_spectrum_tot.at(index) + v_spectrum_tot.at(index);
                        spec_slope_pot.at(index) *= u_spectrum_pot.at(index) + v_spectrum_pot.at(index);
                        spec_slope_tor.at(index) *= u_spectrum_tor.at(index) + v_spectrum_tor.at(index);
                        spec_slope_tot_2.at(index) *= u_spectrum_tot_2.at(index) + v_spectrum_tot_2.at(index);
                        spec_slope_pot_2.at(index) *= u_spectrum_pot_2.at(index) + v_spectrum_pot_2.at(index);
                        spec_slope_tor_2.at(index) *= u_spectrum_tor_2.at(index) + v_spectrum_tor_2.at(index);
                        spec_slope_tot_error.at(index) *= u_spectrum_tot_error.at(index) + v_spectrum_tot_error.at(index);
                        spec_slope_pot_error.at(index) *= u_spectrum_pot_error.at(index) + v_spectrum_pot_error.at(index);
                        spec_slope_tor_error.at(index) *= u_spectrum_tor_error.at(index) + v_spectrum_tor_error.at(index);
                    }
                }
            }

            #if DEBUG >= 1
            if (wRank == 0) { fprintf(stdout, "Beginning post-process routines\n"); }
            fflush(stdout);
            #endif

            Apply_Postprocess_Routines(
                    source_data, postprocess_fields_tor, postprocess_names, OkuboWeiss_tor,
                    scales.at(Iscale), timing_records, "postprocess_toroidal");

            Apply_Postprocess_Routines(
                    source_data, postprocess_fields_pot, postprocess_names, OkuboWeiss_pot,
                    scales.at(Iscale), timing_records, "postprocess_potential");

            Apply_Postprocess_Routines(
                    source_data, postprocess_fields_tot, postprocess_names, OkuboWeiss_tot,
                    scales.at(Iscale), timing_records, "postprocess_full");

            #if DEBUG >= 1
            if (wRank == 0) { fprintf(stdout, "Finished post-process routines\n"); }
            fflush(stdout);
            #endif

        }

        #if DEBUG >= 0
        // Flushing stdout is necessary for SLURM outputs.
        fflush(stdout);
        #endif

        // If we're doing timings, then print out and reset values now
        if (constants::DO_TIMING) { 
            timing_records.print();
            timing_records.reset();
            fflush(stdout);
        }

    }  // end for(scale) block
} // end filtering
