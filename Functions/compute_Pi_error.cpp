#include <vector>
#include <omp.h>
#include "../functions.hpp"
#include "../constants.hpp"
#include "../differentiation_tools.hpp"

/*!
 * \brief Compute the error energy transfer through the current filter scale
 *
 * In particular, computes
 *
 * \f[
 *     \Pi^{(e)}_\ell
 *     =
 *     - \rho_0 \, \overline{S}_{e,ij}
 *     \, \Delta \overline{\tau}_{\ell,ij}
 * \f]
 *
 * where
 *
 * \f[
 *     \overline{S}_{e,ij}
 *     =
 *     \tfrac12 \left(
 *         \partial_j \overline{e}_i
 *         +
 *         \partial_i \overline{e}_j
 *     \right)
 * \f]
 *
 * is the strain-rate tensor of the filtered error velocity
 *
 * \f[
 *     \overline{\mathbf{e}}
 *     =
 *     \overline{\mathbf{u}}_1
 *     -
 *     \overline{\mathbf{u}}_2
 * \f]
 *
 * and
 *
 * \f[
 *     \Delta \overline{\tau}_{\ell,ij}
 *     =
 *     \overline{\tau}_{\ell,ij}(\mathbf{u}_1,\mathbf{u}_1)
 *     -
 *     \overline{\tau}_{\ell,ij}(\mathbf{u}_2,\mathbf{u}_2).
 * \f]
 *
 * The subfilter stress for each field is defined as
 *
 * \f[
 *     \overline{\tau}_{\ell,ij}
 *     =
 *     \overline{u_i u_j}
 *     -
 *     \overline{u}_i \, \overline{u}_j.
 * \f]
 *
 * This computation is applied to the filtered Cartesian velocity components.
 *
 * @param[in,out]   energy_transfer                 where to store the computed values (array)
 * @param[in]       source_data                     dataset class instance containing data (Psi, Phi, etc)
 * @param[in]       ux,uy,uz                        coarse Cartesian first field velocity components
 * @param[in]       uxux,uxuy,uxuz,uyuy,uyuz,uzuz   coarse first field velocity products (e.g. bar(u*v) )  
 * @param[in]       ux_2,uy_2,uz_2                  coarse Cartesian second field velocity components
 * @param[in]       uxux_2,uxuy_2,uxuz_2,uyuy_2,uyuz_2,uzuz_2   coarse second field velocity products (e.g. bar(u*v) )  
 * @param[in]       comm                            MPI communicator object
 *
 */
void compute_Pi_error(
        std::vector<double> & energy_transfer,
        const dataset & source_data,
        const std::vector<double> & ux,
        const std::vector<double> & uy,
        const std::vector<double> & uz,
        const std::vector<double> & uxux,
        const std::vector<double> & uxuy,
        const std::vector<double> & uxuz,
        const std::vector<double> & uyuy,
        const std::vector<double> & uyuz,
        const std::vector<double> & uzuz,
        const std::vector<double> & ux_2,
        const std::vector<double> & uy_2,
        const std::vector<double> & uz_2,
        const std::vector<double> & uxux_2,
        const std::vector<double> & uxuy_2,
        const std::vector<double> & uxuz_2,
        const std::vector<double> & uyuy_2,
        const std::vector<double> & uyuz_2,
        const std::vector<double> & uzuz_2,
        const std::vector<double> * ux_in_tau,
        const std::vector<double> * uy_in_tau,
        const std::vector<double> * uz_in_tau,
        const std::vector<double> * ux_in_tau_2,
        const std::vector<double> * uy_in_tau_2,
        const std::vector<double> * uz_in_tau_2,
        const MPI_Comm comm
        ) {

    const std::vector<bool> &mask = source_data.mask;

    #if DEBUG >= 2
    int wRank, wSize;
    MPI_Comm_rank( comm, &wRank );
    MPI_Comm_size( comm, &wSize );

    if (wRank == 0) { fprintf(stdout, "  Starting Pi_e computation.\n"); }
    #endif

    double pi_tmp;
    int Itime, Idepth, Ilat, Ilon, ii, jj;
    size_t index;
    const size_t Npts = energy_transfer.size();

    std::vector<double> ux_error(Npts), uy_error(Npts), uz_error(Npts);
    for (index = 0; index < Npts; ++index) {
        ux_error[index] = ux[index]  - ux_2[index];
        uy_error[index] = uy[index]  - uy_2[index];
        uz_error[index] = uz[index]  - uz_2[index];
    }

    double ui_j_error, uj_i_error;
    std::vector<double> Delta_tau_ij( Npts );

    // Some convenience handles
    //   note: the pointers aren't constant, but the things
    //         to which they are pointing are
    double ui_loc, uj_loc, uiuj_loc, tau_loc;
    const std::vector<double> *uiuj, *ui, *uj, *ui_in_tau, *uj_in_tau;

    double ui_loc_2, uj_loc_2, uiuj_loc_2, tau_loc_2;
    const std::vector<double> *uiuj_2, *ui_2, *uj_2, *ui_in_tau_2, *uj_in_tau_2;

    const std::vector<double> *ui_error, *uj_error;

    //const std::vector<double> *ux_in_tau = ( ux_for_tau == NULL ) ? &ux[0] : ux_for_tau;
    //const std::vector<double> *uy_in_tau = ( uy_for_tau == NULL ) ? &uy[0] : uy_for_tau;
    //const std::vector<double> *uz_in_tau = ( uz_for_tau == NULL ) ? &uz[0] : uz_for_tau;
    if ( ux_in_tau == NULL ) { ux_in_tau = &ux; }
    if ( uy_in_tau == NULL ) { uy_in_tau = &uy; }
    if ( uz_in_tau == NULL ) { uz_in_tau = &uz; }

    if ( ux_in_tau_2 == NULL ) { ux_in_tau_2 = &ux_2; }
    if ( uy_in_tau_2 == NULL ) { uy_in_tau_2 = &uy_2; }
    if ( uz_in_tau_2 == NULL ) { uz_in_tau_2 = &uz_2; }

    // Set up the derivatives to pass through the differentiation functions

    std::vector<double*> x_deriv_vals_error, y_deriv_vals_error, z_deriv_vals_error;
    std::vector<const std::vector<double>*> deriv_fields_error(2);

    // Zero out energy transfer before we start
    std::fill( energy_transfer.begin(), energy_transfer.end(), 0.);



    for (ii = 0; ii < 3; ii++) {
        for (jj = 0; jj < 3; jj++) {

            //   Assign some handy pointers: uiuj, ui, uj
            //
            //   0 -> x
            //   1 -> y
            //   2 -> z

            // ui
            ui = ( ii == 0 ) ? &ux : ( ii == 1 ) ? &uy : &uz;
            uj = ( jj == 0 ) ? &ux : ( jj == 1 ) ? &uy : &uz;

            ui_in_tau = ( ii == 0 ) ? ux_in_tau : ( ii == 1 ) ? uy_in_tau : uz_in_tau;
            uj_in_tau = ( jj == 0 ) ? ux_in_tau : ( jj == 1 ) ? uy_in_tau : uz_in_tau;

            // ui_2
            ui_2 = ( ii == 0 ) ? &ux_2 : ( ii == 1 ) ? &uy_2 : &uz_2;
            uj_2 = ( jj == 0 ) ? &ux_2 : ( jj == 1 ) ? &uy_2 : &uz_2;

            ui_in_tau_2 = ( ii == 0 ) ? ux_in_tau_2 : ( ii == 1 ) ? uy_in_tau_2 : uz_in_tau_2;
            uj_in_tau_2 = ( jj == 0 ) ? ux_in_tau_2 : ( jj == 1 ) ? uy_in_tau_2 : uz_in_tau_2;

            // ui_error

            ui_error = ( ii == 0 ) ? &ux_error : ( ii == 1 ) ? &uy_error : &uz_error;
            uj_error = ( jj == 0 ) ? &ux_error : ( jj == 1 ) ? &uy_error : &uz_error;

            deriv_fields_error[0] = ui_error;
            deriv_fields_error[1] = uj_error;

            // uiuj (note that they're symmetric i.e. uiuj = ujui)
            uiuj =  ( ii == 0 ) ? ( ( jj == 0 ) ? &uxux : ( jj == 1 ) ? &uxuy : &uxuz ) :
                    ( ii == 1 ) ? ( ( jj == 0 ) ? &uxuy : ( jj == 1 ) ? &uyuy : &uyuz ) :
                                  ( ( jj == 0 ) ? &uxuz : ( jj == 1 ) ? &uyuz : &uzuz ) ;

            // uiuj_2
            uiuj_2 =  ( ii == 0 ) ? ( ( jj == 0 ) ? &uxux_2 : ( jj == 1 ) ? &uxuy_2 : &uxuz_2 ) :
                      ( ii == 1 ) ? ( ( jj == 0 ) ? &uxuy_2 : ( jj == 1 ) ? &uyuy_2 : &uyuz_2 ) :
                                    ( ( jj == 0 ) ? &uxuz_2 : ( jj == 1 ) ? &uyuz_2 : &uzuz_2 ) ;

            // Compute Delta_tau_ij 
            #pragma omp parallel \
            default(none) \
            shared(mask, ui_in_tau, uj_in_tau, uiuj, source_data,\
                   ui_in_tau_2, uj_in_tau_2, uiuj_2, Delta_tau_ij) \
            private(index, uiuj_loc, ui_loc, uj_loc, tau_loc, uiuj_loc_2, ui_loc_2, uj_loc_2, tau_loc_2) \
            firstprivate( Npts )
            {
                #pragma omp for collapse(1) schedule(guided)
                for (index = 0; index < Npts; index++) {

                    if ( constants::FILTER_OVER_LAND or mask.at(index) ) {
                        ui_loc   = ui_in_tau->at(  index);
                        uj_loc   = uj_in_tau->at(  index);
                        uiuj_loc = uiuj->at(index);
                        
                        tau_loc = uiuj_loc - ui_loc * uj_loc;

                        ui_loc_2   = ui_in_tau_2->at(  index);
                        uj_loc_2   = uj_in_tau_2->at(  index);
                        uiuj_loc_2 = uiuj_2->at(index);

                        tau_loc_2 = uiuj_loc_2 - ui_loc_2 * uj_loc_2;

                        Delta_tau_ij.at(index) = tau_loc - tau_loc_2;
                    }
                }
            }

            #pragma omp parallel default(none) \
            shared( source_data, energy_transfer, mask, \
                    ii, jj, ui_error, uj_error, Delta_tau_ij, deriv_fields_error)\
            private( Itime, Idepth, Ilat, Ilon, index, pi_tmp, ui_j_error, uj_i_error, \
                     x_deriv_vals_error, y_deriv_vals_error, z_deriv_vals_error) \
            firstprivate( Npts )
            {

                x_deriv_vals_error.resize(2);
                y_deriv_vals_error.resize(2);
                z_deriv_vals_error.resize(2);

                // Now set the appropriate derivative pointers
                //   in order to compute
                //   ui,j and uj,i
                x_deriv_vals_error.at(0) = (jj == 0) ? &ui_j_error : NULL;
                x_deriv_vals_error.at(1) = (ii == 0) ? &uj_i_error : NULL;

                y_deriv_vals_error.at(0) = (jj == 1) ? &ui_j_error : NULL;
                y_deriv_vals_error.at(1) = (ii == 1) ? &uj_i_error : NULL;

                z_deriv_vals_error.at(0) = (jj == 2) ? &ui_j_error : NULL;
                z_deriv_vals_error.at(1) = (ii == 2) ? &uj_i_error : NULL;

                // Now actually compute Pi_e
                //   in particular, compute S_ij_e * Delta tau_ij
                #pragma omp for collapse(1) schedule(guided)
                for (index = 0; index < Npts; index++) {

                    if ( constants::FILTER_OVER_LAND or mask.at(index) ) {

                        source_data.index1to4_local( index, Itime, Idepth, Ilat, Ilon);

                        // Compute the desired derivatives
                        Cart_derivatives_at_point(
                                x_deriv_vals_error, y_deriv_vals_error, z_deriv_vals_error, deriv_fields_error, 
                                source_data, Itime, Idepth, Ilat, Ilon,
                                1, constants::DiffOrd);

                        double Sij_e = 0.5 * ( ui_j_error + uj_i_error );
                        pi_tmp = - constants::rho0 * Sij_e * Delta_tau_ij.at(index);
                        energy_transfer.at(index) += pi_tmp;
                    }
                }
            }
        }
    }
    #if DEBUG >= 2
    if (wRank == 0) { fprintf(stdout, "     ... done.\n"); }
    #endif
}
