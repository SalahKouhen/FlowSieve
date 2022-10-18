#include <vector>
#include <string>
#include <assert.h>
#include <math.h>
#include "../../differentiation_tools.hpp"
#include "../../constants.hpp"
#include "../../functions.hpp"

void get_diff_vector(
        std::vector<double> & diff_vector,
        int & LB_ret,
        const std::vector<double> & grid,
        const std::string & dim,
        const int Itime,
        const int Idepth,
        const int Ilat,
        const int Ilon,
        const int Ntime,
        const int Ndepth,
        const int Nlat,
        const int Nlon,
        const std::vector<bool> & mask,
        const int order_of_deriv,
        const int diff_ord
        ) {

    // Check which derivative we're taking
    int index;
    const bool do_dep = (dim == "depth");
    const bool do_lat = (dim == "lat");
    const bool do_lon = (dim == "lon");
    assert( do_dep ^ (do_lat ^ do_lon) ); // ^ = xor

    int Iref = do_dep ? Idepth :
               do_lat ? Ilat :
               do_lon ? Ilon : -1;
    const int Nref = grid.size();

    // Determine lowest lower bound (LLB) and upperest upper bound (UUB)
    //   for the integration region. This essentially just depends on periodicity.
    const bool periodic = do_dep ? false : 
                          do_lat ? constants::PERIODIC_Y : 
                          do_lon ? constants::PERIODIC_X : false;
    const int LLB = periodic ? Iref - Nref : 0 ;
    const int UUB = periodic ? Iref + Nref : Nref - 1 ;

    // Differentiation vector
    const int num_deriv_pts = diff_ord + order_of_deriv;
    std::vector<double> ddl(num_deriv_pts);

    // Assuming uniform grid
    const double dl = grid.at(1) - grid.at(0);

    // Build outwards to try and build the stencil, but stop when
    //   we either hit a land cell or have gone far enough.
    // lb (lower case) will be the periodicity-adjusted value of LB 
    int lb, LB = Iref;
    while (LB > LLB) {

        if ( (Iref - LB) >= num_deriv_pts ) { break; }
       
        // Check if the next point would be land
        lb = ( ( LB - 1 ) % Nref + Nref ) % Nref;
        index = Index( Itime, do_dep ? lb : Idepth, do_lat ? lb : Ilat, do_lon ? lb : Ilon,
                       Ntime, Ndepth,               Nlat,               Nlon );
        
        if ( mask.at(index) )   { LB--;  }  // If next point is still water, extend stencil over it
        else                    { break; }  // Otherwise, halt [ without extending stencil ]

    }

    // ub (lower case) will be the periodicity-adjusted value of UB 
    int UB = Iref, ub;
    while (UB < UUB) {
        if ( (UB - Iref) >= num_deriv_pts ) { break; }
       
        // Check if the next point would be land
        ub = ( ( UB + 1 ) % Nref + Nref ) % Nref;
        index = Index( Itime, do_dep ? ub : Idepth, do_lat ? ub : Ilat, do_lon ? ub : Ilon,
                       Ntime, Ndepth,               Nlat,               Nlon );

        if ( mask.at(index) )   { UB++;  }  // If next point is still water, extend stencil over it
        else                    { break; }  // Otherwise, halt [ without extending stencil ]
    }

    // NOTE
    //   In the case of periodicity, LB may be negative and UB may exceed Nref
    //     this means that subtraction still gives the correct number of points
    //     in the stencil, but that a periodicity-adjusted value will be needed
    //     when determining logical indices.

    // We've possibly made too large of a stencil, so now collapse it back down
    while (UB - LB + 1 > num_deriv_pts) {
        if ((UB - Iref > Iref - LB) and (UB >= Iref)) { UB--; }
        else { LB++; }
    }

    // We're including LB and UB in our stencil, so the stencil
    //   has UB - LB + 1 points. The requisit number of points is
    //   num_deriv_pts.
    if (UB - LB + 1 == num_deriv_pts) {

        // If we have enough cells for differentiation, do it
        if ( do_lon or (do_lat and constants::UNIFORM_LAT_GRID)) {
            // Since we're on a uniform grid, we can use pre-computed
            //   differentiation coefficients
            differentiation_vector(ddl, dl, Iref - LB, order_of_deriv, diff_ord);
        } else {
            // We're on a non-uniform grid, so we can't guarantee the
            //   differentiation coefficients a priori, so we need
            //   to actually compute them now.
            // This will get expensive (or ugly...) for higher orders of accuracy.
            // NOTE: This CANNOT handle periodicity
            non_uniform_diff_vector(ddl, grid, Iref, LB, UB, diff_ord);
        }

        diff_vector.clear();
        for (int IND = LB; IND <= UB; IND++) {
            diff_vector.push_back( ddl.at(IND - LB) );
        }

        LB_ret = LB;

    } else if (diff_ord > 2) {

        // If we couldn't build a large enough stencil, then 
        //   try again with a lower order. This will allow us
        //   to fill in smaller areas with at least something,
        //   if not the most accurate something.
        get_diff_vector(
                diff_vector, LB_ret, grid, dim,
                Itime, Idepth, Ilat, Ilon,
                Ntime, Ndepth, Nlat, Nlon,
                mask, order_of_deriv, diff_ord - 2);

    } else {

        // Some back-up plans, in the case nothing else worked
        diff_vector.clear();

        // Back-up plan 1: first derivative with two points?
        //                 then use a first-order derivative
        //                 (not great, but better than nothing (hopefully))
        if ( (order_of_deriv == 1) and (UB - LB + 1 == 2)) {
            diff_vector.push_back( -1. / dl );
            diff_vector.push_back(  1. / dl );
            LB_ret = LB;
        }

        // Back-up plan 2: second derivative with only three points?
        //                 assume the second derivative is constant on those
        //                 three points, and use a classic (1, -2, 1) stencil
        //                 (not great, but better than nothing (hopefully))
        if ( (order_of_deriv == 2) and (UB - LB + 1 == 3)) {
            // For now require uniform grid
            if ( do_lon or (do_lat and constants::UNIFORM_LAT_GRID)) {
                diff_vector.push_back(  1. / pow(dl, 2.) );
                diff_vector.push_back( -2. / pow(dl, 2.) );
                diff_vector.push_back(  1. / pow(dl, 2.) );
                LB_ret = LB;
            }
        }
    }

}
