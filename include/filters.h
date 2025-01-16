#pragma once

#include <memory>
#include <string>

namespace dendroderivs {

enum FilterType { F_NONE = 0, F_KO4, F_KO6, F_KO8 };

class Filters {
   protected:
    unsigned int p_n;
    unsigned int p_pw;
    unsigned int p_ele_order;

    Filters(unsigned int ele_order) : p_ele_order{ele_order} {
        p_n  = p_ele_order * 2 + 1;
        p_pw = p_ele_order / 2;
    }

    Filters(const Filters &obj) {};

   public:
    virtual ~Filters() {};

    virtual std::unique_ptr<Filters> clone() const         = 0;

    virtual void do_full_filter(const double *const input, double *const output,
                                double *const workspace_x,
                                double *const workspace_y,
                                double *const worksapce_z, const double dx,
                                const double dy, const double dz,
                                const double coeff, const unsigned int *sz,
                                const unsigned int bflag)  = 0;

    virtual std::string toString() const                   = 0;

    virtual bool do_filter_before() const                  = 0;

    virtual void set_maximum_block_size(size_t block_size) = 0;
};

}  // namespace dendroderivs
